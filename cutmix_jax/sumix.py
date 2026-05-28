import jax
import jax.numpy as jnp
import jax.nn as jnn


def cross_entropy_with_integer_labels(logits, labels):
    log_probs = jnn.log_softmax(logits, axis=-1)
    batch_indices = jnp.arange(labels.shape[0])
    return -log_probs[batch_indices, labels]


def l2_normalize(x, axis=-1, eps=1e-8):
    norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


def gather_by_label(values, labels):
    batch_indices = jnp.arange(labels.shape[0])
    return values[batch_indices, labels]


def zero_out_label(prob, labels):
    """
    Official-style semantic masking.

    For each sample i:
        prob[i, labels[i]] = 0

    Matches official SUMix:
        semantic_one[i, y_b[i]] = 0
        semantic_one_[i, y_a[i]] = 0
    """
    batch_indices = jnp.arange(labels.shape[0])
    return prob.at[batch_indices, labels].set(0.0)


def estimate_uncertainty(uncertain_logits):
    """
    Official uncertainty branch:
        softmax -> l2_norm
    """
    uncertain_prob = jnn.softmax(uncertain_logits, axis=-1)
    uncertain_prob = l2_normalize(uncertain_prob, axis=-1)
    return uncertain_prob


def estimate_semantic_information(cls_logits):
    """Semantic information from classifier logits."""
    return jnn.softmax(cls_logits, axis=-1)


def estimate_mixup_ratio(
    cls_one,
    uncertain_one,
    cls_mix,
    uncertain_mix,
    labels_a,
    labels_b,
    perm,
    lam_area,
    alpha_scale: float = 1.0,
):
    """
    Official-alignment SUMix ratio correction with tunable alpha scaling.

    Official SUMix uses alpha_scale = batch_size.
    Stable JAX adaptation uses a smaller alpha_scale, e.g. 4.
    """
    batch_size = labels_a.shape[0]

    if jnp.ndim(lam_area) == 0:
        lam_area = jnp.ones((batch_size,), dtype=cls_mix.dtype) * lam_area
    lam_area = lam_area.reshape(-1)

    semantic_one = estimate_semantic_information(cls_one)
    semantic_mix = estimate_semantic_information(cls_mix)

    # Official uses cls_one.clone().detach() and cls_mix.clone().detach().
    semantic_one = jax.lax.stop_gradient(semantic_one)
    semantic_mix = jax.lax.stop_gradient(semantic_mix)

    semantic_b = semantic_one[perm]

    semantic_one_masked = zero_out_label(semantic_one, labels_b)
    semantic_b_masked = zero_out_label(semantic_b, labels_a)

    # Raw semantic discrepancy before alpha scaling.
    raw_alpha_a = l2_normalize(
        jnn.softmax(semantic_mix - semantic_one_masked, axis=-1),
        axis=-1,
    )

    raw_alpha_b = l2_normalize(
        jnn.softmax(semantic_mix - semantic_b_masked, axis=-1),
        axis=-1,
    )

    # Scaled semantic discrepancy.
    # Official-equivalent: alpha_scale = batch_size.
    # Stable JAX adaptation: alpha_scale = 4, 8, 32, etc.
    alpha_a = raw_alpha_a * alpha_scale
    alpha_b = raw_alpha_b * alpha_scale

    uncertain_one = estimate_uncertainty(uncertain_one)
    uncertain_mix = estimate_uncertainty(uncertain_mix)
    uncertain_b = uncertain_one[perm]

    beta_a = uncertain_one + uncertain_mix
    beta_b = uncertain_b + uncertain_mix

    info_a = jnp.exp(-alpha_a)
    info_b = jnp.exp(-alpha_b)

    info_a_y = gather_by_label(info_a, labels_a)
    info_b_y = gather_by_label(info_b, labels_b)

    lam_a = lam_area * info_a_y
    lam_b = (1.0 - lam_area) * info_b_y

    # Official code does not add epsilon, but JAX/TPU can underflow to zero.
    # This tiny epsilon prevents NaNs while preserving the diagnostic behavior.
    lam_sumix = lam_a / (lam_a + lam_b + 1e-12)
    lam_sumix = jnp.clip(lam_sumix, 0.0, 1.0)

    return lam_sumix, {
        "raw_alpha_a": raw_alpha_a,
        "raw_alpha_b": raw_alpha_b,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "beta_a": beta_a,
        "beta_b": beta_b,
        "info_a": info_a,
        "info_b": info_b,
    }


def sumix_loss(
    cls_one,
    uncertain_one,
    cls_mix,
    uncertain_mix,
    labels_a,
    labels_b,
    perm,
    lam_area,
    gamma: float = 0.1,
    alpha_scale: float = 1.0,
):
    """
    SUMix loss with raw-alpha diagnostics.

    Main reproduction behavior:
    - Keeps original scalar CE reduction before lambda weighting.
    - Keeps raw_alpha logging for diagnosing alpha scaling collapse.
    - Uses exp(-alpha) for semantic information weights.
    - Uses exp(-(alpha + beta)) for uncertainty regularization.
    """
    lam_sumix, ratio_info = estimate_mixup_ratio(
        cls_one=cls_one,
        uncertain_one=uncertain_one,
        cls_mix=cls_mix,
        uncertain_mix=uncertain_mix,
        labels_a=labels_a,
        labels_b=labels_b,
        perm=perm,
        lam_area=lam_area,
        alpha_scale=alpha_scale,
    )

    if jnp.ndim(lam_area) == 0:
        lam_area = jnp.ones_like(lam_sumix) * lam_area
    lam_area = lam_area.reshape(-1)

    # Original scalar CE reduction behavior.
    ce_a_scalar = jnp.mean(
        cross_entropy_with_integer_labels(cls_mix, labels_a)
    )
    ce_b_scalar = jnp.mean(
        cross_entropy_with_integer_labels(cls_mix, labels_b)
    )

    cls_loss = jnp.mean(
        ce_a_scalar * lam_sumix + ce_b_scalar * (1.0 - lam_sumix)
    )

    # Official regularization path: INa_f = exp(-(beta + alpha)).
    reg_logits_a = jnp.exp(-(
        ratio_info["alpha_a"] + ratio_info["beta_a"]
    ))
    reg_logits_b = jnp.exp(-(
        ratio_info["alpha_b"] + ratio_info["beta_b"]
    ))

    # Original scalar CE reduction behavior for regularization.
    reg_ce_a_scalar = jnp.mean(
        cross_entropy_with_integer_labels(reg_logits_a, labels_a)
    )
    reg_ce_b_scalar = jnp.mean(
        cross_entropy_with_integer_labels(reg_logits_b, labels_b)
    )

    reg_loss = jnp.mean(
        reg_ce_a_scalar * lam_area + reg_ce_b_scalar * (1.0 - lam_area)
    )

    total_loss = cls_loss + gamma * reg_loss

    return total_loss, {
        "lam": jnp.mean(lam_sumix),
        "lam_min": jnp.min(lam_sumix),
        "lam_max": jnp.max(lam_sumix),
        "lam_std": jnp.std(lam_sumix),
        "cls_loss": cls_loss,
        "reg_loss": reg_loss,
        "total_loss": total_loss,

        # Raw alpha before alpha_scale.
        "raw_alpha_a_y_mean": jnp.mean(
            gather_by_label(ratio_info["raw_alpha_a"], labels_a)
        ),
        "raw_alpha_b_y_mean": jnp.mean(
            gather_by_label(ratio_info["raw_alpha_b"], labels_b)
        ),

        # Scaled alpha after alpha_scale.
        "alpha_a_y_mean": jnp.mean(
            gather_by_label(ratio_info["alpha_a"], labels_a)
        ),
        "alpha_b_y_mean": jnp.mean(
            gather_by_label(ratio_info["alpha_b"], labels_b)
        ),

        # Information weights.
        "info_a_y_mean": jnp.mean(
            gather_by_label(ratio_info["info_a"], labels_a)
        ),
        "info_b_y_mean": jnp.mean(
            gather_by_label(ratio_info["info_b"], labels_b)
        ),
    }