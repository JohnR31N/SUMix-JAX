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

    This matches the idea in official SUMix:
        semantic_one[i, y_b[i]] = 0
        semantic_one_[i, y_a[i]] = 0
    """
    batch_indices = jnp.arange(labels.shape[0])
    return prob.at[batch_indices, labels].set(0.0)


def estimate_uncertainty(uncertain_logits):
    """
    Uncertainty estimation module.

    Official-style:
        softmax -> l2_norm
    """
    uncertain_prob = jnn.softmax(uncertain_logits, axis=-1)
    uncertain_prob = l2_normalize(uncertain_prob, axis=-1)
    return uncertain_prob


def estimate_semantic_information(cls_logits):
    """
    Semantic information from classifier logits.
    """
    semantic_prob = jnn.softmax(cls_logits, axis=-1)
    return semantic_prob


def estimate_mixup_ratio(
    cls_one,
    uncertain_one,
    cls_mix,
    uncertain_mix,
    labels_a,
    labels_b,
    perm,
    lam_area,
):
    """
    SUMix mixup ratio correction.

    This follows the official-style semantic correction logic, but removes
    batch-size scaling from alpha for numerical stability in this JAX version.

        semantic_one  = softmax(cls_one.detach())
        semantic_mix  = softmax(cls_mix.detach())
        semantic_b    = semantic_one[rand_index]

        semantic_one[i, y_b[i]] = 0
        semantic_b[i, y_a[i]] = 0

        alpha_a = l2_norm(softmax(semantic_mix - semantic_one))
        alpha_b = l2_norm(softmax(semantic_mix - semantic_b))

        INa = exp(-alpha_a)
        INb = exp(-alpha_b)

        lam_a = lam * INa[y_a]
        lam_b = (1 - lam) * INb[y_b]
        lam_sumix = lam_a / (lam_a + lam_b)

    beta / uncertainty is returned for regularization, not used directly
    in lambda correction.
    """

    batch_size = labels_a.shape[0]

    if jnp.ndim(lam_area) == 0:
        lam_area = jnp.ones((batch_size,), dtype=cls_mix.dtype) * lam_area

    lam_area = lam_area.reshape(-1)

    semantic_one = estimate_semantic_information(cls_one)
    semantic_mix = estimate_semantic_information(cls_mix)

    semantic_one = jax.lax.stop_gradient(semantic_one)
    semantic_mix = jax.lax.stop_gradient(semantic_mix)

    semantic_b = semantic_one[perm]

    semantic_one_masked = zero_out_label(semantic_one, labels_b)
    semantic_b_masked = zero_out_label(semantic_b, labels_a)

    alpha_a = l2_normalize(
        jnn.softmax(semantic_mix - semantic_one_masked, axis=-1),
        axis=-1,
    )

    alpha_b = l2_normalize(
        jnn.softmax(semantic_mix - semantic_b_masked, axis=-1),
        axis=-1,
    )

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

    lam_sumix = lam_a / (lam_a + lam_b + 1e-8)
    lam_sumix = jnp.clip(lam_sumix, 0.0, 1.0)

    return lam_sumix, {
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
):
    """
    SUMix loss.

    total_loss =
        corrected-lambda mixed classification loss
        + gamma * uncertainty/semantic regularization
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
    )

    loss_a = cross_entropy_with_integer_labels(cls_mix, labels_a)
    loss_b = cross_entropy_with_integer_labels(cls_mix, labels_b)

    cls_loss = jnp.mean(
        lam_sumix * loss_a + (1.0 - lam_sumix) * loss_b
    )

    reg_logits_a = -(
        ratio_info["alpha_a"] + ratio_info["beta_a"]
    )

    reg_logits_b = -(
        ratio_info["alpha_b"] + ratio_info["beta_b"]
    )

    reg_loss_a = cross_entropy_with_integer_labels(
        reg_logits_a,
        labels_a,
    )

    reg_loss_b = cross_entropy_with_integer_labels(
        reg_logits_b,
        labels_b,
    )

    if jnp.ndim(lam_area) == 0:
        lam_area = jnp.ones_like(lam_sumix) * lam_area

    lam_area = lam_area.reshape(-1)

    reg_loss = jnp.mean(
        lam_area * reg_loss_a + (1.0 - lam_area) * reg_loss_b
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
        "info_a_y_mean": jnp.mean(gather_by_label(ratio_info["info_a"], labels_a)),
        "info_b_y_mean": jnp.mean(gather_by_label(ratio_info["info_b"], labels_b)),
    }