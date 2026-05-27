import jax
import jax.numpy as jnp
import jax.nn as jnn


def cross_entropy_with_integer_labels(logits, labels):
    log_probs = jnn.log_softmax(logits, axis=-1)
    batch_indices = jnp.arange(labels.shape[0])
    return -log_probs[batch_indices, labels]


def compute_semantic_lambda(logits_mix, logits_a, logits_b, temperature: float = 1.0):
    """
    SUMix-style semantic lambda.

    If mixed prediction is closer to image A prediction, lambda should be larger.
    If mixed prediction is closer to image B prediction, lambda should be smaller.
    """
    probs_mix = jnn.softmax(logits_mix / temperature, axis=-1)
    probs_a = jax.lax.stop_gradient(jnn.softmax(logits_a / temperature, axis=-1))
    probs_b = jax.lax.stop_gradient(jnn.softmax(logits_b / temperature, axis=-1))

    dist_a = jnp.mean((probs_mix - probs_a) ** 2, axis=-1)
    dist_b = jnp.mean((probs_mix - probs_b) ** 2, axis=-1)

    lam = dist_b / (dist_a + dist_b + 1e-8)
    lam = jnp.clip(lam, 0.0, 1.0)

    return lam


def normalized_entropy(logits):
    probs = jnn.softmax(logits, axis=-1)
    log_probs = jnn.log_softmax(logits, axis=-1)

    entropy = -jnp.sum(probs * log_probs, axis=-1)
    max_entropy = jnp.log(logits.shape[-1])

    return entropy / max_entropy


def sumix_cutmix_loss(
    logits_mix,
    logits_a,
    logits_b,
    labels_a,
    labels_b,
    beta: float = 0.1,
    temperature: float = 1.0,
):
    """
    CutMix + SUMix-style loss.

    Main idea:
    1. Use semantic distance to estimate lambda.
    2. Use this lambda for mixed-label classification loss.
    3. Add uncertainty regularization.
    """
    lam = compute_semantic_lambda(
        logits_mix=logits_mix,
        logits_a=logits_a,
        logits_b=logits_b,
        temperature=temperature,
    )

    loss_a = cross_entropy_with_integer_labels(logits_mix, labels_a)
    loss_b = cross_entropy_with_integer_labels(logits_mix, labels_b)

    cls_loss = jnp.mean(lam * loss_a + (1.0 - lam) * loss_b)

    uncertainty = normalized_entropy(logits_mix)

    target_uncertainty = jax.lax.stop_gradient(
        4.0 * lam * (1.0 - lam)
    )

    uncertainty_loss = jnp.mean((uncertainty - target_uncertainty) ** 2)

    total_loss = cls_loss + beta * uncertainty_loss

    return total_loss, {
        "lam": jnp.mean(lam),
        "cls_loss": cls_loss,
        "uncertainty_loss": uncertainty_loss,
        "uncertainty": jnp.mean(uncertainty),
    }