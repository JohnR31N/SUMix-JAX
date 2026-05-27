import jax.numpy as jnp
import jax.nn as jnn


def cross_entropy_with_integer_labels(logits, labels):
    """
    Args:
        logits: [B, num_classes]
        labels: [B], integer class labels

    Returns:
        loss: [B]
    """
    log_probs = jnn.log_softmax(logits, axis=-1)
    batch_indices = jnp.arange(labels.shape[0])
    return -log_probs[batch_indices, labels]


def classification_loss(logits, labels):
    """
    Standard classification loss for baseline ERM training.
    """
    losses = cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(losses)


def cutmix_loss(logits, info):
    """
    CutMix loss:
        lam * CE(logits, labels_a) + (1 - lam) * CE(logits, labels_b)
    """
    labels_a = info["labels_a"]
    labels_b = info["labels_b"]
    lam = info["lam"]

    loss_a = cross_entropy_with_integer_labels(logits, labels_a)
    loss_b = cross_entropy_with_integer_labels(logits, labels_b)

    losses = lam * loss_a + (1.0 - lam) * loss_b
    return jnp.mean(losses)