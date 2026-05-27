import jax
import jax.numpy as jnp


def rand_bbox(rng, height: int, width: int, lam: jnp.ndarray):
    """
    Generate CutMix bounding box.

    Args:
        rng: JAX random key.
        height: image height.
        width: image width.
        lam: lambda sampled from Beta distribution.

    Returns:
        y1, y2, x1, x2, corrected_lam
    """
    cut_ratio = jnp.sqrt(1.0 - lam)
    cut_h = (height * cut_ratio).astype(jnp.int32)
    cut_w = (width * cut_ratio).astype(jnp.int32)

    rng_y, rng_x = jax.random.split(rng)

    cy = jax.random.randint(rng_y, shape=(), minval=0, maxval=height)
    cx = jax.random.randint(rng_x, shape=(), minval=0, maxval=width)

    y1 = jnp.clip(cy - cut_h // 2, 0, height)
    y2 = jnp.clip(cy + cut_h // 2, 0, height)
    x1 = jnp.clip(cx - cut_w // 2, 0, width)
    x2 = jnp.clip(cx + cut_w // 2, 0, width)

    box_area = (y2 - y1) * (x2 - x1)
    corrected_lam = 1.0 - box_area / (height * width)

    return y1, y2, x1, x2, corrected_lam


def make_cutmix_mask(height: int, width: int, channels: int, y1, y2, x1, x2):
    """
    Create a CutMix mask in NHWC format.

    mask = 1 means keep original image.
    mask = 0 means replace with shuffled image.
    """
    yy = jnp.arange(height)[:, None]
    xx = jnp.arange(width)[None, :]

    box = (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2)
    mask_hw = jnp.where(box, 0.0, 1.0)

    mask = jnp.broadcast_to(mask_hw[:, :, None], (height, width, channels))
    return mask


def cutmix_batch(images, labels, rng, alpha: float = 1.0):
    """
    Apply CutMix to a batch of images.

    Args:
        images: jnp.ndarray, shape [B, H, W, C]
        labels: jnp.ndarray, shape [B] or [B, num_classes]
        rng: JAX random key
        alpha: beta distribution parameter

    Returns:
        mixed_images: jnp.ndarray, shape [B, H, W, C]
        info: dict containing labels_a, labels_b, lam, perm, box
    """
    batch_size, height, width, channels = images.shape

    rng_lam, rng_perm, rng_box = jax.random.split(rng, 3)

    lam = jax.random.beta(rng_lam, alpha, alpha)
    perm = jax.random.permutation(rng_perm, batch_size)

    images_b = images[perm]
    labels_a = labels
    labels_b = labels[perm]

    y1, y2, x1, x2, corrected_lam = rand_bbox(
        rng_box,
        height,
        width,
        lam,
    )

    mask = make_cutmix_mask(height, width, channels, y1, y2, x1, x2)

    mixed_images = images * mask[None, ...] + images_b * (1.0 - mask[None, ...])

    info = {
        "labels_a": labels_a,
        "labels_b": labels_b,
        "lam": corrected_lam,
        "perm": perm,
        "box": (y1, y2, x1, x2),
    }

    return mixed_images, info