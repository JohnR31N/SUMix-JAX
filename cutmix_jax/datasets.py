from typing import Iterator, Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Convert uint8 image [0, 255] to normalized float32 image.

    Input:
        image: [32, 32, 3], uint8

    Output:
        image: [32, 32, 3], float32
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - CIFAR10_MEAN) / CIFAR10_STD
    return image


def preprocess_train(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    image = example["image"]
    label = example["label"]

    # CIFAR-style augmentation: pad 4 then random crop back to 32x32
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)

    image = normalize_image(image)

    return {
        "image": image,
        "label": tf.cast(label, tf.int32),
    }


def preprocess_eval(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    image = normalize_image(example["image"])
    label = tf.cast(example["label"], tf.int32)

    return {
        "image": image,
        "label": label,
    }


def get_cifar10_dataset(
    split: str,
    batch_size: int,
    shuffle: bool = True,
    data_dir: str = "./data",
) -> tf.data.Dataset:
    """
    Create CIFAR-10 tf.data.Dataset.

    Args:
        split: "train" or "test"
        batch_size: batch size
        shuffle: whether to shuffle
        data_dir: local dataset cache path

    Returns:
        tf.data.Dataset yielding:
            {
                "image": [B, 32, 32, 3], float32
                "label": [B], int32
            }
    """
    ds = tfds.load(
        "cifar10",
        split=split,
        data_dir=data_dir,
        as_supervised=False,
    )

    if split == "train":
        if shuffle:
            ds = ds.shuffle(50_000)
        ds = ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def numpy_iterator(ds: tf.data.Dataset) -> Iterator[Dict[str, np.ndarray]]:
    """
    Convert tf.data.Dataset batches to NumPy batches.

    JAX can consume NumPy arrays directly.
    """
    return tfds.as_numpy(ds)