from typing import Iterator, Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)


def normalize_image(image: tf.Tensor, dataset_name: str = "cifar10") -> tf.Tensor:
    """
    Convert uint8 image [0, 255] to normalized float32 image.
    """
    image = tf.cast(image, tf.float32) / 255.0

    if dataset_name == "cifar10":
        image = (image - CIFAR10_MEAN) / CIFAR10_STD
    elif dataset_name == "cifar100":
        image = (image - CIFAR100_MEAN) / CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return image


def preprocess_train(
    example: Dict[str, tf.Tensor],
    dataset_name: str,
) -> Dict[str, tf.Tensor]:
    image = example["image"]
    label = example["label"]

    # CIFAR-style augmentation: pad 4 then random crop back to 32x32.
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)

    image = normalize_image(image, dataset_name)

    return {
        "image": image,
        "label": tf.cast(label, tf.int32),
    }


def preprocess_eval(
    example: Dict[str, tf.Tensor],
    dataset_name: str,
) -> Dict[str, tf.Tensor]:
    image = normalize_image(example["image"], dataset_name)
    label = tf.cast(example["label"], tf.int32)

    return {
        "image": image,
        "label": label,
    }


def get_cifar_dataset(
    dataset_name: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    data_dir: str = "./data",
) -> tf.data.Dataset:
    """
    Create CIFAR-10 or CIFAR-100 tf.data.Dataset.
    """
    if dataset_name not in ["cifar10", "cifar100"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    ds = tfds.load(
        dataset_name,
        split=split,
        data_dir=data_dir,
        as_supervised=False,
    )

    if split == "train":
        if shuffle:
            ds = ds.shuffle(50_000)
        ds = ds.map(
            lambda x: preprocess_train(x, dataset_name),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda x: preprocess_eval(x, dataset_name),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_cifar10_dataset(
    split: str,
    batch_size: int,
    shuffle: bool = True,
    data_dir: str = "./data",
) -> tf.data.Dataset:
    return get_cifar_dataset(
        dataset_name="cifar10",
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        data_dir=data_dir,
    )


def get_cifar100_dataset(
    split: str,
    batch_size: int,
    shuffle: bool = True,
    data_dir: str = "./data",
) -> tf.data.Dataset:
    return get_cifar_dataset(
        dataset_name="cifar100",
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        data_dir=data_dir,
    )


def numpy_iterator(ds: tf.data.Dataset) -> Iterator[Dict[str, np.ndarray]]:
    """
    Convert tf.data.Dataset batches to NumPy batches.
    """
    return tfds.as_numpy(ds)