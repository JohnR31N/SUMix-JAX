import argparse

import optax

from cutmix_jax.datasets import get_cifar10_dataset, get_cifar100_dataset
from cutmix_jax.models.small_cnn import SmallCNN
from cutmix_jax.models.pyramidnet import PyramidNet
from cutmix_jax.models.resnet import ResNet18


def parse_args():
    parser = argparse.ArgumentParser(description="Generic JAX training script")

    parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["small_cnn", "pyramidnet", "resnet18"],
    )
    parser.add_argument(
        "--aug",
        type=str,
        default="none",
        choices=["none", "cutmix", "cutmix_sumix"],
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=["none", "multistep"],
    )
    parser.add_argument("--lr-milestones", type=str, default="150,225")
    parser.add_argument("--lr-gamma", type=float, default=0.1)

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")

    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=1.0)

    parser.add_argument("--sumix-gamma", type=float, default=0.1)
    parser.add_argument("--sumix-alpha-scale", type=float, default=1.0)

    # Use multi-device data parallel training via jax.pmap.
    # Keep --batch-size as the GLOBAL batch size; it must be divisible by local_device_count.
    parser.add_argument("--use-pmap", action="store_true")

    parser.add_argument("--pyramid-depth", type=int, default=20)
    parser.add_argument("--pyramid-alpha", type=int, default=48)
    parser.add_argument("--pyramid-bottleneck", action="store_true")

    return parser.parse_args()


def create_model(args, num_classes: int):
    if args.model == "small_cnn":
        return SmallCNN(num_classes=num_classes)

    if args.model == "pyramidnet":
        return PyramidNet(
            depth=args.pyramid_depth,
            alpha=args.pyramid_alpha,
            num_classes=num_classes,
            bottleneck=args.pyramid_bottleneck,
        )

    if args.model == "resnet18":
        return ResNet18(
            num_classes=num_classes,
            base_width=64,
            use_sumix_head=(args.aug == "cutmix_sumix"),
        )

    raise ValueError(f"Unsupported model: {args.model}")


def create_datasets(dataset_name: str, batch_size: int, data_dir: str):
    if dataset_name == "cifar10":
        train_ds = get_cifar10_dataset(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            data_dir=data_dir,
        )
        test_ds = get_cifar10_dataset(
            split="test",
            batch_size=batch_size,
            shuffle=False,
            data_dir=data_dir,
        )

        num_classes = 10
        num_train_examples = 50000
        num_test_examples = 10000
        return train_ds, test_ds, num_classes, num_train_examples, num_test_examples

    if dataset_name == "cifar100":
        train_ds = get_cifar100_dataset(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            data_dir=data_dir,
        )
        test_ds = get_cifar100_dataset(
            split="test",
            batch_size=batch_size,
            shuffle=False,
            data_dir=data_dir,
        )

        num_classes = 100
        num_train_examples = 50000
        num_test_examples = 10000
        return train_ds, test_ds, num_classes, num_train_examples, num_test_examples

    raise ValueError(f"Unsupported dataset: {dataset_name}")
    if dataset_name == "cifar10":
        train_ds = get_cifar10_dataset(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            data_dir=data_dir,
        )
        test_ds = get_cifar10_dataset(
            split="test",
            batch_size=batch_size,
            shuffle=False,
            data_dir=data_dir,
        )

        num_classes = 10
        num_train_examples = 50000
        num_test_examples = 10000
        return train_ds, test_ds, num_classes, num_train_examples, num_test_examples

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_milestones(milestones: str):
    if milestones.strip() == "":
        return []
    return [int(x.strip()) for x in milestones.split(",") if x.strip()]


def create_lr_schedule(args, steps_per_epoch: int):
    if args.lr_schedule == "none":
        return args.lr

    if args.lr_schedule == "multistep":
        milestones = parse_milestones(args.lr_milestones)
        boundaries_and_scales = {
            milestone * steps_per_epoch: args.lr_gamma
            for milestone in milestones
        }
        return optax.piecewise_constant_schedule(
            init_value=args.lr,
            boundaries_and_scales=boundaries_and_scales,
        )

    raise ValueError(f"Unsupported LR schedule: {args.lr_schedule}")


def create_optimizer(args, steps_per_epoch: int):
    learning_rate = create_lr_schedule(args=args, steps_per_epoch=steps_per_epoch)

    if args.optimizer == "adam":
        return optax.adam(learning_rate=learning_rate)

    if args.optimizer == "sgd":
        sgd = optax.sgd(
            learning_rate=learning_rate,
            momentum=args.momentum,
            nesterov=True,
        )
        if args.weight_decay > 0.0:
            return optax.chain(optax.add_decayed_weights(args.weight_decay), sgd)
        return sgd

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")
