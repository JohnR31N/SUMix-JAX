import argparse
import csv
import os
import time
from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

from cutmix_jax.datasets import get_cifar10_dataset, numpy_iterator
from cutmix_jax.losses import classification_loss, cutmix_loss
from cutmix_jax.cutmix import cutmix_batch
from cutmix_jax.sumix import sumix_loss
from cutmix_jax.models.small_cnn import SmallCNN
from cutmix_jax.models.pyramidnet import PyramidNet
from cutmix_jax.models.resnet import ResNet18


class TrainState(train_state.TrainState):
    batch_stats: Dict[str, Any]


def parse_args():
    parser = argparse.ArgumentParser(description="Generic JAX training script")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])

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
    learning_rate = create_lr_schedule(
        args=args,
        steps_per_epoch=steps_per_epoch,
    )

    if args.optimizer == "adam":
        return optax.adam(
            learning_rate=learning_rate,
        )

    if args.optimizer == "sgd":
        sgd = optax.sgd(
            learning_rate=learning_rate,
            momentum=args.momentum,
            nesterov=True,
        )

        if args.weight_decay > 0.0:
            return optax.chain(
                optax.add_decayed_weights(args.weight_decay),
                sgd,
            )

        return sgd

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def create_train_state(rng, model, args, steps_per_epoch: int):
    dummy_x = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    variables = model.init(rng, dummy_x, train=True)

    tx = create_optimizer(
        args=args,
        steps_per_epoch=steps_per_epoch,
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )


def get_logits(outputs):
    """
    Normal model returns:
        logits

    SUMix model returns:
        cls_logits, uncertain_logits

    This helper extracts cls_logits.
    """
    if isinstance(outputs, tuple):
        return outputs[0]

    return outputs


@jax.jit
def train_step_baseline(state, batch):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    def loss_fn(params):
        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }

        outputs, new_model_state = state.apply_fn(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
        )

        logits = get_logits(outputs)
        loss = classification_loss(logits, labels)

        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return state, {
        "loss": loss,
        "acc": acc,
    }


@partial(jax.jit, static_argnames=("cutmix_alpha",))
def train_step_cutmix(state, batch, rng, cutmix_alpha: float):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    mixed_images, info = cutmix_batch(
        images=images,
        labels=labels,
        rng=rng,
        alpha=cutmix_alpha,
    )

    def loss_fn(params):
        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }

        outputs, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )

        logits = get_logits(outputs)
        loss = cutmix_loss(logits, info)

        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == info["labels_a"])

    return state, {
        "loss": loss,
        "acc": acc,
        "lam": info["lam"],
    }


@partial(
    jax.jit,
    static_argnames=("cutmix_alpha", "sumix_gamma", "sumix_alpha_scale"),
)
def train_step_cutmix_sumix(
    state,
    batch,
    rng,
    cutmix_alpha: float,
    sumix_gamma: float,
    sumix_alpha_scale: float,
):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    mixed_images, info = cutmix_batch(
        images=images,
        labels=labels,
        rng=rng,
        alpha=cutmix_alpha,
    )

    def loss_fn(params):
        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }

        # Official-alignment: keep the original-image forward in train mode.
        # This matches the PyTorch/OpenMixup training path more closely than
        # using eval-mode BatchNorm for the original image branch.
        outputs_one, _ = state.apply_fn(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
        )
        cls_one, uncertain_one = outputs_one

        outputs_mix, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )

        cls_mix, uncertain_mix = outputs_mix

        loss, sumix_info = sumix_loss(
            cls_one=cls_one,
            uncertain_one=uncertain_one,
            cls_mix=cls_mix,
            uncertain_mix=uncertain_mix,
            labels_a=info["labels_a"],
            labels_b=info["labels_b"],
            perm=info["perm"],
            lam_area=info["lam"],
            gamma=sumix_gamma,
            alpha_scale=sumix_alpha_scale,
        )

        return loss, (cls_mix, new_model_state, sumix_info)

    (loss, (logits, new_model_state, sumix_info)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == info["labels_a"])

    return state, {
        "loss": loss,
        "acc": acc,
        "lam": sumix_info["lam"],
        "lam_min": sumix_info["lam_min"],
        "lam_max": sumix_info["lam_max"],
        "lam_std": sumix_info["lam_std"],
        "cls_loss": sumix_info["cls_loss"],
        "reg_loss": sumix_info["reg_loss"],
        "alpha_a_y_mean": sumix_info["alpha_a_y_mean"],
        "alpha_b_y_mean": sumix_info["alpha_b_y_mean"],
        "info_a_y_mean": sumix_info["info_a_y_mean"],
        "info_b_y_mean": sumix_info["info_b_y_mean"],
    }


@jax.jit
def eval_step(state, batch):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }

    outputs = state.apply_fn(
        variables,
        images,
        train=False,
        mutable=False,
    )

    logits = get_logits(outputs)

    loss = classification_loss(logits, labels)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return {
        "loss": loss,
        "acc": acc,
    }


def run_epoch_train(state, train_ds, args, rng, epoch: int, steps_per_epoch: int):
    train_losses = []
    train_accs = []
    train_lams = []

    # SUMix diagnostic metrics.
    train_lam_mins = []
    train_lam_maxs = []
    train_lam_stds = []
    train_cls_losses = []
    train_reg_losses = []
    train_alpha_a_y_means = []
    train_alpha_b_y_means = []
    train_info_a_y_means = []
    train_info_b_y_means = []

    cutmix_count = 0
    total_count = 0

    train_iter = tqdm(
        numpy_iterator(train_ds),
        total=steps_per_epoch,
        desc=f"Train Epoch {epoch:03d}",
        dynamic_ncols=True,
        leave=True,
    )

    for batch in train_iter:
        rng, step_rng, prob_rng = jax.random.split(rng, 3)
        total_count += 1

        if args.aug == "none":
            state, metrics = train_step_baseline(state, batch)

        elif args.aug in ["cutmix", "cutmix_sumix"]:
            use_mix = bool(jax.random.uniform(prob_rng) < args.cutmix_prob)

            if use_mix:
                if args.aug == "cutmix":
                    state, metrics = train_step_cutmix(
                        state,
                        batch,
                        step_rng,
                        args.cutmix_alpha,
                    )
                else:
                    state, metrics = train_step_cutmix_sumix(
                        state,
                        batch,
                        step_rng,
                        args.cutmix_alpha,
                        args.sumix_gamma,
                        args.sumix_alpha_scale,
                    )

                train_lams.append(float(metrics["lam"]))
                cutmix_count += 1
            else:
                state, metrics = train_step_baseline(state, batch)

        else:
            raise ValueError(f"Unsupported augmentation: {args.aug}")

        loss = float(metrics["loss"])
        acc = float(metrics["acc"])

        if args.aug == "cutmix_sumix" and "cls_loss" in metrics:
            train_lam_mins.append(float(metrics["lam_min"]))
            train_lam_maxs.append(float(metrics["lam_max"]))
            train_lam_stds.append(float(metrics["lam_std"]))
            train_cls_losses.append(float(metrics["cls_loss"]))
            train_reg_losses.append(float(metrics["reg_loss"]))
            train_alpha_a_y_means.append(float(metrics["alpha_a_y_mean"]))
            train_alpha_b_y_means.append(float(metrics["alpha_b_y_mean"]))
            train_info_a_y_means.append(float(metrics["info_a_y_mean"]))
            train_info_b_y_means.append(float(metrics["info_b_y_mean"]))

        train_losses.append(loss)
        train_accs.append(acc)

        postfix = {
            "loss": f"{sum(train_losses) / len(train_losses):.4f}",
            "acc": f"{sum(train_accs) / len(train_accs):.4f}",
        }

        if train_lams:
            postfix["lam"] = f"{sum(train_lams) / len(train_lams):.4f}"

        if train_reg_losses:
            postfix["reg"] = f"{sum(train_reg_losses) / len(train_reg_losses):.4f}"
            postfix["lstd"] = f"{sum(train_lam_stds) / len(train_lam_stds):.4f}"
            postfix["alpha"] = f"{sum(train_alpha_a_y_means) / len(train_alpha_a_y_means):.2f}"

        if args.aug in ["cutmix", "cutmix_sumix"]:
            postfix["mix"] = f"{cutmix_count / total_count:.3f}"

        train_iter.set_postfix(postfix)

    output = {
        "loss": sum(train_losses) / len(train_losses),
        "acc": sum(train_accs) / len(train_accs),
        "lam": sum(train_lams) / len(train_lams) if train_lams else None,
        "lam_min": sum(train_lam_mins) / len(train_lam_mins) if train_lam_mins else None,
        "lam_max": sum(train_lam_maxs) / len(train_lam_maxs) if train_lam_maxs else None,
        "lam_std": sum(train_lam_stds) / len(train_lam_stds) if train_lam_stds else None,
        "cls_loss": sum(train_cls_losses) / len(train_cls_losses) if train_cls_losses else None,
        "reg_loss": sum(train_reg_losses) / len(train_reg_losses) if train_reg_losses else None,
        "alpha_a_y_mean": (
            sum(train_alpha_a_y_means) / len(train_alpha_a_y_means)
            if train_alpha_a_y_means
            else None
        ),
        "alpha_b_y_mean": (
            sum(train_alpha_b_y_means) / len(train_alpha_b_y_means)
            if train_alpha_b_y_means
            else None
        ),
        "info_a_y_mean": (
            sum(train_info_a_y_means) / len(train_info_a_y_means)
            if train_info_a_y_means
            else None
        ),
        "info_b_y_mean": (
            sum(train_info_b_y_means) / len(train_info_b_y_means)
            if train_info_b_y_means
            else None
        ),
        "cutmix_rate": cutmix_count / total_count
        if args.aug in ["cutmix", "cutmix_sumix"]
        else 0.0,
    }

    return state, output, rng


def run_epoch_eval(state, test_ds, epoch: int, test_steps: int):
    test_losses = []
    test_accs = []

    test_iter = tqdm(
        numpy_iterator(test_ds),
        total=test_steps,
        desc=f"Eval  Epoch {epoch:03d}",
        dynamic_ncols=True,
        leave=True,
    )

    for batch in test_iter:
        metrics = eval_step(state, batch)

        loss = float(metrics["loss"])
        acc = float(metrics["acc"])

        test_losses.append(loss)
        test_accs.append(acc)

        test_iter.set_postfix(
            {
                "loss": f"{sum(test_losses) / len(test_losses):.4f}",
                "acc": f"{sum(test_accs) / len(test_accs):.4f}",
            }
        )

    return {
        "loss": sum(test_losses) / len(test_losses),
        "acc": sum(test_accs) / len(test_accs),
    }


def get_csv_path(args):
    run_dir = os.path.join(
        args.output_dir,
        args.dataset,
        args.model,
    )
    os.makedirs(run_dir, exist_ok=True)

    if args.model == "pyramidnet":
        model_tag = f"pyramidnet_d{args.pyramid_depth}_a{args.pyramid_alpha}"
        if args.pyramid_bottleneck:
            model_tag += "_bottleneck"
    else:
        model_tag = args.model

    filename = f"{model_tag}_{args.aug}_seed{args.seed}.csv"
    return os.path.join(run_dir, filename)


def create_csv_writer(csv_path):
    csv_file = open(csv_path, mode="w", newline="")

    fieldnames = [
        "epoch",
        "dataset",
        "model",
        "aug",
        "pyramid_depth",
        "pyramid_alpha",
        "pyramid_bottleneck",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "test_error",
        "best_test_acc",
        "avg_lam",
        "lam_min",
        "lam_max",
        "lam_std",
        "cls_loss",
        "reg_loss",
        "alpha_a_y_mean",
        "alpha_b_y_mean",
        "info_a_y_mean",
        "info_b_y_mean",
        "cutmix_rate",
        "epoch_time_sec",
        "total_time_sec",
        "seed",
        "batch_size",
        "learning_rate",
        "optimizer",
        "momentum",
        "weight_decay",
        "lr_schedule",
        "lr_milestones",
        "lr_gamma",
        "cutmix_alpha",
        "cutmix_prob",
        "sumix_gamma",
        "sumix_alpha_scale",
    ]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    return csv_file, writer


def print_config(args, steps_per_epoch: int, test_steps: int):
    print("=" * 80)
    print("Training configuration")
    print(f"Dataset:            {args.dataset}")
    print(f"Model:              {args.model}")

    if args.model == "pyramidnet":
        print(f"Pyramid depth:      {args.pyramid_depth}")
        print(f"Pyramid alpha:      {args.pyramid_alpha}")
        print(f"Pyramid bottleneck: {args.pyramid_bottleneck}")

    print(f"Augmentation:       {args.aug}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Epochs:             {args.epochs}")
    print(f"Train steps/epoch:  {steps_per_epoch}")
    print(f"Eval steps/epoch:   {test_steps}")
    print(f"Learning rate:      {args.lr}")
    print(f"Optimizer:          {args.optimizer}")

    if args.optimizer == "sgd":
        print(f"Momentum:           {args.momentum}")
        print(f"Weight decay:       {args.weight_decay}")
        print(f"LR schedule:        {args.lr_schedule}")
        print(f"LR milestones:      {args.lr_milestones}")
        print(f"LR gamma:           {args.lr_gamma}")

    print(f"Seed:               {args.seed}")
    print(f"Data dir:           {args.data_dir}")
    print(f"Output dir:         {args.output_dir}")

    if args.aug in ["cutmix", "cutmix_sumix"]:
        print(f"CutMix alpha:       {args.cutmix_alpha}")
        print(f"CutMix prob:        {args.cutmix_prob}")

    if args.aug == "cutmix_sumix":
        print(f"SUMix gamma:        {args.sumix_gamma}")
        print(f"SUMix alpha scale:  {args.sumix_alpha_scale}")

    print(f"Device:             {jax.devices()}")
    print("=" * 80)


def main():
    args = parse_args()

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    train_ds, test_ds, num_classes, num_train_examples, num_test_examples = create_datasets(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    steps_per_epoch = num_train_examples // args.batch_size
    test_steps = num_test_examples // args.batch_size

    model = create_model(
        args=args,
        num_classes=num_classes,
    )

    state = create_train_state(
        rng=init_rng,
        model=model,
        args=args,
        steps_per_epoch=steps_per_epoch,
    )

    print_config(args, steps_per_epoch, test_steps)

    csv_path = get_csv_path(args)
    csv_file, csv_writer = create_csv_writer(csv_path)

    print(f"CSV log: {csv_path}")

    best_test_acc = 0.0
    total_start_time = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            state, train_metrics, rng = run_epoch_train(
                state=state,
                train_ds=train_ds,
                args=args,
                rng=rng,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )

            test_metrics = run_epoch_eval(
                state=state,
                test_ds=test_ds,
                epoch=epoch,
                test_steps=test_steps,
            )

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - total_start_time

            train_loss = train_metrics["loss"]
            train_acc = train_metrics["acc"]
            test_loss = test_metrics["loss"]
            test_acc = test_metrics["acc"]
            test_error = 1.0 - test_acc

            best_test_acc = max(best_test_acc, test_acc)

            avg_lam = train_metrics["lam"]
            lam_min = train_metrics["lam_min"]
            lam_max = train_metrics["lam_max"]
            lam_std = train_metrics["lam_std"]
            cls_loss = train_metrics["cls_loss"]
            reg_loss = train_metrics["reg_loss"]
            alpha_a_y_mean = train_metrics["alpha_a_y_mean"]
            alpha_b_y_mean = train_metrics["alpha_b_y_mean"]
            info_a_y_mean = train_metrics["info_a_y_mean"]
            info_b_y_mean = train_metrics["info_b_y_mean"]
            cutmix_rate = train_metrics["cutmix_rate"]

            msg = (
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.4f} | "
                f"train acc {train_acc:.4f} | "
                f"test loss {test_loss:.4f} | "
                f"test acc {test_acc:.4f} | "
                f"test error {test_error:.4f} | "
                f"best test acc {best_test_acc:.4f}"
            )

            if avg_lam is not None:
                msg += f" | avg lam {avg_lam:.4f}"

            if args.aug == "cutmix_sumix" and reg_loss is not None:
                msg += (
                    f" | lam std {lam_std:.4f}"
                    f" | cls {cls_loss:.4f}"
                    f" | reg {reg_loss:.4f}"
                    f" | alpha_a {alpha_a_y_mean:.4f}"
                    f" | alpha_b {alpha_b_y_mean:.4f}"
                    f" | info_a {info_a_y_mean:.4e}"
                    f" | info_b {info_b_y_mean:.4e}"
                )

            if args.aug in ["cutmix", "cutmix_sumix"]:
                msg += f" | mix rate {cutmix_rate:.4f}"

            msg += f" | epoch time {epoch_time:.1f}s"

            print(msg)

            csv_writer.writerow(
                {
                    "epoch": epoch,
                    "dataset": args.dataset,
                    "model": args.model,
                    "aug": args.aug,
                    "pyramid_depth": args.pyramid_depth if args.model == "pyramidnet" else "",
                    "pyramid_alpha": args.pyramid_alpha if args.model == "pyramidnet" else "",
                    "pyramid_bottleneck": args.pyramid_bottleneck if args.model == "pyramidnet" else "",
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_error": test_error,
                    "best_test_acc": best_test_acc,
                    "avg_lam": "" if avg_lam is None else avg_lam,
                    "lam_min": "" if lam_min is None else lam_min,
                    "lam_max": "" if lam_max is None else lam_max,
                    "lam_std": "" if lam_std is None else lam_std,
                    "cls_loss": "" if cls_loss is None else cls_loss,
                    "reg_loss": "" if reg_loss is None else reg_loss,
                    "alpha_a_y_mean": "" if alpha_a_y_mean is None else alpha_a_y_mean,
                    "alpha_b_y_mean": "" if alpha_b_y_mean is None else alpha_b_y_mean,
                    "info_a_y_mean": "" if info_a_y_mean is None else info_a_y_mean,
                    "info_b_y_mean": "" if info_b_y_mean is None else info_b_y_mean,
                    "cutmix_rate": cutmix_rate,
                    "epoch_time_sec": epoch_time,
                    "total_time_sec": total_time,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "optimizer": args.optimizer,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "lr_schedule": args.lr_schedule,
                    "lr_milestones": args.lr_milestones,
                    "lr_gamma": args.lr_gamma,
                    "cutmix_alpha": args.cutmix_alpha,
                    "cutmix_prob": args.cutmix_prob,
                    "sumix_gamma": args.sumix_gamma if args.aug == "cutmix_sumix" else "",
                    "sumix_alpha_scale": args.sumix_alpha_scale if args.aug == "cutmix_sumix" else "",
                }
            )
            csv_file.flush()

    finally:
        csv_file.close()

    final_total_time = time.time() - total_start_time
    print("=" * 80)
    print("Training finished.")
    print(f"Best test acc: {best_test_acc:.4f}")
    print(f"CSV saved to: {csv_path}")
    print(f"Total time: {final_total_time:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()