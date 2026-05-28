import jax
from flax.training.common_utils import shard
from tqdm import tqdm

from cutmix_jax.datasets import numpy_iterator
from cutmix_jax.training.metrics import metric_to_float
from cutmix_jax.training.steps import (
    eval_step,
    train_step_baseline,
    train_step_baseline_pmap,
    train_step_cutmix,
    train_step_cutmix_pmap,
    train_step_cutmix_sumix,
    train_step_cutmix_sumix_pmap,
)


SUMIX_METRIC_KEYS = [
    "lam_min",
    "lam_max",
    "lam_std",
    "cls_loss",
    "reg_loss",
    "raw_alpha_a_y_mean",
    "raw_alpha_b_y_mean",
    "alpha_a_y_mean",
    "alpha_b_y_mean",
    "info_a_y_mean",
    "info_b_y_mean",
]


def _empty_metric_lists():
    return {key: [] for key in SUMIX_METRIC_KEYS}


def _mean_or_none(values):
    return sum(values) / len(values) if values else None


def run_epoch_train(state, train_ds, args, rng, epoch: int, steps_per_epoch: int):
    train_losses = []
    train_accs = []
    train_lams = []
    sumix_lists = _empty_metric_lists()

    cutmix_count = 0
    total_count = 0

    train_iter = tqdm(
        numpy_iterator(train_ds),
        total=steps_per_epoch,
        desc=f"Train Epoch {epoch:03d}",
        dynamic_ncols=True,
        leave=True,
    )

    n_devices = jax.local_device_count()

    for batch in train_iter:
        rng, step_rng, prob_rng = jax.random.split(rng, 3)
        total_count += 1

        step_batch = batch
        step_rng_arg = step_rng

        if args.use_pmap:
            step_batch = shard(batch)
            step_rng_arg = jax.random.split(step_rng, n_devices)

        if args.aug == "none":
            if args.use_pmap:
                state, metrics = train_step_baseline_pmap(state, step_batch)
            else:
                state, metrics = train_step_baseline(state, step_batch)

        elif args.aug in ["cutmix", "cutmix_sumix"]:
            use_mix = bool(jax.random.uniform(prob_rng) < args.cutmix_prob)

            if use_mix:
                if args.aug == "cutmix":
                    if args.use_pmap:
                        state, metrics = train_step_cutmix_pmap(
                            state,
                            step_batch,
                            step_rng_arg,
                            args.cutmix_alpha,
                        )
                    else:
                        state, metrics = train_step_cutmix(
                            state,
                            step_batch,
                            step_rng_arg,
                            args.cutmix_alpha,
                        )
                else:
                    if args.use_pmap:
                        state, metrics = train_step_cutmix_sumix_pmap(
                            state,
                            step_batch,
                            step_rng_arg,
                            args.cutmix_alpha,
                            args.sumix_gamma,
                            args.sumix_alpha_scale,
                        )
                    else:
                        state, metrics = train_step_cutmix_sumix(
                            state,
                            step_batch,
                            step_rng_arg,
                            args.cutmix_alpha,
                            args.sumix_gamma,
                            args.sumix_alpha_scale,
                        )

                train_lams.append(metric_to_float(metrics, "lam"))
                cutmix_count += 1
            else:
                if args.use_pmap:
                    state, metrics = train_step_baseline_pmap(state, step_batch)
                else:
                    state, metrics = train_step_baseline(state, step_batch)

        else:
            raise ValueError(f"Unsupported augmentation: {args.aug}")

        loss = metric_to_float(metrics, "loss")
        acc = metric_to_float(metrics, "acc")

        if args.aug == "cutmix_sumix" and "cls_loss" in metrics:
            for key in SUMIX_METRIC_KEYS:
                if key in metrics:
                    sumix_lists[key].append(metric_to_float(metrics, key))

        train_losses.append(loss)
        train_accs.append(acc)

        postfix = {
            "loss": f"{sum(train_losses) / len(train_losses):.4f}",
            "acc": f"{sum(train_accs) / len(train_accs):.4f}",
        }

        if train_lams:
            postfix["lam"] = f"{sum(train_lams) / len(train_lams):.4f}"

        if sumix_lists["reg_loss"]:
            postfix["reg"] = f"{_mean_or_none(sumix_lists['reg_loss']):.4f}"
            postfix["lstd"] = f"{_mean_or_none(sumix_lists['lam_std']):.4f}"
            postfix["alpha"] = f"{_mean_or_none(sumix_lists['alpha_a_y_mean']):.2f}"
            if sumix_lists["raw_alpha_a_y_mean"]:
                postfix["raw_a"] = f"{_mean_or_none(sumix_lists['raw_alpha_a_y_mean']):.3f}"

        if args.aug in ["cutmix", "cutmix_sumix"]:
            postfix["mix"] = f"{cutmix_count / total_count:.3f}"

        train_iter.set_postfix(postfix)

    output = {
        "loss": sum(train_losses) / len(train_losses),
        "acc": sum(train_accs) / len(train_accs),
        "lam": _mean_or_none(train_lams),
        "cutmix_rate": cutmix_count / total_count
        if args.aug in ["cutmix", "cutmix_sumix"]
        else 0.0,
    }
    output.update({key: _mean_or_none(values) for key, values in sumix_lists.items()})
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
