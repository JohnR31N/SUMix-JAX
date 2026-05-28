import time

import jax
from flax import jax_utils

from cutmix_jax.training.config import create_datasets, create_model, parse_args
from cutmix_jax.training.epoch import run_epoch_eval, run_epoch_train
from cutmix_jax.training.io import create_csv_writer, get_csv_path, print_config
from cutmix_jax.training.state import create_train_state


def _csv_value(value):
    return "" if value is None else value


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

    model = create_model(args=args, num_classes=num_classes)
    state = create_train_state(
        rng=init_rng,
        model=model,
        args=args,
        steps_per_epoch=steps_per_epoch,
    )

    if args.use_pmap:
        n_devices = jax.local_device_count()
        if n_devices < 2:
            raise ValueError("--use-pmap was set, but JAX sees fewer than 2 local devices.")
        if args.batch_size % n_devices != 0:
            raise ValueError(
                f"Global batch size {args.batch_size} must be divisible by local_device_count {n_devices}."
            )
        print(f"Replicating TrainState across {n_devices} devices...")
        state = jax_utils.replicate(state)

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

            eval_state = jax_utils.unreplicate(state) if args.use_pmap else state
            test_metrics = run_epoch_eval(
                state=eval_state,
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
            raw_alpha_a_y_mean = train_metrics["raw_alpha_a_y_mean"]
            raw_alpha_b_y_mean = train_metrics["raw_alpha_b_y_mean"]
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
                )
                if raw_alpha_a_y_mean is not None:
                    msg += (
                        f" | raw_alpha_a {raw_alpha_a_y_mean:.4f}"
                        f" | raw_alpha_b {raw_alpha_b_y_mean:.4f}"
                    )
                msg += (
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
                    "avg_lam": _csv_value(avg_lam),
                    "lam_min": _csv_value(lam_min),
                    "lam_max": _csv_value(lam_max),
                    "lam_std": _csv_value(lam_std),
                    "cls_loss": _csv_value(cls_loss),
                    "reg_loss": _csv_value(reg_loss),
                    "raw_alpha_a_y_mean": _csv_value(raw_alpha_a_y_mean),
                    "raw_alpha_b_y_mean": _csv_value(raw_alpha_b_y_mean),
                    "alpha_a_y_mean": _csv_value(alpha_a_y_mean),
                    "alpha_b_y_mean": _csv_value(alpha_b_y_mean),
                    "info_a_y_mean": _csv_value(info_a_y_mean),
                    "info_b_y_mean": _csv_value(info_b_y_mean),
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
