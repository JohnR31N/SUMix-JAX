import csv
import os

import jax


def get_csv_path(args):
    run_dir = os.path.join(args.output_dir, args.dataset, args.model)
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
        "raw_alpha_a_y_mean",
        "raw_alpha_b_y_mean",
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
    print(f"Use pmap:           {args.use_pmap}")
    print(f"Local devices:      {jax.local_device_count()}")
    if args.use_pmap:
        print(f"Per-device batch:   {args.batch_size // jax.local_device_count()}")
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
