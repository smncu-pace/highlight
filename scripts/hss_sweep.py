from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PATTERNS: list[dict[str, Any]] = [
    {
        "name": "dense",
        "dense": True,
    },
    {
        "name": "C0_2x4_only",
        "rank0": (2, 4),
        "rank1": None,
    },
    {
        "name": "C1_3x4_only",
        "rank0": None,
        "rank1": (3, 4),
    },
    {
        "name": "HSS_C1_3x4_C0_2x4",
        "rank0": (2, 4),
        "rank1": (3, 4),
    },
    {
        "name": "HSS_C1_2x4_C0_2x4",
        "rank0": (2, 4),
        "rank1": (2, 4),
    },
    {
        "name": "HSS_C1_4x8_C0_2x4",
        "rank0": (2, 4),
        "rank1": (4, 8),
    },
]


SUMMARY_COLUMNS = [
    "pattern",
    "rank0",
    "rank1",
    "actual_density",
    "actual_sparsity",
    "estimated_weight_compute_speedup",
    "dense_top1",
    "zero_shot_pruned_top1",
    "best_finetune_top1",
    "accuracy_drop",
    "covered_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Level-1 HSS pattern sweep.")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--dataset", default="imagenet100")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output-dir", default="runs/hss_sweep")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default="data/imagenet-100-hf")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--hf-dataset", default="clane9/imagenet-100")
    parser.add_argument("--device", default="")
    parser.add_argument("--include-linear", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prune-first-conv", action="store_true")
    parser.add_argument("--fixed-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-update-interval", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Extra args forwarded to train.py after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in PATTERNS:
        run_pattern(args, pattern, output_dir)

    write_summary(output_dir)


def run_pattern(args: argparse.Namespace, pattern: dict[str, Any], output_dir: Path) -> None:
    pattern_dir = output_dir / pattern["name"]
    pattern_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "train.py",
        "--arch",
        args.arch,
        "--dataset",
        args.dataset,
        "--resume",
        args.resume,
        "--output-dir",
        str(pattern_dir),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--data-dir",
        args.data_dir,
        "--cache-dir",
        args.cache_dir,
        "--hf-dataset",
        args.hf_dataset,
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.no_progress:
        cmd.append("--no-progress")

    if pattern.get("dense"):
        cmd.append("--eval")
    else:
        cmd += [
            "--hss",
            "--hss-eval-before-finetune",
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--hss-warmup-epochs",
            str(args.warmup_epochs),
            "--hss-mask-update-interval",
            str(args.mask_update_interval),
        ]
        if args.fixed_mask:
            cmd.append("--hss-fixed-mask")
        if args.include_linear:
            cmd.append("--hss-include-linear")
        else:
            cmd.append("--no-hss-include-linear")
        if args.prune_first_conv:
            cmd.append("--hss-prune-first-conv")
        rank0 = pattern["rank0"]
        rank1 = pattern["rank1"]
        if rank0 is not None:
            cmd += ["--hss-rank0-enable", "--hss-rank0-g", str(rank0[0]), "--hss-rank0-h", str(rank0[1])]
        if rank1 is not None:
            cmd += ["--hss-rank1-enable", "--hss-rank1-g", str(rank1[0]), "--hss-rank1-h", str(rank1[1])]

    extra_args = args.train_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd += extra_args

    print("running:", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def write_summary(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for pattern in PATTERNS:
        metrics_path = output_dir / pattern["name"] / "metrics.json"
        if not metrics_path.exists():
            print(f"missing_metrics={metrics_path}", flush=True)
            continue
        metrics = json.loads(metrics_path.read_text())
        rows.append({column: metrics.get(column) for column in SUMMARY_COLUMNS})

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
