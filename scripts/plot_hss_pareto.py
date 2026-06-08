from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HSS Pareto curves from summary.csv.")
    parser.add_argument("--summary", default="runs/hss_sweep/summary.csv")
    parser.add_argument("--output-dir", default="runs/hss_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with your project environment first.") from exc

    rows = read_rows(Path(args.summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter(
        plt,
        rows,
        x_key="actual_sparsity",
        y_key="best_finetune_top1",
        xlabel="actual_sparsity",
        ylabel="best_finetune_top1",
        title="HSS Accuracy vs Sparsity",
        output_path=output_dir / "pareto_accuracy_sparsity.png",
    )
    plot_scatter(
        plt,
        rows,
        x_key="estimated_weight_compute_speedup",
        y_key="best_finetune_top1",
        xlabel="estimated_weight_compute_speedup",
        ylabel="best_finetune_top1",
        title="HSS Accuracy vs Estimated Speedup",
        output_path=output_dir / "pareto_accuracy_speedup.png",
    )
    plot_scatter(
        plt,
        rows,
        x_key="estimated_weight_compute_speedup",
        y_key="accuracy_drop",
        xlabel="estimated_weight_compute_speedup",
        ylabel="accuracy_drop",
        title="HSS Accuracy Drop vs Estimated Speedup",
        output_path=output_dir / "pareto_accuracy_drop_vs_speedup.png",
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            converted: dict[str, Any] = dict(row)
            for key in [
                "actual_density",
                "actual_sparsity",
                "estimated_weight_compute_speedup",
                "dense_top1",
                "zero_shot_pruned_top1",
                "best_finetune_top1",
                "accuracy_drop",
                "covered_ratio",
            ]:
                converted[key] = _to_float(row.get(key))
            rows.append(converted)
    return rows


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def plot_scatter(
    plt: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        label = str(row.get("pattern", "unknown"))
        ax.scatter([x], [y], s=42)
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"saved_plot={output_path}", flush=True)


if __name__ == "__main__":
    main()
