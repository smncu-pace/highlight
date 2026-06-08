from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from torch import nn

from springdance1.hss.convert import (
    apply_all_weight_masks,
    convert_resnet_to_hss,
    freeze_all_hss_masks,
    mask_optimizer_state,
    update_all_hss_masks,
)
from springdance1.hss.metrics import summarize_hss_masks
from springdance1.hss.pattern import HSSPattern


def add_hss_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hss", action="store_true", help="Enable HSS masked Conv2d/Linear layers.")
    parser.add_argument("--hss-macro-block-size", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--hss-rank0-enable", action="store_true", default=None, help="Enable C0 G:H magnitude pruning.")
    parser.add_argument("--hss-rank0-g", type=int, default=2, help="C0 G value.")
    parser.add_argument("--hss-rank0-h", type=int, default=4, help="C0 H value.")
    parser.add_argument("--hss-rank1-enable", action="store_true", default=None, help="Enable C1 G:H block pruning.")
    parser.add_argument("--hss-rank1-g", type=int, default=3, help="C1 G value.")
    parser.add_argument("--hss-rank1-h", type=int, default=4, help="C1 H value over C0 blocks.")
    parser.add_argument("--hss-keep-tail", action=argparse.BooleanOptionalAction, default=True, help="Keep incomplete fiber tails dense.")
    parser.add_argument(
        "--hss-include-linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace Linear layers with HSSLinear.",
    )
    parser.add_argument("--hss-prune-first-conv", action="store_true", help="Also prune the top-level ResNet stem conv1.")
    parser.add_argument("--hss-warmup-epochs", type=int, default=0, help="Epochs to wait before periodic mask updates.")
    parser.add_argument("--hss-mask-update-interval", type=int, default=0, help="Update masks every N epochs after warmup; 0 means initial mask only.")
    parser.add_argument("--hss-fixed-mask", action="store_true", help="Freeze masks after initial generation.")
    parser.add_argument("--hss-output-dir", default="", help="Optional HSS-specific output directory; overrides --output-dir in HSS mode.")
    parser.add_argument("--hss-eval-before-finetune", action="store_true", help="Evaluate dense and zero-shot pruned accuracy before fine-tune.")
    parser.add_argument("--hss-epochs", type=int, default=0, help="Optional HSS-specific epoch count; overrides --epochs in HSS mode.")
    parser.add_argument("--hss-lr", type=float, default=0.0, help="Optional HSS-specific LR; overrides --lr in HSS mode.")
    parser.add_argument("--hss-reg", action="store_true", help="Enable HSS auxiliary regularization in the loss.")
    parser.add_argument("--hss-lambda-value", type=float, default=1e-6, help="Value-level L1 regularization weight.")
    parser.add_argument("--hss-lambda-block", type=float, default=1e-5, help="Macro-block group regularization weight.")
    return parser


def hss_pattern_from_args(args: argparse.Namespace) -> HSSPattern:
    rank0_enable = getattr(args, "hss_rank0_enable", None)
    rank1_enable = getattr(args, "hss_rank1_enable", None)
    if rank0_enable is None and rank1_enable is None:
        rank0_enable = True
        rank1_enable = True
    else:
        rank0_enable = bool(rank0_enable)
        rank1_enable = bool(rank1_enable)
    return HSSPattern(
        rank0_enable=rank0_enable,
        rank0_g=getattr(args, "hss_rank0_g", 2),
        rank0_h=getattr(args, "hss_rank0_h", 4),
        rank1_enable=rank1_enable,
        rank1_g=getattr(args, "hss_rank1_g", 3),
        rank1_h=getattr(args, "hss_rank1_h", 4),
        keep_tail=getattr(args, "hss_keep_tail", True),
    )


def _emit_mask_summary(summary: dict[str, int | float], logger: Callable[[str], Any] | None = None) -> None:
    message = (
        "HSS masks: "
        f"numel={summary['numel']} "
        f"nonzero={summary['nonzero']} "
        f"zero={summary['zero']} "
        f"density={summary['density']:.6f} "
        f"sparsity={summary['sparsity']:.6f} "
        f"covered_ratio={summary.get('covered_ratio', 0.0):.6f} "
        f"estimated_weight_compute_speedup={summary.get('estimated_weight_compute_speedup', 0.0):.6f}"
    )
    if logger is None:
        print(message)
    else:
        logger(message)


def maybe_convert_model_to_hss(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not getattr(args, "hss", False):
        return model
    return convert_resnet_to_hss(
        model,
        pattern=hss_pattern_from_args(args),
        include_linear=args.hss_include_linear,
        prune_first_conv=args.hss_prune_first_conv,
    )


def initialize_hss_masks(
    model: nn.Module,
    args: argparse.Namespace,
    logger: Callable[[str], Any] | None = None,
) -> dict[str, int | float] | None:
    if not getattr(args, "hss", False):
        return None

    update_all_hss_masks(model)
    apply_all_weight_masks(model)
    if getattr(args, "hss_fixed_mask", False):
        freeze_all_hss_masks(model)

    summary = summarize_hss_masks(model)
    _emit_mask_summary(summary, logger)
    return summary


def maybe_update_hss_masks_for_epoch(
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    logger: Callable[[str], Any] | None = None,
) -> dict[str, int | float] | None:
    if not getattr(args, "hss", False):
        return None
    if getattr(args, "hss_fixed_mask", False):
        return None
    interval = getattr(args, "hss_mask_update_interval", 0)
    warmup = getattr(args, "hss_warmup_epochs", 0)
    if interval <= 0 or epoch < warmup:
        return None
    if (epoch - warmup) % interval != 0:
        return None

    update_all_hss_masks(model)
    apply_all_weight_masks(model)
    summary = summarize_hss_masks(model)
    _emit_mask_summary(summary, logger)
    return summary


def enforce_hss_masks_after_step(model: nn.Module, optimizer: Any | None = None) -> None:
    apply_all_weight_masks(model)
    if optimizer is not None:
        mask_optimizer_state(optimizer, model)
