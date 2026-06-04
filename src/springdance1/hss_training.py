from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from torch import nn

from springdance1.hss_convert import (
    convert_resnet_to_hss,
    freeze_all_hss_masks,
    update_all_hss_masks,
)
from springdance1.hss_mask import summarize_hss_masks


def add_hss_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hss", action="store_true", help="Enable HSS masked Conv2d/Linear layers.")
    parser.add_argument("--hss-macro-block-size", type=int, default=16, help="Number of values in each macro block.")
    parser.add_argument("--hss-include-linear", action="store_true", help="Also replace Linear layers with HSSLinear.")
    parser.add_argument("--hss-prune-first-conv", action="store_true", help="Also prune the top-level ResNet stem conv1.")
    parser.add_argument("--hss-warmup-epochs", type=int, default=0, help="Epochs to wait before periodic mask updates.")
    parser.add_argument("--hss-mask-update-interval", type=int, default=0, help="Update masks every N epochs after warmup; 0 means initial mask only.")
    parser.add_argument("--hss-fixed-mask", action="store_true", help="Freeze masks after initial generation.")
    parser.add_argument("--hss-reg", action="store_true", help="Enable HSS auxiliary regularization in the loss.")
    parser.add_argument("--hss-lambda-value", type=float, default=1e-6, help="Value-level L1 regularization weight.")
    parser.add_argument("--hss-lambda-block", type=float, default=1e-5, help="Macro-block group regularization weight.")
    return parser


def _emit_mask_summary(summary: dict[str, int | float], logger: Callable[[str], Any] | None = None) -> None:
    message = (
        "HSS masks: "
        f"numel={summary['numel']} "
        f"nonzero={summary['nonzero']} "
        f"zero={summary['zero']} "
        f"density={summary['density']:.6f} "
        f"sparsity={summary['sparsity']:.6f}"
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
        macro_block_size=args.hss_macro_block_size,
        include_linear=args.hss_include_linear,
        skip_first_conv=not args.hss_prune_first_conv,
    )


def initialize_hss_masks(
    model: nn.Module,
    args: argparse.Namespace,
    logger: Callable[[str], Any] | None = None,
) -> dict[str, int | float] | None:
    if not getattr(args, "hss", False):
        return None

    update_all_hss_masks(model)
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
    summary = summarize_hss_masks(model)
    _emit_mask_summary(summary, logger)
    return summary
