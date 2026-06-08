from __future__ import annotations

from springdance1.hss.convert import (
    apply_all_weight_masks,
    convert_resnet_to_hss,
    freeze_all_hss_masks,
    mask_optimizer_state,
    summarize_all_hss_masks,
    unfreeze_all_hss_masks,
    update_all_hss_masks,
)
from springdance1.hss.layers import HSSConv2d, HSSLinear
from springdance1.hss.mask import (
    generate_hss_mask_conv2d,
    generate_hss_mask_for_fiber,
    generate_hss_mask_linear,
)
from springdance1.hss.metrics import hss_sparsity, summarize_hss_masks
from springdance1.hss.pattern import HSSPattern
from springdance1.hss.training import (
    add_hss_args,
    enforce_hss_masks_after_step,
    hss_pattern_from_args,
    initialize_hss_masks,
    maybe_convert_model_to_hss,
    maybe_update_hss_masks_for_epoch,
)

__all__ = [
    "HSSConv2d",
    "HSSLinear",
    "HSSPattern",
    "add_hss_args",
    "apply_all_weight_masks",
    "convert_resnet_to_hss",
    "enforce_hss_masks_after_step",
    "freeze_all_hss_masks",
    "generate_hss_mask_conv2d",
    "generate_hss_mask_for_fiber",
    "generate_hss_mask_linear",
    "hss_sparsity",
    "hss_pattern_from_args",
    "initialize_hss_masks",
    "mask_optimizer_state",
    "maybe_convert_model_to_hss",
    "maybe_update_hss_masks_for_epoch",
    "summarize_all_hss_masks",
    "summarize_hss_masks",
    "unfreeze_all_hss_masks",
    "update_all_hss_masks",
]
