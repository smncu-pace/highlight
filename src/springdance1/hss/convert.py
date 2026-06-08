from __future__ import annotations

import torch
from torch import nn

from springdance1.hss.layers import HSSConv2d, HSSLinear
from springdance1.hss.metrics import summarize_hss_masks
from springdance1.hss.pattern import HSSPattern


def _copy_parameters(new_module: nn.Module, old_module: nn.Module) -> None:
    new_module.to(device=old_module.weight.device, dtype=old_module.weight.dtype)
    new_module.weight.data.copy_(old_module.weight.data)
    old_bias = getattr(old_module, "bias", None)
    new_bias = getattr(new_module, "bias", None)
    if old_bias is not None and new_bias is not None:
        new_bias.data.copy_(old_bias.data)


def _convert_child(
    qualified_name: str,
    child: nn.Module,
    pattern: HSSPattern,
    include_linear: bool,
    prune_first_conv: bool,
) -> nn.Module:
    if isinstance(child, (HSSConv2d, HSSLinear)):
        return child

    if isinstance(child, nn.Conv2d):
        if not prune_first_conv and qualified_name == "conv1":
            return child
        new_conv = HSSConv2d(
            child.in_channels,
            child.out_channels,
            child.kernel_size,
            stride=child.stride,
            padding=child.padding,
            dilation=child.dilation,
            groups=child.groups,
            bias=child.bias is not None,
            padding_mode=child.padding_mode,
            pattern=pattern,
        )
        _copy_parameters(new_conv, child)
        return new_conv

    if include_linear and isinstance(child, nn.Linear):
        new_linear = HSSLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            pattern=pattern,
        )
        _copy_parameters(new_linear, child)
        return new_linear

    return child


def convert_resnet_to_hss(
    model: nn.Module,
    pattern: HSSPattern | None = None,
    include_linear: bool = True,
    prune_first_conv: bool = False,
    skip_first_conv: bool | None = None,
    macro_block_size: int | None = None,
    _prefix: str = "",
) -> nn.Module:
    """Recursively replace Conv2d/Linear modules with HSS masked modules.

    Dense checkpoints should normally be loaded before conversion. The
    ``macro_block_size`` and ``skip_first_conv`` parameters are accepted for
    compatibility with the previous fixed-pattern implementation.
    """
    del macro_block_size
    if skip_first_conv is not None:
        prune_first_conv = not skip_first_conv
    pattern = pattern or HSSPattern()

    for name, child in list(model.named_children()):
        qualified_name = f"{_prefix}.{name}" if _prefix else name
        converted_child = _convert_child(qualified_name, child, pattern, include_linear, prune_first_conv)
        if converted_child is child:
            convert_resnet_to_hss(
                converted_child,
                pattern=pattern,
                include_linear=include_linear,
                prune_first_conv=prune_first_conv,
                _prefix=qualified_name,
            )
        else:
            setattr(model, name, converted_child)
    return model


def iter_hss_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, (HSSConv2d, HSSLinear)):
            yield module


@torch.no_grad()
def update_all_hss_masks(model: nn.Module) -> None:
    for module in iter_hss_modules(model):
        module.update_hss_mask()


@torch.no_grad()
def apply_all_weight_masks(model: nn.Module) -> None:
    for module in iter_hss_modules(model):
        module.apply_weight_mask()


def freeze_all_hss_masks(model: nn.Module) -> None:
    for module in iter_hss_modules(model):
        module.freeze_mask()


def unfreeze_all_hss_masks(model: nn.Module) -> None:
    for module in iter_hss_modules(model):
        module.unfreeze_mask()


def summarize_all_hss_masks(model: nn.Module) -> dict[str, int | float]:
    return summarize_hss_masks(model)


@torch.no_grad()
def mask_optimizer_state(optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
    """Mask optimizer state for pruned weights.

    This handles SGD momentum buffers and Adam-style ``exp_avg`` / ``exp_avg_sq``
    tensors by masking every state tensor that has the same shape as the weight.
    """
    for module in iter_hss_modules(model):
        state = optimizer.state.get(module.weight)
        if not state:
            continue
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.shape == module.weight.shape:
                value.mul_(module.hss_mask)
