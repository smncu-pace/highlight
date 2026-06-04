from __future__ import annotations

from torch import nn

from springdance1.hss_layers import HSSConv2d, HSSLinear


def _copy_parameters(new_module: nn.Module, old_module: nn.Module) -> None:
    new_module.weight.data.copy_(old_module.weight.data)
    if getattr(old_module, "bias", None) is not None and getattr(new_module, "bias", None) is not None:
        new_module.bias.data.copy_(old_module.bias.data)


def _convert_child(
    qualified_name: str,
    child: nn.Module,
    macro_block_size: int,
    include_linear: bool,
    skip_first_conv: bool,
) -> nn.Module:
    if isinstance(child, (HSSConv2d, HSSLinear)):
        return child

    if isinstance(child, nn.Conv2d):
        if skip_first_conv and qualified_name == "conv1":
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
            macro_block_size=macro_block_size,
        )
        _copy_parameters(new_conv, child)
        return new_conv

    if include_linear and isinstance(child, nn.Linear):
        new_linear = HSSLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            macro_block_size=macro_block_size,
        )
        _copy_parameters(new_linear, child)
        return new_linear

    return child


def convert_resnet_to_hss(
    model: nn.Module,
    macro_block_size: int = 16,
    include_linear: bool = True,
    skip_first_conv: bool = True,
    _prefix: str = "",
) -> nn.Module:
    """Recursively replace Conv2d/Linear modules with HSS masked versions.

    Dense checkpoints can be loaded before conversion, or after conversion with
    ``strict=False`` so missing ``hss_mask`` buffers do not break loading. When
    ``skip_first_conv`` is true, only the top-level ResNet stem ``conv1`` is left
    dense; bottleneck layers named ``conv1`` are still converted.
    """
    for name, child in list(model.named_children()):
        qualified_name = f"{_prefix}.{name}" if _prefix else name
        converted_child = _convert_child(qualified_name, child, macro_block_size, include_linear, skip_first_conv)
        if converted_child is child:
            convert_resnet_to_hss(converted_child, macro_block_size, include_linear, skip_first_conv, qualified_name)
        else:
            setattr(model, name, converted_child)
    return model


def update_all_hss_masks(model: nn.Module) -> None:
    for module in model.modules():
        update = getattr(module, "update_hss_mask", None)
        if callable(update):
            update()


def freeze_all_hss_masks(model: nn.Module) -> None:
    for module in model.modules():
        freeze = getattr(module, "freeze_mask", None)
        if callable(freeze):
            freeze()


def unfreeze_all_hss_masks(model: nn.Module) -> None:
    for module in model.modules():
        unfreeze = getattr(module, "unfreeze_mask", None)
        if callable(unfreeze):
            unfreeze()
