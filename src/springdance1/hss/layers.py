from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from springdance1.hss.mask import generate_hss_mask_conv2d, generate_hss_mask_linear
from springdance1.hss.pattern import HSSPattern


class HSSConv2d(nn.Conv2d):
    def __init__(self, *args: object, pattern: HSSPattern | None = None, **kwargs: object) -> None:
        kwargs.pop("macro_block_size", None)
        super().__init__(*args, **kwargs)
        self.hss_pattern = pattern or HSSPattern()
        self.hss_enabled = True
        self.mask_frozen = False
        self.register_buffer("hss_mask", torch.ones_like(self.weight))
        self.weight.register_hook(self._mask_gradient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.weight * self.hss_mask if self.hss_enabled else self.weight
        return F.conv2d(x, w_eff, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @torch.no_grad()
    def update_hss_mask(self) -> None:
        if self.mask_frozen:
            return
        self.hss_mask.copy_(generate_hss_mask_conv2d(self.weight, self.hss_pattern))
        self.apply_weight_mask()

    @torch.no_grad()
    def apply_weight_mask(self) -> None:
        self.weight.mul_(self.hss_mask)

    @torch.no_grad()
    def freeze_mask(self) -> None:
        self.mask_frozen = True

    @torch.no_grad()
    def unfreeze_mask(self) -> None:
        self.mask_frozen = False

    def _mask_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.hss_enabled:
            return grad
        return grad * self.hss_mask


class HSSLinear(nn.Linear):
    def __init__(self, *args: object, pattern: HSSPattern | None = None, **kwargs: object) -> None:
        kwargs.pop("macro_block_size", None)
        super().__init__(*args, **kwargs)
        self.hss_pattern = pattern or HSSPattern()
        self.hss_enabled = True
        self.mask_frozen = False
        self.register_buffer("hss_mask", torch.ones_like(self.weight))
        self.weight.register_hook(self._mask_gradient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.weight * self.hss_mask if self.hss_enabled else self.weight
        return F.linear(x, w_eff, self.bias)

    @torch.no_grad()
    def update_hss_mask(self) -> None:
        if self.mask_frozen:
            return
        self.hss_mask.copy_(generate_hss_mask_linear(self.weight, self.hss_pattern))
        self.apply_weight_mask()

    @torch.no_grad()
    def apply_weight_mask(self) -> None:
        self.weight.mul_(self.hss_mask)

    @torch.no_grad()
    def freeze_mask(self) -> None:
        self.mask_frozen = True

    @torch.no_grad()
    def unfreeze_mask(self) -> None:
        self.mask_frozen = False

    def _mask_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.hss_enabled:
            return grad
        return grad * self.hss_mask
