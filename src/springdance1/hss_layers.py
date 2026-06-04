from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from springdance1.hss_mask import generate_hss_mask_3_4_2_4


class HSSConv2d(nn.Conv2d):
    def __init__(self, *args: object, macro_block_size: int = 16, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.macro_block_size = macro_block_size
        self.mask_frozen = False
        self.register_buffer("hss_mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.weight * self.hss_mask
        return F.conv2d(x, w_eff, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @torch.no_grad()
    def update_hss_mask(self) -> None:
        if self.mask_frozen:
            return
        new_mask = generate_hss_mask_3_4_2_4(self.weight, self.macro_block_size)
        self.hss_mask.copy_(new_mask)

    @torch.no_grad()
    def freeze_mask(self) -> None:
        self.mask_frozen = True

    @torch.no_grad()
    def unfreeze_mask(self) -> None:
        self.mask_frozen = False


class HSSLinear(nn.Linear):
    def __init__(self, *args: object, macro_block_size: int = 16, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.macro_block_size = macro_block_size
        self.mask_frozen = False
        self.register_buffer("hss_mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.weight * self.hss_mask
        return F.linear(x, w_eff, self.bias)

    @torch.no_grad()
    def update_hss_mask(self) -> None:
        if self.mask_frozen:
            return
        new_mask = generate_hss_mask_3_4_2_4(self.weight, self.macro_block_size)
        self.hss_mask.copy_(new_mask)

    @torch.no_grad()
    def freeze_mask(self) -> None:
        self.mask_frozen = True

    @torch.no_grad()
    def unfreeze_mask(self) -> None:
        self.mask_frozen = False
