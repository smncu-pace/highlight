from __future__ import annotations

import torch
from torch import nn

from springdance1.hss.convert import iter_hss_modules


def hss_regularization(
    model: nn.Module,
    lambda_value: float = 1e-6,
    lambda_block: float = 1e-5,
) -> torch.Tensor:
    total: torch.Tensor | None = None

    for module in iter_hss_modules(model):
        weight = module.weight
        reg = lambda_value * weight.abs().sum()
        pattern = module.hss_pattern
        block_size = pattern.rank0_h
        flat = weight.flatten()
        usable = (flat.numel() // block_size) * block_size
        if usable > 0:
            blocks = flat[:usable].view(-1, block_size)
            block_scores = torch.sqrt(torch.mean(blocks.float().pow(2), dim=1))
            reg = reg + lambda_block * block_scores.sum().to(dtype=weight.dtype)
        total = reg if total is None else total + reg

    if total is not None:
        return total

    try:
        first_param = next(model.parameters())
        return first_param.new_tensor(0.0)
    except StopIteration:
        return torch.tensor(0.0)
