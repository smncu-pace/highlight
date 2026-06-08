from __future__ import annotations

import math

import torch
from torch import nn

from springdance1.hss.layers import HSSConv2d, HSSLinear
from springdance1.hss.mask import covered_elements_for_conv2d, covered_elements_for_linear
from springdance1.hss.pattern import HSSPattern


def hss_sparsity(mask: torch.Tensor) -> dict[str, int | float]:
    numel = int(mask.numel())
    nonzero = int(torch.count_nonzero(mask).item())
    zero = numel - nonzero
    density = float(nonzero / numel) if numel else 0.0
    return {
        "numel": numel,
        "nonzero": nonzero,
        "zero": zero,
        "density": density,
        "sparsity": 1.0 - density if numel else 0.0,
    }


def summarize_hss_masks(model: nn.Module) -> dict[str, int | float]:
    total_numel = 0
    total_nonzero = 0
    total_covered = 0

    for module in model.modules():
        mask = getattr(module, "hss_mask", None)
        pattern = getattr(module, "hss_pattern", None)
        if not isinstance(mask, torch.Tensor):
            continue
        total_numel += int(mask.numel())
        total_nonzero += int(torch.count_nonzero(mask).item())
        if isinstance(pattern, HSSPattern):
            if isinstance(module, HSSConv2d):
                total_covered += covered_elements_for_conv2d(tuple(module.weight.shape), pattern)
            elif isinstance(module, HSSLinear):
                total_covered += covered_elements_for_linear(tuple(module.weight.shape), pattern)
            else:
                total_covered += int(mask.numel())
        else:
            total_covered += int(mask.numel())

    total_zero = total_numel - total_nonzero
    density = float(total_nonzero / total_numel) if total_numel else 0.0
    covered_ratio = float(total_covered / total_numel) if total_numel else 0.0
    return {
        "numel": total_numel,
        "nonzero": total_nonzero,
        "zero": total_zero,
        "density": density,
        "sparsity": 1.0 - density if total_numel else 0.0,
        "covered": total_covered,
        "covered_ratio": covered_ratio,
        "estimated_weight_compute_speedup": estimated_weight_compute_speedup(density),
    }


def estimated_weight_compute_speedup(actual_density: float) -> float:
    """Theoretical dense-multiply reduction estimate, not measured latency."""
    if actual_density <= 0:
        return math.inf
    return 1.0 / actual_density
