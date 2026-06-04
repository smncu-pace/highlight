from __future__ import annotations

import torch
from torch import nn


def generate_hss_mask_3_4_2_4(weight: torch.Tensor, macro_block_size: int = 16) -> torch.Tensor:
    """Generate a hierarchical 3:4 macro x 2:4 micro sparsity mask.

    The returned mask has the same shape, dtype, and device as ``weight``. Tail
    weights that cannot form complete macro or macro-group structures are kept by
    default for shape safety.
    """
    if macro_block_size <= 0:
        raise ValueError("macro_block_size must be positive")
    if macro_block_size % 4 != 0:
        raise ValueError("macro_block_size must be divisible by 4")

    flat = weight.detach().flatten()
    mask_flat = torch.ones_like(flat)
    num_blocks = flat.numel() // macro_block_size
    if num_blocks < 4:
        return mask_flat.view_as(weight)

    grouped_blocks = (num_blocks // 4) * 4
    usable = grouped_blocks * macro_block_size
    if usable == 0:
        return mask_flat.view_as(weight)

    blocks = flat[:usable].view(grouped_blocks, macro_block_size)
    abs_blocks = blocks.abs()

    block_scores = torch.sqrt(torch.mean(blocks.float().pow(2), dim=1))
    score_groups = block_scores.view(-1, 4)
    keep_indices = torch.topk(score_groups, k=3, dim=1).indices

    macro_mask = torch.zeros_like(blocks)
    macro_keep_blocks = torch.zeros((score_groups.shape[0], 4), dtype=torch.bool, device=weight.device)
    macro_keep_blocks.scatter_(1, keep_indices, True)
    macro_mask[macro_keep_blocks.view(-1)] = 1

    micro_groups = abs_blocks.view(grouped_blocks, -1, 4)
    micro_keep_indices = torch.topk(micro_groups, k=2, dim=2).indices
    micro_mask = torch.zeros_like(micro_groups)
    micro_mask.scatter_(2, micro_keep_indices, 1)
    micro_mask = micro_mask.view(grouped_blocks, macro_block_size)

    final_blocks = macro_mask * micro_mask
    mask_flat[:usable] = final_blocks.flatten().to(dtype=weight.dtype)
    # Tail weights are kept by default for shape safety:
    # - incomplete macro blocks
    # - final macro block group with fewer than 4 blocks
    return mask_flat.view_as(weight)


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

    for module in model.modules():
        mask = getattr(module, "hss_mask", None)
        if isinstance(mask, torch.Tensor):
            total_numel += int(mask.numel())
            total_nonzero += int(torch.count_nonzero(mask).item())

    total_zero = total_numel - total_nonzero
    density = float(total_nonzero / total_numel) if total_numel else 0.0
    return {
        "numel": total_numel,
        "nonzero": total_nonzero,
        "zero": total_zero,
        "density": density,
        "sparsity": 1.0 - density if total_numel else 0.0,
    }
