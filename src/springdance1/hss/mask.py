from __future__ import annotations

import torch

from springdance1.hss.pattern import HSSPattern


def generate_hss_mask_for_fiber(fiber: torch.Tensor, pattern: HSSPattern) -> torch.Tensor:
    """Generate an HSS mask for one C-dimension or input-feature fiber."""
    if fiber.ndim != 1:
        raise ValueError("fiber must be a 1D tensor")
    return _generate_hss_mask_for_fibers(fiber.detach().reshape(1, -1), pattern).reshape_as(fiber)


def generate_hss_mask_conv2d(weight: torch.Tensor, pattern: HSSPattern) -> torch.Tensor:
    """Generate HSS masks for Conv2d weights along the input-channel dimension.

    For ``weight`` with shape ``[M, C, R, S]``, every ``W[m, :, r, s]`` fiber is
    pruned independently. Tail channels that cannot form complete HSS groups are
    kept for shape safety and to avoid accidental accuracy collapse.
    """
    if weight.ndim != 4:
        raise ValueError("Conv2d HSS mask expects weight shape [M, C, R, S]")
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    fibers = weight.detach().permute(0, 2, 3, 1).contiguous().reshape(-1, in_channels)
    mask_fibers = _generate_hss_mask_for_fibers(fibers, pattern)
    return (
        mask_fibers.reshape(out_channels, kernel_h, kernel_w, in_channels)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(dtype=weight.dtype, device=weight.device)
    )


def generate_hss_mask_linear(weight: torch.Tensor, pattern: HSSPattern) -> torch.Tensor:
    """Generate HSS masks for Linear weights along each input-feature row."""
    if weight.ndim != 2:
        raise ValueError("Linear HSS mask expects weight shape [out_features, in_features]")
    return _generate_hss_mask_for_fibers(weight.detach(), pattern).to(dtype=weight.dtype, device=weight.device)


def covered_elements_for_fiber(length: int, pattern: HSSPattern) -> int:
    """Number of elements in complete HSS groups for one fiber."""
    if length < 0:
        raise ValueError("length must be non-negative")
    group_size = _coverage_group_size(pattern)
    if group_size <= 0:
        return length
    return (length // group_size) * group_size


def covered_elements_for_conv2d(weight_shape: tuple[int, ...], pattern: HSSPattern) -> int:
    if len(weight_shape) != 4:
        raise ValueError("Conv2d weight shape must be [M, C, R, S]")
    out_channels, in_channels, kernel_h, kernel_w = weight_shape
    return int(out_channels * kernel_h * kernel_w * covered_elements_for_fiber(in_channels, pattern))


def covered_elements_for_linear(weight_shape: tuple[int, ...], pattern: HSSPattern) -> int:
    if len(weight_shape) != 2:
        raise ValueError("Linear weight shape must be [O, I]")
    out_features, in_features = weight_shape
    return int(out_features * covered_elements_for_fiber(in_features, pattern))


def _coverage_group_size(pattern: HSSPattern) -> int:
    if pattern.rank1_enable:
        return pattern.rank1_h * pattern.rank0_h
    if pattern.rank0_enable:
        return pattern.rank0_h
    return 0


def _generate_hss_mask_for_fibers(fibers: torch.Tensor, pattern: HSSPattern) -> torch.Tensor:
    if fibers.ndim != 2:
        raise ValueError("fibers must be a 2D tensor [num_fibers, fiber_length]")

    num_fibers, fiber_length = fibers.shape
    if fiber_length == 0 or (not pattern.rank0_enable and not pattern.rank1_enable):
        return torch.ones_like(fibers)

    rank0_mask = torch.ones_like(fibers)
    rank1_mask = torch.ones_like(fibers)

    if pattern.rank0_enable:
        rank0_usable = _rank0_usable_length(fiber_length, pattern)
        if rank0_usable > 0:
            rank0_blocks = fibers[:, :rank0_usable].abs().reshape(num_fibers, -1, pattern.rank0_h)
            rank0_keep = torch.topk(rank0_blocks, k=pattern.rank0_g, dim=2).indices
            rank0_block_mask = torch.zeros_like(rank0_blocks)
            rank0_block_mask.scatter_(2, rank0_keep, 1)
            rank0_mask[:, :rank0_usable] = rank0_block_mask.reshape(num_fibers, rank0_usable)

    if pattern.rank1_enable:
        group_size = pattern.rank1_h * pattern.rank0_h
        rank1_usable = (fiber_length // group_size) * group_size
        if rank1_usable > 0:
            payload = fibers[:, :rank1_usable] * rank0_mask[:, :rank1_usable]
            blocks = payload.float().reshape(num_fibers, -1, pattern.rank1_h, pattern.rank0_h)
            scores = torch.sqrt(torch.mean(blocks.pow(2), dim=3))
            rank1_keep = torch.topk(scores, k=pattern.rank1_g, dim=2).indices
            rank1_block_mask = torch.zeros_like(scores)
            rank1_block_mask.scatter_(2, rank1_keep, 1)
            rank1_mask[:, :rank1_usable] = (
                rank1_block_mask.unsqueeze(-1)
                .expand(-1, -1, -1, pattern.rank0_h)
                .reshape(num_fibers, rank1_usable)
                .to(dtype=fibers.dtype)
            )

    final_mask = rank0_mask * rank1_mask
    if not pattern.keep_tail:
        covered = covered_elements_for_fiber(fiber_length, pattern)
        if covered < fiber_length:
            final_mask[:, covered:] = 0
    return final_mask.to(dtype=fibers.dtype, device=fibers.device)


def _rank0_usable_length(fiber_length: int, pattern: HSSPattern) -> int:
    if pattern.rank1_enable:
        group_size = pattern.rank1_h * pattern.rank0_h
        return (fiber_length // group_size) * group_size
    return (fiber_length // pattern.rank0_h) * pattern.rank0_h


def generate_hss_mask_3_4_2_4(weight: torch.Tensor, macro_block_size: int = 16) -> torch.Tensor:
    """Backward-compatible wrapper for the default C1(3:4) -> C0(2:4) pattern."""
    if macro_block_size != 16:
        raise ValueError("generic HSS now uses rank arguments; legacy macro_block_size must be 16")
    pattern = HSSPattern(rank0_enable=True, rank0_g=2, rank0_h=4, rank1_enable=True, rank1_g=3, rank1_h=4)
    if weight.ndim == 1:
        return generate_hss_mask_for_fiber(weight, pattern)
    if weight.ndim == 2:
        return generate_hss_mask_linear(weight, pattern)
    if weight.ndim == 4:
        return generate_hss_mask_conv2d(weight, pattern)
    flat_mask = generate_hss_mask_for_fiber(weight.detach().flatten(), pattern)
    return flat_mask.reshape_as(weight)
