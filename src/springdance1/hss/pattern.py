from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HSSPattern:
    """Two-rank hierarchical structured sparsity pattern.

    Rank 0 is the C0 value-level block, for example 2:4. Rank 1 is the C1
    block-selection rank, for example 3:4 over C0 blocks.
    """

    rank0_enable: bool = True
    rank0_g: int = 2
    rank0_h: int = 4
    rank1_enable: bool = True
    rank1_g: int = 3
    rank1_h: int = 4
    keep_tail: bool = True

    def __post_init__(self) -> None:
        if self.rank0_h <= 0:
            raise ValueError("rank0_h must be a positive integer")
        if self.rank1_h <= 0:
            raise ValueError("rank1_h must be a positive integer")
        if self.rank0_enable:
            _validate_g_h("rank0", self.rank0_g, self.rank0_h)
        if self.rank1_enable:
            _validate_g_h("rank1", self.rank1_g, self.rank1_h)
            if self.rank0_h <= 0:
                raise ValueError("rank1 requires rank0_h because C1 payload is built from C0 blocks")

    def density(self) -> float:
        density = 1.0
        if self.rank0_enable:
            density *= self.rank0_g / self.rank0_h
        if self.rank1_enable:
            density *= self.rank1_g / self.rank1_h
        return float(density)

    def sparsity(self) -> float:
        return 1.0 - self.density()

    def name(self) -> str:
        if self.rank0_enable and self.rank1_enable:
            return f"HSS_C1_{self.rank1_g}x{self.rank1_h}_C0_{self.rank0_g}x{self.rank0_h}"
        if self.rank0_enable:
            return f"C0_{self.rank0_g}x{self.rank0_h}_only"
        if self.rank1_enable:
            return f"C1_{self.rank1_g}x{self.rank1_h}_only"
        return "dense"

    def rank0_label(self) -> str:
        return f"{self.rank0_g}:{self.rank0_h}" if self.rank0_enable else "disabled"

    def rank1_label(self) -> str:
        return f"{self.rank1_g}:{self.rank1_h}" if self.rank1_enable else "disabled"


def _validate_g_h(rank_name: str, g: int, h: int) -> None:
    if h <= 0:
        raise ValueError(f"{rank_name}_h must be positive")
    if g <= 0 or g > h:
        raise ValueError(f"{rank_name}_g must satisfy 0 < G <= H")
