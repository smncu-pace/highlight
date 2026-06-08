from __future__ import annotations

import torch

from springdance1.hss.convert import convert_resnet_to_hss, update_all_hss_masks
from springdance1.hss.layers import HSSConv2d, HSSLinear
from springdance1.hss.mask import (
    generate_hss_mask_conv2d,
    generate_hss_mask_for_fiber,
    generate_hss_mask_linear,
)
from springdance1.hss.metrics import hss_sparsity
from springdance1.hss.pattern import HSSPattern
from springdance1.models import resnet50


def assert_binary_mask(mask: torch.Tensor) -> None:
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_hss_c1_3x4_c0_2x4_length_16_keeps_6_values() -> None:
    pattern = HSSPattern(rank0_enable=True, rank0_g=2, rank0_h=4, rank1_enable=True, rank1_g=3, rank1_h=4)
    fiber = torch.arange(1, 17, dtype=torch.float32)
    mask = generate_hss_mask_for_fiber(fiber, pattern)

    assert int(mask.sum().item()) == 6
    assert_binary_mask(mask)


def test_hss_c1_3x4_c0_2x4_length_32_keeps_12_values() -> None:
    pattern = HSSPattern(rank0_enable=True, rank0_g=2, rank0_h=4, rank1_enable=True, rank1_g=3, rank1_h=4)
    fiber = torch.arange(1, 33, dtype=torch.float32)
    mask = generate_hss_mask_for_fiber(fiber, pattern)

    assert int(mask.sum().item()) == 12


def test_rank0_only_2x4_length_16_keeps_8_values() -> None:
    pattern = HSSPattern(rank0_enable=True, rank0_g=2, rank0_h=4, rank1_enable=False)
    fiber = torch.arange(1, 17, dtype=torch.float32)
    mask = generate_hss_mask_for_fiber(fiber, pattern)

    assert int(mask.sum().item()) == 8


def test_rank1_only_3x4_length_16_keeps_12_values() -> None:
    pattern = HSSPattern(rank0_enable=False, rank0_h=4, rank1_enable=True, rank1_g=3, rank1_h=4)
    fiber = torch.arange(1, 17, dtype=torch.float32)
    mask = generate_hss_mask_for_fiber(fiber, pattern)

    assert int(mask.sum().item()) == 12


def test_conv2d_mask_shape_equals_weight_shape() -> None:
    pattern = HSSPattern()
    weight = torch.randn(8, 16, 3, 3)
    mask = generate_hss_mask_conv2d(weight, pattern)

    assert mask.shape == weight.shape
    assert mask.dtype == weight.dtype
    assert_binary_mask(mask)


def test_linear_mask_shape_equals_weight_shape() -> None:
    pattern = HSSPattern()
    weight = torch.randn(5, 32)
    mask = generate_hss_mask_linear(weight, pattern)

    assert mask.shape == weight.shape
    assert mask.dtype == weight.dtype
    assert_binary_mask(mask)


def test_tail_is_kept_by_default() -> None:
    pattern = HSSPattern()
    fiber = torch.arange(1, 19, dtype=torch.float32)
    mask = generate_hss_mask_for_fiber(fiber, pattern)

    assert int(mask[:16].sum().item()) == 6
    assert torch.equal(mask[16:], torch.ones(2))


def test_sparsity_for_exact_default_pattern_is_3_8_density() -> None:
    pattern = HSSPattern()
    weight = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    mask = generate_hss_mask_linear(weight, pattern)

    assert hss_sparsity(mask)["density"] == 0.375


def test_convert_resnet_to_hss_and_forward() -> None:
    model = resnet50(num_classes=10)
    model = convert_resnet_to_hss(model, pattern=HSSPattern(), include_linear=True, prune_first_conv=False)
    update_all_hss_masks(model)

    assert any(isinstance(module, HSSConv2d) for module in model.modules())
    assert isinstance(model.conv1, torch.nn.Conv2d)
    assert not isinstance(model.conv1, HSSConv2d)
    assert isinstance(model.fc, HSSLinear)

    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 10)


def test_masked_positions_have_zero_gradient() -> None:
    pattern = HSSPattern(rank0_enable=True, rank0_g=2, rank0_h=4, rank1_enable=False)
    layer = HSSLinear(8, 2, bias=False, pattern=pattern)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(2, 8))
    layer.update_hss_mask()

    x = torch.ones(3, 8)
    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad is not None
    masked_grad = layer.weight.grad[layer.hss_mask == 0]
    assert torch.equal(masked_grad, torch.zeros_like(masked_grad))
