from __future__ import annotations

import torch

from springdance1.hss_convert import convert_resnet_to_hss, update_all_hss_masks
from springdance1.hss_layers import HSSConv2d
from springdance1.hss_mask import generate_hss_mask_3_4_2_4, hss_sparsity
from springdance1.models import resnet50


def test_hss_mask_shape_binary_and_density() -> None:
    weight = torch.arange(1024, dtype=torch.float32).reshape(64, 16)
    mask = generate_hss_mask_3_4_2_4(weight, macro_block_size=16)

    assert mask.shape == weight.shape
    assert mask.dtype == weight.dtype
    assert mask.device == weight.device
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert hss_sparsity(mask)["density"] == 0.375


def test_hss_macro_group_keeps_three_of_four_blocks() -> None:
    blocks = torch.tensor(
        [
            [1.0] * 16,
            [2.0] * 16,
            [3.0] * 16,
            [4.0] * 16,
        ]
    )
    mask = generate_hss_mask_3_4_2_4(blocks, macro_block_size=16).view(4, 16)
    kept_blocks = (mask.sum(dim=1) > 0).tolist()

    assert sum(kept_blocks) == 3
    assert kept_blocks == [False, True, True, True]


def test_hss_micro_groups_keep_two_of_four_values() -> None:
    block0 = torch.arange(1, 17, dtype=torch.float32)
    block1 = block0 + 100
    block2 = block0 + 200
    block3 = block0 + 300
    weight = torch.stack([block0, block1, block2, block3])
    mask = generate_hss_mask_3_4_2_4(weight, macro_block_size=16).view(4, 4, 4)

    for block_mask in mask[1:]:
        assert torch.equal(block_mask.sum(dim=1), torch.full((4,), 2.0))


def test_convert_resnet_to_hss_and_forward() -> None:
    model = resnet50(num_classes=10)
    model = convert_resnet_to_hss(model, macro_block_size=16, include_linear=True, skip_first_conv=True)
    update_all_hss_masks(model)

    assert any(isinstance(module, HSSConv2d) for module in model.modules())

    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 10)
