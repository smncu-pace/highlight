from __future__ import annotations

import torch

from springdance1.hss.convert import convert_resnet_to_hss, update_all_hss_masks
from springdance1.hss.metrics import summarize_hss_masks
from springdance1.hss.pattern import HSSPattern
from springdance1.models import resnet50


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pattern = HSSPattern()
    model = resnet50(num_classes=1000)
    model = convert_resnet_to_hss(model, pattern=pattern, include_linear=True, prune_first_conv=False).to(device)
    update_all_hss_masks(model)
    model.eval()

    sample = torch.randn(2, 3, 224, 224, device=device)
    with torch.inference_mode():
        output = model(sample)

    print(f"device={device}")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"hss_pattern={pattern.name()}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"hss_summary={summarize_hss_masks(model)}")
