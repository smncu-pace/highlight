
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from springdance1.hss_convert import freeze_all_hss_masks
from springdance1.hss_mask import summarize_hss_masks
from springdance1.hss_training import maybe_convert_model_to_hss
from springdance1.models import resnet50
from train import load_checkpoint_if_needed


class ImageNet100TTADataset(Dataset):
    def __init__(self, hf_dataset, transform: transforms.Compose) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.hf_dataset[index]
        image = sample["image"].convert("RGB")
        crops = self.transform(image)
        label = int(sample["label"])
        return crops, label


def make_tta_transform(resize_size: int, crop_mode: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def stack_crops(crops):
        return torch.stack([normalize(transforms.functional.to_tensor(crop)) for crop in crops])

    if crop_mode == "center":
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(lambda tensor: tensor.unsqueeze(0)),
        ])
    if crop_mode == "five":
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.FiveCrop(224),
            transforms.Lambda(stack_crops),
        ])
    if crop_mode == "ten":
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.TenCrop(224),
            transforms.Lambda(stack_crops),
        ])
    raise ValueError(f"Unknown crop mode: {crop_mode}")


def load_val_dataset(data_dir: str, cache_dir: str, hf_dataset: str):
    data_dir_path = Path(data_dir)
    val_files = sorted((data_dir_path / "data").glob("validation-*.parquet"))
    if val_files:
        return load_dataset("parquet", data_files={"validation": [str(path) for path in val_files]}, cache_dir=cache_dir)["validation"]
    return load_dataset(hf_dataset, cache_dir=cache_dir)["validation"]


def build_model(args: argparse.Namespace) -> nn.Module:
    model = resnet50(num_classes=100)
    if args.hss:
        model = maybe_convert_model_to_hss(model, args)
    load_checkpoint_if_needed(model, args.checkpoint)
    if args.hss and args.hss_fixed_mask:
        freeze_all_hss_masks(model)
    model.to(args.device)
    model.eval()
    if args.hss:
        summary = summarize_hss_masks(model)
        print(
            "HSS masks: "
            f"density={summary[density]:.6f} "
            f"sparsity={summary[sparsity]:.6f} "
            f"nonzero={summary[nonzero]} numel={summary[numel]}"
        )
    return model


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, args: argparse.Namespace) -> tuple[float, float]:
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for crops, labels in tqdm(loader, desc="eval", disable=args.no_progress):
        labels = labels.to(args.device, non_blocking=True)
        bsz, ncrops, channels, height, width = crops.shape
        crops = crops.view(bsz * ncrops, channels, height, width).to(args.device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=args.amp and args.device.startswith("cuda")):
            logits = model(crops).view(bsz, ncrops, -1).mean(dim=1)
            loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        loss_total += float(loss.item())
    return loss_total / max(total, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ImageNet-100 checkpoints with optional TTA.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/imagenet-100-hf")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--hf-dataset", default="clane9/imagenet-100")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-mode", choices=["center", "five", "ten"], default="center")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--hss", action="store_true")
    parser.add_argument("--hss-macro-block-size", type=int, default=16)
    parser.add_argument("--hss-include-linear", action="store_true")
    parser.add_argument("--hss-prune-first-conv", action="store_true")
    parser.add_argument("--hss-fixed-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = make_tta_transform(args.resize_size, args.crop_mode)
    val_dataset = load_val_dataset(args.data_dir, args.cache_dir, args.hf_dataset)
    val_set = ImageNet100TTADataset(val_dataset, transform)
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    model = build_model(args)
    loss, acc = evaluate(model, loader, args)
    print(
        f"checkpoint={args.checkpoint} resize={args.resize_size} crop_mode={args.crop_mode} "
        f"val_loss={loss:.4f} val_acc={acc:.4f}"
    )


if __name__ == "__main__":
    main()
