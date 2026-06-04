
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from springdance1.hss_convert import freeze_all_hss_masks
from springdance1.hss_mask import summarize_hss_masks
from springdance1.hss_training import maybe_convert_model_to_hss
from springdance1.models import resnet50
from train import load_checkpoint_if_needed
from scripts.evaluate import ImageNet100TTADataset, load_val_dataset, make_tta_transform


class EnsembleMember(nn.Module):
    def __init__(self, checkpoint: str, args: argparse.Namespace) -> None:
        super().__init__()
        model = resnet50(num_classes=100)
        if args.hss:
            model = maybe_convert_model_to_hss(model, args)
        load_checkpoint_if_needed(model, checkpoint)
        if args.hss and args.hss_fixed_mask:
            freeze_all_hss_masks(model)
        model.eval()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_models(args: argparse.Namespace) -> list[nn.Module]:
    models: list[nn.Module] = []
    for checkpoint in args.checkpoints:
        member = EnsembleMember(checkpoint, args).to(args.device)
        models.append(member)
    if args.hss:
        summary = summarize_hss_masks(models[0])
        print(
            "HSS masks: "
            f"density={summary[density]:.6f} "
            f"sparsity={summary[sparsity]:.6f} "
            f"nonzero={summary[nonzero]} numel={summary[numel]}"
        )
    return models


@torch.inference_mode()
def evaluate(models: list[nn.Module], loader: DataLoader, args: argparse.Namespace) -> tuple[float, float]:
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for crops, labels in tqdm(loader, desc="eval", disable=args.no_progress):
        labels = labels.to(args.device, non_blocking=True)
        bsz, ncrops, channels, height, width = crops.shape
        crops = crops.view(bsz * ncrops, channels, height, width).to(args.device, non_blocking=True)
        logits_sum = None
        with torch.autocast(device_type="cuda", enabled=args.amp and args.device.startswith("cuda")):
            for model in models:
                logits = model(crops).view(bsz, ncrops, -1).mean(dim=1)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            assert logits_sum is not None
            logits_avg = logits_sum / len(models)
            loss = criterion(logits_avg, labels)
        pred = logits_avg.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        loss_total += float(loss.item())
    return loss_total / max(total, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ImageNet-100 checkpoint ensemble.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
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
    val_set: Dataset = ImageNet100TTADataset(val_dataset, transform)
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    models = build_models(args)
    loss, acc = evaluate(models, loader, args)
    print(
        f"checkpoints={len(args.checkpoints)} resize={args.resize_size} crop_mode={args.crop_mode} "
        f"val_loss={loss:.4f} val_acc={acc:.4f}"
    )


if __name__ == "__main__":
    main()
