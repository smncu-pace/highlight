from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from springdance1.hss_convert import freeze_all_hss_masks
from springdance1.hss_mask import summarize_hss_masks
from springdance1.hss_regularization import hss_regularization
from springdance1.hss_training import (
    add_hss_args,
    initialize_hss_masks,
    maybe_convert_model_to_hss,
    maybe_update_hss_masks_for_epoch,
)
from springdance1.models import resnet50


class ImageNet100Dataset(Dataset):
    def __init__(self, hf_dataset, transform: transforms.Compose) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.hf_dataset[index]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])
        return self.transform(image), label


def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


def build_transforms(train: bool, args: argparse.Namespace | None = None) -> transforms.Compose:
    if train:
        transform_list: list[object] = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        if args is not None and args.randaugment:
            transform_list.append(transforms.RandAugment(num_ops=args.randaugment_ops, magnitude=args.randaugment_magnitude))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        if args is not None and args.random_erasing > 0:
            transform_list.append(transforms.RandomErasing(p=args.random_erasing))
        return transforms.Compose(transform_list)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ImageNet-100 with optional HSS masks.")
    parser.add_argument("--arch", default="resnet50", choices=["resnet50"])
    parser.add_argument("--dataset", default="imagenet100", choices=["imagenet100"])
    parser.add_argument("--hf-dataset", default="clane9/imagenet-100")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--data-dir", default="data/imagenet-100-hf", help="Local snapshot directory with parquet files.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "cosine"])
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-after-hss", action="store_true", help="Load an HSS checkpoint after layer conversion.")
    parser.add_argument("--teacher-checkpoint", default="", help="Dense teacher checkpoint for knowledge distillation.")
    parser.add_argument("--distill-alpha", type=float, default=0.0, help="Weight for teacher KL loss; 0 disables distillation.")
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug limit; 0 means full epoch.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Debug limit; 0 means full validation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", help="Initialize from torchvision ImageNet-1K ResNet-50 weights except fc.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--randaugment-ops", type=int, default=2)
    parser.add_argument("--randaugment-magnitude", type=int, default=9)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--model-ema-decay", type=float, default=0.999)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return add_hss_args(parser).parse_args()


def imagenet100_indices(
    synset_path: str = "metadata/imagenet100.txt",
    class_index_path: str = "metadata/imagenet_class_index.json",
) -> list[int] | None:
    synset_file = Path(synset_path)
    class_index_file = Path(class_index_path)
    if not synset_file.exists() or not class_index_file.exists():
        return None

    synsets = [line.strip() for line in synset_file.read_text().splitlines() if line.strip()]
    class_index = json.loads(class_index_file.read_text())
    synset_to_index = {value[0]: int(key) for key, value in class_index.items()}
    if any(synset not in synset_to_index for synset in synsets):
        return None
    return [synset_to_index[synset] for synset in synsets]


def load_pretrained_if_needed(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    weights = ResNet50_Weights.DEFAULT
    state_dict = weights.get_state_dict(progress=True)
    indices = imagenet100_indices()
    if indices is not None:
        state_dict["fc.weight"] = state_dict["fc.weight"][indices].clone()
        state_dict["fc.bias"] = state_dict["fc.bias"][indices].clone()
        print("Initialized fc from matching ImageNet-1K rows for ImageNet-100 classes.")
    else:
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("fc.")}
        print("ImageNet-100 synset metadata unavailable; loading pretrained backbone except fc.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded torchvision ResNet-50 ImageNet-1K pretrained weights.")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")


def load_checkpoint_if_needed(model: nn.Module, path: str) -> None:
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {path}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")


def make_teacher_if_needed(args: argparse.Namespace) -> nn.Module | None:
    if not args.teacher_checkpoint or args.distill_alpha <= 0:
        return None

    teacher = resnet50(num_classes=100)
    load_checkpoint_if_needed(teacher, args.teacher_checkpoint)
    teacher.to(args.device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("distill-temperature must be positive")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 1:
        return F.cross_entropy(logits, target)
    return torch.sum(-target * F.log_softmax(logits, dim=1), dim=1).mean()


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return F.one_hot(labels, num_classes=num_classes).float()
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    target = torch.full((labels.shape[0], num_classes), off_value, device=labels.device)
    target.scatter_(1, labels.unsqueeze(1), on_value)
    return target


def apply_mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixup_alpha = args.mixup_alpha
    cutmix_alpha = args.cutmix_alpha
    target = smooth_one_hot(labels, num_classes, args.label_smoothing)
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, target

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    if use_cutmix:
        width = images.size(3)
        height = images.size(2)
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    else:
        images = images * lam + images[index] * (1.0 - lam)

    mixed_target = target * lam + target[index] * (1.0 - lam)
    return images, mixed_target


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    return None


def print_hss_summary(model: nn.Module) -> None:
    summary = summarize_hss_masks(model)
    print(
        "HSS masks: "
        f"numel={summary['numel']} "
        f"nonzero={summary['nonzero']} "
        f"zero={summary['zero']} "
        f"density={summary['density']:.6f} "
        f"sparsity={summary['sparsity']:.6f}"
    )


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    train_files = sorted((data_dir / "data").glob("train-*.parquet"))
    val_files = sorted((data_dir / "data").glob("validation-*.parquet"))
    if train_files and val_files:
        dataset = load_dataset(
            "parquet",
            data_files={
                "train": [str(path) for path in train_files],
                "validation": [str(path) for path in val_files],
            },
            cache_dir=str(cache_dir),
        )
    else:
        dataset = load_dataset(args.hf_dataset, cache_dir=str(cache_dir))

    train_set = ImageNet100Dataset(dataset["train"], build_transforms(train=True, args=args))
    val_set = ImageNet100Dataset(dataset["validation"], build_transforms(train=False, args=args))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    teacher: nn.Module | None,
    ema: ModelEma | None,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    loss_total = 0.0
    acc_total = 0.0
    steps = 0
    iterator = tqdm(loader, desc=f"train epoch {epoch}", leave=False, disable=args.no_progress)
    for images, labels in iterator:
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=args.amp and args.device.startswith("cuda")):
            images, targets = apply_mixup_cutmix(images, labels, args, num_classes=100)
            logits = model(images)
            ce_loss = soft_cross_entropy(logits, targets)
            loss = ce_loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                kd_loss = distillation_loss(logits, teacher_logits, args.distill_temperature)
                loss = (1.0 - args.distill_alpha) * ce_loss + args.distill_alpha * kd_loss
            if args.hss_reg:
                loss = loss + hss_regularization(model, args.hss_lambda_value, args.hss_lambda_block)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        loss_total += float(loss.detach().item())
        acc_total += accuracy_top1(logits.detach(), labels)
        steps += 1
        iterator.set_postfix(loss=loss_total / steps, acc=acc_total / steps)
        if args.max_train_batches and steps >= args.max_train_batches:
            break
    return loss_total / max(steps, 1), acc_total / max(steps, 1)


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, args: argparse.Namespace) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    acc_total = 0.0
    steps = 0
    iterator = tqdm(loader, desc="val", leave=False, disable=args.no_progress)
    for images, labels in iterator:
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        loss_total += float(loss.item())
        acc_total += accuracy_top1(logits, labels)
        steps += 1
        iterator.set_postfix(loss=loss_total / steps, acc=acc_total / steps)
        if args.max_val_batches and steps >= args.max_val_batches:
            break
    return loss_total / max(steps, 1), acc_total / max(steps, 1)


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = args.device.startswith("cuda")

    train_loader, val_loader = make_loaders(args)

    model = resnet50(num_classes=100)
    load_pretrained_if_needed(model, args.pretrained)
    if not args.resume_after_hss:
        load_checkpoint_if_needed(model, args.resume)
    model = maybe_convert_model_to_hss(model, args)
    if args.resume_after_hss:
        load_checkpoint_if_needed(model, args.resume)
    model.to(args.device)
    if args.resume_after_hss and args.hss:
        if args.hss_fixed_mask:
            freeze_all_hss_masks(model)
        print_hss_summary(model)
    else:
        initialize_hss_masks(model, args)
    teacher = make_teacher_if_needed(args)
    ema = ModelEma(model, args.model_ema_decay) if args.model_ema else None

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))
    best_acc = 0.0

    for epoch in range(args.epochs):
        maybe_update_hss_masks_for_epoch(model, args, epoch)
        train_loss, train_acc = train_one_epoch(model, teacher, ema, train_loader, criterion, optimizer, scaler, args, epoch)
        val_model = ema.module if ema is not None else model
        val_loss, val_acc = validate(val_model, val_loader, criterion, args)
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6g}",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = output_dir / "best.pth"
            best_state = ema.module.state_dict() if ema is not None else model.state_dict()
            torch.save({"model": best_state, "args": vars(args), "epoch": epoch, "val_acc": val_acc}, best_path)
            print(f"saved_best={best_path} val_acc={val_acc:.4f}", flush=True)

    checkpoint_path = output_dir / "last.pth"
    final_state = ema.module.state_dict() if ema is not None else model.state_dict()
    torch.save({"model": final_state, "args": vars(args), "val_acc": best_acc}, checkpoint_path)
    print(f"saved={checkpoint_path}", flush=True)
    print(f"elapsed_sec={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
