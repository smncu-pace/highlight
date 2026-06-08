# springdance1

PyTorch ResNet-50 ImageNet-100 experiments with a Level 1 software reproduction path for HighLight-style hierarchical structured sparsity (HSS).

This repository does not implement a hardware accelerator or CUDA sparse kernels. HSS is applied by masked PyTorch layers that run dense operators with `weight * hss_mask`.

## Environment

- Python 3.12
- PyTorch / torchvision managed by `uv`
- Dataset default: `clane9/imagenet-100` or local parquet snapshot under `data/imagenet-100-hf`

## Dense Model

```python
from springdance1.models import resnet50

model = resnet50(num_classes=100)
```

## HSS Level 1 Implementation

The configurable two-rank pattern is:

```text
C1(G1:H1) -> C0(G0:H0)
```

Default HSS behavior is `C1(3:4) -> C0(2:4)`, giving theoretical density `3/4 * 2/4 = 3/8` and sparsity `62.5%`.

For Conv2d weights `[M, C, R, S]`, masks are generated independently along the input-channel fiber `W[m, :, r, s]`. For Linear weights `[O, I]`, masks are generated independently along each input row `W[o, :]`.

The pruning order is lower-to-higher rank:

1. C0 magnitude pruning keeps the largest `G0` absolute values in each `H0` block.
2. C1 block pruning scores each C0 block by RMS magnitude after C0 masking, then keeps the largest `G1` C0 blocks in each C1 group.

Tail elements that cannot form complete HSS groups are kept by default for shape safety and to avoid accidental accuracy collapse. Metrics report `covered_ratio` for the fraction of masked-layer parameters covered by complete HSS groups.

## CLI Arguments

Key HSS arguments:

- `--hss`
- `--hss-rank0-enable --hss-rank0-g 2 --hss-rank0-h 4`
- `--hss-rank1-enable --hss-rank1-g 3 --hss-rank1-h 4`
- `--hss-include-linear` / `--no-hss-include-linear`
- `--hss-prune-first-conv`
- `--hss-fixed-mask`
- `--hss-mask-update-interval 0`
- `--hss-warmup-epochs 0`
- `--hss-output-dir runs/hss_xxx`
- `--hss-eval-before-finetune`
- `--hss-epochs 30`
- `--hss-lr 1e-3`

By default, the top-level ResNet stem `conv1` is skipped. Linear `fc` is pruned by default unless `--no-hss-include-linear` is passed.

## Commands

Evaluate a dense checkpoint:

```bash
python train.py \
  --arch resnet50 \
  --dataset imagenet100 \
  --resume path/to/dense_resnet50.pth \
  --eval
```

HSS `C1(3:4) -> C0(2:4)` fixed-mask fine-tune:

```bash
python train.py \
  --arch resnet50 \
  --dataset imagenet100 \
  --resume path/to/dense_resnet50.pth \
  --hss \
  --hss-rank0-enable \
  --hss-rank0-g 2 \
  --hss-rank0-h 4 \
  --hss-rank1-enable \
  --hss-rank1-g 3 \
  --hss-rank1-h 4 \
  --hss-fixed-mask \
  --hss-include-linear \
  --hss-output-dir runs/hss_c1_3x4_c0_2x4 \
  --epochs 30 \
  --lr 1e-3
```

HSS periodic mask update:

```bash
python train.py \
  --arch resnet50 \
  --dataset imagenet100 \
  --resume path/to/dense_resnet50.pth \
  --hss \
  --hss-rank0-enable \
  --hss-rank0-g 2 \
  --hss-rank0-h 4 \
  --hss-rank1-enable \
  --hss-rank1-g 3 \
  --hss-rank1-h 4 \
  --hss-mask-update-interval 5 \
  --hss-include-linear \
  --hss-output-dir runs/hss_periodic_c1_3x4_c0_2x4 \
  --epochs 60 \
  --lr 1e-3
```

Run the sweep:

```bash
python scripts/hss_sweep.py \
  --arch resnet50 \
  --dataset imagenet100 \
  --resume path/to/dense_resnet50.pth \
  --output-dir runs/hss_sweep \
  --epochs 30 \
  --lr 1e-3
```

Plot Pareto curves:

```bash
python scripts/plot_hss_pareto.py \
  --summary runs/hss_sweep/summary.csv \
  --output-dir runs/hss_sweep
```

## Outputs

Each HSS run writes:

- `metrics.json`
- `best_checkpoint.pth`
- `best.pth` compatibility alias
- `last_checkpoint.pth`
- `last.pth` compatibility alias
- `train_log.csv`

Sweep output:

- `runs/hss_sweep/<pattern_name>/metrics.json`
- `runs/hss_sweep/<pattern_name>/best_checkpoint.pth`
- `runs/hss_sweep/<pattern_name>/train_log.csv`
- `runs/hss_sweep/summary.csv`

Pareto plots:

- `pareto_accuracy_sparsity.png`
- `pareto_accuracy_speedup.png`
- `pareto_accuracy_drop_vs_speedup.png`

## Checkpoints

Curated trained checkpoints are kept under `weights/`:

- `dense_adamw_cosine_6epoch_best.pth`
- `dense_pretrained_mix_cutmix_10epoch_best.pth`
- `dense_resume_lightaug_ema_5epoch_best.pth`
- `dense_resume_sgd_cosine_8epoch_best.pth`

Large local datasets and experiment outputs are intentionally excluded from git via `.gitignore`. Recreate or download ImageNet-100 data with the scripts in `scripts/`, then point `--data-dir` at the local snapshot.

## Metrics Note

`estimated_weight_compute_speedup = 1.0 / actual_density`.

This is only a theoretical estimate of weight multiply-count reduction in masked layers. It is not measured latency and does not account for memory movement, dense PyTorch kernels, sparse kernel overhead, accelerator scheduling, or hardware-specific utilization.
