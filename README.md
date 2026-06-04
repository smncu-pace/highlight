# springdance1

A PyTorch project managed by `uv`, currently containing a ResNet-50 image classification model and an HSS 3:4 x 2:4 mask system for Conv2d/Linear sparse training experiments.

## Environment

- Python 3.12
- PyTorch managed by `uv`
- CUDA support follows the installed PyTorch wheel

## Smoke test

```bash
uv run springdance1
```

This constructs ResNet-50, converts eligible layers to HSS layers, updates masks, and runs one fake ImageNet-shaped forward pass.

## Use the dense model

```python
from springdance1.models import resnet50

model = resnet50(num_classes=1000)
```

## Use HSS conversion

```python
from springdance1.hss_convert import convert_resnet_to_hss, update_all_hss_masks
from springdance1.hss_mask import summarize_hss_masks
from springdance1.models import resnet50

model = resnet50(num_classes=1000)
model = convert_resnet_to_hss(
    model,
    macro_block_size=16,
    include_linear=True,
    skip_first_conv=True,
)
update_all_hss_masks(model)
print(summarize_hss_masks(model))
```

Dense checkpoints can be loaded before conversion, or after conversion with `strict=False` so newly registered `hss_mask` buffers do not break checkpoint loading.

## Training integration

The current repository does not include a `train.py` yet. When adding HSS to an existing ImageNet-100 training script, import these helpers instead of rewriting the framework:

```python
from springdance1.hss_regularization import hss_regularization
from springdance1.hss_training import (
    add_hss_args,
    initialize_hss_masks,
    maybe_convert_model_to_hss,
    maybe_update_hss_masks_for_epoch,
)

parser = add_hss_args(parser)

model = build_resnet50()
# Load dense checkpoint here if present.
model = maybe_convert_model_to_hss(model, args)
initialize_hss_masks(model, args)

for epoch in range(start_epoch, args.epochs):
    maybe_update_hss_masks_for_epoch(model, args, epoch)
    ce_loss = criterion(output, target)
    loss = ce_loss
    if args.hss_reg:
        loss = loss + hss_regularization(model, args.hss_lambda_value, args.hss_lambda_block)
```

## Added HSS command-line arguments

- `--hss`
- `--hss-macro-block-size`, default `16`
- `--hss-include-linear`
- `--hss-prune-first-conv`
- `--hss-warmup-epochs`, default `0`
- `--hss-mask-update-interval`, default `0`
- `--hss-fixed-mask`
- `--hss-reg`
- `--hss-lambda-value`, default `1e-6`
- `--hss-lambda-block`, default `1e-5`

## Example experiment commands

Dense baseline:

```bash
python train.py --arch resnet50 --dataset imagenet100 ...
```

HSS fixed-mask fine-tune:

```bash
python train.py --arch resnet50 --dataset imagenet100 \
    --resume path/to/dense_checkpoint.pth \
    --hss \
    --hss-macro-block-size 16 \
    --hss-mask-update-interval 0 \
    --hss-fixed-mask \
    --epochs 30 \
    --lr 0.001
```

HSS periodic-mask training:

```bash
python train.py --arch resnet50 --dataset imagenet100 \
    --resume path/to/dense_checkpoint.pth \
    --hss \
    --hss-macro-block-size 16 \
    --hss-warmup-epochs 0 \
    --hss-mask-update-interval 5 \
    --epochs 60 \
    --lr 0.001
```

HSS periodic-mask training with auxiliary regularization:

```bash
python train.py --arch resnet50 --dataset imagenet100 \
    --resume path/to/dense_checkpoint.pth \
    --hss \
    --hss-macro-block-size 16 \
    --hss-mask-update-interval 5 \
    --hss-reg \
    --hss-lambda-value 1e-6 \
    --hss-lambda-block 1e-5 \
    --epochs 60 \
    --lr 0.001
```

## Default pruning coverage

By default, `convert_resnet_to_hss(..., skip_first_conv=True)` leaves the top-level ResNet stem `conv1` dense. Linear layers are included only when `include_linear=True`; in CLI usage that corresponds to passing `--hss-include-linear`.

The first version does not implement CUDA sparse kernels. It applies sparsity by using `weight * hss_mask` in the dense PyTorch forward path.

## Edge cases handled

- `macro_block_size` must be positive and divisible by 4.
- If fewer than 4 complete macro blocks are available, all weights are kept.
- If the number of macro blocks is not a multiple of 4, the final incomplete macro-block group is kept.
- Tail weights that cannot form a complete macro block are kept by default for shape safety.
- Masks are buffers, not parameters, so optimizers do not update them.
- Dense checkpoints can be migrated by loading before conversion or by using `strict=False` after conversion.
