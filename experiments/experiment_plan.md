# Experiment Plan

This project supports the required experiment pipeline through CLI arguments.
Full training can be expensive, so the commands below are intended for a
configured machine with the dataset available under `underwater_data/`.

## Shared Outputs

- Final experiment summaries: `experiments/results.csv`
- Training curves: `assets/loss_curves.png` and `assets/miou_curves.png`
- Prediction examples: `assets/prediction_examples.png`
- Best checkpoint per run: `checkpoints/best.pth`

## Smoke Test

```bash
python3 -m compileall src tests
pytest
```

## Full Training Commands

### 1. U-Net baseline, lr=1e-4, image_size=256

```bash
python src/train.py \
  --experiment-name unet_lr1e-4_size256 \
  --lr 1e-4 \
  --batch-size 16 \
  --image-size 256 \
  --epochs 10
```

### 2. U-Net lower learning rate, lr=1e-5

```bash
python src/train.py \
  --experiment-name unet_lr1e-5_size256 \
  --lr 1e-5 \
  --batch-size 16 \
  --image-size 256 \
  --epochs 10
```

### 3. U-Net smaller image size, image_size=128

```bash
python src/train.py \
  --experiment-name unet_lr1e-4_size128 \
  --lr 1e-4 \
  --batch-size 16 \
  --image-size 128 \
  --epochs 10
```

### 4. U-Net weighted CrossEntropyLoss

```bash
python src/train.py \
  --experiment-name unet_weighted_ce_size256 \
  --lr 1e-4 \
  --batch-size 16 \
  --image-size 256 \
  --epochs 10 \
  --weighted-loss
```

### 5. Horizontal flip TTA prediction

```bash
python src/predict.py \
  --checkpoint checkpoints/best.pth \
  --images-dir underwater_data/test/images \
  --output submission.txt \
  --tta
```

## Evaluation and Visualization

```bash
python src/evaluate.py \
  --checkpoint checkpoints/best.pth \
  --images-dir underwater_data/train/images \
  --masks-dir underwater_data/train/masks
```

```bash
python src/visualize.py \
  --checkpoint checkpoints/best.pth \
  --images-dir underwater_data/test/images \
  --output assets/prediction_examples.png
```

## Results Tracking

Each training run appends one row to `experiments/results.csv` with:

- experiment name
- model
- learning rate
- batch size
- image size
- epoch count
- weighted loss flag
- best validation mean IoU
- best validation pixel accuracy
- best validation per-class IoU
- checkpoint path
