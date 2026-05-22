# Experiment Plan

This project supports Kaggle-first experiment runs for SUIM semantic
segmentation. The canonical dataset location is:

```text
/kaggle/input/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim
```

The code resolves directories with fallbacks:

- `train/images` first, otherwise `train_val/images`
- `train/masks` first, otherwise `train_val/masks`
- `test/images` first, otherwise `TEST/images`

## Shared Outputs

- Final experiment summaries: `/kaggle/working/experiments/results.csv`
- Training curves: `/kaggle/working/assets/<experiment>/loss_curves.png`
- mIoU curves: `/kaggle/working/assets/<experiment>/miou_curves.png`
- Prediction examples: `/kaggle/working/assets/prediction_examples.png`
- Checkpoints: `/kaggle/working/checkpoints/<experiment>/best.pth`

## Smoke Test

```bash
./scripts/run_kaggle_experiments.sh --smoke
```

Smoke mode runs the full experiment list for 1 epoch.

## Main Kaggle Experiments

The documented default for real experiments is 20 epochs.

```bash
./scripts/run_kaggle_experiments.sh
```

The runner executes:

1. U-Net, `image_size=128`, `epochs=20`
2. U-Net, `image_size=256`, `epochs=20`
3. U-Net weighted CrossEntropyLoss, `image_size=256`, `epochs=20`
4. DeepLabV3 ResNet50, `image_size=256`, `epochs=20`
5. DeepLabV3 ResNet50 pretrained backbone, `image_size=256`, `epochs=20`

## Manual Commands

### U-Net, image_size=128

```bash
python src/train.py \
  --model unet \
  --experiment-name unet_size128_main \
  --image-size 128 \
  --epochs 20
```

### U-Net, image_size=256

```bash
python src/train.py \
  --model unet \
  --experiment-name unet_size256_main \
  --image-size 256 \
  --epochs 20
```

### U-Net weighted CrossEntropyLoss

```bash
python src/train.py \
  --model unet \
  --experiment-name unet_weighted_ce_size256_main \
  --image-size 256 \
  --epochs 20 \
  --weighted-loss
```

### DeepLabV3 ResNet50

```bash
python src/train.py \
  --model deeplabv3_resnet50 \
  --experiment-name deeplabv3_resnet50_size256_main \
  --image-size 256 \
  --epochs 20
```

### DeepLabV3 ResNet50 pretrained backbone

```bash
python src/train.py \
  --model deeplabv3_resnet50 \
  --pretrained-backbone \
  --experiment-name deeplabv3_resnet50_pretrained_size256_main \
  --image-size 256 \
  --epochs 20
```

## Prediction Visualization

```bash
python src/predict.py \
  --checkpoint /kaggle/working/checkpoints/deeplabv3_resnet50_pretrained_size256_main/best.pth \
  --predictions-dir /kaggle/working/predictions \
  --tta
```

`predict.py` writes raw class-id masks to `predictions_raw/` and colored RGB
visualizations to `predictions_color/`.
