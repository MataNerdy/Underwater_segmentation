# AGENTS.md

## Project

This is a PyTorch semantic segmentation project for underwater scene images.

Current notebook:
- `notebooks/underwater_baseline.ipynb`

Goal:
- Refactor notebook code into reusable scripts.
- Train and compare segmentation experiments.
- Save metrics, visualizations, checkpoints and README updates.

## Dataset

Expected local structure:

```text
underwater_data/
├── train/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

Masks are RGB masks converted to 8 class indices using:

mask = (mask > 100)
mask = np.dot(mask, [4, 2, 1]).astype(np.uint8)

Do not commit dataset files, checkpoints, or large artifacts.

## Commands

Install dependencies:

pip install -r requirements.txt

Train baseline:

python src/train.py --model deeplabv3_resnet50 --epochs 10 --batch-size 16 --lr 1e-4

Evaluate checkpoint:

python src/evaluate.py --checkpoint checkpoints/best.pth

Generate predictions:

python src/predict.py --checkpoint checkpoints/best.pth --output submission.txt

## Experiments to run

Run at least these experiments:

DeepLabV3 ResNet50 baseline, lr=1e-4, AdamW.
DeepLabV3 ResNet50, lr=1e-5 fine-tuning.
Compare image size 128 vs 256.
Compare loss:
CrossEntropyLoss
weighted CrossEntropyLoss
Add simple test-time augmentation:
horizontal flip TTA.

For every experiment save:

experiments/results.csv
assets/loss_curves.png
assets/miou_curves.png
assets/prediction_examples.png

## Metrics

Use:

pixel accuracy
mean IoU
per-class IoU

## Code style

No hardcoded Google Drive paths.
No absolute local paths.
All paths must be configurable via CLI arguments.
Keep functions small and reusable.
Add docstrings to public functions.
README must be in Russian.
README should follow STAR structure:
    Situation
    Task
    Action
    Result

## Done means

Scripts run without notebook dependency.
README explains task, dataset, model, experiments and results.
results.csv contains experiment comparison.
GitHub repo is clean and portfolio-ready.