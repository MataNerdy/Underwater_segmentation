from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import UnderwaterSegmentationDataset, denormalize_image
from src.model import build_model
from src.train import get_model_output


def parse_args() -> argparse.Namespace:
    """Parse visualization CLI arguments."""
    parser = argparse.ArgumentParser(description="Save prediction examples for portfolio review.")
    parser.add_argument("--images-dir", default=Path("underwater_data/test/images"), type=Path)
    parser.add_argument("--checkpoint", default=Path("checkpoints/best.pth"), type=Path)
    parser.add_argument("--output", default=Path("assets/prediction_examples.png"), type=Path)
    parser.add_argument("--masks-dir", default=None, type=Path)
    parser.add_argument("--image-size", default=None, type=int)
    parser.add_argument("--num-examples", default=6, type=int)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--features", default=None, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Save a grid with images, predicted masks and optional ground truth masks."""
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = args.num_classes or checkpoint.get("num_classes", 8)
    features = args.features or checkpoint.get("features", 32)
    image_size = args.image_size or checkpoint.get("image_size", 256)

    dataset = UnderwaterSegmentationDataset(args.images_dir, args.masks_dir, image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(num_classes=num_classes, features=features).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    examples = []
    for index, batch in enumerate(loader):
        if index >= args.num_examples:
            break
        image = batch["image"].to(device)
        logits = get_model_output(model, image)
        prediction = logits.argmax(dim=1)[0].cpu()
        examples.append((batch["image"][0], prediction, batch.get("mask")))

    if not examples:
        raise ValueError("No examples found for visualization.")

    columns = 3 if examples[0][2] is not None else 2
    fig, axes = plt.subplots(len(examples), columns, figsize=(4 * columns, 4 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    for row_index, (image, prediction, mask) in enumerate(examples):
        row_axes = axes[row_index]
        row_axes[0].imshow(denormalize_image(image).permute(1, 2, 0))
        row_axes[0].set_title("Image")
        row_axes[1].imshow(prediction, vmin=0, vmax=num_classes - 1, cmap="tab10")
        row_axes[1].set_title("Prediction")
        if mask is not None:
            row_axes[2].imshow(mask[0], vmin=0, vmax=num_classes - 1, cmap="tab10")
            row_axes[2].set_title("Ground truth")
        for axis in row_axes:
            axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
