from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import UnderwaterSegmentationDataset
from src.metrics import SegmentationMeter
from src.model import build_model
from src.train import get_model_output


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate underwater segmentation checkpoint.")
    parser.add_argument("--images-dir", default=Path("underwater_data/train/images"), type=Path)
    parser.add_argument("--masks-dir", default=Path("underwater_data/train/masks"), type=Path)
    parser.add_argument("--checkpoint", default=Path("checkpoints/best.pth"), type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--image-size", default=None, type=int)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--features", default=None, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Load a checkpoint and print validation metrics."""
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model", "unet")
    num_classes = args.num_classes or checkpoint.get("num_classes", 8)
    features = args.features or checkpoint.get("features", 32)
    image_size = args.image_size or checkpoint.get("image_size", 256)

    dataset = UnderwaterSegmentationDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        features=features,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    meter = SegmentationMeter(num_classes=num_classes)

    total_loss = 0.0

    for batch in tqdm(loader, desc="evaluate"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        logits = get_model_output(model, images)

        total_loss += criterion(logits, masks).item() * images.size(0)
        meter.update(logits, masks)

    print(f"loss: {total_loss / max(len(loader.dataset), 1):.4f}")

    for name, value in meter.compute().items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
