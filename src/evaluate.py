from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import UnderwaterSegmentationDataset
from src.metrics import dice_coefficient, iou_score, pixel_accuracy
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate underwater segmentation model.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--features", default=None, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    features = args.features or checkpoint.get("features", 32)
    image_size = checkpoint.get("image_size", args.image_size)

    dataset = UnderwaterSegmentationDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_size=(image_size, image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(features=features).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "accuracy": 0.0}

    for batch in tqdm(loader, desc="evaluate"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        batch_size = images.size(0)

        totals["loss"] += criterion(logits, masks).item() * batch_size
        totals["dice"] += dice_coefficient(logits, masks).item() * batch_size
        totals["iou"] += iou_score(logits, masks).item() * batch_size
        totals["accuracy"] += pixel_accuracy(logits, masks).item() * batch_size

    for name, value in totals.items():
        print(f"{name}: {value / len(dataset):.4f}")


if __name__ == "__main__":
    main()

