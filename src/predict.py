from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import UnderwaterSegmentationDataset
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate segmentation predictions.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/predictions"), type=Path)
    parser.add_argument("--submission-path", default=Path("submission.txt"), type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--features", default=None, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def encode_mask(mask: np.ndarray) -> str:
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(value) for value in changes)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    features = args.features or checkpoint.get("features", 32)
    image_size = checkpoint.get("image_size", args.image_size)

    dataset = UnderwaterSegmentationDataset(
        images_dir=args.images_dir,
        image_size=(image_size, image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(features=features).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    submission_rows = ["image_id,encoded_pixels"]
    for batch in tqdm(loader, desc="predict"):
        images = batch["image"].to(device)
        logits = model(images)
        masks = (torch.sigmoid(logits) > args.threshold).cpu().numpy().astype(np.uint8)

        for image_id, mask in zip(batch["image_id"], masks, strict=True):
            mask_2d = mask.squeeze(0) * 255
            Image.fromarray(mask_2d).save(args.output_dir / f"{image_id}.png")
            submission_rows.append(f"{image_id},{encode_mask(mask.squeeze(0))}")

    args.submission_path.write_text("\n".join(submission_rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

