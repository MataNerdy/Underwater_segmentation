from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import UnderwaterSegmentationDataset
from src.model import build_model
from src.train import get_model_output


def parse_args() -> argparse.Namespace:
    """Parse prediction CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate underwater segmentation submission.")
    parser.add_argument("--images-dir", default=Path("underwater_data/test/images"), type=Path)
    parser.add_argument("--checkpoint", default=Path("checkpoints/best.pth"), type=Path)
    parser.add_argument("--output", default=Path("submission.txt"), type=Path)
    parser.add_argument("--predictions-dir", default=None, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--image-size", default=None, type=int)
    parser.add_argument("--output-size", default=128, type=int)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--features", default=None, type=int)
    parser.add_argument("--tta", action="store_true", help="Use horizontal flip test-time augmentation.")
    parser.add_argument("--mode-filter", action="store_true", help="Apply 3x3 majority filter.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_submission(labels: np.ndarray, output_path: Path) -> None:
    """Save class-index masks in the notebook-compatible submission.txt format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Array shape: {labels.shape}\n")
        for mask in labels:
            np.savetxt(handle, mask, fmt="%d")
            handle.write("# New slice\n")


def load_submission(path: str | Path) -> np.ndarray:
    """Load a notebook-compatible submission.txt file into an array of masks."""
    slices: list[np.ndarray] = []
    current_slice: list[list[int]] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                if current_slice:
                    slices.append(np.array(current_slice, dtype=np.uint8))
                    current_slice = []
                continue

            if line.strip():
                current_slice.append([int(value) for value in line.split()])

    if current_slice:
        slices.append(np.array(current_slice, dtype=np.uint8))

    if not slices:
        raise ValueError(f"No masks found in submission file: {path}")

    return np.stack(slices, axis=0)


def compare_submissions(
    prediction_path: str | Path,
    target_path: str | Path,
    num_classes: int = 8,
) -> dict[str, float]:
    """Compare two submission files with pixel accuracy, mean IoU and per-class IoU."""
    predictions = load_submission(prediction_path)
    targets = load_submission(target_path)

    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

    result: dict[str, float] = {
        "pixel_accuracy": float((predictions == targets).mean())
    }

    ious = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        target_class = targets == class_id

        if not target_class.any():
            result[f"iou_class_{class_id}"] = float("nan")
            continue

        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()
        iou = float(intersection / max(union, 1))

        result[f"iou_class_{class_id}"] = iou
        ious.append(iou)

    result["mean_iou"] = float(np.mean(ious)) if ious else float("nan")

    return result


def mode_filter_3x3(labels: torch.Tensor, num_classes: int = 8) -> torch.Tensor:
    """Smooth class-index masks with the notebook's 3x3 majority vote filter."""
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    kernel = torch.ones((num_classes, 1, 3, 3), device=labels.device)
    counts = F.conv2d(one_hot, kernel, padding=1, groups=num_classes)
    return counts.argmax(dim=1)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, images: torch.Tensor, tta: bool) -> torch.Tensor:
    """Predict logits, optionally averaging original and horizontal-flip outputs."""
    logits = get_model_output(model, images)

    if not tta:
        return logits

    flipped = torch.flip(images, dims=[-1])
    flipped_logits = torch.flip(get_model_output(model, flipped), dims=[-1])

    return (logits + flipped_logits) / 2.0


@torch.no_grad()
def main() -> None:
    """Generate predictions and save submission.txt."""
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = args.num_classes or checkpoint.get("num_classes", 8)
    features = args.features or checkpoint.get("features", 32)
    image_size = args.image_size or checkpoint.get("image_size", 256)

    dataset = UnderwaterSegmentationDataset(
        images_dir=args.images_dir,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(num_classes=num_classes, features=features).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels: list[np.ndarray] = []
    image_ids: list[str] = []

    for batch in tqdm(loader, desc="predict"):
        images = batch["image"].to(device)

        logits = predict_logits(model, images, args.tta)
        logits = F.interpolate(
            logits,
            size=(args.output_size, args.output_size),
            mode="bilinear",
            align_corners=False,
        )

        predictions_tensor = logits.argmax(dim=1)

        if args.mode_filter:
            predictions_tensor = mode_filter_3x3(
                predictions_tensor,
                num_classes=num_classes,
            )

        predictions = predictions_tensor.cpu().numpy().astype(np.uint8)

        labels.extend(predictions)
        image_ids.extend(batch["image_id"])

    labels_array = np.stack(labels, axis=0)
    save_submission(labels_array, args.output)

    if args.predictions_dir:
        args.predictions_dir.mkdir(parents=True, exist_ok=True)

        for image_id, mask in zip(image_ids, labels_array, strict=True):
            Image.fromarray(mask, mode="L").save(args.predictions_dir / f"{image_id}.png")


if __name__ == "__main__":
    main()