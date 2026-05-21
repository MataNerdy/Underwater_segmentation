from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import UnderwaterSegmentationDataset, compute_class_weights
from src.metrics import SegmentationMeter
from src.model import build_model


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train underwater segmentation model.")
    parser.add_argument("--images-dir", default=Path("underwater_data/train/images"), type=Path)
    parser.add_argument("--masks-dir", default=Path("underwater_data/train/masks"), type=Path)
    parser.add_argument("--model", default="unet", choices=["unet"])
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--val-split", default=0.2, type=float)
    parser.add_argument("--num-classes", default=8, type=int)
    parser.add_argument("--features", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--experiment-name", default="unet_8class_baseline")
    parser.add_argument("--checkpoint-dir", default=Path("checkpoints"), type=Path)
    parser.add_argument("--results-csv", default=Path("experiments/results.csv"), type=Path)
    parser.add_argument("--assets-dir", default=Path("assets"), type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_model_output(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return segmentation logits with shape [B, C, H, W]."""
    return model(images)


def make_loss(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Create CrossEntropy loss, optionally with class weights from training masks."""
    if not args.weighted_loss:
        return nn.CrossEntropyLoss()

    weights = compute_class_weights(args.masks_dir, num_classes=args.num_classes).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    num_classes: int = 8,
) -> dict[str, float]:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    meter = SegmentationMeter(num_classes=num_classes)

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc="train" if is_train else "valid", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).long()

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = get_model_output(model, images)
            loss = criterion(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            meter.update(logits.detach(), masks)

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def append_results(path: Path, row: dict[str, float | int | str]) -> None:
    """Append experiment metrics to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_curves(history: list[dict[str, float]], assets_dir: Path) -> None:
    """Save loss and mIoU curves for portfolio reporting."""
    assets_dir.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "loss_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_mean_iou"] for row in history], label="train")
    plt.plot(epochs, [row["val_mean_iou"] for row in history], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "miou_curves.png", dpi=160)
    plt.close()


def main() -> None:
    """Train a segmentation model and save the best checkpoint by validation mIoU."""
    args = parse_args()
    device = torch.device(args.device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnderwaterSegmentationDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_size=args.image_size,
    )

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError("Dataset must contain at least two samples for a validation split.")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=args.num_classes, features=args.features).to(device)
    criterion = make_loss(args, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_miou = -1.0
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    checkpoint_path = args.checkpoint_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            num_classes=args.num_classes,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            num_classes=args.num_classes,
        )

        scheduler.step(val_metrics["mean_iou"])

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mean_iou": train_metrics["mean_iou"],
            "val_mean_iou": val_metrics["mean_iou"],
        }

        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_miou={val_metrics['mean_iou']:.4f} "
            f"val_acc={val_metrics['pixel_accuracy']:.4f}"
        )

        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "num_classes": args.num_classes,
                    "features": args.features,
                    "image_size": args.image_size,
                    "experiment_name": args.experiment_name,
                    "metrics": val_metrics,
                },
                checkpoint_path,
            )

    save_curves(history, args.assets_dir)

    append_results(
        args.results_csv,
        build_results_row(
            args=args,
            checkpoint_path=checkpoint_path,
            best_miou=best_miou,
            best_metrics=best_metrics,
        ),
    )


def build_results_row(
    args: argparse.Namespace,
    checkpoint_path: Path,
    best_miou: float,
    best_metrics: dict[str, float],
) -> dict[str, float | int | str]:
    """Build one experiment summary row for experiments/results.csv."""
    row: dict[str, float | int | str] = {
        "experiment": args.experiment_name,
        "model": args.model,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "weighted_loss": str(args.weighted_loss),
        "features": args.features,
        "best_val_miou": best_miou,
        "best_val_pixel_accuracy": best_metrics.get("pixel_accuracy", float("nan")),
        "checkpoint": str(checkpoint_path),
    }
    for class_id in range(args.num_classes):
        row[f"best_val_iou_class_{class_id}"] = best_metrics.get(
            f"iou_class_{class_id}",
            float("nan"),
        )
    return row


if __name__ == "__main__":
    main()
