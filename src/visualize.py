from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import UnderwaterSegmentationDataset
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save visual examples with predicted masks.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/visualizations"), type=Path)
    parser.add_argument("--masks-dir", default=None, type=Path)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--num-examples", default=6, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--features", default=None, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    features = args.features or checkpoint.get("features", 32)
    image_size = checkpoint.get("image_size", args.image_size)

    dataset = UnderwaterSegmentationDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_size=(image_size, image_size),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(features=features).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for index, batch in enumerate(loader):
        if index >= args.num_examples:
            break

        image = batch["image"].to(device)
        logits = model(image)
        prediction = (torch.sigmoid(logits)[0, 0].cpu() > args.threshold).float()

        has_mask = "mask" in batch
        columns = 3 if has_mask else 2
        fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
        if columns == 2:
            axes = [axes[0], axes[1]]

        axes[0].imshow(tensor_to_image(batch["image"][0]))
        axes[0].set_title("Image")
        axes[1].imshow(prediction, cmap="gray")
        axes[1].set_title("Prediction")
        if has_mask:
            axes[2].imshow(batch["mask"][0, 0].cpu(), cmap="gray")
            axes[2].set_title("Ground truth")

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()
        fig.savefig(args.output_dir / f"{batch['image_id'][0]}_prediction.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

