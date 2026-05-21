from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ImageTransform = Callable[[Image.Image], torch.Tensor]
MaskTransform = Callable[[Image.Image], torch.Tensor]


class UnderwaterSegmentationDataset(Dataset):
    """Dataset for paired underwater images and binary segmentation masks."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path | None = None,
        image_size: tuple[int, int] | None = None,
        image_transform: ImageTransform | None = None,
        mask_transform: MaskTransform | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")

        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        if self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)

        image_tensor = (
            self.image_transform(image) if self.image_transform else image_to_tensor(image)
        )

        sample: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "image_id": image_path.stem,
            "path": str(image_path),
            "original_width": str(original_size[0]),
            "original_height": str(original_size[1]),
        }

        if self.masks_dir:
            mask_path = find_mask_path(self.masks_dir, image_path.stem)
            mask = Image.open(mask_path).convert("L")
            if self.image_size:
                mask = mask.resize(self.image_size, Image.NEAREST)
            sample["mask"] = (
                self.mask_transform(mask) if self.mask_transform else mask_to_tensor(mask)
            )

        return sample


def find_mask_path(masks_dir: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = masks_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Mask for image '{stem}' was not found in {masks_dir}")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.float32)
    array = (array > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)

