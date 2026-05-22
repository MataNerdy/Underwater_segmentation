from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "DATA_DIR",
        "/kaggle/input/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim",
    )
)

CLASS_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [0, 255, 255],
        [255, 0, 0],
        [255, 0, 255],
        [255, 255, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def rgb_mask_to_index(mask: Image.Image | np.ndarray) -> np.ndarray:
    """Convert RGB underwater masks to class indices using 4R + 2G + B coding."""
    array = np.asarray(mask.convert("RGB") if isinstance(mask, Image.Image) else mask)
    binary = array > 100
    return np.dot(binary, [4, 2, 1]).astype(np.int64)


def colorize_mask(mask: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    """Convert a class-index mask to a stable RGB palette visualization."""
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)

    array = np.clip(array.astype(np.int64), 0, len(CLASS_PALETTE) - 1)
    return Image.fromarray(CLASS_PALETTE[array])


def _select_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            print(f"Selected {label}: {path}")
            return path
    fallback = candidates[-1]
    print(f"Selected {label}: {fallback} (not found yet)")
    return fallback


def resolve_dataset_paths(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve SUIM train/test directories with train -> train_val fallback."""
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    print(f"Selected DATA_DIR: {root}")
    return {
        "train_images": _select_existing(
            [root / "train" / "images", root / "train_val" / "images"],
            "train images dir",
        ),
        "train_masks": _select_existing(
            [root / "train" / "masks", root / "train_val" / "masks"],
            "train masks dir",
        ),
        "test_images": _select_existing(
            [root / "test" / "images", root / "TEST" / "images"],
            "test images dir",
        ),
    }


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Resize, convert and normalize an RGB image for the segmentation model."""
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def mask_to_tensor(mask: Image.Image, image_size: int) -> torch.Tensor:
    """Resize an RGB mask and convert it to a long tensor with class ids 0..7."""
    mask = mask.convert("RGB").resize((image_size, image_size), Image.NEAREST)
    return torch.from_numpy(rgb_mask_to_index(mask)).long()


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a normalized image tensor back to display range [0, 1]."""
    return (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


class UnderwaterSegmentationDataset(Dataset):
    """Dataset for underwater images and optional RGB segmentation masks."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path | None = None,
        image_size: int = 256,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.image_size = image_size

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")

        if self.masks_dir and not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory does not exist: {self.masks_dir}")

        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        sample: dict[str, torch.Tensor | str] = {
            "image": image_to_tensor(image, self.image_size),
            "image_id": image_path.stem,
            "path": str(image_path),
        }

        if self.masks_dir:
            mask_path = find_mask_path(self.masks_dir, image_path.stem)
            sample["mask"] = mask_to_tensor(Image.open(mask_path), self.image_size)

        return sample


def find_mask_path(masks_dir: Path, stem: str) -> Path:
    """Find a mask by image stem across common image extensions."""
    for suffix in IMAGE_EXTENSIONS:
        path = masks_dir / f"{stem}{suffix}"
        if path.exists():
            return path

    raise FileNotFoundError(f"Mask for image '{stem}' was not found in {masks_dir}")


def compute_class_weights(
    masks_dir: str | Path,
    num_classes: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute median-frequency class weights from RGB masks."""
    masks_dir = Path(masks_dir)
    counts = np.zeros(num_classes, dtype=np.int64)

    for mask_path in sorted(masks_dir.iterdir()):
        if mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask = rgb_mask_to_index(Image.open(mask_path))
        counts += np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]

    frequencies = counts / max(counts.sum(), 1)
    non_zero = frequencies[frequencies > 0]

    if len(non_zero) == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    median = np.median(non_zero)
    weights = median / (frequencies + eps)

    return torch.tensor(weights, dtype=torch.float32)


def resize_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize segmentation logits without changing the number of classes."""
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
