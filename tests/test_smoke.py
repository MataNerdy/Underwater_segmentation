from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.dataset import UnderwaterSegmentationDataset
from src.metrics import dice_coefficient, iou_score, pixel_accuracy
from src.model import build_model


def test_dataset_model_and_metrics_smoke(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = [20, 120, 180]
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    Image.fromarray(image).save(images_dir / "sample.png")
    Image.fromarray(mask).save(masks_dir / "sample.png")

    dataset = UnderwaterSegmentationDataset(images_dir, masks_dir, image_size=(32, 32))
    sample = dataset[0]

    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)

    model = build_model(features=8)
    logits = model(sample["image"].unsqueeze(0))

    assert logits.shape == (1, 1, 32, 32)
    assert 0 <= dice_coefficient(logits, sample["mask"].unsqueeze(0)).item() <= 1
    assert 0 <= iou_score(logits, sample["mask"].unsqueeze(0)).item() <= 1
    assert 0 <= pixel_accuracy(logits, sample["mask"].unsqueeze(0)).item() <= 1

