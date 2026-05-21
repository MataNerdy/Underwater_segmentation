from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.dataset import UnderwaterSegmentationDataset, rgb_mask_to_index
from src.metrics import SegmentationMeter, mean_iou, pixel_accuracy
from src.model import build_model
from src.predict import compare_submissions, load_submission, mode_filter_3x3, save_submission


def test_rgb_mask_to_index_uses_project_encoding():
    mask = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]])
    encoded = rgb_mask_to_index(mask)
    assert encoded.tolist() == [[4, 2, 1, 7]]


def test_dataset_metrics_and_submission_smoke(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[:8, :, 0] = 255
    mask[8:, :, 2] = 255

    Image.fromarray(image).save(images_dir / "sample.png")
    Image.fromarray(mask).save(masks_dir / "sample.bmp")

    dataset = UnderwaterSegmentationDataset(images_dir, masks_dir, image_size=16)
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["mask"].shape == (16, 16)
    assert sample["mask"].dtype == torch.long

    model = build_model(num_classes=8, features=4)
    model.eval()
    with torch.no_grad():
        model_logits = model(sample["image"].unsqueeze(0))
    assert model_logits.shape == (1, 8, 16, 16)

    logits = torch.zeros(1, 8, 16, 16)
    logits[:, 4, :8, :] = 10
    logits[:, 1, 8:, :] = 10
    target = sample["mask"].unsqueeze(0)

    assert 0 <= pixel_accuracy(logits, target) <= 1
    assert 0 <= mean_iou(logits, target, num_classes=8) <= 1

    meter = SegmentationMeter(num_classes=8)
    meter.update(logits, target)
    assert "mean_iou" in meter.compute()

    output = tmp_path / "submission.txt"
    save_submission(np.zeros((1, 16, 16), dtype=np.uint8), output)
    assert "# Array shape: (1, 16, 16)" in output.read_text(encoding="utf-8")
    assert load_submission(output).shape == (1, 16, 16)
    assert compare_submissions(output, output)["pixel_accuracy"] == 1.0

    noisy = torch.zeros(1, 5, 5, dtype=torch.long)
    noisy[:, 2, 2] = 7
    assert mode_filter_3x3(noisy, num_classes=8)[0, 2, 2].item() == 0
