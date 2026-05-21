from __future__ import annotations

import torch


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate pixel accuracy for multiclass segmentation logits."""
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


@torch.no_grad()
def per_class_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 8,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate IoU for each class and return NaN for classes absent in targets."""
    predictions = logits.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)
    values = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        target_class = targets == class_id

        if target_class.sum() == 0:
            values.append(torch.tensor(float("nan"), device=logits.device))
            continue

        intersection = torch.logical_and(pred_class, target_class).sum().float()
        union = torch.logical_or(pred_class, target_class).sum().float()
        values.append((intersection + eps) / (union + eps))

    return torch.stack(values)


@torch.no_grad()
def mean_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 8) -> float:
    """Calculate mean IoU, ignoring classes absent from the target masks."""
    return torch.nanmean(per_class_iou(logits, targets, num_classes=num_classes)).item()


class SegmentationMeter:
    """Accumulate pixel accuracy, mean IoU and per-class IoU over batches."""

    def __init__(self, num_classes: int = 8) -> None:
        self.num_classes = num_classes
        self.correct_pixels = 0
        self.total_pixels = 0
        self.intersections = torch.zeros(num_classes, dtype=torch.float64)
        self.unions = torch.zeros(num_classes, dtype=torch.float64)
        self.target_counts = torch.zeros(num_classes, dtype=torch.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch of logits and target masks to the running metrics."""
        predictions = logits.argmax(dim=1)
        targets = targets.long()

        self.correct_pixels += int((predictions == targets).sum().item())
        self.total_pixels += targets.numel()

        predictions = predictions.reshape(-1).detach().cpu()
        targets = targets.reshape(-1).detach().cpu()

        for class_id in range(self.num_classes):
            pred_class = predictions == class_id
            target_class = targets == class_id
            self.intersections[class_id] += torch.logical_and(pred_class, target_class).sum()
            self.unions[class_id] += torch.logical_or(pred_class, target_class).sum()
            self.target_counts[class_id] += target_class.sum()

    def compute(self) -> dict[str, float]:
        """Return averaged metrics."""
        valid = self.target_counts > 0
        per_class = torch.full((self.num_classes,), float("nan"), dtype=torch.float64)
        per_class[valid] = self.intersections[valid] / torch.clamp(self.unions[valid], min=1)

        metrics = {
            "pixel_accuracy": self.correct_pixels / max(self.total_pixels, 1),
            "mean_iou": torch.nanmean(per_class).item(),
        }

        metrics.update(
            {f"iou_class_{idx}": value.item() for idx, value in enumerate(per_class)}
        )

        return metrics
