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
        self.total_accuracy = 0.0
        self.total_miou = 0.0
        self.total_batches = 0
        self.class_iou_sum = torch.zeros(num_classes, dtype=torch.float64)
        self.class_iou_count = torch.zeros(num_classes, dtype=torch.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch of logits and target masks to the running metrics."""
        class_iou = per_class_iou(logits, targets, self.num_classes).detach().cpu()
        valid = ~torch.isnan(class_iou)

        self.class_iou_sum[valid] += class_iou[valid].double()
        self.class_iou_count[valid] += 1
        self.total_accuracy += pixel_accuracy(logits, targets)
        self.total_miou += torch.nanmean(class_iou).item()
        self.total_batches += 1

    def compute(self) -> dict[str, float]:
        """Return averaged metrics."""
        batches = max(self.total_batches, 1)
        per_class = self.class_iou_sum / torch.clamp(self.class_iou_count, min=1)

        metrics = {
            "pixel_accuracy": self.total_accuracy / batches,
            "mean_iou": self.total_miou / batches,
        }

        metrics.update(
            {f"iou_class_{idx}": value.item() for idx, value in enumerate(per_class)}
        )

        return metrics