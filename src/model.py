from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution layers used in U-Net encoder and decoder blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block."""
        return self.block(x)


class UNet(nn.Module):
    """Compact U-Net baseline for 8-class semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        features: int = 32,
    ) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.down2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.bridge = DoubleConv(features * 2, features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(features * 4, features * 2)

        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(features * 2, features)

        self.head = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [B, num_classes, H, W]."""
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool1(skip1))

        x = self.bridge(self.pool2(skip2))

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv1(x)

        return self.head(x)


def build_model(num_classes: int = 8, features: int = 32) -> nn.Module:
    """Build the U-Net baseline."""
    return UNet(num_classes=num_classes, features=features)