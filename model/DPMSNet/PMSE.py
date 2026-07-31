from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class DilatedConvBlock(nn.Sequential):
    def __init__(self, channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class AdaptiveScaleFusion(nn.Module):
    """Adaptive Scale Fusion (ASF) described in Eqs. (19)-(23)."""

    def __init__(self, channels: int, num_scales: int = 4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.descriptor_projections = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1, bias=False) for _ in range(num_scales)]
        )

    def forward(self, scale_features: Sequence[torch.Tensor]) -> torch.Tensor:
        descriptors = [
            projection(self.global_pool(features))
            for projection, features in zip(self.descriptor_projections, scale_features)
        ]
        scale_descriptors = torch.cat(descriptors, dim=2)
        scale_weights = torch.softmax(torch.sigmoid(scale_descriptors), dim=2)

        fused = torch.zeros_like(scale_features[0])
        for index, features in enumerate(scale_features):
            weight = scale_weights[:, :, index, :].unsqueeze(2)
            fused = fused + weight * features
        return fused


class ProgressiveMultiScaleFeatureExtraction(nn.Module):
    """Progressive Multi-Scale Feature Extraction (PMSE) module."""

    def __init__(self, channels: int, dilation_rates: Sequence[int] = (1, 2, 3)):
        super().__init__()
        if len(dilation_rates) != 3:
            raise ValueError("PMSE expects exactly three dilation rates")

        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dilated_stages = nn.ModuleList(
            [DilatedConvBlock(channels, rate) for rate in dilation_rates]
        )
        self.asf = AdaptiveScaleFusion(channels)
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale_0 = self.initial_conv(features)
        scale_1 = self.dilated_stages[0](scale_0 + features)
        scale_2 = self.dilated_stages[1](scale_1 + features)
        scale_3 = self.dilated_stages[2](scale_2 + features)
        fused = self.asf((scale_0, scale_1, scale_2, scale_3))
        return self.output_projection(fused + features)
