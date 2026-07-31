from __future__ import annotations

import math

import torch
import torch.nn as nn


def _frozen_batch_norm(num_channels: int) -> nn.BatchNorm2d:
    norm = nn.BatchNorm2d(num_channels)
    for parameter in norm.parameters():
        parameter.requires_grad = False
    return norm


class ScharrOperator(nn.Module):
    """Fixed Scharr filters used by the SSD branch."""

    def __init__(self, channels: int):
        super().__init__()
        kernel_x = torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            dtype=torch.float32,
        )[None, None]
        kernel_y = torch.tensor(
            [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
            dtype=torch.float32,
        )[None, None]

        self.gradient_x = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.gradient_y = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.gradient_x.weight.data.copy_(kernel_x.repeat(channels, 1, 1, 1))
        self.gradient_y.weight.data.copy_(kernel_y.repeat(channels, 1, 1, 1))
        self.gradient_x.weight.requires_grad = False
        self.gradient_y.weight.requires_grad = False
        self.norm = _frozen_batch_norm(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        response_x = self.gradient_x(images)
        response_y = self.gradient_y(images)
        magnitude = torch.sqrt(response_x.square() + response_y.square())
        return self.activation(self.norm(magnitude))


class GaussianOperator(nn.Module):
    """Fixed Gaussian filter used for low-frequency background modeling."""

    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        kernel = self._make_kernel(kernel_size, sigma)
        self.filter = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.filter.weight.data.copy_(kernel.repeat(channels, 1, 1, 1))
        self.filter.weight.requires_grad = False
        self.norm = _frozen_batch_norm(channels)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _make_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        radius = kernel_size // 2
        kernel = torch.tensor(
            [
                [
                    math.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
                    / (2.0 * math.pi * sigma * sigma)
                    for x in range(-radius, radius + 1)
                ]
                for y in range(-radius, radius + 1)
            ],
            dtype=torch.float32,
        )[None, None]
        return kernel / kernel.sum()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        response = self.filter(images)
        return self.activation(self.norm(response))


class LocalResponseContrast(nn.Module):
    """Mean- and extremum-difference LRC operators from Eqs. (1) and (2)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.mean_pool = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        self.extreme_pool = nn.MaxPool2d(kernel_size, stride=1, padding=padding)
        self.mean_norm = nn.BatchNorm2d(channels)
        self.extreme_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_difference = self.mean_norm(images - self.mean_pool(images.detach()))
        extreme_difference = self.extreme_norm(images - self.extreme_pool(images.detach()))
        return mean_difference, extreme_difference


class SpatialStructureDescriptor(nn.Module):
    """Scharr and Gaussian SSD operators from Eqs. (3)-(6)."""

    def __init__(self, channels: int):
        super().__init__()
        self.scharr = ScharrOperator(channels)
        self.gaussian = GaussianOperator(channels)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scharr(images), self.gaussian(images)


class StructuralSaliencyPrior(nn.Module):
    """Structural Saliency Prior (SSP) module from Eqs. (7) and (8)."""

    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.lrc = LocalResponseContrast(input_channels)
        self.ssd = SpatialStructureDescriptor(input_channels)
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels * 4, input_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, 1, kernel_size=3, padding=1, bias=True),
        )
        self.activation = nn.Sigmoid()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mean_difference, extreme_difference = self.lrc(images)
        scharr_response, gaussian_response = self.ssd(images)
        structural_features = torch.cat(
            [mean_difference, extreme_difference, scharr_response, gaussian_response],
            dim=1,
        )
        return self.activation(self.projection(structural_features))
