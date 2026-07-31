from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = channels // reduction
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_response = self.projection(self.average_pool(features))
        maximum_response = self.projection(self.maximum_pool(features))
        return self.activation(average_response + maximum_response)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.projection = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_response = torch.mean(features, dim=1, keepdim=True)
        maximum_response = torch.max(features, dim=1, keepdim=True).values
        descriptors = torch.cat([average_response, maximum_response], dim=1)
        return self.activation(self.projection(descriptors))


class FeatureFusionModule(nn.Module):
    """Fuse encoder details and decoder semantics with parallel attention."""

    def __init__(self, input_channels: int, output_channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = output_channels // reduction
        self.input_projection = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.channel_attention = ChannelAttention(hidden_channels)
        self.spatial_convs = nn.ModuleList(
            [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            ]
        )
        self.spatial_attention = SpatialAttention()
        self.output_projection = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(
        self,
        encoder_features: torch.Tensor,
        decoder_features: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.input_projection(torch.cat([encoder_features, decoder_features], dim=1))
        channel_features = fused * self.channel_attention(fused)
        spatial_features = sum(convolution(fused) for convolution in self.spatial_convs)
        spatial_features = spatial_features * self.spatial_attention(spatial_features)
        return self.output_projection(channel_features + spatial_features)
