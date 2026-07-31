from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FeatureFusion import FeatureFusionModule
from .LPMM import LanguagePriorModulationModule
from .PMSE import ProgressiveMultiScaleFeatureExtraction
from .SSP import StructuralSaliencyPrior


class ResidualBlock(nn.Module):
    """Residual Block (RB) used by the encoder and decoder."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(output_channels)

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features)
        output = self.activation(self.norm1(self.conv1(features)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


def upsample_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        source,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


class DPMSNet(nn.Module):
    """Dual-Prior-Guided Multi-Scale Network for infrared small target detection."""

    def __init__(
        self,
        base_channels: int = 32,
        input_channels: int = 1,
        num_classes: int = 1,
        mode: str = "train",
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.deep_supervision = deep_supervision

        self.ssp = StructuralSaliencyPrior(input_channels=input_channels)
        self.lpmm = LanguagePriorModulationModule(embedding_dim=512)

        self.pool = nn.MaxPool2d(2, 2)
        self.stem = self._make_stage(input_channels, base_channels)
        self.encoder2 = self._make_stage(base_channels, base_channels * 2)
        self.encoder3 = self._make_stage(base_channels * 2, base_channels * 4)
        self.encoder4 = self._make_stage(base_channels * 4, base_channels * 8)
        self.encoder5 = self._make_stage(base_channels * 8, base_channels * 8)

        self.pmse1 = ProgressiveMultiScaleFeatureExtraction(base_channels)
        self.pmse2 = ProgressiveMultiScaleFeatureExtraction(base_channels * 2)
        self.pmse3 = ProgressiveMultiScaleFeatureExtraction(base_channels * 4)
        self.pmse4 = ProgressiveMultiScaleFeatureExtraction(base_channels * 8)
        self.pmse5 = ProgressiveMultiScaleFeatureExtraction(base_channels * 8)

        self.fusion4 = FeatureFusionModule(base_channels * 16, base_channels * 16)
        self.fusion3 = FeatureFusionModule(base_channels * 8, base_channels * 8)
        self.fusion2 = FeatureFusionModule(base_channels * 4, base_channels * 4)
        self.fusion1 = FeatureFusionModule(base_channels * 2, base_channels * 2)

        self.decoder4 = self._make_stage(base_channels * 16, base_channels * 4)
        self.decoder3 = self._make_stage(base_channels * 8, base_channels * 2)
        self.decoder2 = self._make_stage(base_channels * 4, base_channels)
        self.decoder1 = self._make_stage(base_channels * 2, base_channels)
        self.segmentation_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        if self.deep_supervision:
            self.auxiliary_head5 = nn.Conv2d(base_channels * 8, num_classes, kernel_size=1)
            self.auxiliary_head4 = nn.Conv2d(base_channels * 4, num_classes, kernel_size=1)
            self.auxiliary_head3 = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)
            self.auxiliary_head2 = nn.Conv2d(base_channels, num_classes, kernel_size=1)
            self.deep_supervision_fusion = nn.Conv2d(num_classes * 5, num_classes, kernel_size=1)

    @staticmethod
    def _make_stage(
        input_channels: int,
        output_channels: int,
        num_blocks: int = 1,
    ) -> nn.Sequential:
        blocks = [ResidualBlock(input_channels, output_channels)]
        blocks.extend(
            ResidualBlock(output_channels, output_channels)
            for _ in range(num_blocks - 1)
        )
        return nn.Sequential(*blocks)

    def forward(
        self,
        images: torch.Tensor,
        text_descriptions: Optional[Sequence[str]] = None,
    ):
        structural_attention = self.ssp(images)

        encoder1 = self.pmse1(self.stem(images))
        encoder1 = encoder1 + encoder1 * structural_attention
        encoder2 = self.pmse2(self.encoder2(self.pool(encoder1)))
        encoder3 = self.pmse3(self.encoder3(self.pool(encoder2)))
        encoder4 = self.pmse4(self.encoder4(self.pool(encoder3)))
        encoder5 = self.pmse5(self.encoder5(self.pool(encoder4)))

        bottleneck = upsample_like(encoder5, encoder4)
        bottleneck, image_embedding, text_embedding = self.lpmm(
            bottleneck,
            images,
            descriptions=text_descriptions,
            use_text_prior=self.mode == "train",
        )

        decoder4 = self.decoder4(self.fusion4(encoder4, bottleneck))
        decoder3 = self.decoder3(
            self.fusion3(encoder3, upsample_like(decoder4, encoder3))
        )
        decoder2 = self.decoder2(
            self.fusion2(encoder2, upsample_like(decoder3, encoder2))
        )
        decoder1 = self.decoder1(
            self.fusion1(encoder1, upsample_like(decoder2, encoder1))
        )
        logits = self.segmentation_head(decoder1)

        if not self.deep_supervision:
            return torch.sigmoid(logits), image_embedding, text_embedding

        output_size = logits.shape[-2:]
        auxiliary5 = F.interpolate(
            self.auxiliary_head5(encoder5),
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        auxiliary4 = F.interpolate(
            self.auxiliary_head4(decoder4),
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        auxiliary3 = F.interpolate(
            self.auxiliary_head3(decoder3),
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        auxiliary2 = F.interpolate(
            self.auxiliary_head2(decoder2),
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        fused_logits = self.deep_supervision_fusion(
            torch.cat([auxiliary2, auxiliary3, auxiliary4, auxiliary5, logits], dim=1)
        )

        if self.mode == "train":
            predictions = tuple(
                torch.sigmoid(output)
                for output in (
                    auxiliary5,
                    auxiliary4,
                    auxiliary3,
                    auxiliary2,
                    fused_logits,
                    logits,
                )
            )
        else:
            predictions = torch.sigmoid(logits)

        return predictions, image_embedding, text_embedding
