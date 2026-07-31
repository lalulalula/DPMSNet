from __future__ import annotations

from typing import Optional, Sequence

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """Lightweight residual adapter for infrared image embeddings."""

    def __init__(self, embedding_dim: int = 512, reduction: int = 4, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha
        hidden_dim = embedding_dim // reduction
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        adapted = self.layers(embedding)
        return self.alpha * embedding + (1.0 - self.alpha) * adapted


class LanguagePriorEncoder(nn.Module):
    """Frozen CLIP encoders with a trainable infrared-domain adapter."""

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        embedding_dim: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load(model_name, device=self.device_name)
        self.clip_model = self.clip_model.float()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        self.adapter = Adapter(embedding_dim=embedding_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        images = images.repeat(1, 3, 1, 1)
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(images)
        image_embedding = self.adapter(image_embedding.float())
        return F.normalize(image_embedding, dim=-1)

    def encode_text(self, descriptions: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(descriptions).to(self.device_name)
            text_embedding = self.clip_model.encode_text(tokens)
        return F.normalize(text_embedding.float(), dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        descriptions: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        image_embedding = self.encode_image(images)
        text_embedding = None
        if descriptions is not None:
            text_embedding = self.encode_text(descriptions)
        return image_embedding, text_embedding


class SemanticFeatureModulation(nn.Module):
    """Generate the scale and bias in Eq. (13) and apply Eq. (14)."""

    def __init__(self, embedding_dim: int = 512, reduction: int = 8):
        super().__init__()
        hidden_dim = embedding_dim // reduction
        self.parameter_generator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor, semantic_vector: torch.Tensor) -> torch.Tensor:
        scale, bias = self.parameter_generator(semantic_vector).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return features * (1.0 + scale) + bias


class LanguagePriorModulationModule(nn.Module):
    """Language Prior Modulation Module (LPMM) described in the paper."""

    def __init__(self, model_name: str = "ViT-B/32", embedding_dim: int = 512):
        super().__init__()
        self.prior_encoder = LanguagePriorEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
        )
        self.feature_modulation = SemanticFeatureModulation(embedding_dim=embedding_dim)

    def forward(
        self,
        deep_features: torch.Tensor,
        images: torch.Tensor,
        descriptions: Optional[Sequence[str]] = None,
        use_text_prior: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        encoded_descriptions = descriptions if use_text_prior else None
        image_embedding, text_embedding = self.prior_encoder(images, encoded_descriptions)

        if use_text_prior and text_embedding is not None:
            semantic_vector = text_embedding
        else:
            semantic_vector = image_embedding

        modulated_features = self.feature_modulation(
            deep_features,
            semantic_vector.detach(),
        )
        return modulated_features, image_embedding, text_embedding
