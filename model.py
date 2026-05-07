"""High-capacity U-Net style model for binary landscape isolation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

try:  # pragma: no cover - version compatibility shim
    from torchvision.models import ResNet50_Weights, resnet50
except ImportError:  # pragma: no cover
    ResNet50_Weights = None  # type: ignore[assignment]
    from torchvision.models import resnet50  # type: ignore[no-redef]

from config import ModelConfig

LOGGER = logging.getLogger(__name__)


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """Create a stable GroupNorm layer for small segmentation batches."""

    num_groups = min(32, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResidualConvBlock(nn.Module):
    """Residual convolution block used throughout the decoder and bottleneck."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x, inplace=True)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return F.silu(x, inplace=True)


class DecoderBlock(nn.Module):
    """Upsampling block with a skip connection and residual refinement."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = ResidualConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class HighCapacityUNet(nn.Module):
    """A ResNet-50 encoder U-Net tailored for binary segmentation."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.encoder = self._build_encoder(model_config.pretrained_encoder)

        self.stem = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
        )
        self.maxpool = self.encoder.maxpool
        self.encoder_layer1 = self.encoder.layer1
        self.encoder_layer2 = self.encoder.layer2
        self.encoder_layer3 = self.encoder.layer3
        self.encoder_layer4 = self.encoder.layer4

        self.bottleneck = ResidualConvBlock(2048, 1024, dropout=model_config.decoder_dropout)
        self.decoder4 = DecoderBlock(1024, 1024, 512, dropout=model_config.decoder_dropout)
        self.decoder3 = DecoderBlock(512, 512, 256, dropout=model_config.decoder_dropout)
        self.decoder2 = DecoderBlock(256, 256, 128, dropout=model_config.decoder_dropout)
        self.decoder1 = DecoderBlock(128, 64, 64, dropout=model_config.decoder_dropout)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ResidualConvBlock(32, 32, dropout=model_config.decoder_dropout),
        )
        self.segmentation_head = nn.Conv2d(32, model_config.output_channels, kernel_size=1)

        if model_config.freeze_encoder_at_start:
            self.freeze_encoder()

    def _build_encoder(self, pretrained: bool) -> nn.Module:
        """Build the ResNet-50 encoder with a safe fallback to random init."""

        if not pretrained:
            return resnet50(weights=None)

        if ResNet50_Weights is None:
            LOGGER.warning("torchvision weight enums are unavailable; using random initialization.")
            return resnet50(weights=None)

        try:
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception as exc:  # pragma: no cover - environment dependent fallback
            LOGGER.warning(
                "Unable to load pretrained ResNet-50 weights; using random initialization instead: %s",
                exc,
            )
            return resnet50(weights=None)

    def freeze_encoder(self) -> None:
        """Freeze the encoder parameters to stabilize early training."""

        for module in (
            self.stem,
            self.maxpool,
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze the encoder parameters for full fine-tuning."""

        for module in (
            self.stem,
            self.maxpool,
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw logits."""

        original_size = x.shape[-2:]
        stem_features = self.stem(x)
        encoder1 = self.encoder_layer1(self.maxpool(stem_features))
        encoder2 = self.encoder_layer2(encoder1)
        encoder3 = self.encoder_layer3(encoder2)
        encoder4 = self.encoder_layer4(encoder3)

        x = self.bottleneck(encoder4)
        x = self.decoder4(x, encoder3)
        x = self.decoder3(x, encoder2)
        x = self.decoder2(x, encoder1)
        x = self.decoder1(x, stem_features)
        x = self.final_upsample(x)
        logits = self.segmentation_head(x)

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits


def build_model(model_config: ModelConfig) -> HighCapacityUNet:
    """Factory helper for the segmentation model."""

    return HighCapacityUNet(model_config)
