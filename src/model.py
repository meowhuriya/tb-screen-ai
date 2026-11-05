"""Multi-Architecture Clinical CNN Suite for Tuberculosis Detection.

Supports:
1. EfficientNet-B0 (Edge-optimized compound scaling, ~4.01M params, ~16MB)
2. DenseNet-121 (Dense feature connectivity for fine radiological textures, ~6.95M params, ~28MB)
3. ResNet-50 (Deep residual baseline, ~23.5M params, ~95MB)

Includes:
- Two-stage transfer learning hooks (head warmup -> fine-tuning)
- Grad-CAM target layer retrieval
- Parameter counting and latency profiling
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    resnet50,
)


class TBClassifier(nn.Module):
    """Modular clinical chest X-ray classifier supporting multiple CNN backbones."""

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize classifier with chosen backbone.

        Args:
            architecture: 'efficientnet_b0', 'densenet121', or 'resnet50'.
            num_classes: Number of output targets (default: 2).
            pretrained: Load ImageNet pretrained weights.
            dropout_rate: Dropout probability before classification.
        """
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if self.architecture == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features=in_features, out_features=num_classes),
            )
        elif self.architecture == "densenet121":
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features=in_features, out_features=num_classes),
            )
        elif self.architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features=in_features, out_features=num_classes),
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'efficientnet_b0', 'densenet121', or 'resnet50'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze feature extractor for Stage 1 head warmup."""
        if self.architecture == "efficientnet_b0":
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.architecture == "densenet121":
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.architecture == "resnet50":
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def unfreeze_blocks(self, num_blocks: int = 2) -> None:
        """Unfreeze top feature stages for Stage 2 fine-tuning."""
        if self.architecture == "efficientnet_b0":
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
            total = len(self.backbone.features)
            for i, block in enumerate(self.backbone.features):
                is_trainable = i >= max(0, total - num_blocks)
                for p in block.parameters():
                    p.requires_grad = is_trainable
        elif self.architecture == "densenet121":
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
            # In DenseNet121, unfreeze denseblock4 & norm5
            for name, param in self.backbone.features.named_parameters():
                if "denseblock4" in name or "norm5" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.architecture == "resnet50":
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            # In ResNet50, unfreeze layer4
            for name, param in self.backbone.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the optimal target convolutional layer for Grad-CAM."""
        if self.architecture == "efficientnet_b0":
            return self.backbone.features[-1]
        elif self.architecture == "densenet121":
            return self.backbone.features.norm5
        elif self.architecture == "resnet50":
            return self.backbone.layer4[-1].conv3


def get_model(
    architecture: str = "efficientnet_b0",
    pretrained: bool = True,
    num_classes: int = 2,
    dropout_rate: float = 0.3,
) -> TBClassifier:
    """Factory helper to instantiate any supported TBClassifier."""
    return TBClassifier(
        architecture=architecture,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
