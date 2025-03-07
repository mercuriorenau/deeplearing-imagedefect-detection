"""
Binary classifier (defect / no_defect) using ResNet transfer learning.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_resnet(name="resnet18", num_classes=2, pretrained=True):
    """Load ResNet from torchvision and replace last layer for num_classes."""
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet: {name}. Use resnet18, resnet34 or resnet50.")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class DefectClassifier(nn.Module):
    """ResNet-based classifier. Can freeze backbone for first epochs."""

    def __init__(self, backbone_name="resnet18", num_classes=2, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.model = get_resnet(backbone_name, num_classes, pretrained)

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all layers except the last one (fc)."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
