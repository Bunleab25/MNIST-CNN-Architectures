"""
Advanced CNN Architecture Module (Modern Deep ConvNet with GAP)
================================================================
Implements AdvancedMNISTCNN featuring:
1. Dual-Convolution Stages (2x Conv-BN-ReLU per stage).
2. Global Average Pooling (GAP) replacing dense flatten layers.
3. Translation invariance and extreme parameter efficiency.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class DualConvBlock(nn.Module):
    """
    Two consecutive 3x3 Convolutions with Batch Normalization and ReLU,
    followed by optional 2x2 Max Pooling.
    
    Structure:
    Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d -> ReLU -> [MaxPool2d(2x2)]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass 1
        x = self.relu1(self.bn1(self.conv1(x)))
        # Pass 2
        x = self.relu2(self.bn2(self.conv2(x)))
        # Spatial Downsampling
        x = self.maxpool(x)
        return x


class AdvancedMNISTCNN(nn.Module):
    """
    Modern Deep CNN Topology with Global Average Pooling (GAP).
    
    Tensor Flow Dimensions:
    Input:                 (N,   1, 28, 28)
    Stage 1 (Dual 32):     (N,  32, 14, 14)  [Pool 2x2]
    Stage 2 (Dual 64):     (N,  64,  7,  7)  [Pool 2x2]
    Stage 3 (Dual 128):    (N, 128,  7,  7)  [No Pool]
    Global Avg Pool (GAP): (N, 128,  1,  1)
    Flatten:               (N, 128)
    Dropout (p=0.3):       (N, 128)
    Linear Classifier:     (N,  10)          [Raw Logits]
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extraction Stages
        self.stage1 = DualConvBlock(in_channels=in_channels, out_channels=32, pool=True)
        self.stage2 = DualConvBlock(in_channels=32, out_channels=64, pool=True)
        self.stage3 = DualConvBlock(in_channels=64, out_channels=128, pool=False)

        # Global Average Pooling Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extraction stages and GAP classifier.
        """
        x = self.stage1(x)       # -> (N,  32, 14, 14)
        x = self.stage2(x)       # -> (N,  64,  7,  7)
        x = self.stage3(x)       # -> (N, 128,  7,  7)
        x = self.gap(x)          # -> (N, 128,  1,  1)
        x = self.flatten(x)      # -> (N, 128)
        x = self.dropout(x)      # -> (N, 128)
        logits = self.classifier(x)  # -> (N,  10)
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (trainable_parameters, total_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def summary(self) -> None:
        """Prints a clean architectural breakdown."""
        trainable, total = self.count_parameters()
        print("=" * 70)
        print("              ADVANCED MNIST CNN (DUAL-CONV + GAP) SUMMARY")
        print("=" * 70)
        print(f"Input Shape:           (N, {self.in_channels}, 28, 28)")
        print(f"Stage 1 (Dual 32-ch):  (N, 32, 14, 14)  [Pool 2x2]")
        print(f"Stage 2 (Dual 64-ch):  (N, 64,  7,  7)  [Pool 2x2]")
        print(f"Stage 3 (Dual 128-ch): (N, 128, 7,  7)  [Preserves Spatial 7x7]")
        print(f"Global Avg Pool (GAP): (N, 128, 1,  1)  [Slashes Dense Overfitting]")
        print(f"Classifier Output:     (N, {self.num_classes})")
        print("-" * 70)
        print(f"Trainable Parameters:  {trainable:,}")
        print(f"Total Parameters:      {total:,}")
        print("=" * 70)


def sanity_check_advanced_shape(input_shape: Tuple[int, ...] = (2, 1, 28, 28)) -> bool:
    """Verifies that dummy input produces strictly expected output shape."""
    print(f"[Sanity Check] Verifying AdvancedMNISTCNN with shape {input_shape}...")
    model = AdvancedMNISTCNN()
    model.eval()
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (input_shape[0], 10), (
        f"Shape mismatch! Expected {(input_shape[0], 10)}, got {output.shape}"
    )
    print(f"[Sanity Check] SUCCESS! Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    model = AdvancedMNISTCNN()
    model.summary()
    sanity_check_advanced_shape()
