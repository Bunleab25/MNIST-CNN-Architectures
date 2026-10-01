"""
CNN Model Architecture Module
==============================
Defines reusable ConvBlock components and the complete MNISTCNN classifier.
Tracks tensor shapes step-by-step to ensure dimensional consistency.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Modular Convolutional Block consisting of:
    1. 2D Convolution
    2. Batch Normalization (stabilizes internal covariate shift)
    3. ReLU Non-linear Activation
    4. Max Pooling (downsamples spatial dimensions by factor of 2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: bool = True,
    ) -> None:
        super().__init__()
        # bias=False because BatchNorm contains its own learnable affine parameters (beta/gamma)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape:  (N, C_in, H, W)
        x = self.conv(x)  # -> (N, C_out, H', W')
        x = self.bn(x)    # -> (N, C_out, H', W')
        x = self.relu(x)  # -> (N, C_out, H', W')
        x = self.pool(x)  # -> (N, C_out, H'/2, W'/2)
        return x


class MNISTCNN(nn.Module):
    """
    End-to-End Convolutional Neural Network for MNIST digit recognition.
    
    Architecture:
    Input: (N, 1, 28, 28)
    ──► ConvBlock 1 (1 -> 32)   ──► Output: (N, 32, 14, 14)
    ──► ConvBlock 2 (32 -> 64)  ──► Output: (N, 64, 7, 7)
    ──► Flatten                 ──► Output: (N, 3136)
    ──► Dropout(p=0.5)          ──► Output: (N, 3136)
    ──► Linear(3136 -> 128)     ──► Output: (N, 128)
    ──► ReLU                    ──► Output: (N, 128)
    ──► Dropout(p=0.25)         ──► Output: (N, 128)
    ──► Linear(128 -> 10)       ──► Output: (N, 10) [Raw Logits]
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Feature Extraction Backbone
        self.block1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.block2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Classification Head
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input batch tensor of shape (N, 1, 28, 28)
            
        Returns:
            torch.Tensor: Raw logits of shape (N, 10)
        """
        x = self.block1(x)       # (N, 1, 28, 28)  -> (N, 32, 14, 14)
        x = self.block2(x)       # (N, 32, 14, 14) -> (N, 64, 7, 7)
        x = self.flatten(x)      # (N, 64, 7, 7)   -> (N, 3136)
        logits = self.classifier(x)  # (N, 3136)       -> (N, 10)
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """
        Counts trainable and total parameters in the model.
        
        Returns:
            Tuple[int, int]: (trainable_params, total_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def summary(self) -> None:
        """
        Prints architecture summary and parameter counts.
        """
        trainable, total = self.count_parameters()
        print("=" * 65)
        print("                   MNISTCNN ARCHITECTURE SUMMARY")
        print("=" * 65)
        print(self)
        print("-" * 65)
        print(f"Trainable Parameters: {trainable:,}")
        print(f"Total Parameters:     {total:,}")
        print("=" * 65)


def sanity_check_shape(input_shape: Tuple[int, ...] = (2, 1, 28, 28)) -> bool:
    """
    Performs a dry-run forward pass with dummy tensor to verify tensor shape flow.
    
    Args:
        input_shape: Shape of dummy input batch (default: batch size 2, 1 channel, 28x28).
        
    Returns:
        bool: True if output matches (batch_size, 10).
    """
    print(f"[Sanity Check] Testing forward pass with dummy input shape {input_shape}...")
    model = MNISTCNN()
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        output = model(dummy_input)
        
    expected_shape = (input_shape[0], 10)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"
    )
    print(f"[Sanity Check] SUCCESS! Output shape: {output.shape} correctly matches expected.")
    return True


if __name__ == "__main__":
    model = MNISTCNN()
    model.summary()
    sanity_check_shape()
