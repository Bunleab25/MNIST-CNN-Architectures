"""
Exploratory Data Analysis (EDA) Module
======================================
Inspects the raw MNIST dataset:
1. Calculates exact dataset mean and standard deviation.
2. Checks class distribution and balance.
3. Generates sample visual inspection grids.
"""

import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def calculate_dataset_statistics(data_dir: str = "./data") -> Tuple[float, float]:
    """
    Computes global pixel mean and standard deviation on the raw MNIST training dataset.
    
    Args:
        data_dir: Path to directory for storing/loading MNIST data.
        
    Returns:
        Tuple[float, float]: (mean, std)
    """
    print("[EDA] Loading raw dataset for statistical calculation...")
    raw_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    
    loader = torch.utils.data.DataLoader(
        raw_dataset,
        batch_size=len(raw_dataset),
        shuffle=False,
        num_workers=0,
    )
    
    data, _ = next(iter(loader))
    mean = float(data.mean().item())
    std = float(data.std().item())
    
    print(f"[EDA] Dataset Size: {len(raw_dataset)} samples")
    print(f"[EDA] Tensor Shape per image: {data.shape[1:]} (Channels x Height x Width)")
    print(f"[EDA] Pixel Value Range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"[EDA] Calculated Mean: {mean:.4f}")
    print(f"[EDA] Calculated Std:  {std:.4f}")
    return mean, std


def check_class_distribution(data_dir: str = "./data") -> Dict[int, int]:
    """
    Checks sample counts per class (0-9) to detect class imbalance.
    
    Args:
        data_dir: Path to directory for storing/loading MNIST data.
        
    Returns:
        Dict[int, int]: Mapping from class label to sample count.
    """
    raw_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )
    targets = raw_dataset.targets.numpy()
    unique, counts = np.unique(targets, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    
    total = len(targets)
    print("\n[EDA] Class Distribution:")
    print("-" * 35)
    for digit, count in distribution.items():
        percentage = (count / total) * 100
        print(f"Digit {digit}: {count:>5} samples ({percentage:.2f}%)")
    print("-" * 35)
    return distribution


def plot_sample_grid(
    data_dir: str = "./data",
    save_path: str = "./reports/eda_samples.png",
    num_samples_per_row: int = 5,
) -> None:
    """
    Plots a sample image grid showing digits 0-9.
    
    Args:
        data_dir: Path to data directory.
        save_path: Destination path for saved figure.
        num_samples_per_row: Number of columns in subplot grid.
    """
    raw_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Collect first occurrence of each digit 0-9
    found_digits = {}
    for img, label in raw_dataset:
        if label not in found_digits:
            found_digits[label] = img
        if len(found_digits) == 10:
            break
            
    fig, axes = plt.subplots(2, num_samples_per_row, figsize=(10, 4.5))
    fig.suptitle("MNIST Dataset Sample Grid (Digits 0 - 9)", fontsize=14, fontweight="bold")
    
    for digit in range(10):
        row = digit // num_samples_per_row
        col = digit % num_samples_per_row
        ax = axes[row, col]
        img = found_digits[digit].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Digit: {digit}", fontsize=11)
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Sample visual grid saved to: {save_path}")


def run_eda(data_dir: str = "./data", report_dir: str = "./reports") -> None:
    """
    Runs the complete Exploratory Data Analysis suite.
    """
    print("=" * 60)
    print("        EXPLORATORY DATA ANALYSIS (EDA) - MNIST")
    print("=" * 60)
    os.makedirs(report_dir, exist_ok=True)
    calculate_dataset_statistics(data_dir)
    check_class_distribution(data_dir)
    plot_sample_grid(data_dir, save_path=os.path.join(report_dir, "eda_samples.png"))
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
