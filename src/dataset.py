"""
Dataset & DataLoader Pipeline
=============================
Handles data transformations, data augmentation, and PyTorch DataLoader assembly
with train/validation/test splitting.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


def get_transforms(
    augment: bool = True,
    mean: float = 0.1307,
    std: float = 0.3081,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs PyTorch transformation pipelines for training and evaluation.
    
    Args:
        augment: If True, applies data augmentations to training set.
        mean: Normalization mean (derived from EDA).
        std: Normalization standard deviation (derived from EDA).
        
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transform, test_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    return train_transform, test_transform


def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads/loads MNIST and constructs Train, Validation, and Test DataLoaders.
    
    Args:
        data_dir: Storage directory for dataset.
        batch_size: Batch size per iteration.
        val_split: Fraction of training dataset to allocate for validation (e.g., 0.1 = 10%).
        num_workers: Subprocesses for data loading.
        augment: Whether to apply data augmentation during training.
        seed: Random generator seed for reproducible dataset split.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    train_transform, test_transform = get_transforms(augment=augment)

    # Full 60k training dataset
    full_train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # 10k Test dataset
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Train / Validation Split
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"[Dataset] Loaders created successfully:")
    print(f"          - Train samples: {train_size} ({len(train_loader)} batches)")
    print(f"          - Val samples:   {val_size} ({len(val_loader)} batches)")
    print(f"          - Test samples:  {len(test_dataset)} ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader
