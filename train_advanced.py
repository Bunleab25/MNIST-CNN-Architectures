"""
Advanced Training Engine (AdvancedMNISTCNN with Label Smoothing & Cosine Annealing)
==================================================================================
Executes training for AdvancedMNISTCNN featuring:
1. Multi-transform Data Augmentation (Rotation + Scaling + Translation + Perspective).
2. Label Smoothing (0.05) to eliminate overconfidence on ambiguous digits.
3. Cosine Annealing Learning Rate Decay.
4. Checkpointing & Fit Diagnostics.

Usage:
    python train_advanced.py --epochs 12 --batch-size 64 --lr 0.001
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from src.diagnostics import diagnose_fit, plot_fit_diagnostics
from src.model_advanced import AdvancedMNISTCNN, sanity_check_advanced_shape


def get_advanced_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns upgraded training augmentations and clean validation/test transforms.
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.35),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    return train_transform, test_transform


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepares Train, Val, and Test loaders with advanced transforms."""
    train_transform, test_transform = get_advanced_transforms()

    full_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader


def train_advanced_model(
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    label_smoothing: float = 0.05,
    weight_decay: float = 1e-4,
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    report_dir: str = "./reports",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Advanced Train] Active Device: {device}")

    # 1. Sanity Check
    sanity_check_advanced_shape()

    # 2. DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # 3. Model, Loss, Optimizer, Scheduler
    model = AdvancedMNISTCNN().to(device)
    model.summary()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_advanced_model.pth")

    print(f"\n[Advanced Train] Starting {epochs} Epochs with Cosine Annealing & Label Smoothing ({label_smoothing})")
    print("=" * 75)
    print(f"{'Epoch':^7} | {'LR':^9} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Val Loss':^12} | {'Val Acc (%)':^14}")
    print("-" * 75)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        current_lr = scheduler.get_last_lr()[0]

        # Training Pass
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validation Pass
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / val_total
        val_acc = (val_correct / val_total) * 100.0

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history,
            }, best_checkpoint_path)
            saved_tag = "⭐ (Best Saved)"

        print(f"{epoch:^7d} | {current_lr:^9.6f} | {train_loss:^12.4f} | {train_acc:^14.2f} | {val_loss:^12.4f} | {val_acc:^14.2f} {saved_tag}")

    elapsed = time.time() - start_time
    print("=" * 75)
    print(f"[Advanced Train] Completed in {elapsed:.2f}s. Best val loss: {best_val_loss:.4f}")

    # Save history JSON
    with open(os.path.join(checkpoint_dir, "advanced_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot Diagnostics
    report = diagnose_fit(history)
    report.print_summary()
    plot_fit_diagnostics(history, report, save_path=os.path.join(report_dir, "fit_diagnostic_advanced.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AdvancedMNISTCNN")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()

    train_advanced_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        label_smoothing=args.label_smoothing,
    )
