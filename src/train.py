"""
Training & Validation Engine
============================
Contains the training loop, validation logic, single-batch sanity check,
and learning curve plotting.
"""

import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Executes one full training epoch across the dataset using the 4-step workflow.
    
    Returns:
        Tuple[float, float]: (average_epoch_loss, epoch_accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # ----------------------------------------------------
        # 4-STEP TRAINING WORKFLOW
        # ----------------------------------------------------
        # Step 1: Clear prior gradients (set_to_none is faster than zeroing)
        optimizer.zero_grad(set_to_none=True)

        # Step 2: Forward pass through model
        outputs = model(images)

        # Step 3: Compute scalar loss and backpropagate gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Step 4: Update trainable weights via optimizer rule
        optimizer.step()
        # ----------------------------------------------------

        # Metric tracking
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates model performance on the validation set without computing gradients.
    
    Returns:
        Tuple[float, float]: (average_val_loss, val_accuracy_percent)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc


def overfit_single_batch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int = 40,
) -> bool:
    """
    Sanity test: Attempts to overfit a single mini-batch.
    If the training loss fails to drop close to 0.0, there is a bug in backprop or model capacity.
    """
    print(f"\n[Sanity Check] Running Single-Batch Overfit Test ({steps} steps)...")
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == steps:
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item() * 100
            print(f"  Step {step:02d}/{steps:02d} - Loss: {loss.item():.4f} | Batch Acc: {acc:.2f}%")

    if loss.item() < 0.1:
        print("[Sanity Check] PASSED! Model successfully memorized single batch.")
        return True
    else:
        print("[Sanity Check] WARNING: Model struggled to overfit single batch. Check learning rate.")
        return False


def plot_learning_curves(history: Dict[str, List[float]], save_path: str = "./reports/learning_curves.png") -> None:
    """
    Plots standard publication-grade Training vs Validation Loss and Accuracy graphs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss)) + 1
    min_val_loss = val_loss[best_epoch - 1]
    max_val_acc = val_acc[best_epoch - 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 1. Loss Curve
    ax1.plot(epochs, train_loss, "o-", label="Training Loss", color="#1f77b4", linewidth=2.2, markersize=6)
    ax1.plot(epochs, val_loss, "s--", label="Validation Loss", color="#d62728", linewidth=2.2, markersize=6)
    ax1.axvline(x=best_epoch, color="#2ca02c", linestyle=":", linewidth=1.8, alpha=0.85, label=f"Best Epoch ({best_epoch})")
    ax1.scatter([best_epoch], [min_val_loss], color="#2ca02c", s=120, zorder=5, marker="*")
    
    ax1.set_title("Training vs. Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    # 2. Accuracy Curve
    ax2.plot(epochs, train_acc, "o-", label="Training Accuracy", color="#1f77b4", linewidth=2.2, markersize=6)
    ax2.plot(epochs, val_acc, "s--", label="Validation Accuracy", color="#d62728", linewidth=2.2, markersize=6)
    ax2.axvline(x=best_epoch, color="#2ca02c", linestyle=":", linewidth=1.8, alpha=0.85, label=f"Best Epoch ({best_epoch})")
    ax2.scatter([best_epoch], [max_val_acc], color="#2ca02c", s=120, zorder=5, marker="*")

    ax2.set_title("Training vs. Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    fig.suptitle("Model Learning Curves (Train vs. Validation)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Training] Standard learning curves saved to: {save_path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: str = "./checkpoints",
    report_dir: str = "./reports",
) -> Dict[str, List[float]]:
    """
    Full training loop with checkpointing on lowest validation loss.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    print(f"\n[Training] Starting {epochs} Epochs on Device: {device}")
    print("=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Val Loss':^12} | {'Val Acc (%)':^14}")
    print("-" * 70)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # LR Scheduler step
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpoint if best model
        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "history": history,
                },
                best_checkpoint_path,
            )
            saved_tag = "⭐ (Best Saved)"

        print(
            f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc:^14.2f} | {val_loss:^12.4f} | {val_acc:^14.2f} {saved_tag}"
        )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[Training] Completed in {elapsed:.2f} seconds.")
    print(f"[Training] Best validation loss: {best_val_loss:.4f} saved to {best_checkpoint_path}")

    # Save raw history to JSON
    import json
    history_json_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot final curves
    plot_learning_curves(history, save_path=os.path.join(report_dir, "learning_curves.png"))

    # Run Automatic Fit Diagnostics
    try:
        from .diagnostics import diagnose_fit, plot_fit_diagnostics
        diagnostic_report = diagnose_fit(history)
        diagnostic_report.print_summary()
        plot_fit_diagnostics(
            history,
            diagnostic_report,
            save_path=os.path.join(report_dir, "fit_diagnostic_analysis.png"),
        )
    except Exception as e:
        print(f"[Diagnostics] Warning: could not run fit diagnostics: {e}")

    return history

