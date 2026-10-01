"""
Model Diagnostics & Evaluation Module
=====================================
Performs hold-out test set evaluation:
1. Calculates overall top-1 test accuracy and loss.
2. Computes per-class precision and recall.
3. Generates Confusion Matrix heatmaps.
4. Plots misclassified test images to diagnose failure modes.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .model import MNISTCNN
    from .dataset import get_mnist_dataloaders
except (ImportError, ValueError):
    try:
        from src.model import MNISTCNN
        from src.dataset import get_mnist_dataloaders
    except ImportError:
        from model import MNISTCNN
        from dataset import get_mnist_dataloaders


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluates the model on unseen test data.
    
    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: (test_loss, test_acc, all_preds, all_targets)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    test_loss = running_loss / total
    test_acc = (correct / total) * 100.0

    return test_loss, test_acc, np.array(all_preds), np.array(all_targets)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Computes confusion matrix without external dependencies.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
    save_path: str = "./reports/confusion_matrix.png",
) -> None:
    """
    Draws and saves a confusion matrix heatmap.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted Digit", fontsize=11, labelpad=10)
    ax.set_ylabel("True Digit", fontsize=11, labelpad=10)
    ax.set_title("Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {save_path}")


def plot_misclassified_examples(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_path: str = "./reports/misclassifications.png",
    max_samples: int = 16,
) -> None:
    """
    Collects and visualizes incorrectly predicted test samples.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            mask = preds != labels
            if mask.sum() > 0:
                bad_imgs = images[mask].cpu()
                bad_labels = labels[mask].cpu()
                bad_preds = preds[mask].cpu()

                for i in range(bad_imgs.size(0)):
                    misclassified_images.append(bad_imgs[i])
                    misclassified_labels.append(bad_labels[i].item())
                    misclassified_preds.append(bad_preds[i].item())

                    if len(misclassified_images) >= max_samples:
                        break
            if len(misclassified_images) >= max_samples:
                break

    if not misclassified_images:
        print("[Evaluation] Perfect score! No misclassified samples found.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_to_plot = min(len(misclassified_images), max_samples)
    cols = 4
    rows = (num_to_plot + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.5))
    fig.suptitle("Top Misclassified Test Samples (Error Diagnostics)", fontsize=13, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < num_to_plot:
            img = misclassified_images[i].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            true_lbl = misclassified_labels[i]
            pred_lbl = misclassified_preds[i]
            ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl}", color="red", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Misclassified sample diagnostics saved to: {save_path}")


def run_evaluation(
    checkpoint_path: str,
    test_loader: DataLoader,
    device: torch.device,
    report_dir: str = "./reports",
) -> Dict[str, float]:
    """
    Loads saved checkpoint and runs full evaluation.
    """
    print(f"\n[Evaluation] Loading model checkpoint: {checkpoint_path}")
    model = MNISTCNN().to(device)

    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(str(PROJECT_ROOT), checkpoint_path.lstrip("./"))
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, all_preds, all_targets = evaluate_test_set(model, test_loader, device)

    print("=" * 60)
    print("                 FINAL TEST SET EVALUATION")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=" * 60)

    plot_confusion_matrix(
        all_targets,
        all_preds,
        num_classes=10,
        save_path=os.path.join(report_dir, "confusion_matrix.png"),
    )
    plot_misclassified_examples(
        model,
        test_loader,
        device,
        save_path=os.path.join(report_dir, "misclassifications.png"),
    )

    return {"test_loss": test_loss, "test_accuracy": test_acc}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Resolve default paths relative to working directory or project root
    data_dir = "./data" if os.path.exists("./data") else str(PROJECT_ROOT / "data")
    ckpt_path = "./checkpoints/best_model.pth" if os.path.exists("./checkpoints/best_model.pth") else str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
    report_dir = "./reports" if os.path.exists("./reports") else str(PROJECT_ROOT / "reports")

    _, _, test_loader = get_mnist_dataloaders(data_dir=data_dir, batch_size=64, num_workers=2)
    run_evaluation(checkpoint_path=ckpt_path, test_loader=test_loader, device=device, report_dir=report_dir)

