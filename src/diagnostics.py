"""
Model Fit Diagnostics & Overfitting/Underfitting Analyzer
=========================================================
Automatically analyzes training and validation histories to measure:
1. Fit Status: (Good Fit, Overfitting, Underfitting, Under-trained).
2. Generalization Gap (Loss & Accuracy).
3. Inflection / Best Epoch (where Val Loss hit minimum).
4. Overfitting Severity Percentage.
5. Actionable Engineering Recommendations.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class FitDiagnosticReport:
    fit_status: str
    status_description: str
    best_epoch: int
    min_val_loss: float
    final_train_loss: float
    final_val_loss: float
    final_train_acc: float
    final_val_acc: float
    generalization_gap_loss: float
    generalization_gap_acc: float
    overfitting_increase_pct: float
    loss_trajectory_slope: float
    recommendations: List[str]

    def print_summary(self) -> None:
        """Prints a structured ASCII diagnostic report to terminal."""
        print("\n" + "=" * 70)
        print("                 🧠 DEEP LEARNING MODEL FIT DIAGNOSTIC REPORT")
        print("=" * 70)
        print(f"Overall Diagnosis :  \033[1;32m{self.fit_status}\033[0m" if "GOOD" in self.fit_status or "REGULARIZED" in self.fit_status
              else f"Overall Diagnosis :  \033[1;31m{self.fit_status}\033[0m")
        print(f"Details           :  {self.status_description}")
        print("-" * 70)
        print("📈 KEY METRICS:")
        print(f"  • Best Validation Epoch      :  Epoch {self.best_epoch}")
        print(f"  • Minimum Validation Loss    :  {self.min_val_loss:.4f}")
        print(f"  • Final Train / Val Loss     :  {self.final_train_loss:.4f} / {self.final_val_loss:.4f}")
        print(f"  • Final Train / Val Accuracy :  {self.final_train_acc:.2f}% / {self.final_val_acc:.2f}%")
        print(f"  • Generalization Gap (Loss)  :  {self.generalization_gap_loss:+.4f} (Val Loss - Train Loss)")
        print(f"  • Generalization Gap (Acc)   :  {self.generalization_gap_acc:+.2f}% (Train Acc - Val Acc)")
        print(f"  • Post-Min Loss Rise         :  {self.overfitting_increase_pct:.2f}%")
        print("-" * 70)
        print("💡 ACTIONABLE RECOMMENDATIONS:")
        for idx, rec in enumerate(self.recommendations, 1):
            print(f"  {idx}. {rec}")
        print("=" * 70 + "\n")


def diagnose_fit(history: Dict[str, List[float]]) -> FitDiagnosticReport:
    """
    Analyzes training history dictionary and computes mathematical metrics to measure fit.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        
    Returns:
        FitDiagnosticReport: Object containing detailed diagnostic metrics.
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    total_epochs = len(train_loss)

    if total_epochs < 2:
        raise ValueError("At least 2 epochs of history are required to evaluate fit trends.")

    # 1. Optimal Validation Epoch
    min_val_loss = min(val_loss)
    best_epoch = val_loss.index(min_val_loss) + 1  # 1-indexed

    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]

    # 2. Generalization Gaps
    gen_gap_loss = final_val_loss - final_train_loss
    gen_gap_acc = final_train_acc - final_val_acc

    # 3. Overfitting Post-Min Increase
    if min_val_loss > 0:
        overfitting_increase_pct = ((final_val_loss - min_val_loss) / min_val_loss) * 100.0
    else:
        overfitting_increase_pct = 0.0

    # 4. Late-stage Loss Trajectory (Slope of last 3 epochs of val loss)
    window = min(3, total_epochs)
    x = np.arange(window)
    val_loss_recent = np.array(val_loss[-window:])
    slope = float(np.polyfit(x, val_loss_recent, 1)[0])

    # 5. Fit Classification Logic
    recommendations: List[str] = []

    # Case A: Severe Underfitting (high loss, low accuracy on both)
    if final_train_loss > 0.4 or final_train_acc < 90.0:
        status = "UNDERFITTING"
        description = "Model capacity is insufficient or learning rate is misconfigured."
        recommendations.append("Increase model capacity (add more convolutional channels or layers).")
        recommendations.append("Adjust the initial learning rate or optimizer.")
        recommendations.append("Reduce regularization (lower dropout probability or weight decay).")

    # Case B: Under-Trained (Loss is still decreasing rapidly across the last epochs)
    elif slope < -0.015 and best_epoch == total_epochs and final_val_acc < 99.2:
        status = "UNDER-TRAINED (NEEDS MORE EPOCHS)"
        description = "Validation loss is still dropping rapidly at the final epoch."
        recommendations.append(f"Model hasn't converged yet. Extend training from {total_epochs} to {total_epochs * 2} epochs.")
        recommendations.append("The learning curves show no signs of plateau or divergence.")

    # Case C: Overfitting (Val loss rose significantly past its minimum)
    elif overfitting_increase_pct > 15.0 or (best_epoch <= total_epochs - 3 and slope > 0.01):
        status = "OVERFITTING (DIVERGENCE DETECTED)"
        description = f"Validation loss hit minimum at Epoch {best_epoch} and rose by {overfitting_increase_pct:.1f}% afterward."
        recommendations.append(f"Use the checkpoint from Epoch {best_epoch} (lowest validation loss).")
        recommendations.append("Increase Dropout probability (e.g. p=0.5 -> p=0.6).")
        recommendations.append("Increase data augmentation strength or add L2 weight decay.")
        recommendations.append("Implement Early Stopping with patience=3.")

    # Case D: Regularized Good Fit (Val loss is lower than Train loss due to Dropout/Augmentation)
    elif final_val_loss <= final_train_loss and final_val_acc >= 97.0:
        status = "REGULARIZED GOOD FIT (OPTIMAL)"
        description = "Validation performance is excellent and surpasses train performance due to active Dropout & Data Augmentation during training."
        recommendations.append(f"Model generalized exceptionally well! Best weights saved at Epoch {best_epoch}.")
        recommendations.append("No changes needed. Proceed to final hold-out test set evaluation (`--mode eval`).")

    # Case E: Standard Balanced Good Fit
    elif gen_gap_loss <= 0.05 and final_val_acc >= 98.0:
        status = "GOOD FIT (BALANCED CONVERGENCE)"
        description = "Train and Validation curves converged closely with minimal generalization gap."
        recommendations.append("Model is well-balanced and ready for test evaluation and deployment.")

    else:
        status = "STABLE / MODERATE GENERALIZATION"
        description = "Loss curves are stable with minor generalization variance."
        recommendations.append("Performance is acceptable. Review confusion matrix for any class-specific errors.")

    return FitDiagnosticReport(
        fit_status=status,
        status_description=description,
        best_epoch=best_epoch,
        min_val_loss=min_val_loss,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_train_acc=final_train_acc,
        final_val_acc=final_val_acc,
        generalization_gap_loss=gen_gap_loss,
        generalization_gap_acc=gen_gap_acc,
        overfitting_increase_pct=overfitting_increase_pct,
        loss_trajectory_slope=slope,
        recommendations=recommendations,
    )


def plot_fit_diagnostics(
    history: Dict[str, List[float]],
    report: FitDiagnosticReport,
    save_path: str = "./reports/fit_diagnostic_analysis.png",
) -> None:
    """
    Plots an annotated 2-panel diagnostic chart with the optimal epoch marker and fit status box.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Panel 1: Loss Diagnostic
    ax1.plot(epochs, history["train_loss"], "o-", label="Training Loss", color="#1f77b4", linewidth=2.2, markersize=6)
    ax1.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#d62728", linewidth=2.2, markersize=6)
    ax1.axvline(
        x=report.best_epoch,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label=f"Best Epoch ({report.best_epoch})",
    )
    ax1.scatter([report.best_epoch], [report.min_val_loss], color="#2ca02c", s=120, zorder=5, marker="*")
    ax1.set_title("Training vs. Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    # Panel 2: Accuracy Diagnostic
    ax2.plot(epochs, history["train_acc"], "o-", label="Training Accuracy", color="#1f77b4", linewidth=2.2, markersize=6)
    ax2.plot(epochs, history["val_acc"], "s--", label="Validation Accuracy", color="#d62728", linewidth=2.2, markersize=6)
    ax2.axvline(
        x=report.best_epoch,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label=f"Best Epoch ({report.best_epoch})",
    )
    ax2.scatter([report.best_epoch], [report.final_val_acc], color="#2ca02c", s=120, zorder=5, marker="*")
    ax2.set_title("Training vs. Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    # Super Title with Diagnosed Status
    status_color = "#2ca02c" if "GOOD" in report.fit_status or "REGULARIZED" in report.fit_status else "#d62728"
    fig.suptitle(
        f"Model Fit Diagnosis: {report.fit_status}\nBest Val Loss: {report.min_val_loss:.4f} @ Epoch {report.best_epoch}",
        fontsize=14,
        fontweight="bold",
        color=status_color,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Diagnostics] Annotated fit analysis plot saved to: {save_path}")


def load_history_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, List[float]]]:
    """Attempts to load history from checkpoint file or companion json."""
    # 1. Check for companion training_history.json
    json_path = os.path.join(os.path.dirname(checkpoint_path), "training_history.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    # 2. Check within PyTorch checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "history" in checkpoint:
            return checkpoint["history"]
    return None
