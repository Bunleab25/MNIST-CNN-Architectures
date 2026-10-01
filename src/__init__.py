"""
MNIST CNN Package
=================
Modular PyTorch implementation of an end-to-end CNN for MNIST digit classification.
"""

from .eda import run_eda, calculate_dataset_statistics, check_class_distribution
from .dataset import get_mnist_dataloaders, get_transforms
from .model import MNISTCNN, ConvBlock
from .train import train_model, train_one_epoch, validate, overfit_single_batch
from .evaluate import evaluate_test_set, plot_confusion_matrix, plot_misclassified_examples
from .diagnostics import diagnose_fit, plot_fit_diagnostics, FitDiagnosticReport

__all__ = [
    "run_eda",
    "calculate_dataset_statistics",
    "check_class_distribution",
    "get_mnist_dataloaders",
    "get_transforms",
    "MNISTCNN",
    "ConvBlock",
    "train_model",
    "train_one_epoch",
    "validate",
    "overfit_single_batch",
    "evaluate_test_set",
    "plot_confusion_matrix",
    "plot_misclassified_examples",
    "diagnose_fit",
    "plot_fit_diagnostics",
    "FitDiagnosticReport",
]
