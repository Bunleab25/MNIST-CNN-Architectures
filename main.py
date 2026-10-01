"""
MNIST CNN Classifier - Main Entry Point
=======================================
Command Line Interface (CLI) to orchestrate:
1. EDA & Data Inspection (--mode eda)
2. Sanity Checks & Shape Dry-Runs (--mode sanity)
3. Model Training & Validation (--mode train)
4. Test Set Evaluation & Diagnostics (--mode eval)
5. Complete Pipeline (--mode all)
"""

import argparse
import os
import torch

from src.dataset import get_mnist_dataloaders
from src.eda import run_eda
from src.evaluate import run_evaluation
from src.model import MNISTCNN, sanity_check_shape
from src.train import overfit_single_batch, train_model


def get_device(force_cpu: bool = False) -> torch.device:
    """Detects best available device (CUDA, MPS, or CPU)."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch MNIST CNN End-to-End Deep Learning Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["eda", "sanity", "train", "eval", "diagnose", "all"],
        help="Pipeline phase: 'eda', 'sanity', 'train', 'eval', 'diagnose', or 'all'",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer initial learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="L2 regularization weight decay"
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set split ratio")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./reports",
        help="Directory to save visual diagnostics and figures",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Disables CUDA even if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(force_cpu=args.no_cuda)
    print(f"[System] Active Compute Device: {device}")

    # 1. EXPLORATORY DATA ANALYSIS (EDA)
    if args.mode in ["eda", "all"]:
        print("\n" + "=" * 60)
        print("PHASE 1: Exploratory Data Analysis")
        print("=" * 60)
        run_eda(data_dir=args.data_dir, report_dir=args.report_dir)

    # 2. SANITY CHECK (Shape verification & Single-batch overfitting)
    if args.mode in ["sanity", "all"]:
        print("\n" + "=" * 60)
        print("PHASE 2: Architectural Sanity Checks")
        print("=" * 60)
        # Check forward pass shape flow
        sanity_check_shape(input_shape=(2, 1, 28, 28))
        
        # Test single-batch memorization
        train_loader, _, _ = get_mnist_dataloaders(
            data_dir=args.data_dir, batch_size=32, val_split=0.1, num_workers=0
        )
        test_model = MNISTCNN().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3)
        overfit_single_batch(test_model, train_loader, criterion, optimizer, device, steps=30)

    # 3. TRAINING & VALIDATION
    if args.mode in ["train", "all"]:
        print("\n" + "=" * 60)
        print("PHASE 3: Model Training & Validation")
        print("=" * 60)
        train_loader, val_loader, _ = get_mnist_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=2,
            augment=True,
        )
        model = MNISTCNN()
        model.summary()

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            report_dir=args.report_dir,
        )

    # 4. EVALUATION & ERROR ANALYSIS
    if args.mode in ["eval", "all"]:
        print("\n" + "=" * 60)
        print("PHASE 4: Final Test Set Evaluation & Diagnostics")
        print("=" * 60)
        _, _, test_loader = get_mnist_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=2,
        )
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        run_evaluation(
            checkpoint_path=checkpoint_path,
            test_loader=test_loader,
            device=device,
            report_dir=args.report_dir,
        )

    # 5. FIT DIAGNOSTICS (Overfitting / Underfitting Analyzer)
    if args.mode in ["diagnose"]:
        print("\n" + "=" * 60)
        print("PHASE 5: Model Fit & Generalization Diagnostics")
        print("=" * 60)
        from src.diagnostics import diagnose_fit, load_history_from_checkpoint, plot_fit_diagnostics
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        history = load_history_from_checkpoint(checkpoint_path)
        if history is None:
            # Fallback to demo/latest metrics
            from measure_fit import get_default_history_if_none
            history = get_default_history_if_none()
        report = diagnose_fit(history)
        report.print_summary()
        plot_fit_diagnostics(
            history,
            report,
            save_path=os.path.join(args.report_dir, "fit_diagnostic_analysis.png"),
        )


if __name__ == "__main__":
    main()

