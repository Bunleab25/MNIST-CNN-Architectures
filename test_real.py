#!/usr/bin/env python3
"""
Real Image Testing & Inference Module
=====================================
Performs inference on real-world test images stored in a dataset directory
(e.g., './datasets') using trained MNIST CNN models (AdvancedMNISTCNN or baseline).

Key Features:
1. Automated image discovery (.png, .jpg, .jpeg, .bmp, .webp) in target folder.
2. Robust real-world preprocessing:
   - Grayscale conversion.
   - Intelligent auto-inversion (adapts black-on-white drawings/photos to MNIST white-on-black).
   - Aspect-ratio preserving digit bounding box extraction and 28x28 centering.
   - Exact MNIST mean/std normalization.
3. Checkpoint auto-detection (AdvancedMNISTCNN or baseline MNISTCNN).
4. Terminal tabular metrics (Predicted class, Confidence %, Top-3 probabilities).
5. High-resolution diagnostic visualization figure saved to disk.

Usage:
    python test_real.py
    python test_real.py --image-dir ./datasets
    python test_real.py --checkpoint ./checkpoints/best_advanced_model.pth --save-plot ./reports/real_test_predictions.png
"""

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import MNISTCNN
from src.model_advanced import AdvancedMNISTCNN


def get_default_device() -> torch.device:
    """Detects available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str,
    model_type: str = "auto",
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, str]:
    """
    Loads model checkpoint with automatic architecture detection.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        model_type: 'auto', 'advanced', or 'baseline'.
        device: PyTorch device.

    Returns:
        Tuple of (model, detected_architecture_name)
    """
    if device is None:
        device = get_default_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Detect architecture if auto
    if model_type == "auto":
        is_advanced = any(k.startswith("stage1.") or k.startswith("gap.") for k in state_dict.keys())
        model_type = "advanced" if is_advanced else "baseline"

    if model_type == "advanced":
        model = AdvancedMNISTCNN().to(device)
        arch_name = "AdvancedMNISTCNN (Dual-Conv + GAP)"
    else:
        model = MNISTCNN().to(device)
        arch_name = "Baseline MNISTCNN"

    model.load_state_dict(state_dict)
    model.eval()
    return model, arch_name


def preprocess_image(
    image_path: str,
    invert_mode: str = "auto",
) -> Tuple[torch.Tensor, np.ndarray, Image.Image]:
    """
    Preprocesses a real-world image to match the MNIST input distribution.

    Steps:
    1. Convert to single-channel 8-bit grayscale.
    2. Check background intensity:
       - Real photos/scans usually have dark ink on white paper (bg > 128).
       - MNIST requires bright digit stroke on dark background (bg ~ 0).
       - Auto-inverts if background brightness exceeds threshold.
    3. Isolate digit bounding box, scale down preserving aspect ratio to fit inside
       a 20x20 region, and center inside a 28x28 canvas (exact MNIST standard).
    4. Normalize with MNIST mean (0.1307) and std (0.3081).

    Args:
        image_path: Path to image file.
        invert_mode: 'auto', 'invert', or 'no-invert'.

    Returns:
        Tuple of:
          - normalized_tensor: Shape (1, 1, 28, 28) ready for model forward pass.
          - canvas_28x28: Shape (28, 28) numpy array in range [0, 1] for visualization.
          - raw_pil_image: Original PIL image for display.
    """
    raw_img = Image.open(image_path)
    gray_img = raw_img.convert("L")
    arr = np.array(gray_img, dtype=np.float32)

    # Inversion Handling
    if invert_mode == "auto":
        # Sample border pixels (top, bottom, left, right edges) to determine background
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        if np.mean(border) > 127.5:
            arr = 255.0 - arr
    elif invert_mode == "invert":
        arr = 255.0 - arr
    elif invert_mode == "no-invert":
        pass

    # Clip negative or spurious values
    arr = np.clip(arr, 0.0, 255.0)

    # Bounding Box Extraction & Centering (20x20 in 28x28)
    # Threshold stroke pixels
    stroke_threshold = max(35.0, np.percentile(arr, 70))
    coords = np.argwhere(arr > stroke_threshold)

    canvas = np.zeros((28, 28), dtype=np.float32)

    if len(coords) > 0 and (arr.shape[0] > 28 or arr.shape[1] > 28):
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = arr[y_min : y_max + 1, x_min : x_max + 1]

        h, w = cropped.shape
        # Scale factor to fit inside 20x20
        scale = 20.0 / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        cropped_pil = Image.fromarray(cropped.astype(np.uint8))
        resized_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
        resized_arr = np.array(resized_pil, dtype=np.float32)

        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_arr
    else:
        # Image is already small or uniform: direct resize
        resized_pil = Image.fromarray(arr.astype(np.uint8)).resize((28, 28), Image.Resampling.BILINEAR)
        canvas = np.array(resized_pil, dtype=np.float32)

    # Scale to [0, 1]
    canvas = canvas / 255.0

    # Convert to Tensor and Normalize with MNIST constants
    tensor = torch.tensor(canvas, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    normalized_tensor = (tensor - 0.1307) / 0.3081

    return normalized_tensor, canvas, raw_img


def run_inference_on_folder(
    image_dir: str,
    checkpoint_path: str,
    model_type: str = "auto",
    invert_mode: str = "auto",
    save_plot_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> List[dict]:
    """
    Discovers all images in image_dir, runs inference, prints table, and saves plot.
    """
    if device is None:
        device = get_default_device()

    # Load Model
    model, arch_name = load_model(checkpoint_path, model_type=model_type, device=device)

    # Find Images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"\n[Error] No image files found in folder: '{image_dir}'")
        print(f"Supported formats: {extensions}\n")
        return []

    print("=" * 85)
    print("               REAL IMAGE MNIST CNN INFERENCE TEST ENGINE")
    print("=" * 85)
    print(f"  Model Architecture : {arch_name}")
    print(f"  Checkpoint Path    : {checkpoint_path}")
    print(f"  Compute Device     : {device}")
    print(f"  Target Image Folder: {image_dir}")
    print(f"  Total Images Found : {len(image_paths)}")
    print(f"  Color Invert Mode  : {invert_mode}")
    print("=" * 85)

    results = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        tensor, canvas, raw_pil = preprocess_image(img_path, invert_mode=invert_mode)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        predicted_digit = int(np.argmax(probs))
        confidence = float(probs[predicted_digit]) * 100.0

        # Top 3 predictions
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_info = [(int(idx), float(probs[idx]) * 100.0) for idx in top3_indices]

        results.append({
            "path": img_path,
            "filename": filename,
            "prediction": predicted_digit,
            "confidence": confidence,
            "top3": top3_info,
            "probabilities": probs,
            "canvas": canvas,
            "raw_pil": raw_pil,
        })

    # Print Formatted Results Table
    print(f"\n{'FILENAME':<24} | {'PRED':<5} | {'CONFIDENCE':<10} | {'TOP-3 PREDICTIONS (Digit : Prob%)'}")
    print("-" * 85)
    for res in results:
        top3_str = ", ".join([f"Digit {d}: {p:.1f}%" for d, p in res["top3"]])
        print(
            f"{res['filename']:<24} | "
            f"  {res['prediction']}   | "
            f"{res['confidence']:>8.2f}% | "
            f"{top3_str}"
        )
    print("-" * 85)
    print(f"  Successfully processed {len(results)} images.\n")

    # Generate Visualization Plot
    if save_plot_path:
        plot_inference_results(results, save_plot_path, arch_name)

    return results


def plot_inference_results(results: List[dict], save_path: str, arch_name: str) -> None:
    """
    Plots a multi-panel visual diagnostic figure showing raw images,
    preprocessed 28x28 tensors, and complete 10-class probability distributions.
    """
    n = len(results)
    if n == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(n, 3, figsize=(13, 3.4 * n), squeeze=False)
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    for i, res in enumerate(results):
        # 1. Raw Input Image
        axes[i, 0].imshow(res["raw_pil"])
        axes[i, 0].set_title(f"Raw Input: {res['filename']}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. Preprocessed 28x28 Tensor
        axes[i, 1].imshow(res["canvas"], cmap="gray", interpolation="nearest")
        axes[i, 1].set_title(
            f"Preprocessed (28x28)\nPred: {res['prediction']} ({res['confidence']:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color="#0b6623" if res["confidence"] > 70 else "#b22222",
        )
        axes[i, 1].axis("off")

        # 3. Class Probability Bar Chart
        probs = res["probabilities"] * 100.0
        classes = list(range(10))
        bars = axes[i, 2].bar(classes, probs, color="#4682b4", edgecolor="#1c3b5e", width=0.7)
        # Highlight top predicted class
        bars[res["prediction"]].set_color("#2e8b57")
        bars[res["prediction"]].set_edgecolor("#1b5233")

        axes[i, 2].set_xticks(classes)
        axes[i, 2].set_ylim(0, 105)
        axes[i, 2].set_xlabel("Digit Class", fontsize=10)
        axes[i, 2].set_ylabel("Probability (%)", fontsize=10)
        axes[i, 2].grid(axis="y", linestyle="--", alpha=0.5)

        # Label top bar percentage
        top_digit = res["prediction"]
        top_prob = probs[top_digit]
        axes[i, 2].text(
            top_digit,
            min(top_prob + 3.0, 102.0),
            f"{top_prob:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1b5233",
        )

    plt.suptitle(f"Real Image Inference Evaluation — {arch_name}", fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Visual prediction report saved to: {save_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MNIST CNN inference on real images in a dataset folder."
    )
    parser.add_argument(
        "--image-dir",
        "--dataset-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "datasets"),
        help="Path to folder containing real images (default: ./datasets)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "best_advanced_model.pth"),
        help="Path to trained model checkpoint .pth (default: ./checkpoints/best_advanced_model.pth)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "advanced", "baseline"],
        help="Model architecture: 'auto' (detects from weights), 'advanced', or 'baseline'",
    )
    parser.add_argument(
        "--invert",
        type=str,
        default="auto",
        choices=["auto", "invert", "no-invert"],
        help="Color inversion mode: 'auto' (detects white background), 'invert', or 'no-invert'",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=os.path.join(PROJECT_ROOT, "reports", "real_test_predictions.png"),
        help="Path to save diagnostic prediction plot (default: ./reports/real_test_predictions.png)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force inference on CPU even if CUDA/MPS is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.no_cuda else get_default_device()

    # Fallback checkpoint detection if default doesn't exist
    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        fallback = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
        if os.path.exists(fallback):
            print(f"[Warning] {checkpoint} not found. Falling back to: {fallback}")
            checkpoint = fallback
        else:
            print(f"[Error] Checkpoint not found at {checkpoint}")
            sys.exit(1)

    run_inference_on_folder(
        image_dir=args.image_dir,
        checkpoint_path=checkpoint,
        model_type=args.model_type,
        invert_mode=args.invert,
        save_plot_path=args.save_plot,
        device=device,
    )


if __name__ == "__main__":
    main()
