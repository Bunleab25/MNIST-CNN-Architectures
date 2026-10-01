"""
Architecture Diagram & Computational Graph Generator
=====================================================
Generates publication-quality architectural flowcharts and comparison diagrams
for MNISTCNN (Baseline) and AdvancedMNISTCNN (Dual-Conv + GAP).

Usage:
    python plot_architecture_graph.py
"""

import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_standard_architecture_diagram(save_path: str = "./reports/architecture_diagram.png") -> None:
    """
    Renders a standard, publication-grade block flow diagram for AdvancedMNISTCNN.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    c_input = "#E0E0E0"
    c_conv = "#4A90E2"
    c_pool = "#F5A623"
    c_gap = "#9013FE"
    c_fc = "#7ED321"
    c_text = "#FFFFFF"

    # Blocks definition: (name, subtext, shape, color, x_pos)
    blocks = [
        ("Input", "Grayscale", "(1, 28, 28)", c_input, "#333333", 0.6),
        ("Stage 1\nDual-Conv", "2x [Conv3x3 + BN + ReLU]\n32 Filters", "(32, 28, 28)", c_conv, c_text, 2.7),
        ("MaxPool 1", "Kernel 2x2, Stride 2", "(32, 14, 14)", c_pool, c_text, 5.0),
        ("Stage 2\nDual-Conv", "2x [Conv3x3 + BN + ReLU]\n64 Filters", "(64, 14, 14)", c_conv, c_text, 7.1),
        ("MaxPool 2", "Kernel 2x2, Stride 2", "(64, 7, 7)", c_pool, c_text, 9.4),
        ("Stage 3\nDual-Conv", "2x [Conv3x3 + BN + ReLU]\n128 Filters", "(128, 7, 7)", c_conv, c_text, 11.5),
        ("GAP Head", "AdaptiveAvgPool2d(1,1)\nDropout(p=0.3)", "(128, 1, 1)", c_gap, c_text, 13.6),
        ("Output", "Linear (128 -> 10)\nRaw Logits (0-9)", "(10)", c_fc, c_text, 15.3),
    ]

    width = 1.6
    y_center = 4.0
    height = 2.8

    # Title
    ax.text(
        8.0, 7.3,
        "AdvancedMNISTCNN: Dual-Convolution + Global Average Pooling Architecture",
        fontsize=15, fontweight="bold", ha="center", va="center", color="#1A1A1A"
    )
    ax.text(
        8.0, 6.8,
        "End-to-End Tensor Flow, Receptive Field Expansion & Spatial Downsampling",
        fontsize=10.5, fontstyle="italic", ha="center", va="center", color="#555555"
    )

    for i, (name, subtext, shape, col, text_col, x) in enumerate(blocks):
        # Draw block box
        h = height if "Stage" in name or "Input" in name or "Head" in name else height * 0.75
        y = y_center - h / 2
        w = width if i != 0 and i != len(blocks) - 1 else 1.1

        box = patches.FancyBboxPatch(
            (x - w / 2, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            facecolor=col, edgecolor="#333333", linewidth=1.5,
            mutation_scale=1.0, zorder=3
        )
        ax.add_patch(box)

        # Block text
        ax.text(x, y_center + 0.35, name, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_col, zorder=4)
        ax.text(x, y_center - 0.25, subtext, ha="center", va="center", fontsize=7.5, color=text_col, zorder=4)

        # Shape annotation below
        ax.text(
            x, y_center - (h / 2) - 0.55, f"Shape:\n{shape}",
            ha="center", va="top", fontsize=8.5, fontweight="bold", color="#111111"
        )

        # Draw connecting arrow
        if i < len(blocks) - 1:
            next_x = blocks[i + 1][5]
            next_w = width if i + 1 != len(blocks) - 1 else 1.1
            arrow_start = x + (w / 2) + 0.05
            arrow_end = next_x - (next_w / 2) - 0.05

            ax.annotate(
                "",
                xy=(arrow_end, y_center),
                xytext=(arrow_start, y_center),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=2, color="#333333"),
                zorder=2
            )

    # Legend / Key Info at bottom
    info_box = patches.FancyBboxPatch(
        (0.8, 0.4), 14.4, 0.9,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        facecolor="#F8F9FA", edgecolor="#CCCCCC", linewidth=1.0, zorder=1
    )
    ax.add_patch(info_box)

    ax.text(
        8.0, 0.85,
        "Key Upgrade: Global Average Pooling (GAP) replaces 401k Flatten weights with 1.2k weights, completely removing dense memorization.",
        fontsize=9.5, fontweight="bold", ha="center", va="center", color="#2C3E50"
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Architecture Graph] Publication diagram saved to: {save_path}")


if __name__ == "__main__":
    plot_standard_architecture_diagram()
