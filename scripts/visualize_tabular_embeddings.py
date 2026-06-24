#!/usr/bin/env python3
"""Visualize TabBERT-style embeddings colored by TERMINACION and CALIBRATION_SEGMENT.

Produces two PCA scatter plots:
  1. Points colored by TERMINACION (termination type)
  2. Points colored by CALIBRATION_SEGMENT (IRB calibration segment)

Usage:
    python scripts/visualize_tabular_embeddings.py [--method pca|tsne|umap] [--output-dir data/plots]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def _get_colors(labels: list[str]) -> dict[str, str]:
    """Assign distinct colors to unique labels using a standard palette."""
    unique = sorted(set(labels))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    return {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique)}


def make_plot(
    coords: np.ndarray,
    labels: list[str],
    color_field: str,
    output_path: Path,
    title: str | None = None,
    sample: int = 5000,
) -> None:
    """Create and save a scatter plot colored by a categorical field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = coords.shape[0]
    if n > sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample, replace=False)
        coords = coords[idx]
        labels = [labels[i] for i in idx]

    color_map = _get_colors(labels)
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, alpha=0.6, s=12, edgecolors="none",
    )
    ax.set_title(title or f"Tabular Embeddings colored by {color_field}",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        Patch(facecolor=color_map[lbl], label=f"{lbl} ({cnt})")
        for lbl, cnt in Counter(labels).most_common()
    ]
    # Split legend into two columns if many categories
    ncol = 2 if len(legend_elements) > 8 else 1
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=8, ncol=ncol, framealpha=0.9,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plotted = coords.shape[0]
    print(f"Saved plot ({plotted} points{' — sampled from ' + str(n) if plotted < n else ''}) → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize tabular embeddings colored by segment/termination"
    )
    parser.add_argument(
        "--method", type=str, default="pca",
        choices=["pca", "tsne", "umap"],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "plots",
    )
    parser.add_argument("--sample", type=int, default=5000,
                        help="Max points to plot (random subsample)")
    args = parser.parse_args()

    # Load pre-built embeddings
    from src.embeddings.tabular.embedder import TabularEmbeddingSet

    emb_path = (
        PROJECT_ROOT / "data" / "embeddings" / "tabular" / "embeddings.json"
    )
    emb = TabularEmbeddingSet.load(emb_path)
    if emb is None:
        print("No embeddings found. Run train_tabular_embeddings.py first.")
        sys.exit(1)

    print(f"Loaded {len(emb)} embeddings ({emb.vectors.shape[1]} dim)")

    # Project
    from src.embeddings import project

    coords = project(emb.vectors, method=args.method, n_components=2)

    # Extract labels
    terminaciones = [
        str(m.get("TERMINACION", "NONE")) for m in emb.metadata
    ]
    calib_segments = [
        str(m.get("CALIBRATION_SEGMENT", "NONE")) for m in emb.metadata
    ]

    # Plot 1: TERMINACION
    make_plot(
        coords, terminaciones,
        color_field="TERMINACION",
        output_path=args.output_dir / f"embeddings_by_terminacion_{args.method}.png",
        title=f"TabBERT-style Embeddings colored by TERMINACION ({args.method.upper()})",
        sample=args.sample,
    )

    # Plot 2: CALIBRATION_SEGMENT
    make_plot(
        coords, calib_segments,
        color_field="CALIBRATION_SEGMENT",
        output_path=args.output_dir / f"embeddings_by_calibration_segment_{args.method}.png",
        title=f"TabBERT-style Embeddings colored by CALIBRATION_SEGMENT ({args.method.upper()})",
        sample=args.sample,
    )

    print("Done.")


if __name__ == "__main__":
    main()
