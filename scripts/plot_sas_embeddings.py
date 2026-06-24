#!/usr/bin/env python3
"""Load SAS-generated embeddings and produce PCA/t-SNE scatter plots.

Reads the output CSV from tabular_embeddings.sas (or simulate_sas_embeddings.py),
detects embedding columns (EMB_*) and metadata columns automatically, and
generates scatter plots colored by any metadata field.

Usage:
    python scripts/plot_sas_embeddings.py \
        --csv data/embeddings/tabular/sas_embeddings.csv \
        --color-field TERMINACION
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_sas_embeddings(csv_path: Path) -> tuple[np.ndarray, list[dict], list[str]]:
    import csv
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise SystemExit("No data found in SAS output CSV")

    cols = list(rows[0].keys())
    embed_cols = sorted(
        [k for k in cols if k.startswith("EMB_")],
        key=lambda x: int(x.split("_")[1]),
    )
    meta_cols = [c for c in cols
                 if not c.startswith("EMB_") and c != "ROW_ID"]
    embed_dim = len(embed_cols)
    n = len(rows)

    vectors = np.zeros((n, embed_dim), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, col in enumerate(embed_cols):
            vectors[i, j] = float(r.get(col, 0) or 0)

    print(f"Loaded {n} SAS embeddings ({embed_dim} dim)")
    print(f"Metadata columns: {meta_cols}")
    return vectors, [{c: r.get(c, "") for c in meta_cols} for r in rows], meta_cols


def _get_colors(labels: list[str]) -> dict[str, str]:
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
    ax.scatter(coords[:, 0], coords[:, 1],
               c=colors, alpha=0.6, s=12, edgecolors="none")
    ax.set_title(title or f"SAS TabBERT Embeddings colored by {color_field}",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor=color_map[lbl], label=f"{lbl} ({cnt})")
        for lbl, cnt in Counter(labels).most_common()
    ]
    ncol = 2 if len(legend_elements) > 8 else 1
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8, ncol=ncol, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plotted = coords.shape[0]
    print(f"Saved plot ({plotted} points{' — sampled' if plotted < n else ''}) → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SAS-generated tabular embeddings (generic)"
    )
    parser.add_argument("--csv", type=Path,
                        default=PROJECT_ROOT / "data" / "embeddings" / "tabular" / "sas_embeddings.csv")
    parser.add_argument("--method", type=str, default="pca",
                        choices=["pca", "tsne", "umap"])
    parser.add_argument("--color-field", type=str, default=None,
                        help="Metadata field to color by (default: first non-id field)")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "plots")
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"SAS embeddings not found at {args.csv}")
        print("Run sas scripts/tabular_embeddings.sas first")
        sys.exit(1)

    vectors, metadata, meta_cols = _load_sas_embeddings(args.csv)

    from src.embeddings import project
    coords = project(vectors, method=args.method, n_components=2)

    # Determine which metadata field to color by
    color_fields = [args.color_field] if args.color_field else (
        [c for c in meta_cols if "ID" not in c.upper()][:3]
        or meta_cols[:3]
    )

    for field in color_fields:
        labels = [m.get(field, "") for m in metadata]
        make_plot(
            coords, labels,
            color_field=field,
            output_path=args.output_dir / f"sas_embeddings_by_{field}_{args.method}.png",
            title=f"SAS TabBERT Embeddings by {field} ({args.method.upper()})",
            sample=args.sample,
        )

    print("Done.")


if __name__ == "__main__":
    main()
