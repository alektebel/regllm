"""Dimensionality reduction (PCA / t-SNE / UMAP) for the embedding visualizer."""

from __future__ import annotations

import numpy as np

try:
    import umap  # type: ignore

    _HAS_UMAP = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_UMAP = False

PROJECTION_METHODS = ["pca", "tsne"] + (["umap"] if _HAS_UMAP else [])


def project(
    vectors: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce ``vectors`` (n_samples, n_features) to ``n_components`` dims."""
    n_samples = vectors.shape[0]
    if n_samples == 0:
        return np.zeros((0, n_components))
    if n_samples <= n_components:
        # Not enough points to project meaningfully; pad with zeros.
        out = np.zeros((n_samples, n_components))
        out[:, : vectors.shape[1]] = vectors[:, :n_components]
        return out

    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=random_state).fit_transform(vectors)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perp = max(2.0, min(perplexity, (n_samples - 1) / 3))
        return TSNE(
            n_components=n_components,
            perplexity=perp,
            random_state=random_state,
            init="pca",
        ).fit_transform(vectors)

    if method == "umap":
        if not _HAS_UMAP:
            raise ValueError(
                "umap-learn is not available. "
                "Install it with: pip install 'umap-learn>=0.5,<0.6' 'pynndescent>=0.5,<0.6'"
            )
        neighbors = max(2, min(n_neighbors, n_samples - 1))
        try:
            return umap.UMAP(
                n_components=n_components,
                n_neighbors=neighbors,
                min_dist=min_dist,
                random_state=random_state,
            ).fit_transform(vectors)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"UMAP failed due to a missing dependency ({e}). "
                "Pin pynndescent<0.6 to avoid the kuzu requirement: "
                "pip install 'pynndescent>=0.5,<0.6' 'umap-learn>=0.5,<0.6'"
            ) from e

    raise ValueError(f"Unknown projection method {method!r}; choose from {PROJECTION_METHODS}")
