"""Embedding quality inspection tools.

Provides anisotropy measurement, nearest-neighbor audit, cluster analysis
(HDBSCAN or k-means), cosine similarity heatmap data, and a k-NN graph
for the Obsidian-style force-directed visualization.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any


# ── Anisotropy ───────────────────────────────────────────────────────────────

def anisotropy(vectors: np.ndarray, n_samples: int = 500) -> dict[str, Any]:
    """Measure embedding anisotropy via average pairwise cosine similarity.

    Returns avg_cosine, std, and a verdict. Values >0.7 indicate the space
    is collapsing into a narrow cone and needs whitening.
    """
    if vectors.shape[0] < 2:
        return {"avg_cosine": 0.0, "std": 0.0, "n_pairs": 0, "verdict": "too_few_points"}

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms

    n = normed.shape[0]
    if n <= n_samples:
        sim_matrix = normed @ normed.T
        # Upper triangle (exclude diagonal)
        iu = np.triu_indices(n, k=1)
        cosines = sim_matrix[iu]
    else:
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, n, size=n_samples)
        idx_b = rng.integers(0, n, size=n_samples)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        cosines = np.sum(normed[idx_a] * normed[idx_b], axis=1)

    avg = float(np.mean(cosines))
    std = float(np.std(cosines))
    if avg > 0.7:
        verdict = "anisotropic — consider whitening / PCA centering"
    elif avg > 0.4:
        verdict = "moderate — acceptable for most tasks"
    else:
        verdict = "isotropic — embedding space is well-spread"

    return {"avg_cosine": round(avg, 4), "std": round(std, 4),
            "n_pairs": len(cosines), "verdict": verdict}


# ── Nearest-neighbor audit ───────────────────────────────────────────────────

def nn_audit(
    vectors: np.ndarray,
    ids: list[str],
    headings: list[str],
    k: int = 5,
) -> list[dict[str, Any]]:
    """For each point, return its k nearest neighbors with cosine distances.

    Useful for spotting mis-embeddings where unrelated content is too close.
    """
    if vectors.shape[0] < 2:
        return []

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms
    sim = normed @ normed.T

    results = []
    for i in range(sim.shape[0]):
        row = sim[i].copy()
        row[i] = -2.0  # exclude self
        top_idx = np.argsort(row)[::-1][:k]
        neighbors = [
            {"id": ids[j], "heading": headings[j], "cosine": round(float(row[j]), 4)}
            for j in top_idx
        ]
        results.append({
            "id": ids[i],
            "heading": headings[i],
            "neighbors": neighbors,
        })
    return results


# ── Cluster analysis ─────────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    labels: np.ndarray        # cluster id per point (-1 = noise for HDBSCAN)
    n_clusters: int
    method: str
    silhouette: float | None  # None if <2 clusters

    def summary(self, ids: list[str], headings: list[str]) -> dict[str, Any]:
        clusters: dict[int, list[dict[str, str]]] = {}
        for i, label in enumerate(self.labels):
            c = int(label)
            clusters.setdefault(c, []).append({"id": ids[i], "heading": headings[i]})
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "silhouette": round(self.silhouette, 4) if self.silhouette is not None else None,
            "clusters": clusters,
        }


def cluster(
    vectors: np.ndarray,
    method: str = "hdbscan",
    n_clusters: int = 8,
    min_cluster_size: int = 3,
) -> ClusterResult:
    """Cluster embeddings via HDBSCAN or k-means."""
    from sklearn.metrics import silhouette_score

    if vectors.shape[0] < 3:
        return ClusterResult(labels=np.zeros(vectors.shape[0], dtype=int),
                             n_clusters=1, method=method, silhouette=None)

    if method == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN as _HDBSCAN
            model = _HDBSCAN(min_cluster_size=min_cluster_size)
            labels = model.fit_predict(vectors)
        except ImportError:
            # Fall back to k-means if HDBSCAN unavailable
            return cluster(vectors, method="kmeans", n_clusters=n_clusters)
    else:
        from sklearn.cluster import KMeans
        k = min(n_clusters, vectors.shape[0])
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(vectors)

    unique = set(labels)
    unique.discard(-1)
    nc = len(unique)

    sil = None
    if nc >= 2:
        mask = labels != -1
        if mask.sum() >= 2:
            sil = float(silhouette_score(vectors[mask], labels[mask]))

    return ClusterResult(labels=labels, n_clusters=nc, method=method, silhouette=sil)


# ── Cosine similarity heatmap ────────────────────────────────────────────────

def similarity_matrix(
    vectors: np.ndarray,
    ids: list[str],
    max_points: int = 200,
) -> dict[str, Any]:
    """Compute full cosine similarity matrix (capped for frontend performance)."""
    n = vectors.shape[0]
    if n == 0:
        return {"ids": [], "matrix": []}

    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        vectors = vectors[idx]
        ids = [ids[i] for i in idx]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms
    sim = (normed @ normed.T).tolist()

    return {"ids": ids, "matrix": sim}


# ── k-NN graph (for Obsidian-like force view) ───────────────────────────────

def knn_graph(
    vectors: np.ndarray,
    ids: list[str],
    headings: list[str],
    paths: list[str],
    cluster_labels: np.ndarray | None = None,
    k: int = 5,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Build a k-NN graph: nodes + edges where cosine similarity > threshold.

    Returns {nodes: [...], edges: [...]} ready for d3-force rendering.
    """
    n = vectors.shape[0]
    if n == 0:
        return {"nodes": [], "edges": []}

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms
    sim = normed @ normed.T

    nodes = []
    for i in range(n):
        nodes.append({
            "id": ids[i],
            "heading": headings[i],
            "path": paths[i],
            "cluster": int(cluster_labels[i]) if cluster_labels is not None else 0,
        })

    edges = []
    seen = set()
    for i in range(n):
        row = sim[i].copy()
        row[i] = -2.0
        top_idx = np.argsort(row)[::-1][:k]
        for j in top_idx:
            score = float(row[j])
            if score < threshold:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({
                "source": ids[i],
                "target": ids[int(j)],
                "weight": round(score, 4),
            })

    return {"nodes": nodes, "edges": edges}
