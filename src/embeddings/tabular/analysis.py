"""Analysis helpers for tabular embedding demos — cluster separation & outliers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def top_separating_variables(
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    top_k: int = 5,
    skip_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank metadata fields by how well they separate cluster labels."""
    skip = skip_fields or set()
    if len(metadata) == 0:
        return []

    unique = set(int(x) for x in labels if int(x) >= 0)
    if len(unique) < 2:
        return []

    df = pd.DataFrame(metadata)
    scores: list[dict[str, Any]] = []

    for col in df.columns:
        if col in skip:
            continue
        try:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                x = series.fillna(series.median()).astype(float).values.reshape(-1, 1)
                if float(np.std(x)) < 1e-12:
                    continue
                f_stat, _ = f_classif(x, labels)
                scores.append({"field": col, "score": round(float(f_stat[0]), 4), "kind": "numeric"})
            else:
                le = LabelEncoder()
                x = le.fit_transform(series.fillna("").astype(str)).reshape(-1, 1)
                if len(set(x.ravel())) < 2:
                    continue
                mi = mutual_info_classif(x, labels, random_state=42)[0]
                scores.append({"field": col, "score": round(float(mi), 4), "kind": "categorical"})
        except Exception:
            continue

    scores.sort(key=lambda s: -s["score"])
    return scores[:top_k]


def outlier_scores(vectors: np.ndarray, k: int = 10) -> np.ndarray:
    """Higher score = more anomalous (farther from local neighborhood in cosine space)."""
    n = vectors.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms
    sim = normed @ normed.T

    k = min(k, n - 1)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = sim[i].copy()
        row[i] = -2.0
        top = np.sort(row)[::-1][:k]
        scores[i] = 1.0 - float(np.mean(top))
    return scores
