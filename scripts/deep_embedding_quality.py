#!/usr/bin/env python3
"""Deep quality analysis of TabBERT embeddings.

Runs 5 independent analyses and writes results to
  data/embeddings/tabular/analysis/

Usage:
    python scripts/deep_embedding_quality.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHT_DIR = PROJECT_ROOT / "data" / "embeddings" / "tabular"
DATA_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_DIR = WEIGHT_DIR / "analysis"
INPUT_CSV = DATA_DIR / "recuperatory_cycles.csv"
EMB_CSV = WEIGHT_DIR / "sas_embeddings.csv"
OUTLIER_PCT = 0.01

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_projection(path: Path) -> np.ndarray:
    import csv
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        embed_dim = len(header) - 1
        rows_data = []
        for row in reader:
            rows_data.append([float(x) for x in row[1:]])
    return np.array(rows_data, dtype=np.float64)


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _embedding_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("EMB_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _non_embedding_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("EMB_") and c != "ROW_ID"]


# ---------------------------------------------------------------------------
# 1. Missing-value impact
# ---------------------------------------------------------------------------

def analyze_missing_value_impact(
    emb_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    embed_cols: list[str],
    outlier_pct: float,
) -> pd.DataFrame:
    """Test if missing/empty values correlate with outlier status."""
    # Align lengths (embeddings may be subset of original data)
    if len(emb_df) < len(orig_df):
        # Use ROW_ID to align
        if "ROW_ID" in emb_df.columns and "ROW_ID" in orig_df.columns:
            orig_df = orig_df.iloc[:len(emb_df)].copy()  # same ordering assumption
        else:
            orig_df = orig_df.iloc[:len(emb_df)].copy()

    # Identify outliers (farthest from centroid)
    emb_values = emb_df[embed_cols].values
    centroid = emb_values.mean(axis=0)
    dists = np.linalg.norm(emb_values - centroid, axis=1)
    threshold = np.quantile(dists, 1 - outlier_pct)
    is_outlier = dists >= threshold

    results: list[dict] = []
    for col in orig_df.columns:
        is_missing = orig_df[col].isna() | orig_df[col].astype(str).str.strip().eq("")
        if is_missing.sum() < 5:
            continue  # too few missing to test

        # Contingency table
        tbl = pd.crosstab(is_missing, is_outlier)
        if tbl.shape != (2, 2):
            continue
        _, p_val, _, expected = chi2_contingency(tbl, correction=False)

        # Odds ratio: odds of outlier given missing vs odds of outlier given not missing
        a = tbl.iloc[1, 1]  # missing & outlier
        b = tbl.iloc[1, 0]  # missing & not outlier
        c = tbl.iloc[0, 1]  # not missing & outlier
        d = tbl.iloc[0, 0]  # not missing & not outlier
        odds_ratio = (a / b) / (c / d) if b > 0 and c > 0 and d > 0 else float("inf")

        # Fraction of missing rows that are outliers
        missing_n = is_missing.sum()
        missing_outlier_n = is_missing[is_outlier].sum()
        frac_outlier = missing_outlier_n / missing_n if missing_n > 0 else 0.0

        results.append({
            "column": col,
            "n_missing": int(missing_n),
            "n_outlier_missing": int(missing_outlier_n),
            "frac_outlier_given_missing": round(frac_outlier, 4),
            "baseline_outlier_rate": round(outlier_pct, 4),
            "odds_ratio": round(odds_ratio, 2),
            "p_value": f"{p_val:.4e}",
            "significant_005": p_val < 0.05,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("odds_ratio", ascending=False)
    result_df.to_csv(OUTPUT_DIR / "missing_value_impact.csv", index=False)
    return result_df


# ---------------------------------------------------------------------------
# 2. Time-series consistency
# ---------------------------------------------------------------------------

def analyze_temporal_consistency(
    emb_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    embed_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-contract embedding smoothness.

    Contracts with multiple temporal snapshots should have smoothly varying
    embeddings.  High consecutive-step variance = possible data quality issue.
    """
    # Find contract-level ID column (one that groups multiple snapshots)
    n = len(orig_df)
    id_col = None
    for col in orig_df.columns:
        if "ID" in col.upper():
            nu = orig_df[col].nunique()
            if 10 < nu < n * 0.5:  # groups 3+ rows, but not unique-per-row
                id_col = col
                break
    if id_col is None:
        print("  [TS] No contract-grouping ID column found — skipping")
        return pd.DataFrame(), pd.DataFrame()

    # Find a time-ordering column
    time_candidates = [c for c in orig_df.columns if "MES_CICLO" in c.upper()
                       or "FECHA" in c.upper() or "DATE" in c.upper()]
    time_col = time_candidates[0] if time_candidates else None

    # Align rows
    n_min = min(len(emb_df), len(orig_df))
    emb_values = emb_df[embed_cols].values[:n_min]
    contract_ids = orig_df[id_col].values[:n_min]

    all_diffs: list[float] = []
    contract_stats: list[dict] = []
    for cid in np.unique(contract_ids):
        mask = contract_ids == cid
        idxs = np.where(mask)[0]

        # Order by time if available
        if time_col and time_col in orig_df.columns:
            times = pd.to_numeric(orig_df[time_col].iloc[idxs], errors="coerce").values
            order = np.argsort(times)
            idxs = idxs[order]

        if len(idxs) < 2:
            continue

        # Consecutive embedding differences
        contract_emb = emb_values[idxs]
        diffs = np.linalg.norm(np.diff(contract_emb, axis=0), axis=1)
        all_diffs.extend(diffs.tolist())

        contract_stats.append({
            "contract_id": cid,
            "n_snapshots": int(len(idxs)),
            "mean_consecutive_diff": float(diffs.mean()),
            "max_consecutive_diff": float(diffs.max()),
            "std_consecutive_diff": float(diffs.std()) if len(diffs) > 1 else 0.0,
        })

    if not contract_stats:
        return pd.DataFrame(), pd.DataFrame()

    stats_df = pd.DataFrame(contract_stats)

    # Flag erratic contracts (top 5% mean consecutive diff)
    threshold = stats_df["mean_consecutive_diff"].quantile(0.95)
    erratic = stats_df[stats_df["mean_consecutive_diff"] >= threshold].copy()
    erratic["flagged_reason"] = (
        f"mean_consecutive_diff >= P95 ({threshold:.4f})"
    )
    erratic.to_csv(OUTPUT_DIR / "erratic_contracts.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "per_contract_variance.csv", index=False)

    print(f"  [TS] {len(stats_df)} contracts analyzed, {len(erratic)} erratic flagged")
    return stats_df, erratic


# ---------------------------------------------------------------------------
# 3. Column contribution to outliers
# ---------------------------------------------------------------------------

def analyze_outlier_contributions(
    emb_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    config: dict,
    proj: np.ndarray,
    outlier_pct: float,
) -> pd.DataFrame:
    """For each outlier, decompose centroid distance into per-feature contributions.

    Maps from 69-dim feature space back to human-readable feature names
    (e.g., "SEGMENTO=SME", "PD_ESTIMADA").
    """
    # Build feature names
    cat_values = config["cat_values"]
    num_cols = config["numeric_cols"]
    feat_names: list[str] = []
    for col, vals in cat_values.items():
        for val in vals:
            feat_names.append(f"{col}={val}")
    for col in num_cols:
        feat_names.append(col)

    # Build feature vectors for all rows
    from scripts.simulate_sas_embeddings import build_features
    _, _, all_rows, _ = build_features(
        [dict(zip(orig_df.columns, row)) for row in orig_df.itertuples(index=False)],
        config,
        batch_size=5000,
    )

    # Actually, build_features returns the rows as-is. Let me re-build features properly.
    # The function signature has changed. Let me use the original build_features_from_config.
    def _build_features(rows, config):
        cat_cols = config["categorical_cols"]
        num_cols_cfg = config["numeric_cols"]
        cat_vals = config["cat_values"]
        num_params = config["num_params"]

        n = len(rows)
        n_feat = sum(len(v) for v in cat_vals.values()) + len(num_cols_cfg)
        X = np.zeros((n, n_feat), dtype=np.float64)
        for i, row in enumerate(rows):
            idx = 0
            for col in cat_cols:
                vals = cat_vals[col]
                val = row.get(col, "")
                if val in vals:
                    X[i, idx + vals.index(val)] = 1.0
                idx += len(vals)
            for col in num_cols_cfg:
                v = float(row.get(col, 0) or 0)
                info = num_params[col]
                rng = info["range"]
                X[i, idx] = (v - info["min"]) / rng if rng > 0 else 0.0
                idx += 1
        return X

    orig_rows = orig_df.iloc[:len(emb_df)].to_dict("records")
    X = _build_features(orig_rows, config)

    # Compute centroid in feature space and per-feature contribution
    emb_values = emb_df[[c for c in emb_df.columns if c.startswith("EMB_")]].values
    centroid_emb = emb_values.mean(axis=0)

    outlier_dists = np.linalg.norm(emb_values - centroid_emb, axis=1)
    threshold = np.quantile(outlier_dists, 1 - outlier_pct)
    outlier_mask = outlier_dists >= threshold
    outlier_indices = np.where(outlier_mask)[0]

    results: list[dict] = []
    for i in outlier_indices[:200]:  # limit to 200 for readability
        # Contribution of each feature: feats[j] * proj[j,:] projects to embedding space
        # Weight = abs(feat_val) * ||proj_row||  (heuristic: feature influence on centroid distance)
        feat_contrib = np.abs(X[i]) * np.linalg.norm(proj, axis=1)
        top_k = min(5, len(feat_names))
        top_idx = np.argsort(feat_contrib)[::-1][:top_k]

        results.append({
            "ROW_ID": int(emb_df.iloc[i].get("ROW_ID", i + 1)),
            "centroid_distance": round(float(outlier_dists[i]), 4),
            f"top1_feature": feat_names[top_idx[0]],
            f"top1_contribution": round(float(feat_contrib[top_idx[0]]), 4),
            f"top2_feature": feat_names[top_idx[1]] if top_k > 1 else "",
            f"top3_feature": feat_names[top_idx[2]] if top_k > 2 else "",
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "outlier_feature_contributions.csv", index=False)
    print(f"  [Contrib] Analyzed {len(results)} outliers")
    return result_df


# ---------------------------------------------------------------------------
# 4. Silhouette width against known labels
# ---------------------------------------------------------------------------

def analyze_silhouette(
    emb_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    embed_cols: list[str],
    config: dict,
    min_groups: int = 2,
    max_groups: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-point silhouette width for each suitable metadata field.

    Negative silhouette = point closer to a DIFFERENT group than its own →
    possible mislabel or genuinely ambiguous register.
    """
    # Build combined metadata
    meta_df = emb_df[_non_embedding_cols(emb_df)].copy()
    emb_values = emb_df[embed_cols].values
    n = len(emb_values)

    # Augment with original data if available
    if orig_df is not None:
        n_min = min(len(meta_df), len(orig_df))
        for col in orig_df.columns:
            if col not in meta_df.columns and col != "ROW_ID":
                meta_df[col] = orig_df[col].values[:n_min]

    # Use only genuinely categorical fields (from config or reasonable cardinality)
    cat_from_config = set(config.get("categorical_cols", []))
    label_candidates = [
        c for c in meta_df.columns
        if c in cat_from_config
        or (
            not c.startswith("EMB_") and c != "ROW_ID"
            and "ID" not in c.upper()
            and "FECHA" not in c.upper() and "DATE" not in c.upper()
            and meta_df[c].nunique() < 20
            and meta_df[c].nunique() >= 2
        )
    ]

    all_results: list[dict] = []
    neg_sil_rows: list[dict] = []

    for field in label_candidates:
        labels = meta_df[field].astype(str).str.strip()
        label_counts = labels.value_counts()
        n_groups = len(label_counts)

        if n_groups < min_groups or n_groups > max_groups:
            continue

        # Use all labels
        unique_labels = list(label_counts.index)
        valid_labels = set(unique_labels)
        mask = labels.isin(valid_labels)
        sub_emb = emb_values[mask]
        sub_labels = labels[mask].values
        label_to_code = {lbl: i for i, lbl in enumerate(unique_labels)}
        codes = np.array([label_to_code[l] for l in sub_labels])
        m = len(sub_emb)

        if m < 100:
            continue

        # Sample if too large (silhouette_samples scales O(n²))
        if m > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(m, size=5000, replace=False)
            sub_emb = sub_emb[idx]
            codes = codes[idx]
            m = 5000

        # Per-point silhouette (use sklearn's optimized C implementation)
        from sklearn.metrics import silhouette_samples
        sil_scores = silhouette_samples(sub_emb, codes, metric="euclidean")

        avg_sil = float(np.mean(sil_scores))
        neg_mask = sil_scores < -0.05
        n_neg = int(neg_mask.sum())
        pct_neg = n_neg / m * 100

        all_results.append({
            "field": field,
            "n_groups": n_groups,
            "n_total": int(m),
            "avg_silhouette": round(avg_sil, 4),
            "n_negative": n_neg,
            "pct_negative": round(pct_neg, 2),
        })

        # Negative silhouette details
        if n_neg > 0:
            neg_idxs = np.where(neg_mask)[0]
            for idx in neg_idxs[:50]:
                neg_sil_rows.append({
                    "field": field,
                    "label": sub_labels[idx],
                    "silhouette": round(float(sil_scores[idx]), 4),
                })

    if all_results:
        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / "silhouette_scores.csv", index=False
        )
    if neg_sil_rows:
        pd.DataFrame(neg_sil_rows).to_csv(
            OUTPUT_DIR / "negative_silhouette.csv", index=False
        )
    return pd.DataFrame(all_results), pd.DataFrame(neg_sil_rows)


# ---------------------------------------------------------------------------
# 5. PCA reconstruction error
# ---------------------------------------------------------------------------

def analyze_pca_reconstruction(
    emb_df: pd.DataFrame,
    embed_cols: list[str],
    variance_target: float = 0.90,
) -> pd.DataFrame:
    """PCA reconstruction error per point.

    Project to top-K components (K s.t. cumul. variance > variance_target),
    back-project, and measure L2 error.  High error = not explained by the
    main data structure.
    """
    from sklearn.decomposition import PCA

    emb_values = emb_df[embed_cols].values
    n_components = min(emb_values.shape[1], 50)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(emb_values)

    # Determine K for target variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, variance_target) + 1)

    # Reconstruct
    truncated = transformed[:, :k]
    reconstructed = pca.inverse_transform(
        np.column_stack([truncated, np.zeros((len(truncated), n_components - k))])
    )
    errors = np.linalg.norm(emb_values - reconstructed, axis=1)

    # Flag top 5% highest error
    error_threshold = np.quantile(errors, 0.95)
    is_high_error = errors >= error_threshold

    result_df = pd.DataFrame({
        "ROW_ID": emb_df["ROW_ID"].values if "ROW_ID" in emb_df.columns
                  else np.arange(1, len(errors) + 1),
        "reconstruction_error": errors.round(6),
        "is_high_error": is_high_error,
    })

    mean_norm = float(np.mean(np.linalg.norm(emb_values, axis=1)))
    result_df["relative_error_pct"] = (
        result_df["reconstruction_error"] / mean_norm * 100
    ).round(2)

    result_df.to_csv(OUTPUT_DIR / "pca_reconstruction_error.csv", index=False)

    # PCA summary
    pca_summary = pd.DataFrame({
        "component": np.arange(1, n_components + 1),
        "eigenvalue": pca.explained_variance_.round(6),
        "variance_pct": (pca.explained_variance_ratio_ * 100).round(2),
        "cumulative_pct": (cum_var * 100).round(2),
    })
    pca_summary.to_csv(OUTPUT_DIR / "pca_detailed.csv", index=False)

    print(f"  [PCA] K={k} for {variance_target*100:.0f}% variance, "
          f"{int(is_high_error.sum())} high-error points flagged")
    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Deep Embedding Quality Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading embeddings...")
    emb_df = pd.read_csv(EMB_CSV)
    embed_cols = _embedding_cols(emb_df)
    print(f"  {len(emb_df)} rows × {len(embed_cols)} embedding dims")

    print("Loading original data...")
    orig_df = pd.read_csv(INPUT_CSV)
    print(f"  {len(orig_df)} rows × {len(orig_df.columns)} cols")

    config = _load_config(WEIGHT_DIR / "sas_config.json")
    proj = _load_projection(WEIGHT_DIR / "sas_projection.csv")

    # 1. Missing-value impact
    print("\n── 1. Missing-value impact ──")
    mvi = analyze_missing_value_impact(
        emb_df, orig_df, embed_cols, OUTLIER_PCT,
    )
    if not mvi.empty:
        sig = mvi[mvi["significant_005"]]
        if not sig.empty:
            print(f"  {len(sig)} columns with significant association to outlier status")
            for _, row in sig.head(5).iterrows():
                print(f"    {row['column']}: OR={row['odds_ratio']}, "
                      f"p={row['p_value']}")
        else:
            print("  No significant associations found")

    # 2. Time-series consistency
    print("\n── 2. Temporal consistency ──")
    ts_stats, ts_erratic = analyze_temporal_consistency(
        emb_df, orig_df, embed_cols,
    )
    if not ts_erratic.empty:
        print(f"  Top erratic contracts:")
        for _, row in ts_erratic.head(5).iterrows():
            print(f"    {row['contract_id']}: mean diff={row['mean_consecutive_diff']:.4f} "
                  f"({row['n_snapshots']} snapshots)")

    # 3. Column contributions
    print("\n── 3. Feature contribution to outliers ──")
    contrib = analyze_outlier_contributions(
        emb_df, orig_df, config, proj, OUTLIER_PCT,
    )
    if not contrib.empty:
        top_feats = Counter(contrib["top1_feature"]).most_common(10)
        print(f"  Top features driving outliers:")
        for feat, cnt in top_feats:
            print(f"    {feat}: {cnt}x top contributor")

    # 4. Silhouette
    print("\n── 4. Silhouette width ──")
    sil_all, sil_neg = analyze_silhouette(
        emb_df, orig_df, embed_cols, config,
    )
    if not sil_all.empty:
        for _, row in sil_all.sort_values("avg_silhouette", ascending=False).iterrows():
            print(f"    {row['field']}: avg={row['avg_silhouette']:.3f}, "
                  f"{row['pct_negative']:.1f}% negative (n={row['n_negative']})")
    if not sil_neg.empty:
        print(f"  Total negative-silhouette cases: {len(sil_neg)}")

    # 5. PCA reconstruction error
    print("\n── 5. PCA reconstruction error ──")
    pca_err = analyze_pca_reconstruction(emb_df, embed_cols)
    print(f"  Mean reconstruction error: {pca_err['reconstruction_error'].mean():.6f}")
    print(f"  High-error points: {pca_err['is_high_error'].sum()} (top 5%)")

    print("\n" + "=" * 60)
    print(f"  All outputs → {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
