#!/usr/bin/env python3
"""Integration test: verify SAS embedding pipeline produces expected outputs.

Builds feature vectors and embeddings from first principles for a small
sample of rows, then compares against the `simulate_sas_embeddings.py`
output. Run this whenever the pipeline changes:

    pytest scripts/test_sas_embedding_pipeline.py -v

Or directly:
    python scripts/test_sas_embedding_pipeline.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHT_DIR = PROJECT_ROOT / "data" / "embeddings" / "tabular"
DATA_DIR = PROJECT_ROOT / "data" / "samples"
N_TEST_ROWS = 10
RTOL = 1e-8   # CSV roundtrip precision (10 decimal places → ~5e-9 error)
ATOL = 1e-12


# ── helpers ──────────────────────────────────────────────────────────

def _load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    return cols, rows


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v.strip()) if v.strip() else default
    except (ValueError, AttributeError):
        return default


def _load_projection(proj_path: Path) -> np.ndarray:
    with open(proj_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        embed_dim = len(header) - 1
        rows_data = []
        for row in reader:
            rows_data.append([float(x) for x in row[1:]])
    return np.array(rows_data, dtype=np.float64)


# ── feature construction (ground truth) ──────────────────────────────

def build_features_from_config(
    rows: list[dict],
    config: dict,
) -> np.ndarray:
    """Build feature vectors identically to simulate_sas_embeddings.py."""
    cat_cols = config["categorical_cols"]
    num_cols = config["numeric_cols"]
    cat_values = config["cat_values"]
    num_params = config["num_params"]

    n = len(rows)
    n_feat = sum(len(v) for v in cat_values.values()) + len(num_cols)
    X = np.zeros((n, n_feat), dtype=np.float64)

    for i, row in enumerate(rows):
        idx = 0
        for col in cat_cols:
            vals = cat_values[col]
            val = row.get(col, "")
            if val in vals:
                X[i, idx + vals.index(val)] = 1.0
            idx += len(vals)

        for col in num_cols:
            v = _safe_float(row.get(col, ""))
            info = num_params[col]
            rng = info["range"]
            X[i, idx] = (v - info["min"]) / rng if rng > 0 else 0.0
            idx += 1

    return X


def build_features_direct(
    rows: list[dict],
    config: dict,
) -> np.ndarray:
    """Alternative implementation: same logic, redundant structure.

    Used to cross-verify the feature builder itself (no shared code
    path except the config data).
    """
    cat_values = config["cat_values"]
    num_params = config["num_params"]

    n = len(rows)
    n_feat = sum(len(v) for v in cat_values.values()) + len(num_params)
    X = np.zeros((n, n_feat), dtype=np.float64)

    ptr = 0
    # Categoricals (in column order from cat_values dict)
    for col, vals in cat_values.items():
        for j, val in enumerate(vals):
            for i, row in enumerate(rows):
                if row.get(col, "") == val:
                    X[i, ptr + j] = 1.0
        ptr += len(vals)

    # Numerics
    for col, info in num_params.items():
        mn = info["min"]
        rng = info["range"]
        denom = rng if rng != 0 else 1.0
        for i, row in enumerate(rows):
            v = _safe_float(row.get(col, ""))
            X[i, ptr] = (v - mn) / denom
        ptr += 1

    return X


# ── tests ────────────────────────────────────────────────────────────

def test_column_detection_stable() -> None:
    """Auto-detected column roles should match expected values."""
    config_path = WEIGHT_DIR / "sas_config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    config = json.loads(config_path.read_text())

    expected_cat = {
        "SEGMENTO", "CALIBRATION_SEGMENT", "PRODUCTO",
        "TERMINACION", "CAUSA_DEFAULT", "COLATERAL_TIPO",
        "ADJUDICACION_TIPO", "ESTADO_CICLO",
    }
    expected_num = {
        "OR_DISPTO", "VALOR_COLATERAL_INICIAL", "TIPO_INTERES_ORIGINAL",
        "MES_CICLO", "PD_ESTIMADA", "LGD_ESTIMADA", "LGD_REALIZADA",
        "EAD", "ECL", "PROVISION", "STAGE_IFRS9", "CURE_FLAG",
        "DPDS", "RATING_GRADO", "SALDO_PENDIENTE",
        "TIPO_INTERES_ACTUAL", "INTERESES_ACUMULADOS",
        "FLUJO_RECUPERADO", "COSTE_RECUPERACION", "FLUJO_NETO",
        "RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO",
        "ADJUDICACION_FLAG", "ADJUDICACION_VALOR", "VALOR_COLATERAL",
        "LTV", "TASA_DESCUENTO", "RWA",
    }
    expected_meta = {"ID_FUSION_FINAL", "ID_CONTR_CICLO_LGD",
                     "FECHA_DEFAULT", "FECHA_REFERENCIA"}

    got_cat = set(config["categorical_cols"])
    got_num = set(config["numeric_cols"])
    got_meta = set(config["metadata_cols"])

    assert got_cat == expected_cat, (
        f"Categorical mismatch\n"
        f"  extra:   {got_cat - expected_cat}\n"
        f"  missing: {expected_cat - got_cat}"
    )
    assert got_num == expected_num, (
        f"Numeric mismatch\n"
        f"  extra:   {got_num - expected_num}\n"
        f"  missing: {expected_num - got_num}"
    )
    assert got_meta == expected_meta, (
        f"Metadata mismatch\n"
        f"  extra:   {got_meta - expected_meta}\n"
        f"  missing: {expected_meta - got_meta}"
    )


def test_feature_dimension() -> None:
    """Total feature count should match config."""
    config = json.loads((WEIGHT_DIR / "sas_config.json").read_text())
    proj = _load_projection(WEIGHT_DIR / "sas_projection.csv")

    n_cat = sum(len(v) for v in config["cat_values"].values())
    n_num = len(config["numeric_cols"])
    expected = n_cat + n_num

    assert proj.shape[0] == expected, (
        f"Projection rows ({proj.shape[0]}) != cat+num ({expected})"
    )
    assert int(proj.shape[1]) == config.get("embed_dim", 50), (
        f"Projection cols ({proj.shape[1]}) != expected embed dim"
    )


def test_feature_builders_agree() -> None:
    """Two independent feature builders should produce identical matrices."""
    config = json.loads((WEIGHT_DIR / "sas_config.json").read_text())
    _, rows = _load_csv(DATA_DIR / "recuperatory_cycles.csv")
    sample = rows[:N_TEST_ROWS]

    X1 = build_features_from_config(sample, config)
    X2 = build_features_direct(sample, config)

    assert X1.shape == X2.shape, f"Shape mismatch: {X1.shape} vs {X2.shape}"
    diff = np.abs(X1 - X2).max()
    assert diff < ATOL, f"Feature builder disagreement: max diff = {diff}"


def test_feature_onehot_correct() -> None:
    """Categorical one-hot positions should be correct for known values."""
    config = json.loads((WEIGHT_DIR / "sas_config.json").read_text())
    _, rows = _load_csv(DATA_DIR / "recuperatory_cycles.csv")

    # Pick a known row and verify its one-hot positions
    row = rows[0]
    X = build_features_from_config([row], config)
    X_flat = X[0]
    ptr = 0

    for col, vals in config["cat_values"].items():
        val = row.get(col, "")
        expected_idx = ptr + (vals.index(val) if val in vals else -1)
        for j in range(len(vals)):
            if ptr + j == expected_idx:
                assert X_flat[ptr + j] == 1.0, (
                    f"{col}={val}: one-hot at {ptr + j} should be 1, got {X_flat[ptr + j]}"
                )
            else:
                assert X_flat[ptr + j] == 0.0, (
                    f"{col}={val}: one-hot at {ptr + j} should be 0, got {X_flat[ptr + j]}"
                )
        ptr += len(vals)


def test_embedding_reproducibility() -> None:
    """Embeddings from feature matrix × projection should match saved output."""
    config = json.loads((WEIGHT_DIR / "sas_config.json").read_text())
    proj = _load_projection(WEIGHT_DIR / "sas_projection.csv")
    _, rows = _load_csv(DATA_DIR / "recuperatory_cycles.csv")
    sample = rows[:N_TEST_ROWS]

    # Build features and compute
    X = build_features_from_config(sample, config)
    expected_embeddings = X @ proj

    # Load the pre-computed simulation output
    output_path = WEIGHT_DIR / "sas_embeddings.csv"
    assert output_path.exists(), (
        f"Run simulation first: python scripts/simulate_sas_embeddings.py "
        f"--csv {DATA_DIR / 'recuperatory_cycles.csv'} --sample {N_TEST_ROWS}"
    )
    with open(output_path) as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)

    assert len(out_rows) >= N_TEST_ROWS, (
        f"Output has {len(out_rows)} rows, expected at least {N_TEST_ROWS}"
    )

    embed_cols = sorted(
        [k for k in out_rows[0].keys() if k.startswith("EMB_")],
        key=lambda x: int(x.split("_")[1]),
    )
    actual_embeddings = np.array([
        [float(r[c]) for c in embed_cols] for r in out_rows[:N_TEST_ROWS]
    ], dtype=np.float64)

    assert actual_embeddings.shape == expected_embeddings.shape, (
        f"Shape mismatch: actual {actual_embeddings.shape} vs expected {expected_embeddings.shape}"
    )

    diff = np.abs(actual_embeddings - expected_embeddings).max()
    assert diff < RTOL, (
        f"Embedding mismatch: max absolute diff = {diff:.6e}\n"
        f"If this fails, re-run: python scripts/simulate_sas_embeddings.py --sample {N_TEST_ROWS}"
    )


def test_projection_is_orthonormal() -> None:
    """Random projection matrix should be near-orthonormal (columns unit-norm)."""
    config = json.loads((WEIGHT_DIR / "sas_config.json").read_text())
    proj = _load_projection(WEIGHT_DIR / "sas_projection.csv")

    # Column norms
    norms = np.linalg.norm(proj, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-6), (
        f"Projection columns not unit-norm: {norms}"
    )

    # Orthogonality: off-diagonals of Gram matrix near 0
    gram = proj.T @ proj
    off_diag = gram - np.eye(gram.shape[0])
    max_off = np.abs(off_diag).max()
    assert max_off < 1e-6, (
        f"Projection columns not orthogonal: max off-diag = {max_off}"
    )


def test_sas_generated_has_correct_dimensions() -> None:
    """sas_generated.sas macro vars should match metadata from config."""
    sas_path = WEIGHT_DIR / "sas_generated.sas"
    assert sas_path.exists(), f"Missing: {sas_path}"

    config = json.loads((WEIGHT_DIR / "sas_config.json").read_text())
    proj = _load_projection(WEIGHT_DIR / "sas_projection.csv")
    n_feat_cfg = sum(len(v) for v in config["cat_values"].values()) + len(config["numeric_cols"])
    embed_dim_cfg = proj.shape[1]

    sas_text = sas_path.read_text()

    def _extract_macro(name: str) -> int:
        import re
        m = re.search(rf"%let\s+{name}\s*=\s*(\d+)\s*;", sas_text)
        assert m, f"Macro &{name} not found in {sas_path}"
        return int(m.group(1))

    assert _extract_macro("N_FEATURES") == n_feat_cfg
    assert _extract_macro("N_CAT_COLS") == len(config["categorical_cols"])
    assert _extract_macro("N_NUM_COLS") == len(config["numeric_cols"])
    assert _extract_macro("EMBED_DIM") == embed_dim_cfg


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
