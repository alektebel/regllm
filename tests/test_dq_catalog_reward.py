"""Tests for DQC catalog eval harness."""

from __future__ import annotations

import json

from training.dq.catalog_reward import score_catalog
from training.dq.catalog_schema import CatalogDocument, CatalogEntry, extract_catalog, load_field_universe
from training.dq.coherence_rules import PK_COLUMN, TABLE_NAME


def _minimal_entry(dqc_id: str, fields: list[str], sql: str, tipo: str = "consistencia") -> CatalogEntry:
    where = " AND ".join(f"{f} IS NOT NULL" for f in fields[:2]) or "1=0"
    return CatalogEntry(
        dqc_id=dqc_id,
        descripcion=f"Check {dqc_id}",
        campos_entrada=fields,
        tipo=tipo,
        severidad="bloqueante",
        regla_sql=sql or f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE {where}",
    )


def test_full_coverage_catalog_scores_high():
    universe = load_field_universe()
    fields = list(universe.auditable_fields)
    # One mega-DQC covering all fields (coherence will be lower but coverage maxes)
    catalog = CatalogDocument(
        table=TABLE_NAME,
        dqcs=[_minimal_entry("DQC_ALL_001", fields, "")],
    )
    bd = score_catalog(catalog, universe, min_coverage=1.0)
    assert bd.components["r_coverage"] == 1.0
    assert bd.missing_fields == []


def test_partial_coverage_reports_missing():
    universe = load_field_universe()
    catalog = CatalogDocument(
        table=TABLE_NAME,
        dqcs=[_minimal_entry("DQC_PD_001", ["PD_ESTIMADA", "STAGE_IFRS9"], "")],
    )
    bd = score_catalog(catalog, universe)
    assert bd.components["r_coverage"] < 0.1
    assert "PD_ESTIMADA" in bd.covered_fields
    assert len(bd.missing_fields) > 10


def test_coherent_adjudicacion_entry():
    universe = load_field_universe()
    sql = (
        f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
        "WHERE ADJUDICACION_VALOR > 0 AND ADJUDICACION_FLAG = '0'"
    )
    catalog = CatalogDocument(
        table=TABLE_NAME,
        dqcs=[_minimal_entry("DQC_ADJ_001", ["ADJUDICACION_VALOR", "ADJUDICACION_FLAG"], sql)],
    )
    bd = score_catalog(catalog, universe)
    assert bd.entry_reports[0].score >= 0.8


def test_single_field_consistencia_penalized():
    universe = load_field_universe()
    catalog = CatalogDocument(
        table=TABLE_NAME,
        dqcs=[_minimal_entry("DQC_BAD_001", ["EAD"], "", tipo="consistencia")],
    )
    bd = score_catalog(catalog, universe)
    assert any("≥2 campos" in issue for issue in bd.entry_reports[0].issues)


def test_extract_catalog_from_fence():
    payload = {"dqcs": [{"dqc_id": "DQC_X_001", "descripcion": "test", "campos_entrada": ["EAD"]}]}
    text = f"Here is the catalog:\n```json\n{json.dumps(payload)}\n```"
    cat = extract_catalog(text)
    assert cat is not None
    assert cat.dqcs[0].dqc_id == "DQC_X_001"


def test_build_manifest_runs():
    from training.dq.build_catalog_eval_manifest import build_manifest

    rows = build_manifest()
    assert len(rows) == 1
    assert rows[0]["n_auditable_fields"] == 39
    assert "messages" in rows[0]


def test_judge_parse_deterministic_rubric():
    from training.dq.catalog_judge import parse_judge_response

    raw = {
        "puntuacion_total": 72,
        "criterios": {
            "cobertura_campos": {"puntos": 18, "comentario": "cubre pocos campos"},
            "coherencia_campos": {"puntos": 20, "comentario": "sql alineado"},
            "calidad_regulatoria": {"puntos": 14, "comentario": "ok"},
            "calidad_sql": {"puntos": 12, "comentario": "ok"},
            "completitud_catalogo": {"puntos": 8, "comentario": "ids ok"},
        },
        "campos_sin_cubrir": ["LTV"],
        "fortalezas": ["buen sql adjudicacion"],
        "debilidades": ["falta cobertura"],
        "resumen": "catalogo parcial",
    }
    result = parse_judge_response(raw)
    assert result.score == 72
    assert result.criteria["cobertura_campos"]["puntos"] == 18
