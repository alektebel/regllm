"""LLM-as-judge for DQC catalogs using a deterministic rubric template.

The judge prompt is fixed (same criteria weights every run). The model fills
the rubric JSON and returns a 0–100 ``puntuacion_total``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from training.dq.catalog_schema import CatalogDocument, catalog_template, load_field_universe

# Fixed rubric — deterministic template the judge must fill.
JUDGE_RUBRIC_TEMPLATE: dict[str, Any] = {
    "puntuacion_total": 0,
    "criterios": {
        "cobertura_campos": {
            "max_puntos": 30,
            "puntos": 0,
            "comentario": "",
            "guia": "¿Cuántos campos auditable del universo aparecen en campos_entrada? 30=todos.",
        },
        "coherencia_campos": {
            "max_puntos": 25,
            "puntos": 0,
            "comentario": "",
            "guia": "¿campos_entrada coincide con regla_sql? ¿reglas multi-campo son coherentes?",
        },
        "calidad_regulatoria": {
            "max_puntos": 20,
            "puntos": 0,
            "comentario": "",
            "guia": "¿descripciones y referencias regulatorias son específicas y no genéricas?",
        },
        "calidad_sql": {
            "max_puntos": 15,
            "puntos": 0,
            "comentario": "",
            "guia": "¿SQL es portable, detecta violaciones reales y usa la tabla correcta?",
        },
        "completitud_catalogo": {
            "max_puntos": 10,
            "puntos": 0,
            "comentario": "",
            "guia": "¿IDs únicos, severidad/tipo/justificación presentes en la mayoría de entradas?",
        },
    },
    "campos_sin_cubrir": [],
    "fortalezas": [],
    "debilidades": [],
    "resumen": "",
}

JUDGE_SYSTEM_PROMPT = """\
Eres evaluador/a de catálogos DQC. NO regeneres ni copies entradas DQC.

Tu única tarea: rellenar la rúbrica de evaluación y devolver UN objeto JSON con:
- ``puntuacion_total`` (0–100, suma de criterios)
- ``criterios`` con claves: cobertura_campos, coherencia_campos, calidad_regulatoria, \
  calidad_sql, completitud_catalogo — cada una con ``puntos`` (entero) y ``comentario``
- ``campos_sin_cubrir``, ``fortalezas``, ``debilidades``, ``resumen``

Pesos máximos: cobertura_campos=30, coherencia_campos=25, calidad_regulatoria=20, \
calidad_sql=15, completitud_catalogo=10.

Responde SOLO con ese JSON de rúbrica. Sin markdown, sin entradas DQC sueltas.
"""


@dataclass
class JudgeResult:
    score: int
    criteria: dict[str, dict[str, Any]]
    uncovered_fields: list[str]
    strengths: list[str]
    weaknesses: list[str]
    summary: str
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "criteria": self.criteria,
            "uncovered_fields": self.uncovered_fields,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "summary": self.summary,
            "error": self.error,
            "fallback": self.fallback,
        }


def _clamp_score(value: Any) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _is_rubric_response(raw: dict[str, Any]) -> bool:
    if not isinstance(raw, dict):
        return False
    if "dqc_id" in raw and "puntuacion_total" not in raw and "criterios" not in raw:
        return False
    return "puntuacion_total" in raw or "score" in raw or "criterios" in raw or "criteria" in raw


def programmatic_judge_score(programmatic: dict[str, Any]) -> int:
    """Deterministic 0–100 fallback aligned with catalog_reward weights."""
    comps = programmatic.get("components") or {}
    raw = (
        0.35 * float(comps.get("r_coverage", 0))
        + 0.30 * float(comps.get("r_coherence", 0))
        + 0.15 * float(comps.get("r_schema", 0))
        + 0.10 * float(comps.get("r_groups", 0))
        + 0.10 * float(comps.get("r_parse", 0))
    )
    return _clamp_score(round(raw * 100))


def build_judge_user_prompt(
    catalog: CatalogDocument | None,
    programmatic: dict[str, Any],
    *,
    catalog_json: str | None = None,
) -> str:
    universe = load_field_universe()
    reference = catalog_template(universe)

    n_dqcs = len(catalog.dqcs) if catalog else programmatic.get("n_dqcs", 0)
    sample_entries = []
    if catalog:
        for e in catalog.dqcs[:5]:
            sample_entries.append({
                "dqc_id": e.dqc_id,
                "descripcion": e.descripcion[:120],
                "campos_entrada": e.campos_entrada,
                "tipo": e.tipo,
            })

    prog_summary = {
        "coverage_pct": programmatic.get("coverage_pct"),
        "missing_fields_count": len(programmatic.get("missing_fields", [])),
        "missing_fields_sample": programmatic.get("missing_fields", [])[:15],
        "components": programmatic.get("components", {}),
        "n_dqcs": n_dqcs,
    }

    rubric_skeleton = {
        "puntuacion_total": 0,
        "criterios": {
            k: {"puntos": 0, "comentario": ""}
            for k in JUDGE_RUBRIC_TEMPLATE["criterios"]
        },
        "campos_sin_cubrir": [],
        "fortalezas": [],
        "debilidades": [],
        "resumen": "",
    }

    user = (
        "Evalúa el catálogo DQC generado y DEVUELVE SOLO la rúbrica rellena.\n\n"
        f"Referencia de estructura DQC esperada:\n"
        f"{json.dumps(reference['dqcs'][0], ensure_ascii=False)}\n\n"
        f"Campos auditable totales: {len(universe.auditable_fields)}\n"
        f"Métricas programáticas (LÍMITES DUROS — no superes estos techos por criterio):\n"
        f"- cobertura_campos máx ≈ {round(30 * float(prog_summary['components'].get('r_coverage', 0)))} "
        f"(coverage_pct={prog_summary['coverage_pct']})\n"
        f"- coherencia_campos máx ≈ {round(25 * float(prog_summary['components'].get('r_coherence', 0)))}\n"
        f"- calidad_sql máx ≈ {round(15 * float(prog_summary['components'].get('r_parse', 0)))}\n"
        f"- completitud_catalogo máx ≈ {round(10 * float(prog_summary['components'].get('r_schema', 0)))}\n"
        f"{json.dumps(prog_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Resumen catálogo (n={n_dqcs}):\n"
        f"{json.dumps(sample_entries, ensure_ascii=False, indent=2)}\n\n"
        f"Esqueleto de rúbrica a rellenar:\n"
        f"{json.dumps(rubric_skeleton, ensure_ascii=False, indent=2)}"
    )
    return user


def _cap_judge_to_programmatic(
    result: JudgeResult,
    programmatic: dict[str, Any],
) -> JudgeResult:
    """Ensure judge cannot exceed deterministic evidence (coverage/coherence caps)."""
    comps = programmatic.get("components") or {}
    caps = {
        "cobertura_campos": round(30 * float(comps.get("r_coverage", 0))),
        "coherencia_campos": round(25 * float(comps.get("r_coherence", 0))),
        "calidad_sql": round(15 * float(comps.get("r_parse", 0))),
        "completitud_catalogo": round(10 * float(comps.get("r_schema", 0))),
    }
    total = 0
    for key, spec in result.criteria.items():
        max_allowed = caps.get(key, spec.get("max_puntos", 100))
        if key in caps:
            spec["puntos"] = min(spec["puntos"], max_allowed)
        total += spec["puntos"]
    # calidad_regulatoria uncapped (qualitative)
    result.score = _clamp_score(total)
    return result


def parse_judge_response(raw: dict[str, Any]) -> JudgeResult:
    criteria_raw = raw.get("criterios") or raw.get("criteria") or {}
    criteria: dict[str, dict[str, Any]] = {}
    total_from_criteria = 0

    for key, spec in JUDGE_RUBRIC_TEMPLATE["criterios"].items():
        entry = criteria_raw.get(key, {})
        if not isinstance(entry, dict):
            entry = {}
        max_pts = spec["max_puntos"]
        pts = _clamp_score(entry.get("puntos", 0))
        pts = min(pts, max_pts)
        total_from_criteria += pts
        criteria[key] = {
            "max_puntos": max_pts,
            "puntos": pts,
            "comentario": str(entry.get("comentario", entry.get("comment", ""))),
        }

    declared_total = _clamp_score(raw.get("puntuacion_total", raw.get("score", total_from_criteria)))
    # Prefer sum of criteria if declared total diverges wildly
    if abs(declared_total - total_from_criteria) > 5:
        score = total_from_criteria
    else:
        score = declared_total

    return JudgeResult(
        score=score,
        criteria=criteria,
        uncovered_fields=[str(x) for x in (raw.get("campos_sin_cubrir") or raw.get("missing_fields") or [])],
        strengths=[str(x) for x in (raw.get("fortalezas") or raw.get("strengths") or [])],
        weaknesses=[str(x) for x in (raw.get("debilidades") or raw.get("weaknesses") or [])],
        summary=str(raw.get("resumen") or raw.get("summary") or raw.get("rationale") or ""),
        raw=raw,
    )


def judge_catalog(
    catalog: CatalogDocument | None,
    programmatic: dict[str, Any],
    *,
    catalog_json: str | None = None,
    client=None,
) -> JudgeResult:
    """Call local LLM to score catalog 0–100 using deterministic rubric."""
    if client is None:
        from src.knowledge import get_client
        client = get_client()

    user = build_judge_user_prompt(catalog, programmatic, catalog_json=catalog_json)
    try:
        from src.knowledge.llm_client import _safe_json

        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        resp = client.chat(msgs, temperature=0.0, max_tokens=2048, json_mode=False)
        raw = _safe_json(resp.text)
        if not _is_rubric_response(raw):
            # One retry with explicit correction
            retry_user = (
                "Tu respuesta anterior NO era una rúbrica. "
                "Devuelve SOLO el JSON de rúbrica con puntuacion_total y criterios.\n\n"
                + user
            )
            resp = client.chat(
                [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": retry_user}],
                temperature=0.0,
                max_tokens=2048,
                json_mode=False,
            )
            raw = _safe_json(resp.text)

        if not _is_rubric_response(raw):
            fb = programmatic_judge_score(programmatic)
            return JudgeResult(
                score=fb,
                criteria={
                    k: {"max_puntos": v["max_puntos"], "puntos": 0, "comentario": "fallback programático"}
                    for k, v in JUDGE_RUBRIC_TEMPLATE["criterios"].items()
                },
                uncovered_fields=programmatic.get("missing_fields", []),
                strengths=[],
                weaknesses=["El modelo no devolvió la rúbrica; puntuación derivada de métricas programáticas"],
                summary=f"Fallback programático: {fb}/100",
                fallback=True,
                error="model did not return rubric JSON",
            )
        return _cap_judge_to_programmatic(parse_judge_response(raw), programmatic)
    except Exception as exc:
        fb = programmatic_judge_score(programmatic)
        return JudgeResult(
            score=fb,
            criteria={},
            uncovered_fields=programmatic.get("missing_fields", []),
            strengths=[],
            weaknesses=[],
            summary=f"Fallback programático: {fb}/100",
            fallback=True,
            error=str(exc),
        )
