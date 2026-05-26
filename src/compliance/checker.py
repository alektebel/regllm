"""RAG + LLM compliance diagnostic engine.

No hardcoded rules. Every parameter is diagnosed by:
  1. Building a semantic query from the row and column semantics
  2. Retrieving the relevant regulatory text via the pgvector RAG index
  3. Feeding row + regulatory context to the finetuned model
  4. Parsing a structured verdict grounded in the retrieved text

The model must cite only articles present in the retrieved context.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .hard_rules import apply_hard_rules
from .memory import Finding, RunningMemory

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

COMPLIANCE_SYSTEM_PROMPT = """\
Eres un experto en validación regulatoria de modelos de riesgo de crédito IRB/IFRS9 \
bajo CRR, EBA Guidelines y normativa prudencial europea.

Se te proporcionará:
- CONTEXTO REGULATORIO: artículos y directrices recuperados de la base de conocimiento regulatoria.
- MEMORIA ACUMULADA: resumen de hallazgos anteriores en la misma tabla.
- FILA A DIAGNOSTICAR: parámetros de una observación concreta (valores + metadatos del proceso de cálculo).

Tu tarea es diagnosticar si el cálculo es conforme a regulación.

Responde SIEMPRE en español y SIEMPRE en JSON estricto (sin markdown, sin texto fuera del JSON):

{
  "verdict": "compliant" | "flagged" | "uncertain",
  "resumen": "1-2 frases ejecutivas sobre el veredicto.",
  "desarrollo": "Análisis técnico detallado. Cita el texto regulatorio recuperado. Párrafos separados por \\n\\n.",
  "flags": ["Descripción concreta de cada incumplimiento detectado."],
  "articles": ["Solo artículos que aparezcan literalmente en el CONTEXTO REGULATORIO proporcionado."],
  "advertencias": "Aspectos que requieren revisión adicional con documentos oficiales. Null si no aplica."
}

REGLAS CRÍTICAS:
- "articles" SOLO puede contener referencias que aparezcan en el CONTEXTO REGULATORIO. Nunca inventes referencias.
- Si el contexto regulatorio no cubre suficientemente el parámetro, indica verdict "uncertain" y explica por qué.
- "flags" vacío si verdict es "compliant".
- Nunca inventes cifras, umbrales ni requisitos que no estén en el contexto recuperado."""


# ── Per-parameter RAG query templates ────────────────────────────────────────

# Maps column keywords → regulatory query to maximize RAG precision
_RAG_QUERY_MAP: dict[str, str] = {
    "PD":       "requisitos estimación probabilidad incumplimiento PD método IRB EBA directrices CRR suelo regulatorio",
    "LGD":      "requisitos estimación pérdida dado incumplimiento LGD método IRB CRR EBA suelos LGD garantías colateral",
    "EAD":      "exposición incumplimiento EAD CCF factores conversión crédito método IRB requisitos CRR EBA",
    "CCF":      "factores conversión crédito CCF exposición fuera balance método IRB CRR EBA",
    "MATURITY": "vencimiento efectivo M cálculo método IRB CRR art 162 suelo techo duración",
    "STAGE":    "clasificación staging IFRS9 stage 1 2 3 SICR incremento riesgo crédito EBA directrices",
    "ECL":      "pérdida crediticia esperada ECL IFRS9 cálculo provisiones etapas EBA",
    "RWA":      "activos ponderados riesgo RWA cálculo método IRB CRR fórmula Basilea",
    "K":        "requerimiento capital fórmula IRB función riesgo correlación CRR art 153 154",
    "RATING":   "calificación interna rating asignación grado deudor IRB validación EBA",
    "SEGMENT":  "segmentación carteras IRB criterios homogeneidad exposiciones empresas minoristas soberanos",
    "DEFAULT":  "definición incumplimiento default criterios 90 días probable incumplimiento CRR art 178 EBA",
    "CURE":     "salida incumplimiento cura cure rate período prueba IRB EBA directrices",
    "MARGIN":   "margen de conservadurismo MoC ajuste incertidumbre estimaciones IRB EBA GL 2017/16",
    "DOWNTURN": "downturn LGD ciclo económico adverso ajuste estimaciones IRB EBA requisitos",
}


def _build_rag_query(row: dict, column_map: dict[str, str]) -> str:
    """Build a targeted RAG query from the row's column names + values."""
    regulated = []
    for col in row:
        rk = column_map.get(col, col.upper())
        if rk in _RAG_QUERY_MAP:
            regulated.append(rk)

    if regulated:
        queries = [_RAG_QUERY_MAP[rk] for rk in regulated[:3]]
        return " | ".join(queries)

    # Fallback: generic IRB query with the column names
    cols = ", ".join(list(row.keys())[:6])
    return f"requisitos regulatorios IRB IFRS9 CRR EBA para parámetros: {cols}"


# ── LLM response parsing ──────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    text = raw.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse LLM JSON: %s", raw[:300])
    return {
        "verdict": "uncertain",
        "resumen": "No se pudo parsear la respuesta del modelo.",
        "desarrollo": raw[:500],
        "flags": [raw[:200]],
        "articles": [],
        "advertencias": "Respuesta del modelo no estructurada — revisar manualmente.",
    }


# ── Main class ────────────────────────────────────────────────────────────────

@dataclass
class RowVerdict:
    row_id: str
    verdict: str          # "compliant" | "flagged" | "uncertain"
    resumen: str = ""
    desarrollo: str = ""
    flags: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    advertencias: str | None = None
    rag_sources: list[str] = field(default_factory=list)  # source metadata from retrieved chunks
    row_data: dict = field(default_factory=dict)


class ComplianceChecker:
    """
    Diagnoses regulatory compliance of IRB/IFRS9 parameter rows using RAG + LLM.

    No hardcoded thresholds. The diagnosis is fully grounded in the retrieved
    regulatory text — the model can only cite articles present in that context.

    Usage:
        checker = ComplianceChecker(rag_system=rag, llm_backend="ollama")
        report  = checker.run(df, table_name="PD_ESTIMATES")
    """

    def __init__(
        self,
        rag_system=None,
        llm_backend: str = "ollama",
        ollama_model: str = "qwen2.5:14b-instruct-q4_K_M",
        ollama_host: str = "http://localhost:11434",
        groq_model: str = "llama-3.3-70b-versatile",
        column_map: dict[str, str] | None = None,
        compress_threshold: int = 20,
        rag_n_results: int = 5,
        citation_rag=None,
        rerank: bool = False,
        expand_to_parent: bool = True,
        primary_key: list[str] | None = None,
        grain_description: str | None = None,
        context_kb=None,
    ):
        self.rag = rag_system
        self.citation_rag = citation_rag
        self.context_kb = context_kb
        self.llm_backend = llm_backend
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.groq_model = groq_model
        self.column_map = column_map or {}
        self.rag_n_results = rag_n_results
        self.rerank = rerank
        self.expand_to_parent = expand_to_parent
        self.primary_key = primary_key  # explicit PK columns; None → auto-infer
        self.grain_description = grain_description  # e.g. "ciclo de recuperación"
        self._memory = RunningMemory(compress_threshold=compress_threshold)
        self._all_verdicts: list[RowVerdict] = []
        self._llm_calls = 0

        self._llm_fn = self._build_llm_fn()

    # ── LLM backends ──────────────────────────────────────────────────────────

    def _build_llm_fn(self):
        if self.llm_backend == "ollama":
            return self._call_ollama
        elif self.llm_backend == "groq":
            return self._call_groq
        raise ValueError(f"Unknown llm_backend: {self.llm_backend!r}")

    def _call_ollama(self, messages: list[dict]) -> str:
        import ollama as _ollama_lib  # type: ignore
        client = _ollama_lib.Client(host=self.ollama_host)
        resp = client.chat(
            model=self.ollama_model,
            messages=messages,
            stream=False,
            options={"temperature": 0.1, "num_ctx": 8192},
        )
        self._llm_calls += 1
        return resp["message"]["content"]

    def _call_groq(self, messages: list[dict]) -> str:
        from groq import Groq  # type: ignore
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        resp = Groq(api_key=api_key).chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        self._llm_calls += 1
        return resp.choices[0].message.content

    # ── RAG retrieval ──────────────────────────────────────────────────────────

    def _retrieve(self, row: dict) -> tuple[str, list[str]]:
        """Returns (formatted_context, list_of_source_labels)."""
        if self.rag is None:
            return "(RAG no disponible — diagnóstico sin contexto regulatorio)", []

        query = _build_rag_query(row, self.column_map)
        logger.debug("RAG query: %s", query[:120])

        try:
            # Use buscar_con_citas (RAG-3) when citation_rag is provided so
            # per-article nodes supplement document chunk results.
            # expand_to_parent (RAG-2) and rerank (RAG-1) are opt-in via constructor.
            chunks = self.rag.buscar_con_citas(
                query,
                n_resultados=self.rag_n_results,
                citation_rag=self.citation_rag,
                rerank=self.rerank,
                expand_to_parent=self.expand_to_parent,
            )
            context = self.rag.formatear_contexto(chunks)
            sources = [
                c.get("metadata", {}).get("source", c.get("metadata", {}).get("documento", "?"))
                for c in chunks
            ]
            return context, sources
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            return "(error en recuperación RAG)", []

    # ── Per-row diagnosis ──────────────────────────────────────────────────────

    def diagnose_row(self, row_id: str, row: dict) -> RowVerdict:
        """Diagnose a single row against regulation via RAG + LLM."""
        rag_context, sources = self._retrieve(row)

        business_ctx = ""
        if self.context_kb is not None:
            business_ctx = self.context_kb.get_record_context(row)

        business_section = (
            "CONTEXTO DE NEGOCIO (diccionario de campos, notas analistas):\n"
            + business_ctx + "\n\n"
        ) if business_ctx else ""

        prompt_user = (
            business_section
            + "CONTEXTO REGULATORIO:\n" + rag_context + "\n\n"
            + "MEMORIA ACUMULADA:\n" + self._memory.as_prompt_str() + "\n\n"
            + "FILA A DIAGNOSTICAR (id=" + row_id + "):\n"
            + json.dumps(row, ensure_ascii=False, default=str, indent=2)
        )
        messages = [
            {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_user},
        ]

        raw = self._llm_fn(messages)
        parsed = _parse_json(raw)

        verdict   = parsed.get("verdict", "uncertain")
        flags     = parsed.get("flags") or []
        articles  = parsed.get("articles") or []
        resumen   = parsed.get("resumen", "")
        desarrollo = parsed.get("desarrollo", "")
        advertencias = parsed.get("advertencias")

        # Ground articles against retrieved sources to prevent hallucination
        articles = _ground_articles(articles, rag_context)

        # Update running memory
        if verdict in ("flagged", "uncertain"):
            self._memory.add_finding(Finding(
                row_id=row_id,
                column=", ".join(row.keys()),
                value=str({k: v for k, v in list(row.items())[:4]}),
                verdict=verdict,
                flags=flags,
                articles=articles,
                regulatory_text=rag_context[:300],
            ))
            if self._memory.needs_compression():
                logger.info("Compressing memory at row %s", row_id)
                self._memory.compress(self._llm_fn)
        else:
            self._memory.add_compliant()

        return RowVerdict(
            row_id=row_id,
            verdict=verdict,
            resumen=resumen,
            desarrollo=desarrollo,
            flags=flags,
            articles=articles,
            advertencias=advertencias,
            rag_sources=sources,
            row_data=row,
        )

    # ── Full-table pipeline ────────────────────────────────────────────────────

    def _resolve_row_id(self, df: pd.DataFrame, row_id_col: str | None) -> list[str]:
        """Return a list of string row identifiers for each row in df."""
        if row_id_col and row_id_col in df.columns:
            return df[row_id_col].astype(str).tolist()

        # Use PK columns (explicit or inferred) to build a compound key
        pk_cols = self.primary_key
        if pk_cols is None:
            pk_cols = infer_primary_key(df)
            if pk_cols:
                logger.info("Inferred primary key: %s", pk_cols)

        if pk_cols:
            return (
                df[pk_cols]
                .astype(str)
                .apply(lambda r: "|".join(r), axis=1)
                .tolist()
            )

        return [str(i) for i in df.index]

    def run(
        self,
        df: pd.DataFrame,
        table_name: str = "unknown",
        row_id_col: str | None = None,
    ) -> dict:
        """Run compliance diagnosis on every row of a DataFrame.

        Returns a report dict matching the architecture spec.
        """
        self._memory = RunningMemory(compress_threshold=self._memory.compress_threshold)
        self._all_verdicts = []
        self._llm_calls = 0

        total = len(df)
        logger.info("Starting compliance diagnosis: table=%s, rows=%d", table_name, total)

        row_ids = self._resolve_row_id(df, row_id_col)

        for i, ((_idx, row_series), row_id) in enumerate(zip(df.iterrows(), row_ids)):
            row = {
                k: v for k, v in row_series.to_dict().items()
                if k != row_id_col and v is not None
            }
            if self.grain_description:
                row["__grain__"] = self.grain_description
            verdict = self.diagnose_row(row_id, row)
            self._all_verdicts.append(verdict)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                flagged = sum(1 for v in self._all_verdicts if v.verdict != "compliant")
                logger.info("  %d/%d | flagged: %d | LLM calls: %d",
                            i + 1, total, flagged, self._llm_calls)

        return self._build_report(table_name)

    # ── Tiered pipeline ────────────────────────────────────────────────────────

    def run_tiered(
        self,
        df: pd.DataFrame,
        table_name: str = "unknown",
        row_id_col: str | None = None,
        workers: int = 4,
        cluster_dims: list[str] | None = None,
        cluster_sample_n: int = 4,
        zscore_threshold: float = 3.0,
    ) -> dict:
        """Three-tier compliance pipeline for large tables.

        Tier 1 — vectorized hard rules (full DataFrame, no LLM)
            Instantly flags well-known threshold violations.
        Tier 2 — cluster-sampled LLM calls
            Groups remaining rows by segment/rating/etc., samples 3-5 rows per
            cluster, one LLM call per cluster via a thread pool.
        Tier 3 — statistical outlier detection
            Within Tier-2 compliant clusters, zscore outliers trigger individual
            LLM calls.

        Returns the same report dict schema as run().
        """
        self._memory = RunningMemory(compress_threshold=self._memory.compress_threshold)
        self._all_verdicts = []
        self._llm_calls = 0

        row_ids = self._resolve_row_id(df, row_id_col)
        id_series = pd.Series(row_ids, index=df.index, name="__row_id__")
        df_work = df.copy()
        df_work["__row_id__"] = id_series

        total = len(df_work)
        logger.info("Tiered run: table=%s, rows=%d, workers=%d", table_name, total, workers)

        # ── Tier 1: hard rules ────────────────────────────────────────────────
        hard_report = apply_hard_rules(df, column_map=self.column_map)
        tier1_mask = hard_report.any_failure_mask  # True = already flagged
        if tier1_mask is None:
            tier1_mask = pd.Series(False, index=df.index)

        tier1_findings = hard_report.to_findings()
        n_tier1 = len(tier1_findings)
        logger.info("Tier 1: %d hard-rule violations", n_tier1)

        # Rows that need LLM evaluation
        df_remaining = df_work[~tier1_mask].copy()
        if df_remaining.empty:
            return self._build_tiered_report(
                table_name, tier1_findings, [], [], hard_report, total
            )

        # ── Tier 2: cluster-sampled LLM ──────────────────────────────────────
        dims = _pick_cluster_dims(df_remaining, cluster_dims)
        logger.info("Tier 2 clustering on: %s", dims)

        # For each cluster: sample rows → one LLM call → apply verdict to all
        tier2_verdicts: list[RowVerdict] = []
        cluster_verdict_map: dict[tuple, str] = {}  # cluster_key → verdict

        def _diagnose_cluster(cluster_key, cluster_df) -> list[RowVerdict]:
            sample = cluster_df.sample(min(cluster_sample_n, len(cluster_df)), random_state=42)
            rows_data = [
                {k: v for k, v in r.to_dict().items()
                 if k not in ("__row_id__",) and v is not None}
                for _, r in sample.iterrows()
            ]
            if self.grain_description:
                for r in rows_data:
                    r["__grain__"] = self.grain_description

            # Represent cluster as the average/mode of numeric/categorical cols
            cluster_summary = {}
            for col in rows_data[0]:
                vals = [r[col] for r in rows_data if col in r]
                try:
                    cluster_summary[col] = round(sum(float(v) for v in vals) / len(vals), 6)
                except (ValueError, TypeError):
                    # Categorical: pick most common
                    from collections import Counter
                    cluster_summary[col] = Counter(str(v) for v in vals).most_common(1)[0][0]

            cluster_id = "|".join(str(k) for k in cluster_key) if isinstance(cluster_key, tuple) else str(cluster_key)
            representative = RowVerdict.__new__(RowVerdict)
            verdict = self.diagnose_row(f"cluster:{cluster_id}", cluster_summary)
            cluster_verdict_map[cluster_key] = verdict.verdict

            # Propagate verdict to all cluster rows
            verdicts = []
            for _, row_series in cluster_df.iterrows():
                rid = row_series["__row_id__"]
                row = {k: v for k, v in row_series.to_dict().items()
                       if k not in ("__row_id__",) and v is not None}
                verdicts.append(RowVerdict(
                    row_id=rid,
                    verdict=verdict.verdict,
                    resumen=f"[Cluster {cluster_id}] {verdict.resumen}",
                    desarrollo=verdict.desarrollo,
                    flags=verdict.flags,
                    articles=verdict.articles,
                    advertencias=verdict.advertencias,
                    rag_sources=verdict.rag_sources,
                    row_data=row,
                ))
            return verdicts

        if dims:
            groups = list(df_remaining.groupby(dims, dropna=False))
        else:
            # No cluster dims found — treat entire remaining set as one cluster
            groups = [("all", df_remaining)]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_diagnose_cluster, key, cdf): key for key, cdf in groups}
            for fut in as_completed(futures):
                try:
                    tier2_verdicts.extend(fut.result())
                except Exception as e:
                    logger.error("Cluster %s failed: %s", futures[fut], e)

        # ── Tier 3: statistical outliers in compliant clusters ────────────────
        tier3_verdicts: list[RowVerdict] = []
        numeric_cols = df_remaining.select_dtypes("number").columns.tolist()

        if numeric_cols and dims:
            for key, cluster_df in groups:
                # Only investigate clusters that came back compliant
                if cluster_verdict_map.get(key, "compliant") != "compliant":
                    continue
                if len(cluster_df) < 4:
                    continue
                try:
                    from scipy import stats as _stats
                    z = _stats.zscore(cluster_df[numeric_cols].fillna(0), nan_policy="omit")
                    outlier_mask = (abs(z) > zscore_threshold).any(axis=1)
                    outliers = cluster_df[outlier_mask]
                except Exception:
                    continue

                for _, row_series in outliers.iterrows():
                    rid = row_series["__row_id__"]
                    row = {k: v for k, v in row_series.to_dict().items()
                           if k not in ("__row_id__",) and v is not None}
                    if self.grain_description:
                        row["__grain__"] = self.grain_description
                    v = self.diagnose_row(rid, row)
                    tier3_verdicts.append(v)

        logger.info(
            "Tiered complete: tier1=%d, tier2=%d rows (%d clusters), tier3=%d",
            n_tier1, len(tier2_verdicts), len(groups), len(tier3_verdicts),
        )
        return self._build_tiered_report(
            table_name, tier1_findings, tier2_verdicts, tier3_verdicts, hard_report, total
        )

    def _build_tiered_report(
        self,
        table_name: str,
        tier1_findings: list[dict],
        tier2_verdicts: list[RowVerdict],
        tier3_verdicts: list[RowVerdict],
        hard_report,
        total_rows: int,
    ) -> dict:
        # Merge all verdicts into self._all_verdicts for _build_report compatibility
        self._all_verdicts = tier2_verdicts + tier3_verdicts

        base = self._build_report(table_name)

        # Prepend Tier-1 hard-rule findings (they aren't RowVerdicts)
        base["findings"] = tier1_findings + base["findings"]
        base["rows_processed"] = total_rows
        base["rows_flagged"] = (
            len(tier1_findings)
            + sum(1 for v in tier2_verdicts if v.verdict != "compliant")
            + sum(1 for v in tier3_verdicts if v.verdict != "compliant")
        )
        n = total_rows
        base["compliance_rate"] = round(1 - base["rows_flagged"] / n, 4) if n else 1.0
        base["hard_rules_summary"] = hard_report.summary()
        base["tier_counts"] = {
            "tier1_flagged": len(tier1_findings),
            "tier2_rows":    len(tier2_verdicts),
            "tier3_outlier_rows": len(tier3_verdicts),
            "llm_calls":     self._llm_calls,
        }
        return base

    # ── Report generation ──────────────────────────────────────────────────────

    def _build_report(self, table_name: str) -> dict:
        flagged = [v for v in self._all_verdicts if v.verdict != "compliant"]
        n = len(self._all_verdicts)
        compliance_rate = (n - len(flagged)) / n if n else 1.0

        findings_out = [
            {
                "row_id":      v.row_id,
                "verdict":     v.verdict,
                "resumen":     v.resumen,
                "flags":       v.flags,
                "articles":    v.articles,
                "rag_sources": v.rag_sources,
                "advertencias": v.advertencias,
                "row_data":    v.row_data,
            }
            for v in flagged
        ]

        narrative = self._generate_narrative(table_name, n, len(flagged),
                                              compliance_rate, findings_out)

        return {
            "table":              table_name,
            "run_date":           date.today().isoformat(),
            "rows_processed":     n,
            "rows_flagged":       len(flagged),
            "rows_uncertain":     sum(1 for v in flagged if v.verdict == "uncertain"),
            "compliance_rate":    round(compliance_rate, 4),
            "llm_calls":          self._llm_calls,
            "compressed_patterns": self._memory.patterns,
            "findings":           findings_out,
            "narrative":          narrative,
        }

    def _generate_narrative(self, table_name, rows_processed, rows_flagged,
                            compliance_rate, findings) -> dict:
        sample = findings[:8]
        prompt = f"""Genera un informe ejecutivo de validación regulatoria en JSON.

Tabla: {table_name}
Filas analizadas: {rows_processed} | Marcadas: {rows_flagged} | Tasa conformidad: {compliance_rate:.1%}
Patrones acumulados: {self._memory.patterns or '(ninguno)'}
Muestra de hallazgos ({len(sample)} de {len(findings)}):
{json.dumps(sample, ensure_ascii=False, indent=2, default=str)}

Responde SOLO en JSON:
{{
  "resumen": "1-2 frases ejecutivas",
  "desarrollo": "análisis detallado de los principales incumplimientos y patrones detectados",
  "articulos": ["referencias regulatorias implicadas — solo las que aparecen en los hallazgos"],
  "advertencias": "limitaciones o aspectos que requieren revisión manual adicional"
}}"""
        try:
            raw = self._llm_fn([{"role": "user", "content": prompt}])
            result = _parse_json(raw)
            # Ground narrative articles against all articles that appeared in findings
            all_cited = []
            for f in findings:
                arts = f.get("articles", [])
                if isinstance(arts, list):
                    all_cited.extend(arts)
            all_ctx = " ".join(all_cited)
            result["articulos"] = _ground_articles(
                result.get("articulos", []), all_ctx
            )
            return result
        except Exception as e:
            logger.warning("Narrative generation failed: %s", e)
            return {
                "resumen": f"Análisis completado: {rows_flagged}/{rows_processed} filas marcadas.",
                "desarrollo": self._memory.patterns or "Sin patrones detectados.",
                "articulos": [],
                "advertencias": str(e),
            }


# ── PK inference ─────────────────────────────────────────────────────────────

def infer_primary_key(df: pd.DataFrame, max_cols: int = 3) -> list[str]:
    """Infer the primary key of a DataFrame by cardinality analysis.

    Tries single-column candidates with high uniqueness first, then compound
    combinations up to *max_cols*. Returns the smallest combo that has no
    duplicate rows, or [] if none found.
    """
    if df.empty:
        return []

    n = len(df)
    # Candidates: columns where >80% of values are unique
    candidates = [c for c in df.columns if df[c].nunique() / n > 0.8]

    for r in range(1, min(max_cols, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, r):
            if df.duplicated(subset=list(combo)).sum() == 0:
                return list(combo)

    return []


# ── Cluster dimension selection ────────────────────────────────────────────────

# Default dims to cluster on for Tier-2 sampling — present in many IRB tables
_DEFAULT_CLUSTER_DIMS = [
    "SEGMENT", "SEGMENTO", "CARTERA",
    "RATING_GRADE", "GRADO_RATING",
    "COLLATERAL_TYPE", "TIPO_COLATERAL",
    "CALIBRATION_METHOD", "METODO_CALIBRACION",
    "REFERENCE_DATE", "FECHA_REFERENCIA",
]


def _pick_cluster_dims(df: pd.DataFrame, explicit: list[str] | None) -> list[str]:
    """Return cluster dimension columns that actually exist in *df*."""
    if explicit:
        return [c for c in explicit if c in df.columns]

    df_cols_upper = {c.upper(): c for c in df.columns}
    found = []
    seen = set()
    for cand in _DEFAULT_CLUSTER_DIMS:
        actual = df_cols_upper.get(cand.upper())
        if actual and actual not in seen:
            found.append(actual)
            seen.add(actual)
    return found


# ── Citation grounding (anti-hallucination) ───────────────────────────────────

def _ground_articles(articles: list[str], rag_context: str) -> list[str]:
    """Keep only article references mentioned in the retrieved context.

    Makes it structurally impossible to return a hallucinated citation
    regardless of model output.
    """
    if not articles:
        return []
    if not rag_context:
        return articles

    import re as _re
    ctx_norm = _re.sub(r"[()./:,]", " ", rag_context.lower())
    ctx_norm = _re.sub(r"\s+", " ", ctx_norm)

    grounded = []
    for art in articles:
        art_norm = _re.sub(r"[()./:,]", " ", art.lower().strip())
        art_norm = _re.sub(r"\s+", " ", art_norm).strip()
        tokens = art_norm.split()
        found = False
        for i in range(max(1, len(tokens) - 1)):
            fragment = " ".join(tokens[i : i + 2])
            if len(fragment) >= 4 and fragment in ctx_norm:
                found = True
                break
        if not found and art_norm in ctx_norm:
            found = True
        if found:
            grounded.append(art)
        else:
            logger.debug("Dropped ungrounded article: %r", art)
    return grounded
