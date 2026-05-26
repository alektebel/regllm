"""
IRB validation themes — keyword definitions and chunk tagging.

Themes are cross-document semantic clusters. Each chunk can belong to
multiple themes with a relevance score (keyword + optional embedding).
"""

from __future__ import annotations
import json
import re
import logging
import psycopg2
import psycopg2.extras
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ── Theme definitions ────────────────────────────────────────────────────────

@dataclass
class Theme:
    id: str
    label: str
    description: str
    keywords: list[str]
    parent_id: str | None = None


THEMES: list[Theme] = [
    Theme(
        id="pd_estimation",
        label="PD Estimation",
        description="Probability of Default estimation: calibration, long-run average, through-the-cycle, rating grades, observed default rates",
        keywords=[
            "probability of default", "PD estimation", "long-run average default rate",
            "through-the-cycle", "calibration", "rating grade", "observed default rate",
            "estimación de PD", "probabilidad de incumplimiento", "ciclo económico",
            "master scale", "central tendency",
        ],
    ),
    Theme(
        id="lgd_estimation",
        label="LGD Estimation",
        description="Loss Given Default estimation: workout LGD, downturn adjustment, recovery rates, collateral haircuts",
        keywords=[
            "loss given default", "LGD estimation", "workout LGD", "downturn LGD",
            "recovery rate", "collateral", "haircut", "cure rate", "time in default",
            "estimación de LGD", "pérdida en caso de incumplimiento", "recuperación",
            "indirect costs", "discounting", "economic loss",
        ],
    ),
    Theme(
        id="ead_ccf",
        label="EAD / CCF Estimation",
        description="Exposure at Default and Credit Conversion Factors for off-balance sheet items",
        keywords=[
            "exposure at default", "EAD", "credit conversion factor", "CCF",
            "off-balance sheet", "undrawn commitment", "limit", "outstanding amount",
            "estimación de EAD", "exposición en el momento del incumplimiento",
        ],
    ),
    Theme(
        id="default_definition",
        label="Default Definition",
        description="Definition of default: 90 days past due, unlikeliness to pay, technical default, contagion, cure",
        keywords=[
            "definition of default", "default definition", "90 days past due",
            "unlikeliness to pay", "distressed restructuring", "contagion",
            "cure", "re-default", "días de mora", "definición de incumplimiento",
            "impago", "reestructuración en dificultades", "contagio",
            "Article 178", "Artículo 178",
        ],
    ),
    Theme(
        id="model_governance",
        label="Model Governance & Validation",
        description="Model governance framework: independent validation, model risk management, governance structure, senior management oversight",
        keywords=[
            "model governance", "independent validation", "model risk",
            "validation function", "senior management", "management body",
            "internal audit", "gobernanza del modelo", "validación independiente",
            "función de validación", "alta dirección", "órgano de dirección",
            "oversight", "model inventory", "model owner",
        ],
    ),
    Theme(
        id="backtesting",
        label="Backtesting & Benchmarking",
        description="Backtesting of PD, LGD, EAD estimates; benchmarking against external data; statistical tests",
        keywords=[
            "backtesting", "benchmarking", "statistical test", "Binomial test",
            "traffic light", "discriminatory power", "Gini", "AUC", "ROC",
            "Brier score", "mean absolute deviation", "MAD", "back-test",
            "prueba estadística", "contraste estadístico",
        ],
    ),
    Theme(
        id="data_requirements",
        label="Data Requirements & Quality",
        description="Historical data requirements: minimum observation periods, data quality, data governance, missing data",
        keywords=[
            "historical data", "observation period", "data quality", "data governance",
            "minimum five years", "minimum seven years", "data collection",
            "datos históricos", "período de observación", "calidad de datos",
            "representativeness", "data representativeness", "data cleaning",
        ],
    ),
    Theme(
        id="stress_testing",
        label="Stress Testing & Economic Downturn",
        description="Stress scenarios for IRB parameters; economic downturn conditions; capital planning under stress",
        keywords=[
            "stress test", "stress scenario", "economic downturn", "adverse scenario",
            "downturn conditions", "capital planning", "macroeconomic stress",
            "prueba de estrés", "escenario adverso", "recesión económica",
            "downturn adjustment", "cycle adjustment",
        ],
    ),
    Theme(
        id="use_test",
        label="Use Test",
        description="Internal use of IRB estimates in risk management, pricing, limit setting, credit approval",
        keywords=[
            "use test", "internal use", "risk management", "credit decision",
            "pricing", "limit setting", "credit approval", "RAROC",
            "uso interno", "gestión del riesgo", "decisiones de crédito", "precio",
            "apetito al riesgo", "risk appetite",
        ],
    ),
    Theme(
        id="rating_systems",
        label="Rating Systems & Segmentation",
        description="Rating system design: grade structure, obligor vs facility grades, segmentation, human override",
        keywords=[
            "rating system", "rating model", "grade structure", "obligor grade",
            "facility grade", "segmentation", "override", "pooling",
            "sistema de calificación", "modelo de rating", "grado de calificación",
            "segmentación", "override humano", "asignación de grados",
        ],
    ),
    Theme(
        id="supervisory_assessment",
        label="Supervisory Assessment & IRB Permission",
        description="Competent authority assessment methodology, IRB model approval, sequential rollout, partial use",
        keywords=[
            "competent authority", "assessment methodology", "IRB permission",
            "supervisory review", "model approval", "sequential rollout", "partial use",
            "autoridad competente", "metodología de evaluación", "aprobación del modelo",
            "despliegue secuencial", "uso parcial", "autorización IRB",
        ],
    ),
]

# ── Chunk tagging ────────────────────────────────────────────────────────────

def _keyword_score(text: str, keywords: list[str]) -> float:
    """Count keyword hits normalised to 0-1 (caps at 1.0 after 3 hits)."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return min(1.0, hits / 3.0)


class ThemeTagger:
    def __init__(self, dsn: str):
        self.dsn = dsn

    # ── seed themes ──────────────────────────────────────────────────────────

    def seed_themes(self) -> None:
        """Insert THEMES into validation_themes (upsert on id)."""
        conn = psycopg2.connect(self.dsn)
        try:
            with conn, conn.cursor() as cur:
                for t in THEMES:
                    cur.execute(
                        """
                        INSERT INTO validation_themes (id, label, description, keywords, parent_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                            SET label = EXCLUDED.label,
                                description = EXCLUDED.description,
                                keywords = EXCLUDED.keywords
                        """,
                        (t.id, t.label, t.description,
                         json.dumps(t.keywords), t.parent_id),
                    )
            log.info("Seeded %d validation themes", len(THEMES))
        finally:
            conn.close()

    # ── tag chunks ───────────────────────────────────────────────────────────

    def tag_all_chunks(self, min_score: float = 0.33) -> int:
        """
        Score every chunk against every theme via keyword matching.
        Inserts/updates chunk_themes rows for scores >= min_score.
        Returns number of (chunk, theme) pairs inserted.
        """
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor(name="chunks_cur") as cur:
                cur.execute("SELECT chunk_id, texto FROM document_chunks")
                rows = cur.fetchall()

            inserted = 0
            with conn, conn.cursor() as wcur:
                for chunk_id, texto in rows:
                    for theme in THEMES:
                        score = _keyword_score(texto, theme.keywords)
                        if score < min_score:
                            continue
                        wcur.execute(
                            """
                            INSERT INTO chunk_themes (chunk_id, theme_id, score, method)
                            VALUES (%s, %s, %s, 'keyword')
                            ON CONFLICT (chunk_id, theme_id) DO UPDATE
                                SET score = GREATEST(chunk_themes.score, EXCLUDED.score),
                                    method = EXCLUDED.method
                            """,
                            (chunk_id, theme.id, score),
                        )
                        inserted += 1

            log.info("Tagged %d (chunk, theme) pairs", inserted)
            return inserted
        finally:
            conn.close()

    # ── embedding-based theme tagging ─────────────────────────────────────

    def embed_themes(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> None:
        """Compute and store embeddings for each theme's description."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        descriptions = [t.description for t in THEMES]
        embeddings = model.encode(descriptions, normalize_embeddings=True)

        conn = psycopg2.connect(self.dsn)
        try:
            with conn, conn.cursor() as cur:
                for theme, emb in zip(THEMES, embeddings):
                    cur.execute(
                        "UPDATE validation_themes SET embedding = %s WHERE id = %s",
                        (emb.tolist(), theme.id),
                    )
            log.info("Stored embeddings for %d themes", len(THEMES))
        finally:
            conn.close()

    def tag_by_embedding(self, min_score: float = 0.45) -> int:
        """
        Cosine-similarity tagging using pgvector. Requires theme embeddings and
        chunk embeddings to be populated. Uses SQL to avoid loading all vectors
        into Python.
        Returns number of (chunk, theme) pairs inserted/updated.
        """
        conn = psycopg2.connect(self.dsn)
        try:
            with conn, conn.cursor() as cur:
                # For each theme with an embedding, find similar chunks
                cur.execute("SELECT id FROM validation_themes WHERE embedding IS NOT NULL")
                theme_ids = [r[0] for r in cur.fetchall()]

            inserted = 0
            for theme_id in theme_ids:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chunk_themes (chunk_id, theme_id, score, method)
                        SELECT
                            dc.chunk_id,
                            %s,
                            1 - (dc.embedding <=> vt.embedding) AS score,
                            'embedding'
                        FROM document_chunks dc, validation_themes vt
                        WHERE vt.id = %s
                          AND dc.embedding IS NOT NULL
                          AND 1 - (dc.embedding <=> vt.embedding) >= %s
                        ON CONFLICT (chunk_id, theme_id) DO UPDATE
                            SET score = GREATEST(chunk_themes.score, EXCLUDED.score),
                                method = CASE
                                    WHEN EXCLUDED.score > chunk_themes.score
                                    THEN 'embedding' ELSE chunk_themes.method
                                END
                        """,
                        (theme_id, theme_id, min_score),
                    )
                    inserted += cur.rowcount

            log.info("Embedding-tagged %d (chunk, theme) pairs", inserted)
            return inserted
        finally:
            conn.close()
