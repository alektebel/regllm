"""
ChatEngine — shared query pipeline for RegLLM apps.

Extracts context building, prompt assembly, response parsing, reference
enrichment, and Gradio rendering logic so all app variants (local, groq,
ollama) share the same pipeline and only differ in their generate() call.
"""

import json
import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Topic guard ──────────────────────────────────────────────────────────────

# Layer 1 — Regex: instantly reject obvious non-professional / creative requests.
_OFFTOPIC_RE = re.compile(
    r"\b(poes[ií]a|poema|estrofa|rima|verso"
    r"|chiste|broma|humorada"
    r"|cuento|f[aá]bula|novela|historia\s+de\s+amor"
    r"|canc[ií]n|letra\s+de\s+m[uú]sica"
    r"|receta|ingredientes|cocina"
    r"|dibuj[ao]|ilustra[cr]"
    r"|traduc[ei]|traduc[cz]i[oó]n"
    r"|escr[ií]be[nm]e\s+un(?:a)?\s+(?!pregunta|respuesta|informe|an[aá]lisis)"
    r"|h[aá]zme\s+un(?:a)?\s+(?!pregunta|an[aá]lisis|informe|resumen\s+de\s+la\s+norma)"
    r")\b",
    re.IGNORECASE,
)

# Layer 2 — Embedding guardrail: curated seed sentences covering every major
# sub-domain of credit-risk banking regulation.  An incoming question must
# reach _TOPIC_GUARD_THRESHOLD cosine similarity against at least one seed
# to be considered on-topic; anything below is rejected.

_TOPIC_SEEDS = [
    # Capital requirements & CRR/CRD
    "requisitos de capital CET1 Tier 1 Tier 2 activos ponderados por riesgo RWA CRR regulación bancaria",
    "ratio de capital mínimo Basilea III CRD IV CRD V normativa prudencial entidades crédito supervisión",
    "capital ordinario nivel 1 CET1 absorción pérdidas buffer de capital conservación",
    "requerimientos de capital adicional Pilar 2 SREP supervisor proceso revisión",
    # Credit risk — IRB / SA
    "riesgo de crédito probabilidad de impago PD pérdida en caso de impago LGD exposición EAD parámetros IRB",
    "método estándar avanzado riesgo crédito ponderación exposición calificación externa crediticia",
    "enfoque basado calificaciones internas IRB modelos internos riesgo crédito validación supervisora",
    "concentración riesgo crédito grandes exposiciones límite regulatorio CRR contraparte",
    # IFRS 9 / provisioning
    "pérdidas esperadas IFRS 9 provisiones crediticias deterioro activos financieros ECL modelo",
    "clasificación y medición activos financieros IFRS 9 etapas deterioro riesgo crédito",
    # Liquidity ratios
    "ratio cobertura de liquidez LCR activos líquidos alta calidad HQLA salidas netas Basilea III",
    "ratio financiación estable neta NSFR fuentes financiación estable disponible requerida Basilea",
    # Supervisory framework
    "ICAAP evaluación interna adecuación capital proceso supervisor Pilar 2 pruebas estrés",
    "directrices EBA BCE normativa prudencial entidades de crédito banca europea supervisión",
    "Basilea III IV revisión fundamental estándares internacionales capital riesgo comité supervisión bancaria",
    # NPL / asset quality
    "préstamos dudosos NPL ratio calidad activos cobertura provisiones banco exposición deteriorada",
    "exposiciones dudosas NPE clasificación deterioro crediticio entidad crediticia reestructuración",
    # Leverage & systemic risk
    "ratio apalancamiento Tier 1 exposición total activo balance Basilea III límite regulatorio",
    "entidades sistémicas importancia global GSIB colchón sistémico capital adicional resolución",
    # IFRS 9 / ECB / EBA documents
    "directrices EBA gestión riesgo crédito concesión préstamos monitoring seguimiento NPL",
    "artículo CRR reglamento capital requisitos ratio solvencia cálculo regulatorio europeo",
]

_TOPIC_GUARD_THRESHOLD = 0.30  # min cosine similarity to accept as on-topic
_EMBED_MODEL_CE = "paraphrase-multilingual-MiniLM-L12-v2"

_tg_model = None
_tg_seed_vecs = None
_tg_lock = threading.Lock()


def _get_topic_guard_vecs():
    """Lazy-load embed model and pre-compute seed vectors once per process."""
    global _tg_model, _tg_seed_vecs
    with _tg_lock:
        if _tg_seed_vecs is None:
            from sentence_transformers import SentenceTransformer

            _tg_model = SentenceTransformer(_EMBED_MODEL_CE)
            _tg_seed_vecs = _tg_model.encode(_TOPIC_SEEDS, convert_to_numpy=True)
    return _tg_model, _tg_seed_vecs


def _topic_embedding_score(question: str) -> float:
    """
    Return max cosine similarity of question against credit-risk/regulation seed sentences.
    Returns 1.0 (fail-open) if the embed model cannot be loaded, so the guard
    never blocks users due to a missing dependency.
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        model, seed_vecs = _get_topic_guard_vecs()
        q_vec = model.encode([question], convert_to_numpy=True)
        sims = cosine_similarity(q_vec, seed_vecs)[0]
        return float(sims.max())
    except Exception as e:
        logger.warning(f"Topic embedding guard failed: {e}")
        return 1.0  # fail-open: don't block if guard is unavailable


REJECTION_CARD = (
    '<div class="resp-card">'
    '<div class="resp-section" style="background:#FFF5F5;">'
    '<div class="resp-label" style="color:#b91c1c;">Consulta fuera de ámbito</div>'
    '<div class="resp-body">'
    "Este asistente responde exclusivamente preguntas sobre "
    "<strong>regulación bancaria, riesgo de crédito y normativa financiera</strong> "
    "(CRR, CRD IV/V, IFRS 9, Basilea III/IV, directrices EBA/BCE, etc.).<br><br>"
    "Por favor, reformule su pregunta en ese contexto."
    "</div></div></div>"
)

# ─── Shared UI constants (used by all app variants) ──────────────────────────

CSS = """
/* ── Reset & base ── */
body, .gradio-container { background: #f8fafc !important; font-family: 'Inter', system-ui, sans-serif; }
footer { display: none !important; }

/* ── App header ── */
.regllm-header {
    background: #fff;
    border-bottom: 1px solid #e2e8f0;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 0;
}
.regllm-logo {
    width: 48px;
    height: 48px;
    border-radius: 8px;
    object-fit: cover;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.regllm-header h1 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e293b;
    letter-spacing: -0.5px;
}
.regllm-header .subtitle {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #1e40af;
    color: #fff;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
}
.badge-green { background: #166534; }

/* ── Chat area ── */
.chatbot-wrap { background: #fff; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
#chatbot { background: transparent; border: none; }

/* ── User bubbles ── */
.message.user { justify-content: flex-end; }
.message.user .bubble-wrap {
    background: #dbeafe;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
}

/* ── Assistant cards ── */
.message.bot .bubble-wrap {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 4px 18px 18px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    max-width: 90%;
}

/* ── Response card ──────────────────────────────────── */
.resp-card {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
    margin: 4px 0;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.resp-section {
    padding: 12px 16px;
    border-bottom: 1px solid #E2E8F0;
}
.resp-section:last-child { border-bottom: none; }
.resp-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #1E40AF;
    margin-bottom: 6px;
}
.resp-body { font-size: 0.9rem; color: #0F172A; line-height: 1.55; }
.resp-section--refs { background: #F0F4FF; }
.ref-item { margin: 3px 0; padding-left: 4px; }
.resp-section--conf { background: #F0FDF4; }
.conf-bar-wrap {
    background: #E2E8F0;
    border-radius: 4px;
    height: 6px;
    margin: 6px 0 8px 0;
    overflow: hidden;
}
.conf-fill { height: 6px; background: #1E40AF; border-radius: 4px; }
.conf-pct { font-size: 0.82rem; font-weight: 600; color: #1E40AF; }
.conf-justif { font-size: 0.82rem; color: #475569; margin-top: 2px; }

/* ── Input row ── */
.input-row { background: #fff; border-top: 1px solid #e2e8f0; padding: 12px 16px; border-radius: 0 0 12px 12px; }
#send-btn { min-width: 80px; }

/* ── Settings accordion ── */
.settings-panel { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; }
"""

EXAMPLES = [
    "¿Qué establece el artículo 92 del CRR sobre requisitos de capital?",
    "Explica el ICAAP según las directrices de la EBA",
    "¿Cómo se calcula el ratio de apalancamiento en Basilea III?",
    "¿Cuál es el ratio CET1 de Santander en 2023?",
    "Diferencias entre Pilar 1 y Pilar 2 en Basilea",
]

# ─── Structured system prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un asistente especializado exclusivamente en regulación bancaria y riesgo de crédito.
Ámbito: CRR, CRD IV/V, IFRS 9, Basilea III/IV, directrices EBA/BCE, normativa prudencial.

Responde SIEMPRE en español y SIEMPRE en el siguiente formato JSON (sin markdown, sin texto fuera del JSON):

{
  "respuesta": "Texto completo de la respuesta en español, técnico y preciso."
}

REGLAS:
- Responde solo preguntas de regulación bancaria/financiera. Si la pregunta está fuera de ámbito, indícalo.
- Nunca inventes artículos, normativas ni cifras. Si no sabes, dilo explícitamente.
- No incluyas referencias ni puntuaciones: el sistema las obtiene automáticamente de la base de datos regulatoria."""


class ChatEngine:
    """
    Shared RAG + parse + render pipeline.

    Usage:
        engine = ChatEngine(rag_system, citation_rag=citation_rag)
        context, sources = engine.build_context(question, n_sources, hybrid)
        messages = engine.build_messages(question, context, history)
        # caller runs: raw = generate(messages)
        parsed = engine.parse_response(raw)
        parsed = engine.enrich_references(parsed, question)
        md = engine.render_message(parsed)
        engine.log(question, md, latency_ms)
    """

    def __init__(self, rag_system, citation_rag=None, db_source: str = "chat_engine"):
        self.rag = rag_system
        self.citation_rag = citation_rag
        self.db_source = db_source

    # ── Context building ──────────────────────────────────────────────────────

    def build_context(
        self, question: str, n_sources: int = 5, hybrid: bool = True
    ) -> tuple[str, list]:
        """Run RAG retrieval; return (context_string, raw_results)."""
        mode = "hybrid" if hybrid else "semantic"
        logger.debug(f"RAG {mode} search — n_sources={n_sources}")
        try:
            if hybrid:
                results = self.rag.buscar_hibrida(question, n_resultados=n_sources)
            else:
                results = self.rag.buscar_contexto(question, n_resultados=n_sources)
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            results = []

        if not results:
            logger.warning("RAG returned 0 results")
            return (
                "No se encontraron documentos relevantes en la base de conocimiento.",
                [],
            )

        logger.info(f"RAG {mode}: {len(results)} docs retrieved")
        for i, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source", "?")
            dist = r.get("distancia")
            dist_str = f" dist={dist:.3f}" if dist is not None else ""
            logger.debug(f"  [{i}] {src}{dist_str}")

        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "desconocido")
            text = r.get("texto", r.get("text", r.get("document", "")))
            parts.append(f"[{i}] Fuente: {source}\n{text}")

        return "\n\n---\n\n".join(parts), results

    # ── Prompt assembly ───────────────────────────────────────────────────────

    def build_messages(self, question: str, context: str, history: list) -> list:
        """Assemble messages list for chat template / API call."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in history[-8:]:
            role = turn["role"] if isinstance(turn, dict) else turn[0]
            content = turn["content"] if isinstance(turn, dict) else turn[1]
            text = _content_to_text(content)
            if text:
                messages.append({"role": role, "content": text})

        user_content = (
            f"Contexto regulatorio recuperado:\n{context}\n\nPregunta: {question}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── Response parsing ──────────────────────────────────────────────────────

    def parse_response(self, raw: str) -> dict:
        """
        Extract structured JSON from model output.
        Falls back to plain-text respuesta on parse failure.
        """
        try:
            # Strip markdown code fences
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            # Extract outermost {...}
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
                # Normalise keys
                result = {
                    "respuesta": str(data.get("respuesta", raw)),
                    "referencias": data.get("referencias", []),
                    "confianza": data.get("confianza"),
                    "justificacion_confianza": data.get("justificacion_confianza", ""),
                }
                logger.debug(
                    f"Response parsed as JSON ({len(result['respuesta'])} chars)"
                )
                return result
        except Exception as e:
            logger.debug(f"JSON parse failed ({e}); using raw text as respuesta")

        return {
            "respuesta": raw,
            "referencias": [],
            "confianza": None,
            "justificacion_confianza": "",
        }

    # ── Topic guard ───────────────────────────────────────────────────────────

    def check_topic(self, question: str) -> bool:
        """
        Return True if the question is on-topic (credit-risk banking regulation).

        Two-layer check:
          1. Regex — instantly rejects obvious creative/personal requests.
          2. Embedding guardrail — computes max cosine similarity against 20
             curated credit-risk/regulation seed sentences. Questions scoring
             below _TOPIC_GUARD_THRESHOLD are rejected regardless of phrasing.
        """
        # Layer 1: regex fast path
        if _OFFTOPIC_RE.search(question):
            logger.info("Topic guard: rejected by regex")
            return False

        # Layer 2: embedding similarity against regulatory seed sentences
        score = _topic_embedding_score(question)
        logger.info(
            f"Topic guard: embedding_score={score:.3f} threshold={_TOPIC_GUARD_THRESHOLD}"
        )
        if score < _TOPIC_GUARD_THRESHOLD:
            logger.info("Topic guard: rejected by embedding (off-topic)")
            return False

        return True

    # ── Reference enrichment + confidence from vector DB ─────────────────────

    def enrich_references(self, parsed: dict, question: str, top_k: int = 5) -> dict:
        """
        Replace model referencias entirely with CitationRAG hits (DB-only, no hallucination).
        Also computes confianza as the mean cosine similarity of the model answer
        against the top-k citation nodes — grounded in vector distance, not self-report.
        """
        if self.citation_rag is None:
            parsed["referencias"] = []
            parsed["confianza"] = None
            parsed["justificacion_confianza"] = ""
            return parsed

        # ── Step 1: citation hits for the question → referencias panel ────────
        try:
            q_hits = self.citation_rag.search(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"CitationRAG question search failed: {e}")
            q_hits = []

        parsed["referencias"] = [
            {
                "documento": h.get("documento", ""),
                "articulo": h.get("articulo", ""),
                "paragrafo": h.get("paragrafo", ""),
                "descripcion": h.get("text", "")[:120],
            }
            for h in q_hits
        ]

        # ── Step 2: confidence from answer ↔ citation similarity ──────────────
        answer_text = parsed.get("respuesta", "").strip()
        try:
            a_hits = (
                self.citation_rag.search(answer_text[:500], top_k=top_k)
                if answer_text
                else []
            )
        except Exception as e:
            logger.warning(f"CitationRAG answer search failed: {e}")
            a_hits = []

        if a_hits:
            scores = [h["score"] for h in a_hits if h.get("score") is not None]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            pct = max(0, min(100, int(round(mean_score * 100))))
            top = a_hits[0]
            top_ref = " · ".join(
                filter(
                    None,
                    [
                        top.get("documento", ""),
                        top.get("articulo", ""),
                        top.get("paragrafo", ""),
                    ],
                )
            )
            justif = (
                f"Similitud media respuesta↔base regulatoria: {mean_score:.2f}. "
                f"Referencia más próxima: {top_ref}."
            )
            parsed["confianza"] = pct
            parsed["justificacion_confianza"] = justif
        else:
            parsed["confianza"] = None
            parsed["justificacion_confianza"] = ""

        return parsed

    # ── Gradio markdown rendering ─────────────────────────────────────────────

    def render_message(self, parsed: dict) -> str:
        """
        Format a parsed structured response into stacked HTML card sections for Gradio chatbot.
        Three always-visible rows: Respuesta, Referencias, Confianza.
        """
        respuesta = parsed.get("respuesta", "").strip()
        referencias = parsed.get("referencias") or []
        confianza = parsed.get("confianza")
        justificacion = parsed.get("justificacion_confianza", "").strip()

        # Row 0 — Respuesta
        row0 = (
            '<div class="resp-section">'
            '<div class="resp-label">Respuesta</div>'
            f'<div class="resp-body">{respuesta}</div>'
            "</div>"
        )

        # Row 1 — Referencias
        if referencias:
            items_html = ""
            for ref in referencias:
                doc = ref.get("documento", "")
                art = ref.get("articulo", "")
                par = ref.get("paragrafo", "")
                desc = ref.get("descripcion", "")
                label = " · ".join(filter(None, [doc, art, par]))
                item = f"<strong>{label}</strong>" + (f" — {desc}" if desc else "")
                items_html += f'<div class="ref-item">• {item}</div>'
            row1 = (
                '<div class="resp-section resp-section--refs">'
                '<div class="resp-label">Referencias</div>'
                f'<div class="resp-body">{items_html}</div>'
                "</div>"
            )
        else:
            row1 = (
                '<div class="resp-section resp-section--refs">'
                '<div class="resp-label">Referencias</div>'
                '<div class="resp-body" style="color:#94a3b8;font-style:italic;">'
                "Sin referencias disponibles</div>"
                "</div>"
            )

        # Row 2 — Confianza
        if confianza is not None:
            pct = int(confianza)
            bar = (
                '<div class="conf-bar-wrap">'
                f'<div class="conf-fill" style="width:{pct}%"></div>'
                "</div>"
            )
            justif_html = (
                f'<div class="conf-justif">{justificacion}</div>'
                if justificacion
                else ""
            )
            row2 = (
                '<div class="resp-section resp-section--conf">'
                '<div class="resp-label">Confianza</div>'
                '<div class="resp-body">'
                f'<span class="conf-pct">{pct}%</span>'
                f"{bar}{justif_html}"
                "</div></div>"
            )
        else:
            row2 = (
                '<div class="resp-section resp-section--conf">'
                '<div class="resp-label">Confianza</div>'
                '<div class="resp-body" style="color:#94a3b8;font-style:italic;">'
                "No disponible</div>"
                "</div>"
            )

        return f'<div class="resp-card">{row0}{row1}{row2}</div>'

    # ── DB logging ────────────────────────────────────────────────────────────

    def log(self, question: str, answer: str, latency_ms: Optional[int] = None) -> None:
        """Fire-and-forget: log QA interaction to PostgreSQL (non-blocking)."""
        try:
            from src.db import log_qa_interaction, log_async

            log_async(
                log_qa_interaction(
                    question=question,
                    model_answer=answer,
                    category="live",
                    source=self.db_source,
                    latency_ms=latency_ms,
                )
            )
        except Exception as e:
            logger.warning(f"DB log skipped: {e}")


# ─── Shared helpers ───────────────────────────────────────────────────────────


def _content_to_text(content) -> str:
    """
    Normalise Gradio 6 content (list of content-dicts or plain string) to str.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
            if not isinstance(item, dict) or item.get("type") == "text"
        )
    return str(content)
