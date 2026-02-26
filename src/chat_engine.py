"""
ChatEngine — shared query pipeline for RegLLM apps.

Extracts context building, prompt assembly, response parsing, reference
enrichment, and Gradio rendering logic so all app variants (local, groq,
ollama) share the same pipeline and only differ in their generate() call.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Structured system prompt ────────────────────────────────────────────────

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
    gap: 12px;
    margin-bottom: 0;
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

SYSTEM_PROMPT = """Eres un experto asistente en regulación bancaria y riesgo de crédito.
Responde SIEMPRE en español y SIEMPRE en el siguiente formato JSON (sin markdown, sin texto fuera del JSON):

{
  "respuesta": "Texto completo de la respuesta en español...",
  "referencias": [
    {"documento": "CRR", "articulo": "Art. 92", "paragrafo": "§1(a)", "descripcion": "Capital requirements ratio"}
  ],
  "confianza": 85,
  "justificacion_confianza": "Alta confianza porque el artículo está directamente citado en las fuentes."
}

Nunca inventes artículos o normativas. Si no sabes, refléjalo en la confianza y justificación.
Si no encuentras referencias relevantes, devuelve "referencias": []."""


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

    def build_context(self, question: str, n_sources: int = 5, hybrid: bool = True) -> tuple[str, list]:
        """Run RAG retrieval; return (context_string, raw_results)."""
        try:
            if hybrid:
                results = self.rag.buscar_hibrida(question, n_resultados=n_sources)
            else:
                results = self.rag.buscar_contexto(question, n_resultados=n_sources)
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            results = []

        if not results:
            return "No se encontraron documentos relevantes en la base de conocimiento.", []

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

        user_content = f"Contexto regulatorio recuperado:\n{context}\n\nPregunta: {question}"
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
                return {
                    "respuesta": str(data.get("respuesta", raw)),
                    "referencias": data.get("referencias", []),
                    "confianza": data.get("confianza"),
                    "justificacion_confianza": data.get("justificacion_confianza", ""),
                }
        except Exception as e:
            logger.debug(f"JSON parse failed ({e}); using raw text as respuesta")

        return {
            "respuesta": raw,
            "referencias": [],
            "confianza": None,
            "justificacion_confianza": "",
        }

    # ── Reference enrichment via citation RAG ─────────────────────────────────

    def enrich_references(self, parsed: dict, question: str, top_k: int = 5) -> dict:
        """
        Augment model-extracted referencias with vector-matched citation hits.
        Deduplicates by (documento, articulo) pair.
        """
        if self.citation_rag is None:
            return parsed

        try:
            hits = self.citation_rag.search(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"Citation RAG search failed: {e}")
            return parsed

        existing_keys = {
            (r.get("documento", ""), r.get("articulo", ""))
            for r in parsed["referencias"]
        }

        for hit in hits:
            key = (hit.get("documento", ""), hit.get("articulo", ""))
            if key not in existing_keys:
                parsed["referencias"].append({
                    "documento": hit.get("documento", ""),
                    "articulo": hit.get("articulo", ""),
                    "paragrafo": hit.get("paragrafo", ""),
                    "descripcion": hit.get("text", "")[:120],
                })
                existing_keys.add(key)

        return parsed

    # ── Gradio markdown rendering ─────────────────────────────────────────────

    def render_message(self, parsed: dict) -> str:
        """
        Format a parsed structured response into stacked HTML card sections for Gradio chatbot.
        Three always-visible rows: Respuesta, Referencias, Confianza.
        """
        respuesta    = parsed.get("respuesta", "").strip()
        referencias  = parsed.get("referencias") or []
        confianza    = parsed.get("confianza")
        justificacion = parsed.get("justificacion_confianza", "").strip()

        # Row 0 — Respuesta
        row0 = (
            '<div class="resp-section">'
            '<div class="resp-label">Respuesta</div>'
            f'<div class="resp-body">{respuesta}</div>'
            '</div>'
        )

        # Row 1 — Referencias
        if referencias:
            items_html = ""
            for ref in referencias:
                doc   = ref.get("documento", "")
                art   = ref.get("articulo", "")
                par   = ref.get("paragrafo", "")
                desc  = ref.get("descripcion", "")
                label = " · ".join(filter(None, [doc, art, par]))
                item  = f"<strong>{label}</strong>" + (f" — {desc}" if desc else "")
                items_html += f'<div class="ref-item">• {item}</div>'
            row1 = (
                '<div class="resp-section resp-section--refs">'
                '<div class="resp-label">Referencias</div>'
                f'<div class="resp-body">{items_html}</div>'
                '</div>'
            )
        else:
            row1 = (
                '<div class="resp-section resp-section--refs">'
                '<div class="resp-label">Referencias</div>'
                '<div class="resp-body" style="color:#94a3b8;font-style:italic;">'
                'Sin referencias disponibles</div>'
                '</div>'
            )

        # Row 2 — Confianza
        if confianza is not None:
            pct = int(confianza)
            bar = (
                '<div class="conf-bar-wrap">'
                f'<div class="conf-fill" style="width:{pct}%"></div>'
                '</div>'
            )
            justif_html = f'<div class="conf-justif">{justificacion}</div>' if justificacion else ""
            row2 = (
                '<div class="resp-section resp-section--conf">'
                '<div class="resp-label">Confianza</div>'
                '<div class="resp-body">'
                f'<span class="conf-pct">{pct}%</span>'
                f'{bar}{justif_html}'
                '</div></div>'
            )
        else:
            row2 = (
                '<div class="resp-section resp-section--conf">'
                '<div class="resp-label">Confianza</div>'
                '<div class="resp-body" style="color:#94a3b8;font-style:italic;">'
                'No disponible</div>'
                '</div>'
            )

        return f'<div class="resp-card">{row0}{row1}{row2}</div>'

    # ── DB logging ────────────────────────────────────────────────────────────

    def log(self, question: str, answer: str, latency_ms: Optional[int] = None) -> None:
        """Fire-and-forget: log QA interaction to PostgreSQL (non-blocking)."""
        try:
            from src.db import log_qa_interaction, log_async
            log_async(log_qa_interaction(
                question=question,
                model_answer=answer,
                category="live",
                source=self.db_source,
                latency_ms=latency_ms,
            ))
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
