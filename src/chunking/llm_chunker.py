"""
LLM-based regulation chunker.

Instead of a sliding window over article text, this module:
  1. Uses regex to split a regulation document into rough article segments
     (fast, handles 95% of Artículo / Article patterns correctly)
  2. Sends batches of 4 segments to a local Ollama model, which returns
     clean JSON: {article_number, title, content} with PDF artifacts removed
  3. Falls back to the original regex+sliding-window chunker if the LLM
     is unavailable or returns bad output

The result is one chunk per article — clean, deduped, no overlapping windows —
which is what the assertion extractor needs to avoid duplicate rules.

Usage:
    from src.chunking.llm_chunker import chunk_regulation

    chunks = chunk_regulation(raw_text, source_metadata, model="qwen2.5:14b-instruct-q4_K_M")
    # chunks: [{"texto": "...", "metadata": {..., "articulo": "Artículo 160", ...}}, ...]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Article boundary patterns (ES + EN) ──────────────────────────────────────
# Ordered from most to least specific so the first match wins.
_ARTICLE_PATTERNS = [
    re.compile(r"(Art[ií]culo\s+\d+[a-z]?(?:\s+bis)?\b[.\s])", re.IGNORECASE),  # ES EUR-Lex
    re.compile(r"(Article\s+\d+[a-z]?\b[.\s])", re.IGNORECASE),                  # EN EBA PDFs
    re.compile(r"(Sección\s+\d+[.\s])", re.IGNORECASE),                           # ES sections
    re.compile(r"(Section\s+\d+[.\s])", re.IGNORECASE),                           # EN sections
]

_HIERARCHY = re.compile(
    r"(T[ií]tulo\s+[IVXLCDM\d]+[.\s][^\n]*|"
    r"Cap[ií]tulo\s+[IVXLCDM\d]+[.\s][^\n]*|"
    r"Secci[oó]n\s+\d+[.\s][^\n]*|"
    r"Title\s+[IVXLCDM\d]+[.\s][^\n]*|"       # EN Title
    r"Chapter\s+[IVXLCDM\d]+[.\s][^\n]*|"     # EN Chapter
    r"Section\s+\d+[.\s][^\n]*)",              # EN Section
    re.IGNORECASE,
)

# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM = """\
Eres un procesador de textos regulatorios. Tu única tarea es limpiar y \
estructurar artículos de regulación bancaria de la UE.

Para cada artículo proporcionado, devuelve un objeto JSON con:
- "article_number": número de artículo exacto (ej. "Artículo 160", "Article 4")
- "title": título del artículo si está presente, si no cadena vacía
- "content": texto íntegro del artículo limpio (sin números de página, cabeceras, \
  pies de página ni artefactos de PDF; conserva TODOS los requisitos normativos, \
  números y condiciones)

Responde ÚNICAMENTE con un array JSON válido.\
"""

_USER_TEMPLATE = """\
Procesa los siguientes {n} artículos. Devuelve un objeto JSON con clave "articles" \
que contenga un array de {n} objetos (uno por artículo).

{articles_block}

Formato de respuesta:
{{"articles": [{{"article_number": "...", "title": "...", "content": "..."}}]}}\
"""


def _build_articles_block(segments: list[tuple[str, str]]) -> str:
    """Format (header, body) pairs for the prompt."""
    parts = []
    for i, (header, body) in enumerate(segments, 1):
        parts.append(f"=== Artículo {i} (cabecera: {header.strip()!r}) ===\n{body[:2000]}")
    return "\n\n".join(parts)


# ── Regex-based fallback chunker ──────────────────────────────────────────────

def _regex_split(text: str) -> list[tuple[str, str]]:
    """Split text into (article_header, article_body) pairs using regex."""
    for pat in _ARTICLE_PATTERNS:
        parts = pat.split(text)
        if len(parts) > 3:  # at least 2 articles found
            segments = []
            for i in range(1, len(parts) - 1, 2):
                header = parts[i]
                body = parts[i + 1] if i + 1 < len(parts) else ""
                segments.append((header, body))
            return segments
    # No article patterns found — treat entire text as one block
    return [("(documento)", text)]


def _regex_chunks_only(text: str, metadata: dict) -> list[dict]:
    """Pure regex chunking — returns one dict per article, no sliding window."""
    segments = _regex_split(text)
    results = []
    hier = {"titulo": "", "capitulo": "", "seccion": ""}

    for header, body in segments:
        # Update hierarchy markers from body text
        for m in _HIERARCHY.finditer(body):
            txt = m.group(1).strip()[:80]
            t = txt.lower()
            if "tít" in t or "tit" in t:
                hier["titulo"] = txt
            elif "cap" in t:
                hier["capitulo"] = txt
            elif "secc" in t or "sec" in t:
                hier["seccion"] = txt

        article_text = f"{header.strip()} {body.strip()}"
        if len(article_text.strip()) < 80:
            continue

        meta = dict(metadata)
        meta.update({
            "articulo":  header.strip(),
            "titulo":    hier["titulo"],
            "capitulo":  hier["capitulo"],
            "seccion":   hier["seccion"],
            "longitud":  len(article_text),
            "chunker":   "regex_fallback",
        })
        results.append({"texto": article_text.strip(), "metadata": meta})

    return results


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_ollama(segments: list[tuple[str, str]], model: str) -> list[dict] | None:
    """Send a batch to Ollama. Returns list of parsed article dicts, or None on failure."""
    try:
        import ollama  # type: ignore
    except ImportError:
        logger.warning("ollama package not installed — falling back to regex chunker")
        return None

    articles_block = _build_articles_block(segments)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER_TEMPLATE.format(
            n=len(segments), articles_block=articles_block
        )},
    ]
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": 0.0, "num_predict": 4096},
        )
        raw = resp["message"]["content"]
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return None

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.debug("LLM output not parseable: %r", raw[:300])
                return None
        else:
            return None

    # Unwrap envelope {"articles": [...]} or bare list or bare single object
    if isinstance(parsed, dict):
        for key in ("articles", "artículos", "articulos", "items", "results"):
            if key in parsed:
                parsed = parsed[key]
                break
        else:
            # Single article returned as a dict — wrap it
            if "article_number" in parsed or "content" in parsed:
                parsed = [parsed]
            else:
                return None

    if not isinstance(parsed, list):
        return None

    return parsed


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_regulation(
    text: str,
    metadata: dict[str, Any],
    model: str = "qwen2.5:14b-instruct-q4_K_M",
    batch_size: int = 4,
    fallback_on_error: bool = True,
) -> list[dict[str, Any]]:
    """
    Split a regulation document into one clean chunk per article.

    Uses an Ollama model to clean text and extract article structure.
    Falls back to pure regex chunking if the model is unavailable.

    Args:
        text:              Raw regulation text (from fetch_source).
        metadata:          Source metadata dict (documento_id, source, framework, etc.).
        model:             Ollama model name.
        batch_size:        Articles per LLM call (4 is a good balance).
        fallback_on_error: If True, use regex chunker when LLM fails.

    Returns:
        List of {"texto": ..., "metadata": {..., "articulo": ..., "chunker": ...}}
    """
    segments = _regex_split(text)
    logger.info(
        "llm_chunker: %d article segments found in %s",
        len(segments), metadata.get("documento_id", "?"),
    )

    if not segments:
        return []

    results: list[dict] = []
    hier = {"titulo": "", "capitulo": "", "seccion": ""}
    failed_batches = 0

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start: batch_start + batch_size]
        llm_articles = _call_ollama(batch, model)

        if llm_articles is None or len(llm_articles) != len(batch):
            # LLM failed or returned wrong count — fall back for this batch
            failed_batches += 1
            if fallback_on_error:
                for header, body in batch:
                    for m in _HIERARCHY.finditer(body):
                        txt = m.group(1).strip()[:80]
                        t = txt.lower()
                        if "tít" in t or "tit" in t:
                            hier["titulo"] = txt
                        elif "cap" in t:
                            hier["capitulo"] = txt
                        elif "secc" in t or "sec" in t:
                            hier["seccion"] = txt
                    article_text = f"{header.strip()} {body.strip()}"
                    if len(article_text.strip()) < 80:
                        continue
                    meta = dict(metadata)
                    meta.update({
                        "articulo": header.strip(),
                        **hier,
                        "longitud": len(article_text),
                        "chunker":  "regex_fallback",
                    })
                    results.append({"texto": article_text.strip(), "metadata": meta})
            continue

        for (header, body), art in zip(batch, llm_articles):
            # Update hierarchy from raw body (LLM may have stripped these)
            for m in _HIERARCHY.finditer(body):
                txt = m.group(1).strip()[:80]
                t = txt.lower()
                if "tít" in t or "tit" in t:
                    hier["titulo"] = txt
                elif "cap" in t:
                    hier["capitulo"] = txt
                elif "secc" in t or "sec" in t:
                    hier["seccion"] = txt

            content = str(art.get("content", "")).strip()
            if not content or len(content) < 50:
                continue

            article_num = str(art.get("article_number", header)).strip()
            title = str(art.get("title", "")).strip()

            meta = dict(metadata)
            meta.update({
                "articulo":  article_num,
                "titulo":    hier["titulo"],
                "capitulo":  hier["capitulo"],
                "seccion":   hier["seccion"],
                "art_title": title,
                "longitud":  len(content),
                "chunker":   "llm",
            })
            results.append({"texto": content, "metadata": meta})

        logger.info(
            "  batch %d–%d → %d chunks  (total so far: %d)",
            batch_start + 1, batch_start + len(batch), len(llm_articles), len(results),
        )

    if failed_batches:
        logger.warning(
            "llm_chunker: %d/%d batches fell back to regex",
            failed_batches, (len(segments) + batch_size - 1) // batch_size,
        )

    logger.info(
        "llm_chunker: %d clean article chunks from %s",
        len(results), metadata.get("documento_id", "?"),
    )
    return results
