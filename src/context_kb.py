"""Context Knowledge Base — ingests field dictionaries, emails, notes, and table
metadata into pgvector, providing domain context for the compliance checker.

This is separate from the regulation RAG (which indexes EBA/CRR documents).
It answers questions like: "What does PROVISION_PERIOD_MONTHS mean in this
institution's data model? Are there known quirks?"

Two ingestion modes:
  - Structured (CSV): each row → one embedding (field name + description + metadata)
  - Unstructured (TXT/PDF): chunk → SLM extracts facts → embed chunk + facts

Usage:
    kb = ContextKB(rag_system=rag, llm_fn=llm_fn)
    kb.ingest_directory(Path("data/samples/"))
    context = kb.get_record_context({"PD_ESTIMADA": 0.001, "CICLO_ID": "C001"})
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

# pgvector table for context chunks (separate from regulatory corpus)
_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS context_chunks (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(768)
);
CREATE INDEX IF NOT EXISTS context_chunks_embedding_idx
    ON context_chunks USING hnsw (embedding vector_cosine_ops);
"""

_FACT_EXTRACTION_PROMPT = """\
Extrae hechos estructurados del siguiente texto como JSON.
Responde SOLO con JSON, sin texto adicional.
Formato: {{"facts": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
Máximo 8 hechos. Céntrate en definiciones de campos, reglas de negocio y anomalías conocidas.

TEXTO:
{text}
"""

_CHUNK_SIZE = 400   # tokens (~1600 chars)
_CHUNK_OVERLAP = 80


class ContextKB:
    """
    Domain context knowledge base stored in pgvector.

    Args:
        rag_system: RegulatoryRAGSystem (reused for embeddings + DB connection)
        llm_fn: callable(messages) → str for fact extraction from unstructured text.
                If None, only raw chunks are embedded (no fact extraction).
        table: pgvector table name (default: context_chunks)
    """

    def __init__(self, rag_system=None, llm_fn=None, table: str = "context_chunks"):
        self.rag = rag_system
        self.llm_fn = llm_fn
        self.table = table
        self._ensure_table()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_directory(self, path: Path | str) -> int:
        """Ingest all supported files under path. Returns total chunks stored."""
        path = Path(path)
        total = 0
        for f in sorted(path.rglob("*")):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext == ".csv":
                total += self._ingest_csv(f)
            elif ext in (".txt", ".md"):
                total += self._ingest_text(f)
            elif ext == ".pdf":
                total += self._ingest_pdf(f)
        logger.info("ContextKB: ingested %d chunks from %s", total, path)
        return total

    def get_field_context(self, field_name: str, n: int = 3) -> str:
        """Retrieve context for a specific field name."""
        query = f"campo {field_name} definición descripción dominio valores"
        return self._search(query, n, filter_field=field_name)

    def get_record_context(self, record: dict, n_per_field: int = 2) -> str:
        """Retrieve context for all fields in a data record."""
        parts = []
        for field_name in list(record.keys())[:8]:  # cap at 8 fields
            ctx = self.get_field_context(field_name, n=n_per_field)
            if ctx and "(sin contexto)" not in ctx:
                parts.append(f"[{field_name}] {ctx}")
        return "\n".join(parts) if parts else "(sin contexto de dominio disponible)"

    # ── Ingestion: structured CSV ─────────────────────────────────────────────

    def _ingest_csv(self, path: Path) -> int:
        import csv
        count = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return 0
            rows = list(reader)

        # Detect if this is a field dictionary (has field_name + description cols)
        cols_lower = [c.lower() for c in (reader.fieldnames or [])]
        is_field_dict = any(c in cols_lower for c in ("field_name", "campo", "column_name", "columna"))

        for row in rows:
            text = self._csv_row_to_text(row, is_field_dict)
            field_name = None
            for k in row:
                if k.lower() in ("field_name", "campo", "column_name", "columna"):
                    field_name = row[k].strip().upper()
                    break
            metadata = {
                "source": str(path.name),
                "source_type": "field_dict" if is_field_dict else "table_metadata",
                **({"field_name": field_name} if field_name else {}),
            }
            self._store(text, metadata)
            count += 1

        logger.info("ContextKB: %d rows from %s", count, path.name)
        return count

    def _csv_row_to_text(self, row: dict, is_field_dict: bool) -> str:
        if is_field_dict:
            parts = []
            for k, v in row.items():
                if v and v.strip():
                    parts.append(f"{k}: {v.strip()}")
            return ". ".join(parts)
        # Generic: key=value pairs
        return " | ".join(f"{k}={v}" for k, v in row.items() if v and str(v).strip())

    # ── Ingestion: unstructured text ──────────────────────────────────────────

    def _ingest_text(self, path: Path) -> int:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_text(text)
        count = 0
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": str(path.name),
                "source_type": "email" if "email" in path.parent.name.lower() else "notes",
                "chunk_index": i,
            }
            # Store raw chunk
            self._store(chunk, metadata)
            count += 1

            # Extract facts via SLM and store them too
            facts = self._extract_facts(chunk)
            for fact in facts:
                fact_text = f"{fact.get('subject','')} {fact.get('predicate','')} {fact.get('object','')}"
                if len(fact_text.strip()) > 10:
                    self._store(fact_text, {**metadata, "source_type": "extracted_fact"})
                    count += 1

        logger.info("ContextKB: %d chunks+facts from %s", count, path.name)
        return count

    def _ingest_pdf(self, path: Path) -> int:
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.warning("PDF read failed for %s: %s", path, e)
            return 0
        # Treat as plain text after extraction
        tmp = path.with_suffix(".txt")
        tmp.write_text(text, encoding="utf-8")
        count = self._ingest_text(tmp)
        tmp.unlink()
        return count

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = _CHUNK_SIZE - _CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + _CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _extract_facts(self, text: str) -> list[dict]:
        if self.llm_fn is None:
            return []
        prompt = _FACT_EXTRACTION_PROMPT.format(text=text[:1200])
        try:
            raw = self.llm_fn([{"role": "user", "content": prompt}])
            raw = raw.strip()
            # Strip markdown fences
            for fence in ("```json", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
            if raw.endswith("```"):
                raw = raw[:-3]
            data = json.loads(raw.strip())
            return data.get("facts", [])
        except Exception as e:
            logger.debug("Fact extraction failed: %s", e)
            return []

    # ── Search ────────────────────────────────────────────────────────────────

    def _search(self, query: str, n: int, filter_field: str | None = None) -> str:
        if self.rag is None:
            return "(ContextKB sin RAG)"
        try:
            embedding = self.rag._embed([query])[0]
            conn = self.rag._get_connection()
            cur = conn.cursor()
            if filter_field:
                cur.execute(
                    f"""
                    SELECT content FROM {self.table}
                    WHERE metadata->>'field_name' ILIKE %s
                       OR content ILIKE %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (filter_field, f"%{filter_field}%", embedding, n),
                )
            else:
                cur.execute(
                    f"SELECT content FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                    (embedding, n),
                )
            rows = cur.fetchall()
            conn.close()
            if not rows:
                return "(sin contexto)"
            return "\n".join(r[0] for r in rows)
        except Exception as e:
            logger.warning("ContextKB search failed: %s", e)
            return "(error en búsqueda de contexto)"

    # ── Storage ───────────────────────────────────────────────────────────────

    def _store(self, text: str, metadata: dict) -> None:
        if self.rag is None:
            return
        try:
            embedding = self.rag._embed([text])[0]
            conn = self.rag._get_connection()
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO {self.table} (content, metadata, embedding) VALUES (%s, %s, %s)",
                (text, json.dumps(metadata), embedding),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("ContextKB store failed: %s", e)

    def _ensure_table(self) -> None:
        if self.rag is None:
            return
        try:
            conn = self.rag._get_connection()
            cur = conn.cursor()
            cur.execute(_TABLE_DDL)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("ContextKB table init failed: %s", e)
