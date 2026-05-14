"""
CitationRAG — per-article/paragraph citation vector database.

Each node in the regulation tree (or chunk in a raw regulation doc) is stored
as a separate vector.  At query time, the question is embedded and the closest
citations are returned, independently of the main RAG retrieval.

Backed by PostgreSQL + pgvector (citation_chunks table, dim=384).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _conn_kwargs() -> dict:
    return dict(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


def _get_conn():
    conn = psycopg2.connect(**_conn_kwargs())
    register_vector(conn)
    return conn


# ─── Main class ───────────────────────────────────────────────────────────────

class CitationRAG:
    """
    Vector store for regulation citations (one vector per article/paragraph).
    Stores embeddings in the citation_chunks table (dim=384).

    Usage:
        crag = CitationRAG()
        crag.index_citation_tree("data/citation_trees/eu_regulations.json")
        hits = crag.search("CET1 ratio requirements", top_k=5)
    """

    def __init__(self, chroma_path: str = None, embed_model: str = EMBED_MODEL):
        # chroma_path kept for API compat, ignored
        self.encoder = SentenceTransformer(embed_model, device="cpu")
        logger.info(f"CitationRAG ready — {self.count()} citation chunks in DB")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_citation_tree(self, json_path: str) -> int:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Citation tree not found: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            tree = json.load(f)

        nodes = tree.get("nodes", {})
        texts, metadatas, ids = [], [], []

        for node_id, node in nodes.items():
            text = node.get("text", "").strip()
            if len(text) < 30:
                continue

            meta_raw = node.get("metadata", {})
            reference = node.get("reference", node_id)
            documento, articulo, paragrafo = _parse_reference(reference, node.get("type", ""))

            meta = {
                "reference": reference,
                "documento": documento,
                "articulo": articulo,
                "paragrafo": paragrafo,
                "source_type": "tree",
                "url": meta_raw.get("url", ""),
                "language": meta_raw.get("language", "en"),
                "node_type": node.get("type", ""),
            }
            texts.append(text)
            metadatas.append(meta)
            ids.append(f"tree_{node_id}")

        return self._batch_add(texts, metadatas, ids, source_label=str(path.name))

    def index_regulation_doc(self, doc_path: str, doc_name: str) -> int:
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Doc not found: {doc_path}")

        with open(path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        texts, metadatas, ids = [], [], []

        for doc_idx, doc in enumerate(documents):
            raw_text = doc.get("text", doc.get("texto", ""))
            url = doc.get("url", "")

            chunks = _segment_by_article(raw_text)
            for chunk_idx, (articulo, chunk_text) in enumerate(chunks):
                if len(chunk_text) < 50:
                    continue
                item_id = f"doc_{doc_name}_{doc_idx}_{chunk_idx}"
                if item_id in ids:
                    continue
                meta = {
                    "reference": f"{doc_name} {articulo}".strip(),
                    "documento": doc_name,
                    "articulo": articulo,
                    "paragrafo": "",
                    "source_type": "doc_chunk",
                    "url": url,
                    "language": "es",
                }
                texts.append(chunk_text[:1500])
                metadatas.append(meta)
                ids.append(item_id)

        return self._batch_add(texts, metadatas, ids, source_label=doc_name)

    def _batch_add(self, texts: list, metadatas: list, ids: list, source_label: str = "") -> int:
        if not texts:
            logger.warning(f"No items to index from {source_label}")
            return 0

        # Skip IDs that already exist
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id FROM citation_chunks WHERE chunk_id = ANY(%s)", (ids,)
                )
                existing = {r[0] for r in cur.fetchall()}

        new_texts, new_metas, new_ids = [], [], []
        for t, m, i in zip(texts, metadatas, ids):
            if i not in existing:
                new_texts.append(t)
                new_metas.append(m)
                new_ids.append(i)

        if not new_texts:
            logger.info(f"{source_label}: all {len(ids)} items already indexed, skipping")
            return 0

        logger.info(f"Embedding {len(new_texts)} new items from {source_label}…")
        embeddings = self.encoder.encode(new_texts, show_progress_bar=True, batch_size=64)

        rows = [
            (new_ids[i], new_texts[i], json.dumps(new_metas[i]), embeddings[i].tolist())
            for i in range(len(new_texts))
        ]

        with _get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO citation_chunks (chunk_id, texto, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    rows,
                    template="(%s, %s, %s::jsonb, %s::vector)",
                )
            conn.commit()

        logger.info(f"Added {len(new_texts)} items from {source_label}")
        return len(new_texts)

    def clear(self) -> None:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM citation_chunks")
            conn.commit()
        logger.info("citation_chunks table cleared")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        if self.count() == 0:
            return []

        q_emb = self.encoder.encode([query_text])[0].tolist()

        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT texto, metadata,
                           (embedding <=> %s::vector) AS distance
                    FROM citation_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_emb, q_emb, top_k),
                )
                rows = cur.fetchall()

        hits = []
        for doc_text, meta, dist in rows:
            meta = meta or {}
            score = round(1.0 - float(dist), 4) if dist is not None else None
            hits.append({
                "reference": meta.get("reference", ""),
                "documento": meta.get("documento", ""),
                "articulo": meta.get("articulo", ""),
                "paragrafo": meta.get("paragrafo", ""),
                "text": doc_text,
                "url": meta.get("url", ""),
                "score": score,
            })

        return hits

    def count(self) -> int:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM citation_chunks")
                    return cur.fetchone()[0]
        except Exception:
            return 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_reference(reference: str, node_type: str) -> tuple[str, str, str]:
    reference = reference.strip()

    par_match = re.search(r"(§[\w()\-\.]+)", reference)
    paragrafo = par_match.group(1) if par_match else ""

    art_match = re.search(r"(Art(?:icle|ículo|\.)\s*[\d\w]+)", reference, re.IGNORECASE)
    articulo = art_match.group(1) if art_match else ""

    if art_match:
        documento = reference[: art_match.start()].strip().rstrip(",- ")
    elif reference:
        parts = reference.split()
        documento = parts[0] if parts else reference
    else:
        documento = reference

    if not articulo and node_type in ("article", "paragraph", "point"):
        articulo = reference

    return documento, articulo, paragrafo


def _segment_by_article(text: str) -> list[tuple[str, str]]:
    if not text or len(text.strip()) < 50:
        return []

    patron = r"(Art[ií]culo\s+\d+[a-z]?[.\s])"
    partes = re.split(patron, text, flags=re.IGNORECASE)

    chunks = []
    current_art = ""
    current_text = ""

    for parte in partes:
        if re.match(patron, parte, re.IGNORECASE):
            if current_text.strip():
                chunks.append((current_art, current_text.strip()))
            current_art = parte.strip()
            current_text = ""
        else:
            current_text += parte

    if current_text.strip():
        chunks.append((current_art, current_text.strip()))

    if not chunks:
        chunks = [("", text.strip())]

    return chunks
