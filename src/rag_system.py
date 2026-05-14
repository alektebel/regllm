"""
RAG (Retrieval-Augmented Generation) System for Banking Regulatory Assistant.
Provides semantic search and hybrid search using PostgreSQL + pgvector.

ChromaDB has been replaced with pgvector stored in the same PostgreSQL instance,
eliminating the need for a separate EFS volume in production.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


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


# ─── Compat proxy ─────────────────────────────────────────────────────────────

class _CollectionProxy:
    """Keeps rag.collection.count() working for code that reads it."""

    def count(self) -> int:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM document_chunks")
                    return cur.fetchone()[0]
        except Exception:
            return 0


# ─── Main class ───────────────────────────────────────────────────────────────

class RegulatoryRAGSystem:
    """
    Sistema RAG completo para consultas regulatorias en español.
    Stores document chunks in PostgreSQL (pgvector) instead of ChromaDB.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        persist_directory: str = None,  # kept for API compat, ignored
    ):
        logger.info(f"Initializing RAG system — model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name, device="cpu")
        self.collection = _CollectionProxy()

        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.corpus_ids: List[str] = []

        try:
            self._build_bm25_index()
        except Exception as e:
            logger.warning(f"BM25 index not built at startup: {e}")

        logger.info(f"RAG system ready — {self.collection.count()} document chunks in DB")

    # ── Document ingestion ────────────────────────────────────────────────────

    def procesar_documentos(self, documentos: List[Dict[str, Any]]) -> int:
        textos, metadatas, ids = [], [], []

        for i, doc in enumerate(documentos):
            chunks = self._segmentar_documento(
                doc.get("text", doc.get("texto", "")),
                doc.get("metadata", doc),
            )
            for j, chunk in enumerate(chunks):
                chunk_id = (
                    f"{doc.get('metadata', doc).get('documento_id', doc.get('source', f'doc_{i}'))}_{j}"
                )
                if chunk_id not in ids:
                    textos.append(chunk["texto"])
                    metadatas.append(chunk["metadata"])
                    ids.append(chunk_id)

        if not textos:
            logger.warning("No texts to process")
            return 0

        logger.info(f"Generating embeddings for {len(textos)} chunks…")
        embeddings = self.embedder.encode(textos, show_progress_bar=True)

        rows = [
            (ids[i], textos[i], json.dumps(metadatas[i]), embeddings[i].tolist())
            for i in range(len(textos))
        ]

        with _get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO document_chunks (chunk_id, texto, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE
                      SET texto    = EXCLUDED.texto,
                          metadata = EXCLUDED.metadata,
                          embedding = EXCLUDED.embedding
                    """,
                    rows,
                    template="(%s, %s, %s::jsonb, %s::vector)",
                )
            conn.commit()

        logger.info(f"Upserted {len(textos)} chunks from {len(documentos)} documents")
        self._build_bm25_index()
        return len(textos)

    # ── Segmentation (unchanged logic) ───────────────────────────────────────

    def _segmentar_documento(self, texto: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        if not texto or len(texto.strip()) < 100:
            return chunks

        patron = r"(Art[ií]culo\s+\d+[a-z]?[.\s])"
        partes = re.split(patron, texto, flags=re.IGNORECASE)

        articulo_actual = None
        contenido_actual = ""

        for parte in partes:
            if re.match(patron, parte, re.IGNORECASE):
                if articulo_actual and contenido_actual.strip():
                    self._add_article_chunks(chunks, articulo_actual, contenido_actual, metadata)
                articulo_actual = parte.strip()
                contenido_actual = ""
            elif articulo_actual:
                contenido_actual += parte

        if articulo_actual and contenido_actual.strip():
            self._add_article_chunks(chunks, articulo_actual, contenido_actual, metadata)

        if not chunks:
            chunks = self._chunk_by_paragraphs(texto, metadata)

        return chunks

    def _add_article_chunks(self, chunks, articulo, contenido, metadata):
        max_chunk_size = 1500
        parrafos = contenido.split("\n")
        chunk_actual = f"{articulo}\n"

        for parrafo in parrafos:
            parrafo = parrafo.strip()
            if not parrafo or len(parrafo) < 20:
                continue
            if len(chunk_actual) + len(parrafo) > max_chunk_size:
                if len(chunk_actual) > 100:
                    meta = (metadata.copy() if isinstance(metadata, dict) else {})
                    meta["articulo"] = articulo
                    meta["longitud"] = len(chunk_actual)
                    chunks.append({"texto": chunk_actual.strip(), "metadata": meta})
                chunk_actual = f"{articulo}\n{parrafo}\n"
            else:
                chunk_actual += f"{parrafo}\n"

        if len(chunk_actual) > 100:
            meta = (metadata.copy() if isinstance(metadata, dict) else {})
            meta["articulo"] = articulo
            meta["longitud"] = len(chunk_actual)
            chunks.append({"texto": chunk_actual.strip(), "metadata": meta})

    def _chunk_by_paragraphs(self, texto, metadata):
        chunks = []
        max_chunk_size = 1500
        parrafos = texto.split("\n\n")
        chunk_actual = ""

        for parrafo in parrafos:
            parrafo = parrafo.strip()
            if not parrafo or len(parrafo) < 50:
                continue
            if len(chunk_actual) + len(parrafo) > max_chunk_size:
                if len(chunk_actual) > 100:
                    meta = (metadata.copy() if isinstance(metadata, dict) else {})
                    meta["longitud"] = len(chunk_actual)
                    chunks.append({"texto": chunk_actual.strip(), "metadata": meta})
                chunk_actual = parrafo + "\n\n"
            else:
                chunk_actual += parrafo + "\n\n"

        if len(chunk_actual) > 100:
            meta = (metadata.copy() if isinstance(metadata, dict) else {})
            meta["longitud"] = len(chunk_actual)
            chunks.append({"texto": chunk_actual.strip(), "metadata": meta})

        return chunks

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _build_bm25_index(self):
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, texto FROM document_chunks ORDER BY id")
                rows = cur.fetchall()

        if not rows:
            logger.warning("No documents in DB for BM25 indexing")
            return

        self.corpus_ids = [r[0] for r in rows]
        self.corpus = [r[1] for r in rows]
        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(self.corpus)} documents")

    # ── Search ────────────────────────────────────────────────────────────────

    def buscar_contexto(self, pregunta: str, n_resultados: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([pregunta])[0].tolist()

        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, texto, metadata,
                           (embedding <=> %s::vector) AS distance
                    FROM document_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_emb, q_emb, n_resultados),
                )
                rows = cur.fetchall()

        return [
            {
                "texto": r[1],
                "metadata": r[2] if r[2] else {},
                "distancia": float(r[3]) if r[3] is not None else None,
                "id": r[0],
            }
            for r in rows
        ]

    def buscar_hibrida(
        self, pregunta: str, n_resultados: int = 5, peso_semantico: float = 0.7
    ) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus:
            logger.warning("BM25 index not available, falling back to semantic search")
            return self.buscar_contexto(pregunta, n_resultados)

        resultados_semanticos = self.buscar_contexto(pregunta, n_resultados=n_resultados * 2)

        tokenized_query = pregunta.lower().split()
        scores_bm25 = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(scores_bm25) if max(scores_bm25) > 0 else 1

        scores_combinados = []
        for chunk in resultados_semanticos:
            try:
                idx = self.corpus.index(chunk["texto"])
                max_dist = (
                    max(c["distancia"] for c in resultados_semanticos if c["distancia"])
                    if resultados_semanticos
                    else 1
                )
                score_sem = (
                    1 - (chunk["distancia"] / max_dist)
                    if chunk["distancia"] and max_dist > 0
                    else 0.5
                )
                score_bm25_norm = scores_bm25[idx] / max_bm25
                score_final = peso_semantico * score_sem + (1 - peso_semantico) * score_bm25_norm
                scores_combinados.append({"chunk": chunk, "score": score_final})
            except ValueError:
                scores_combinados.append({"chunk": chunk, "score": 0.5})

        scores_combinados.sort(key=lambda x: x["score"], reverse=True)
        return [s["chunk"] for s in scores_combinados[:n_resultados]]

    def formatear_contexto(self, chunks: List[Dict[str, Any]]) -> str:
        partes = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            fuente = f"[{meta.get('documento', meta.get('source', 'Desconocido'))} {meta.get('articulo', '')}]"
            partes.append(f"Fuente {i}: {fuente}\n\"{chunk['texto']}\"\n")
        return "\n".join(partes)

    def load_from_json(self, json_path: str) -> int:
        import json as _json
        logger.info(f"Loading documents from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            documents = _json.load(f)

        formatted = [
            {
                "texto": doc.get("text", ""),
                "metadata": {
                    "documento": doc.get("title", "Unknown"),
                    "documento_id": doc.get("url", "unknown"),
                    "source": doc.get("source", "Unknown"),
                    "tipo": doc.get("type", "unknown"),
                    "url": doc.get("url", ""),
                    "keywords": doc.get("keywords", []),
                },
            }
            for doc in documents
        ]
        return self.procesar_documentos(formatted)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": self.collection.count(),
            "embedding_model": str(self.embedder),
            "bm25_indexed": self.bm25 is not None,
            "corpus_size": len(self.corpus),
        }

    def clear_collection(self):
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks")
            conn.commit()
        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []
        logger.info("document_chunks table cleared")


class HybridSearch:
    """Wrapper kept for API compatibility."""

    def __init__(self, rag_system: RegulatoryRAGSystem):
        self.rag_system = rag_system

    def search(self, query: str, n_results: int = 5, semantic_weight: float = 0.7):
        return self.rag_system.buscar_hibrida(query, n_results, semantic_weight)


def create_rag_system(embedding_model: str = None, persist_dir: str = None) -> RegulatoryRAGSystem:
    kwargs = {}
    if embedding_model:
        kwargs["embedding_model_name"] = embedding_model
    return RegulatoryRAGSystem(**kwargs)
