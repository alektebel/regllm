"""
RAG (Retrieval-Augmented Generation) System for Banking Regulatory Assistant.
Provides semantic search and hybrid search using PostgreSQL + pgvector.

ChromaDB has been replaced with pgvector stored in the same PostgreSQL instance,
eliminating the need for a separate EFS volume in production.
"""

import json
import logging
import os
import pickle
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── RAG improvement constants ─────────────────────────────────────────────────

# RAG-1: Cross-encoder reranker (lazy-loaded on first use)
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None
_reranker_lock = threading.Lock()

# RAG-4: BM25 disk cache path (avoids full DB scan on every restart)
_BM25_CACHE_PATH = os.getenv("BM25_CACHE_PATH", "/tmp/regllm_bm25.pkl")


def _get_reranker(model_name: str = _RERANKER_MODEL):
    """Lazy-load the cross-encoder reranker once per process."""
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(model_name, device="cpu")
            logger.info("Cross-encoder reranker loaded: %s", model_name)
    return _reranker


def _load_bm25_cache(path: str) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_bm25_cache(path: str, data: dict) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug("BM25 cache saved to %s (%d docs)", path, data.get("count", 0))
    except Exception as e:
        logger.warning("Could not save BM25 cache: %s", e)

# ── Hierarchy detection patterns (module-level) ───────────────────────────────

_TITULO_PAT = re.compile(r"(T[ií]tulo\s+[IVXLCDM\d]+[.\s][^\n]*)", re.IGNORECASE)
_CAPITULO_PAT = re.compile(r"(Cap[ií]tulo\s+[IVXLCDM\d]+[.\s][^\n]*)", re.IGNORECASE)
_SECCION_PAT = re.compile(r"(Secci[oó]n\s+\d+[.\s][^\n]*)", re.IGNORECASE)
_ARTICULO_PAT = re.compile(r"(Art[ií]culo\s+\d+[a-z]?(?:\s+bis)?[.\s])", re.IGNORECASE)


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
        bm25_cache_path: str = _BM25_CACHE_PATH,
        build_bm25_on_init: bool = True,
    ):
        logger.info(f"Initializing RAG system — model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name, device="cpu")
        self.collection = _CollectionProxy()
        self._bm25_cache_path = bm25_cache_path

        self.bm25: Optional[BM25Okapi] = None
        self.corpus_ids: List[str] = []
        self.corpus_id_to_idx: Dict[str, int] = {}

        if not build_bm25_on_init:
            logger.info("Skipping BM25 build at init (will rebuild explicitly after ingestion)")
            return

        try:
            self._build_bm25_index()
        except Exception as e:
            logger.warning(f"BM25 index not built at startup: {e}")

        logger.info(f"RAG system ready — {self.collection.count()} document chunks in DB")

    # ── Document ingestion ────────────────────────────────────────────────────

    def procesar_documentos(
        self,
        documentos: List[Dict[str, Any]],
        embed_batch_size: int = 32,
        rebuild_bm25: bool = True,
    ) -> int:
        """Ingest documents into pgvector with RAM-bounded batched embedding.

        Chunks are encoded and inserted in batches of `embed_batch_size` so that
        peak RAM is O(batch) rather than O(total_chunks).
        """
        total_upserted = 0
        seen_ids: set = set()

        for doc_i, doc in enumerate(documentos):
            chunks = self._segmentar_documento(
                doc.get("text", doc.get("texto", "")),
                doc.get("metadata", doc),
            )
            if not chunks:
                continue

            # Build (chunk_id, texto, metadata) triples, deduplicating
            triples = []
            for j, chunk in enumerate(chunks):
                chunk_id = (
                    f"{doc.get('metadata', doc).get('documento_id', doc.get('source', f'doc_{doc_i}'))}_{j}"
                )
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    triples.append((chunk_id, chunk["texto"], chunk["metadata"]))

            if not triples:
                continue

            # Encode + insert in bounded batches to cap RAM usage
            for batch_start in range(0, len(triples), embed_batch_size):
                batch = triples[batch_start: batch_start + embed_batch_size]
                batch_texts = [t[1] for t in batch]

                logger.info(
                    "Embedding batch %d-%d / %d  (doc %d/%d)",
                    batch_start + 1,
                    batch_start + len(batch),
                    len(triples),
                    doc_i + 1,
                    len(documentos),
                )
                import torch
                import gc
                torch.set_num_threads(1)
                with torch.inference_mode():
                    embeddings = self.embedder.encode(
                        batch_texts, show_progress_bar=False, batch_size=embed_batch_size
                    )
                gc.collect()

                rows = [
                    (batch[i][0], batch[i][1], json.dumps(batch[i][2]), embeddings[i].tolist())
                    for i in range(len(batch))
                ]

                with _get_conn() as conn:
                    with conn.cursor() as cur:
                        execute_values(
                            cur,
                            """
                            INSERT INTO document_chunks (chunk_id, texto, metadata, embedding)
                            VALUES %s
                            ON CONFLICT (chunk_id) DO UPDATE
                              SET texto     = EXCLUDED.texto,
                                  metadata  = EXCLUDED.metadata,
                                  embedding = EXCLUDED.embedding
                            """,
                            rows,
                            template="(%s, %s, %s::jsonb, %s::vector)",
                        )
                    conn.commit()

                total_upserted += len(batch)

                # Explicitly release the embedding array so GC can reclaim RAM
                del embeddings, rows

            logger.info(
                "Doc %d/%d done — %d chunks upserted so far",
                doc_i + 1, len(documentos), total_upserted,
            )

        if total_upserted == 0:
            logger.warning("No chunks were processed")
            return 0

        logger.info("Total upserted: %d chunks from %d documents", total_upserted, len(documentos))
        if rebuild_bm25:
            self._build_bm25_index()
        return total_upserted

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _extract_hierarchy(self, text: str) -> dict:
        """Scan text for the last seen Título/Capítulo/Sección before position."""
        titulo = capitulo = seccion = ""
        for m in _TITULO_PAT.finditer(text):
            titulo = m.group(1).strip()[:80]
        for m in _CAPITULO_PAT.finditer(text):
            capitulo = m.group(1).strip()[:80]
        for m in _SECCION_PAT.finditer(text):
            seccion = m.group(1).strip()[:80]
        return {"titulo": titulo, "capitulo": capitulo, "seccion": seccion}

    def _breadcrumb(self, hier: dict) -> str:
        parts = [v for v in [hier.get("titulo"), hier.get("capitulo"), hier.get("seccion")] if v]
        return " | ".join(parts)

    def _sliding_window(self, text: str, max_size: int = 800, overlap: int = 150) -> list[str]:
        """Split text into overlapping windows."""
        if len(text) <= max_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_size
            chunk = text[start:end]
            # Try to break at sentence boundary
            last_period = max(chunk.rfind(". "), chunk.rfind(".\n"))
            if last_period > max_size // 2:
                chunk = chunk[:last_period + 1]
            chunks.append(chunk.strip())
            start += len(chunk) - overlap
        return [c for c in chunks if len(c) > 50]

    def _segmentar_documento(self, texto: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        if not texto or len(texto.strip()) < 100:
            return chunks

        # Scan for hierarchy context (título/capítulo/sección) in full text
        hier = self._extract_hierarchy(texto)

        # Split by article boundaries
        partes = _ARTICULO_PAT.split(texto)

        articulo_actual = None
        hier_actual = dict(hier)
        chunk_index = 0

        for parte in partes:
            if _ARTICULO_PAT.match(parte):
                # Update hierarchy context up to this article
                pos = texto.find(parte)
                if pos >= 0:
                    ctx = texto[:pos]
                    for m in _TITULO_PAT.finditer(ctx):
                        hier_actual["titulo"] = m.group(1).strip()[:80]
                    for m in _CAPITULO_PAT.finditer(ctx):
                        hier_actual["capitulo"] = m.group(1).strip()[:80]
                    for m in _SECCION_PAT.finditer(ctx):
                        hier_actual["seccion"] = m.group(1).strip()[:80]
                articulo_actual = parte.strip()
            elif articulo_actual and parte.strip():
                new_chunks = self._add_article_chunks(
                    articulo_actual, parte, metadata, dict(hier_actual), chunk_index
                )
                chunks.extend(new_chunks)
                chunk_index += len(new_chunks)

        if not chunks:
            chunks = self._chunk_by_paragraphs(texto, metadata)

        return chunks

    def _add_article_chunks(
        self, articulo: str, contenido: str, metadata: Dict, hier: dict, base_index: int
    ) -> List[Dict]:
        source = metadata.get("source", metadata.get("documento_id", ""))
        framework = metadata.get("framework", "")
        breadcrumb = self._breadcrumb(hier)
        prefix = " | ".join(p for p in [framework, breadcrumb, articulo] if p)

        sub_chunks = self._sliding_window(contenido.strip(), max_size=800, overlap=150)

        # RAG-2: if the article produces multiple sub-chunks, store the full article
        # text as parent_text in each child's metadata. Retrieval with
        # expand_to_parent=True replaces the child snippet with the richer full text,
        # giving the LLM complete article context without sacrificing retrieval precision.
        full_article_text = (
            (f"{prefix}\n\n{contenido.strip()}" if prefix else contenido.strip())
            if len(sub_chunks) > 1
            else None
        )

        result = []
        for i, sub in enumerate(sub_chunks):
            text_with_context = f"{prefix}\n\n{sub}" if prefix else sub
            meta = (metadata.copy() if isinstance(metadata, dict) else {})
            meta.update({
                "articulo": articulo,
                "titulo": hier.get("titulo", ""),
                "capitulo": hier.get("capitulo", ""),
                "seccion": hier.get("seccion", ""),
                "chunk_index": base_index + i,
                "longitud": len(text_with_context),
            })
            if full_article_text:
                meta["parent_text"] = full_article_text
            result.append({"texto": text_with_context.strip(), "metadata": meta})
        return result

    def _chunk_by_paragraphs(self, texto: str, metadata: Dict) -> List[Dict]:
        chunks = []
        source = metadata.get("source", "")
        sub_chunks = self._sliding_window(texto, max_size=600, overlap=100)
        for i, sub in enumerate(sub_chunks):
            meta = (metadata.copy() if isinstance(metadata, dict) else {})
            meta.update({"chunk_index": i, "longitud": len(sub)})
            chunks.append({"texto": sub, "metadata": meta})
        return chunks

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _build_bm25_index(self):
        # RAG-4: try loading from disk cache first; rebuild if stale or missing
        db_count = self.collection.count()
        cache = _load_bm25_cache(self._bm25_cache_path)
        if (cache and cache.get("count") == db_count and db_count > 0
                and "corpus_id_to_idx" in cache and "bm25" in cache):
            self.corpus_ids = cache["corpus_ids"]
            self.corpus_id_to_idx = cache["corpus_id_to_idx"]
            self.bm25 = cache["bm25"]
            logger.info("BM25 index loaded from cache (%d docs)", len(self.corpus_ids))
            return

        ids = []
        tokenized = []
        # Server-side cursor streams rows in batches — avoids fetchall() loading
        # the entire corpus into RAM at once.
        conn = _get_conn()
        try:
            with conn.cursor("bm25_stream") as cur:
                cur.itersize = 500
                cur.execute("SELECT chunk_id, texto FROM document_chunks ORDER BY id")
                for chunk_id, texto in cur:
                    ids.append(chunk_id)
                    tokenized.append(texto.lower().split())
        finally:
            conn.close()

        if not ids:
            logger.warning("No documents in DB for BM25 indexing")
            return

        self.corpus_ids = ids
        self.corpus_id_to_idx = {cid: i for i, cid in enumerate(ids)}
        self.bm25 = BM25Okapi(tokenized)
        del tokenized  # release tokenized list — BM25 model has what it needs
        logger.info("BM25 index built with %d documents", len(self.corpus_ids))

        _save_bm25_cache(self._bm25_cache_path, {
            "count": len(ids),
            "corpus_ids": self.corpus_ids,
            "corpus_id_to_idx": self.corpus_id_to_idx,
            "bm25": self.bm25,
        })

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
        self,
        pregunta: str,
        n_resultados: int = 5,
        peso_semantico: float = 0.7,
        rerank: bool = False,
        expand_to_parent: bool = False,
    ) -> List[Dict[str, Any]]:
        """Hybrid BM25 + semantic search.

        Args:
            rerank: If True, retrieve 4× candidates and rerank with a cross-encoder
                (RAG-1). Improves precision at the cost of ~50 ms extra latency.
            expand_to_parent: If True, replace each retrieved sub-chunk with its
                full parent article text when available (RAG-2). Gives the LLM
                richer context without sacrificing retrieval precision.
        """
        # Fetch more candidates when reranking so the reranker has material to work with
        fetch_n = n_resultados * 4 if rerank else n_resultados * 2

        if not self.bm25 or not self.corpus:
            logger.warning("BM25 index not available, falling back to semantic search")
            candidates = self.buscar_contexto(pregunta, fetch_n if rerank else n_resultados)
        else:
            resultados_semanticos = self.buscar_contexto(pregunta, n_resultados=fetch_n)

            tokenized_query = pregunta.lower().split()
            scores_bm25 = self.bm25.get_scores(tokenized_query)
            max_bm25 = max(scores_bm25) if max(scores_bm25) > 0 else 1

            max_dist = (
                max((c["distancia"] for c in resultados_semanticos if c["distancia"]), default=1)
            )
            scores_combinados = []
            for chunk in resultados_semanticos:
                idx = self.corpus_id_to_idx.get(chunk["id"])
                if idx is None:
                    scores_combinados.append({"chunk": chunk, "score": 0.5})
                    continue
                score_sem = (
                    1 - (chunk["distancia"] / max_dist)
                    if chunk["distancia"] and max_dist > 0
                    else 0.5
                )
                score_bm25_norm = scores_bm25[idx] / max_bm25
                score_final = peso_semantico * score_sem + (1 - peso_semantico) * score_bm25_norm
                scores_combinados.append({"chunk": chunk, "score": score_final})

            scores_combinados.sort(key=lambda x: x["score"], reverse=True)
            candidates = [s["chunk"] for s in scores_combinados]

        # RAG-1: cross-encoder reranking
        if rerank and candidates:
            try:
                reranker = _get_reranker()
                pairs = [(pregunta, c["texto"]) for c in candidates]
                scores = reranker.predict(pairs)
                ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
                candidates = [c for _, c in ranked]
                logger.debug("Cross-encoder reranked %d candidates → top %d", len(candidates), n_resultados)
            except Exception as e:
                logger.warning("Cross-encoder reranking failed, using hybrid scores: %s", e)

        results = candidates[:n_resultados]

        # RAG-2: expand child chunks to full parent article text
        if expand_to_parent:
            results = self._expand_to_parent(results)

        return results

    def _expand_to_parent(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace sub-chunks with their full parent article text where available.

        Articles that fit in a single window have no parent_text and are returned
        unchanged. Articles split across multiple windows have parent_text stored
        in their metadata at ingestion time.
        """
        expanded = []
        for chunk in chunks:
            parent_text = chunk.get("metadata", {}).get("parent_text")
            if parent_text:
                chunk = dict(chunk)  # shallow copy — don't mutate caller's list
                chunk["texto"] = parent_text
                chunk["_expanded"] = True
            expanded.append(chunk)
        return expanded

    def buscar_con_citas(
        self,
        pregunta: str,
        n_resultados: int = 5,
        citation_rag=None,
        n_citas: int = 3,
        rerank: bool = False,
        expand_to_parent: bool = False,
    ) -> List[Dict[str, Any]]:
        """Hybrid search over document_chunks merged with CitationRAG hits (RAG-3).

        Runs standard hybrid retrieval, then appends citation_chunks results
        (per-article/paragraph nodes) that are not already covered by document_chunks.
        Useful for the compliance checker where precise article-level grounding matters.

        Args:
            citation_rag: A CitationRAG instance. If None, falls back to buscar_hibrida.
            n_citas: Number of citation chunks to add after deduplication.
        """
        doc_results = self.buscar_hibrida(
            pregunta, n_resultados, rerank=rerank, expand_to_parent=expand_to_parent
        )

        if citation_rag is None:
            return doc_results

        try:
            cit_hits = citation_rag.search(pregunta, top_k=n_citas * 3)
        except Exception as e:
            logger.warning("CitationRAG search failed in buscar_con_citas: %s", e)
            return doc_results

        # Collect articles already covered by document_chunks results
        covered_articles: set[str] = set()
        for c in doc_results:
            art = c.get("metadata", {}).get("articulo", "")
            if art:
                covered_articles.add(art.lower().strip())

        # Normalize citation hits to the same dict shape and append non-duplicate ones
        added = 0
        for hit in cit_hits:
            art = hit.get("articulo", "").lower().strip()
            if art and art in covered_articles:
                continue
            doc_results.append({
                "texto": hit.get("text", ""),
                "metadata": {
                    "documento": hit.get("documento", ""),
                    "articulo": hit.get("articulo", ""),
                    "source": hit.get("reference", ""),
                    "url": hit.get("url", ""),
                },
                "distancia": (1.0 - hit["score"]) if hit.get("score") is not None else None,
                "id": f"cit_{hit.get('reference', '')}",
            })
            if art:
                covered_articles.add(art)
            added += 1
            if added >= n_citas:
                break

        if added:
            logger.debug("buscar_con_citas: added %d citation hits to %d doc results", added, len(doc_results) - added)

        return doc_results

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
            "corpus_size": len(self.corpus_id_to_idx),
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
