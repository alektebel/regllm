"""
CitationRAG — per-article/paragraph citation vector database.

Each node in the regulation tree (or chunk in a raw regulation doc) is stored
as a separate vector.  At query time, the question is embedded and the closest
citations are returned, independently of the main RAG retrieval.

Collection: "regulation_citations"
Embed model: paraphrase-multilingual-MiniLM-L12-v2  (dim=384, CPU)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "regulation_citations"


class CitationRAG:
    """
    Vector store for regulation citations (one vector per article/paragraph).

    Usage:
        crag = CitationRAG()
        crag.index_citation_tree("data/citation_trees/eu_regulations.json")
        hits = crag.search("CET1 ratio requirements", top_k=5)
    """

    def __init__(
        self,
        chroma_path: str = "./vector_db/chroma_db",
        embed_model: str = EMBED_MODEL,
    ):
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.encoder = SentenceTransformer(embed_model, device="cpu")
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Per-article regulation citations", "hnsw:space": "cosine"},
        )
        logger.info(
            f"CitationRAG ready — collection '{COLLECTION_NAME}' has {self.collection.count()} items"
        )

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_citation_tree(self, json_path: str) -> int:
        """
        Walk eu_regulations.json tree; index one vector per node.
        Skips nodes with no meaningful text (< 30 chars).

        Returns number of items added.
        """
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

            # Parse "documento" and "articulo" from reference string
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

            item_id = f"tree_{node_id}"
            texts.append(text)
            metadatas.append(meta)
            ids.append(item_id)

        return self._batch_add(texts, metadatas, ids, source_label=str(path.name))

    def index_regulation_doc(self, doc_path: str, doc_name: str) -> int:
        """
        Re-chunk a regulation JSON doc at article level; one vector per chunk.
        Accepts the same format as RegulatoryRAGSystem.load_from_json().

        Returns number of items added.
        """
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Doc not found: {doc_path}")

        with open(path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        texts, metadatas, ids = [], [], []

        for doc_idx, doc in enumerate(documents):
            raw_text = doc.get("text", doc.get("texto", ""))
            doc_meta = {
                "documento": doc_name,
                "source": doc.get("source", doc_name),
                "url": doc.get("url", ""),
            }

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
                    "url": doc_meta["url"],
                    "language": "es",
                }
                texts.append(chunk_text[:1500])
                metadatas.append(meta)
                ids.append(item_id)

        return self._batch_add(texts, metadatas, ids, source_label=doc_name)

    def _batch_add(self, texts: list, metadatas: list, ids: list, source_label: str = "") -> int:
        """Embed and upsert a batch of items, skipping existing IDs."""
        if not texts:
            logger.warning(f"No items to index from {source_label}")
            return 0

        # Check which IDs already exist to avoid duplicate-key errors
        existing = set(self.collection.get(ids=ids)["ids"])
        new_texts, new_metas, new_ids = [], [], []
        for t, m, i in zip(texts, metadatas, ids):
            if i not in existing:
                new_texts.append(t)
                new_metas.append(m)
                new_ids.append(i)

        if not new_texts:
            logger.info(f"{source_label}: all {len(ids)} items already indexed, skipping")
            return 0

        logger.info(f"Embedding {len(new_texts)} new items from {source_label}...")
        embeddings = self.encoder.encode(new_texts, show_progress_bar=True, batch_size=64)

        batch_size = 5000
        for start in range(0, len(new_texts), batch_size):
            end = min(start + batch_size, len(new_texts))
            self.collection.add(
                embeddings=embeddings[start:end].tolist(),
                documents=new_texts[start:end],
                metadatas=new_metas[start:end],
                ids=new_ids[start:end],
            )

        logger.info(f"Added {len(new_texts)} items from {source_label}")
        return len(new_texts)

    def clear(self) -> None:
        """Delete the collection and re-create it (full rebuild)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Per-article regulation citations", "hnsw:space": "cosine"},
        )
        logger.info("Citation collection cleared and recreated")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Embed query and return top_k most similar citation nodes.

        Returns list of dicts with keys:
          reference, documento, articulo, paragrafo, text, url, score
        """
        if self.collection.count() == 0:
            return []

        q_emb = self.encoder.encode([query_text])[0].tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, self.collection.count()),
        )

        hits = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else [None] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            score = round(1.0 - dist, 4) if dist is not None else None
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
        return self.collection.count()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_reference(reference: str, node_type: str) -> tuple[str, str, str]:
    """
    Split a reference string like "CRR Art. 92 §1(a)" into (documento, articulo, paragrafo).
    Returns best-effort values.
    """
    reference = reference.strip()

    # Try to extract paragraph §...
    par_match = re.search(r"(§[\w()\-\.]+)", reference)
    paragrafo = par_match.group(1) if par_match else ""

    # Try to extract article
    art_match = re.search(r"(Art(?:icle|ículo|\.)\s*[\d\w]+)", reference, re.IGNORECASE)
    articulo = art_match.group(1) if art_match else ""

    # The documento is everything before the article reference (or the first token)
    if art_match:
        documento = reference[: art_match.start()].strip().rstrip(",- ")
    elif reference:
        parts = reference.split()
        documento = parts[0] if parts else reference
    else:
        documento = reference

    # If no article found and this IS an article node, use the whole reference
    if not articulo and node_type in ("article", "paragraph", "point"):
        articulo = reference

    return documento, articulo, paragrafo


def _segment_by_article(text: str) -> list[tuple[str, str]]:
    """
    Split regulation text by Spanish article pattern.
    Returns list of (articulo_header, chunk_text) tuples.
    """
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

    # Fallback: no article structure → single chunk
    if not chunks:
        chunks = [("", text.strip())]

    return chunks
