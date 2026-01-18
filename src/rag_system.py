"""
RAG (Retrieval-Augmented Generation) System for Banking Regulatory Assistant.
Provides semantic search and hybrid search capabilities for regulatory documents.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from rank_bm25 import BM25Okapi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegulatoryRAGSystem:
    """
    Sistema RAG completo para consultas regulatorias en espanol.
    Complete RAG system for regulatory queries in Spanish.
    """

    def __init__(self,
                 embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 persist_directory: str = "./vector_db/chroma_db"):
        """
        Initialize the RAG system.

        Args:
            embedding_model_name: Name of the sentence transformer model for embeddings
            persist_directory: Directory to persist the ChromaDB database
        """
        logger.info(f"Initializing RAG system with embedding model: {embedding_model_name}")

        # Modelo de embeddings multilingue (espanol incluido)
        self.embedder = SentenceTransformer(embedding_model_name)

        # Base de datos vectorial
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Coleccion para documentos regulatorios
        self.collection = self.client.get_or_create_collection(
            name="regulacion_bancaria",
            metadata={"description": "Documentos regulatorios bancarios en espanol"}
        )

        # BM25 index for hybrid search
        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []

        logger.info(f"RAG system initialized. Collection has {self.collection.count()} documents.")

    def procesar_documentos(self, documentos: List[Dict[str, Any]]) -> int:
        """
        Process and store regulatory documents.

        Args:
            documentos: List of dicts with 'texto', 'metadata'

        Returns:
            Number of chunks processed
        """
        textos = []
        metadatas = []
        ids = []

        for i, doc in enumerate(documentos):
            # Segmentar documento en chunks
            chunks = self._segmentar_documento(doc.get('text', doc.get('texto', '')),
                                               doc.get('metadata', doc))

            for j, chunk in enumerate(chunks):
                chunk_id = f"{doc.get('metadata', doc).get('documento_id', doc.get('source', f'doc_{i}'))}_{j}"

                # Avoid duplicates
                if chunk_id not in ids:
                    textos.append(chunk['texto'])
                    metadatas.append(chunk['metadata'])
                    ids.append(chunk_id)

        if not textos:
            logger.warning("No texts to process")
            return 0

        # Generar embeddings
        logger.info(f"Generating embeddings for {len(textos)} chunks...")
        embeddings = self.embedder.encode(textos, show_progress_bar=True)

        # Almacenar en ChromaDB
        batch_size = 5000  # ChromaDB batch limit
        for start in range(0, len(textos), batch_size):
            end = min(start + batch_size, len(textos))
            self.collection.add(
                embeddings=embeddings[start:end].tolist(),
                documents=textos[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )

        logger.info(f"Processed {len(textos)} chunks from {len(documentos)} documents")

        # Update BM25 index
        self._build_bm25_index()

        return len(textos)

    def _segmentar_documento(self, texto: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Segment document by articles/paragraphs while maintaining context.

        Args:
            texto: Full text of the document
            metadata: Document metadata

        Returns:
            List of chunks with text and metadata
        """
        chunks = []

        if not texto or len(texto.strip()) < 100:
            return chunks

        # Detect articles using Spanish patterns
        patron = r'(Art[ií]culo\s+\d+[a-z]?[.\s])'
        partes = re.split(patron, texto, flags=re.IGNORECASE)

        articulo_actual = None
        contenido_actual = ""

        for parte in partes:
            if re.match(patron, parte, re.IGNORECASE):
                # Save previous article if exists
                if articulo_actual and contenido_actual.strip():
                    self._add_article_chunks(chunks, articulo_actual, contenido_actual, metadata)

                articulo_actual = parte.strip()
                contenido_actual = ""
            elif articulo_actual:
                contenido_actual += parte

        # Save last article
        if articulo_actual and contenido_actual.strip():
            self._add_article_chunks(chunks, articulo_actual, contenido_actual, metadata)

        # If no articles found, chunk by paragraphs
        if not chunks:
            chunks = self._chunk_by_paragraphs(texto, metadata)

        return chunks

    def _add_article_chunks(self, chunks: List[Dict], articulo: str, contenido: str, metadata: Dict):
        """Add article chunks, splitting if too long."""
        max_chunk_size = 1500  # characters

        parrafos = contenido.split('\n')
        chunk_actual = f"{articulo}\n"

        for parrafo in parrafos:
            parrafo = parrafo.strip()
            if not parrafo or len(parrafo) < 20:
                continue

            if len(chunk_actual) + len(parrafo) > max_chunk_size:
                if len(chunk_actual) > 100:
                    chunk_metadata = metadata.copy() if isinstance(metadata, dict) else {}
                    chunk_metadata['articulo'] = articulo
                    chunk_metadata['longitud'] = len(chunk_actual)

                    chunks.append({
                        'texto': chunk_actual.strip(),
                        'metadata': chunk_metadata
                    })
                chunk_actual = f"{articulo}\n{parrafo}\n"
            else:
                chunk_actual += f"{parrafo}\n"

        # Add remaining content
        if len(chunk_actual) > 100:
            chunk_metadata = metadata.copy() if isinstance(metadata, dict) else {}
            chunk_metadata['articulo'] = articulo
            chunk_metadata['longitud'] = len(chunk_actual)

            chunks.append({
                'texto': chunk_actual.strip(),
                'metadata': chunk_metadata
            })

    def _chunk_by_paragraphs(self, texto: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs when no article structure is found."""
        chunks = []
        max_chunk_size = 1500

        parrafos = texto.split('\n\n')
        chunk_actual = ""

        for parrafo in parrafos:
            parrafo = parrafo.strip()
            if not parrafo or len(parrafo) < 50:
                continue

            if len(chunk_actual) + len(parrafo) > max_chunk_size:
                if len(chunk_actual) > 100:
                    chunk_metadata = metadata.copy() if isinstance(metadata, dict) else {}
                    chunk_metadata['longitud'] = len(chunk_actual)

                    chunks.append({
                        'texto': chunk_actual.strip(),
                        'metadata': chunk_metadata
                    })
                chunk_actual = parrafo + "\n\n"
            else:
                chunk_actual += parrafo + "\n\n"

        # Add remaining content
        if len(chunk_actual) > 100:
            chunk_metadata = metadata.copy() if isinstance(metadata, dict) else {}
            chunk_metadata['longitud'] = len(chunk_actual)

            chunks.append({
                'texto': chunk_actual.strip(),
                'metadata': chunk_metadata
            })

        return chunks

    def _build_bm25_index(self):
        """Build BM25 index for keyword search."""
        all_docs = self.collection.get()

        if not all_docs['documents']:
            logger.warning("No documents in collection for BM25 indexing")
            return

        self.corpus = all_docs['documents']
        self.corpus_ids = all_docs['ids']

        # Tokenize for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 index built with {len(self.corpus)} documents")

    def buscar_contexto(self, pregunta: str, n_resultados: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant fragments for a question using semantic search.

        Args:
            pregunta: User question in Spanish
            n_resultados: Number of chunks to retrieve

        Returns:
            List of chunks with text and metadata
        """
        # Embed query
        query_embedding = self.embedder.encode([pregunta])[0]

        # Search in ChromaDB
        resultados = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_resultados
        )

        chunks = []
        for i in range(len(resultados['documents'][0])):
            chunks.append({
                'texto': resultados['documents'][0][i],
                'metadata': resultados['metadatas'][0][i] if resultados['metadatas'] else {},
                'distancia': resultados['distances'][0][i] if 'distances' in resultados and resultados['distances'] else None,
                'id': resultados['ids'][0][i] if resultados['ids'] else None
            })

        return chunks

    def buscar_hibrida(self, pregunta: str, n_resultados: int = 5, peso_semantico: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic search with BM25 keyword search.

        Args:
            pregunta: User query
            n_resultados: Number of final results
            peso_semantico: Weight for semantic search (0-1), rest goes to BM25

        Returns:
            Re-ranked list of chunks
        """
        if not self.bm25 or not self.corpus:
            logger.warning("BM25 index not available, falling back to semantic search")
            return self.buscar_contexto(pregunta, n_resultados)

        # 1. Semantic search
        resultados_semanticos = self.buscar_contexto(pregunta, n_resultados=n_resultados*2)

        # 2. BM25 search
        tokenized_query = pregunta.lower().split()
        scores_bm25 = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores
        max_bm25 = max(scores_bm25) if max(scores_bm25) > 0 else 1

        # 3. Combine scores
        scores_combinados = []

        for chunk in resultados_semanticos:
            try:
                idx = self.corpus.index(chunk['texto'])

                # Normalize semantic score (distance to similarity)
                max_dist = max(c['distancia'] for c in resultados_semanticos if c['distancia']) if resultados_semanticos else 1
                score_sem = 1 - (chunk['distancia'] / max_dist) if chunk['distancia'] and max_dist > 0 else 0.5
                score_bm25_norm = scores_bm25[idx] / max_bm25

                score_final = (peso_semantico * score_sem +
                              (1 - peso_semantico) * score_bm25_norm)

                scores_combinados.append({
                    'chunk': chunk,
                    'score': score_final
                })
            except ValueError:
                # Chunk not in corpus (shouldn't happen but handle gracefully)
                scores_combinados.append({
                    'chunk': chunk,
                    'score': 0.5
                })

        # 4. Re-rank and return top-N
        scores_combinados.sort(key=lambda x: x['score'], reverse=True)

        return [s['chunk'] for s in scores_combinados[:n_resultados]]

    def formatear_contexto(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks for the prompt.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted context string
        """
        contexto_partes = []

        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get('metadata', {})
            fuente = f"[{meta.get('documento', meta.get('source', 'Desconocido'))} {meta.get('articulo', '')}]"

            contexto_partes.append(
                f"Fuente {i}: {fuente}\n\"{chunk['texto']}\"\n"
            )

        return "\n".join(contexto_partes)

    def load_from_json(self, json_path: str) -> int:
        """
        Load documents from a JSON file (scraped data format).

        Args:
            json_path: Path to JSON file with scraped documents

        Returns:
            Number of chunks processed
        """
        logger.info(f"Loading documents from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        # Convert to expected format
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                'texto': doc.get('text', ''),
                'metadata': {
                    'documento': doc.get('title', 'Unknown'),
                    'documento_id': doc.get('url', 'unknown'),
                    'source': doc.get('source', 'Unknown'),
                    'tipo': doc.get('type', 'unknown'),
                    'url': doc.get('url', ''),
                    'keywords': doc.get('keywords', [])
                }
            })

        return self.procesar_documentos(formatted_docs)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            'total_documents': self.collection.count(),
            'embedding_model': str(self.embedder),
            'bm25_indexed': self.bm25 is not None,
            'corpus_size': len(self.corpus) if self.corpus else 0
        }

    def clear_collection(self):
        """Clear all documents from the collection."""
        # Get all IDs
        all_docs = self.collection.get()
        if all_docs['ids']:
            self.collection.delete(ids=all_docs['ids'])

        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []

        logger.info("Collection cleared")


class HybridSearch:
    """
    Combines semantic search (embeddings) with BM25 (keywords).
    Wrapper class for more advanced search scenarios.
    """

    def __init__(self, rag_system: RegulatoryRAGSystem):
        self.rag_system = rag_system

    def search(self, query: str, n_results: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search.

        Args:
            query: User query
            n_results: Number of results to return
            semantic_weight: Weight for semantic vs keyword search

        Returns:
            List of relevant chunks
        """
        return self.rag_system.buscar_hibrida(query, n_results, semantic_weight)


# Convenience function for quick initialization
def create_rag_system(embedding_model: str = None, persist_dir: str = None) -> RegulatoryRAGSystem:
    """
    Create a RAG system with default or custom configuration.

    Args:
        embedding_model: Optional custom embedding model
        persist_dir: Optional custom persistence directory

    Returns:
        Configured RegulatoryRAGSystem instance
    """
    kwargs = {}
    if embedding_model:
        kwargs['embedding_model_name'] = embedding_model
    if persist_dir:
        kwargs['persist_directory'] = persist_dir

    return RegulatoryRAGSystem(**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Creating RAG system...")
    rag = create_rag_system()

    print(f"\nRAG System Stats: {rag.get_stats()}")

    # Example: Load data from scraped JSON
    import glob
    json_files = glob.glob("data/raw/*.json")

    if json_files:
        print(f"\nFound {len(json_files)} JSON files. Loading most recent...")
        latest_file = max(json_files)
        chunks_loaded = rag.load_from_json(latest_file)
        print(f"Loaded {chunks_loaded} chunks")

    # Example query
    if rag.collection.count() > 0:
        print("\n" + "="*50)
        print("Testing search...")
        results = rag.buscar_contexto("ratio de capital CET1", n_resultados=3)

        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"Text: {result['texto'][:200]}...")
