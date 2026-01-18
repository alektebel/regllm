"""
FastAPI server for Banking Regulatory Assistant.
Provides REST API endpoints for querying regulatory information.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system import RegulatoryRAGSystem
from src.verification import SistemaVerificacion, presentar_respuesta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Asistente Regulatorio Bancario",
    description="API para consultas sobre regulacion bancaria en espanol",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_system: Optional[RegulatoryRAGSystem] = None
verificador: Optional[SistemaVerificacion] = None
model = None
tokenizer = None


# ============================================================================
# Pydantic Models
# ============================================================================

class PreguntaRequest(BaseModel):
    """Request model for questions."""
    pregunta: str = Field(..., description="Pregunta sobre regulacion bancaria", min_length=5)
    n_fuentes: Optional[int] = Field(5, description="Numero de fuentes a recuperar", ge=1, le=20)
    usar_busqueda_hibrida: Optional[bool] = Field(True, description="Usar busqueda hibrida (semantica + BM25)")
    peso_semantico: Optional[float] = Field(0.7, description="Peso de busqueda semantica (0-1)", ge=0, le=1)


class FuenteResponse(BaseModel):
    """Response model for sources."""
    documento: str
    articulo: Optional[str] = None
    texto: str
    relevancia: Optional[float] = None
    url: Optional[str] = None


class VerificacionResponse(BaseModel):
    """Response model for verification results."""
    score_confianza: float
    nivel_confianza: str
    citaciones_encontradas: int
    citaciones_verificadas: int
    coherencia: str
    es_espanol: bool


class RespuestaResponse(BaseModel):
    """Response model for query results."""
    pregunta: str
    respuesta: str
    fuentes: List[FuenteResponse]
    verificacion: VerificacionResponse
    advertencias: List[str] = []
    timestamp: str


class DocumentRequest(BaseModel):
    """Request model for adding documents."""
    texto: str = Field(..., description="Texto del documento")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos del documento")


class ScrapeRequest(BaseModel):
    """Request model for scraping URLs."""
    urls: List[str] = Field(..., description="URLs a scrapear")
    include_linkedin: Optional[bool] = Field(False, description="Usar scraper mejorado para LinkedIn")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    modelo: str
    vector_db: str
    total_documentos: int
    timestamp: str


class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_documentos: int
    embedding_model: str
    bm25_indexed: bool
    corpus_size: int


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global rag_system, verificador

    logger.info("Inicializando sistema RAG...")
    try:
        rag_system = RegulatoryRAGSystem()
        verificador = SistemaVerificacion()

        # Try to load existing data
        import glob
        json_files = sorted(glob.glob("data/raw/*.json"), reverse=True)
        if json_files and rag_system.collection.count() == 0:
            logger.info(f"Loading data from {json_files[0]}...")
            rag_system.load_from_json(json_files[0])

        logger.info(f"Sistema inicializado. Documentos en BD: {rag_system.collection.count()}")

    except Exception as e:
        logger.error(f"Error inicializando sistema: {e}")
        rag_system = RegulatoryRAGSystem()
        verificador = SistemaVerificacion()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Cerrando sistema...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "mensaje": "Bienvenido al Asistente Regulatorio Bancario",
        "documentacion": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health."""
    return HealthResponse(
        status="healthy" if rag_system else "initializing",
        modelo="operativo" if model else "no_cargado",
        vector_db="conectada" if rag_system else "desconectada",
        total_documentos=rag_system.collection.count() if rag_system else 0,
        timestamp=datetime.now().isoformat()
    )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """Get system statistics."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    stats = rag_system.get_stats()
    return StatsResponse(
        total_documentos=stats['total_documents'],
        embedding_model=stats['embedding_model'],
        bm25_indexed=stats['bm25_indexed'],
        corpus_size=stats['corpus_size']
    )


@app.post("/consultar", response_model=RespuestaResponse, tags=["Queries"])
async def consultar(request: PreguntaRequest):
    """
    Answer a question about banking regulation.

    - **pregunta**: Question about banking regulation in Spanish
    - **n_fuentes**: Number of sources to retrieve (1-20)
    - **usar_busqueda_hibrida**: Use hybrid search (semantic + BM25)
    - **peso_semantico**: Weight for semantic search (0-1)
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        # Search for relevant sources
        if request.usar_busqueda_hibrida:
            fuentes = rag_system.buscar_hibrida(
                request.pregunta,
                n_resultados=request.n_fuentes,
                peso_semantico=request.peso_semantico
            )
        else:
            fuentes = rag_system.buscar_contexto(
                request.pregunta,
                n_resultados=request.n_fuentes
            )

        if not fuentes:
            return RespuestaResponse(
                pregunta=request.pregunta,
                respuesta="No se encontraron fuentes relevantes para responder esta pregunta.",
                fuentes=[],
                verificacion=VerificacionResponse(
                    score_confianza=0.0,
                    nivel_confianza="BAJA",
                    citaciones_encontradas=0,
                    citaciones_verificadas=0,
                    coherencia="N/A",
                    es_espanol=True
                ),
                advertencias=["No se encontraron documentos relevantes"],
                timestamp=datetime.now().isoformat()
            )

        # Format context for response generation
        contexto = rag_system.formatear_contexto(fuentes)

        # Generate response (placeholder - integrate with LLM)
        respuesta = _generar_respuesta_placeholder(request.pregunta, fuentes)

        # Verify response
        resultado_verificacion = verificador.verificar_respuesta(
            request.pregunta,
            respuesta,
            fuentes
        )

        # Format sources for response
        fuentes_response = []
        for fuente in fuentes[:request.n_fuentes]:
            meta = fuente.get('metadata', {})
            fuentes_response.append(FuenteResponse(
                documento=meta.get('documento', meta.get('source', 'Desconocido')),
                articulo=meta.get('articulo'),
                texto=fuente.get('texto', '')[:500],
                relevancia=1 - fuente.get('distancia', 0.5) if fuente.get('distancia') else None,
                url=meta.get('url')
            ))

        # Generate warnings
        advertencias = []
        verif = resultado_verificacion['verificaciones']

        if resultado_verificacion['score_confianza'] < 0.7:
            advertencias.append("Confianza por debajo del umbral recomendado. Verificar respuesta.")
        if not verif['idioma']['es_español']:
            advertencias.append("ADVERTENCIA: Respuesta no detectada como espanol.")
        if verif['hallucination']['hallucination_detectada']:
            advertencias.append("Posible informacion no verificada en la respuesta.")

        return RespuestaResponse(
            pregunta=request.pregunta,
            respuesta=respuesta,
            fuentes=fuentes_response,
            verificacion=VerificacionResponse(
                score_confianza=resultado_verificacion['score_confianza'],
                nivel_confianza=resultado_verificacion['nivel_confianza'],
                citaciones_encontradas=verif['citaciones']['citaciones_encontradas'],
                citaciones_verificadas=verif['citaciones']['citaciones_verificadas'],
                coherencia=verif['coherencia']['nivel'],
                es_espanol=verif['idioma']['es_español']
            ),
            advertencias=advertencias,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/buscar", tags=["Search"])
async def buscar(pregunta: str, n_resultados: int = 5):
    """
    Search for relevant document chunks without generating a response.

    - **pregunta**: Search query
    - **n_resultados**: Number of results to return
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        fuentes = rag_system.buscar_hibrida(pregunta, n_resultados=n_resultados)

        return {
            "query": pregunta,
            "resultados": [
                {
                    "texto": f.get('texto', '')[:500],
                    "metadata": f.get('metadata', {}),
                    "score": 1 - f.get('distancia', 0.5) if f.get('distancia') else None
                }
                for f in fuentes
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documentos/agregar", tags=["Documents"])
async def agregar_documento(request: DocumentRequest):
    """
    Add a single document to the vector database.

    - **texto**: Document text
    - **metadata**: Document metadata
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        chunks_added = rag_system.procesar_documentos([{
            'texto': request.texto,
            'metadata': request.metadata
        }])

        return {
            "mensaje": f"Documento procesado. {chunks_added} chunks agregados.",
            "chunks_agregados": chunks_added,
            "total_documentos": rag_system.collection.count()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documentos/cargar-json", tags=["Documents"])
async def cargar_json(filepath: str):
    """
    Load documents from a JSON file.

    - **filepath**: Path to the JSON file
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        chunks_added = rag_system.load_from_json(filepath)

        return {
            "mensaje": f"Archivo cargado. {chunks_added} chunks agregados.",
            "chunks_agregados": chunks_added,
            "total_documentos": rag_system.collection.count()
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {filepath}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape", tags=["Scraping"])
async def scrape_urls(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Scrape URLs and add to vector database.

    - **urls**: List of URLs to scrape
    - **include_linkedin**: Use enhanced scraper for LinkedIn
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        if request.include_linkedin:
            from src.scraper import EnhancedScraper
            scraper = EnhancedScraper()
            documents = scraper.scrape_multiple(request.urls)
        else:
            from src.scraper import RegulationScraper
            scraper = RegulationScraper()
            documents = []
            for url in request.urls:
                docs = scraper.scrape_page(url)
                documents.extend(docs)

        # Process documents
        if documents:
            chunks_added = rag_system.procesar_documentos(documents)
            return {
                "mensaje": f"Scraping completado. {len(documents)} documentos, {chunks_added} chunks.",
                "documentos_scrapeados": len(documents),
                "chunks_agregados": chunks_added,
                "total_documentos": rag_system.collection.count()
            }
        else:
            return {
                "mensaje": "No se pudieron obtener documentos de las URLs proporcionadas.",
                "documentos_scrapeados": 0,
                "chunks_agregados": 0
            }

    except Exception as e:
        logger.error(f"Error en scraping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documentos/limpiar", tags=["Documents"])
async def limpiar_documentos():
    """Clear all documents from the vector database."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        rag_system.clear_collection()
        return {
            "mensaje": "Base de datos limpiada",
            "total_documentos": 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

def _generar_respuesta_placeholder(pregunta: str, fuentes: List[Dict]) -> str:
    """
    Generate a placeholder response based on sources.
    Replace this with actual LLM integration.
    """
    if not fuentes:
        return "No se encontraron fuentes relevantes para responder esta pregunta."

    # Build response from sources
    respuesta_parts = [
        f"Basado en los documentos regulatorios disponibles, he encontrado la siguiente informacion relevante:\n"
    ]

    for i, fuente in enumerate(fuentes[:3], 1):
        meta = fuente.get('metadata', {})
        texto = fuente.get('texto', '')[:300]
        doc_name = meta.get('documento', meta.get('source', 'Documento'))
        articulo = meta.get('articulo', '')

        if articulo:
            respuesta_parts.append(f"\n[{doc_name} - {articulo}]:\n\"{texto}...\"\n")
        else:
            respuesta_parts.append(f"\n[{doc_name}]:\n\"{texto}...\"\n")

    respuesta_parts.append(
        "\n\nNota: Esta respuesta es un extracto de las fuentes. "
        "Para una respuesta mas elaborada, integre un modelo de lenguaje."
    )

    return ''.join(respuesta_parts)


# ============================================================================
# Run Server
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Asistente Regulatorio API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
