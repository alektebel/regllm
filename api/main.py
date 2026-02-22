"""
FastAPI server for Banking Regulatory Assistant.
Provides REST API endpoints for querying regulatory information.
"""

import os
import time

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
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
from src.db import init_db, dispose_engine, log_query, get_query_logs, get_db_stats
from config import MODEL, INFERENCE

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
model_path_used: str = "placeholder"


# ============================================================================
# Model Loading
# ============================================================================

# Map config model names to HuggingFace paths
AVAILABLE_MODELS = {
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-3b': 'Qwen/Qwen2.5-3B-Instruct',
    'phi-3-mini': 'microsoft/Phi-3-mini-4k-instruct',
    'stablelm-3b': 'stabilityai/stablelm-3b-4e1t',
    'phi-2': 'microsoft/phi-2',
    'gemma-2b': 'google/gemma-2b-it',
    'qwen-1.8b': 'Qwen/Qwen2-1.8B-Instruct',
}


def _find_model_path() -> Optional[str]:
    """Find the best available finetuned model path."""
    # 1. Explicit env var
    env_path = os.getenv("MODEL_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    project_root = Path(__file__).parent.parent

    # 2. GRPO final model
    grpo_path = project_root / "models" / "grpo" / "final_model"
    if grpo_path.exists():
        return str(grpo_path)

    # 3. Latest SFT finetuned run
    finetuned_dir = project_root / "models" / "finetuned"
    if finetuned_dir.exists():
        runs = sorted(finetuned_dir.glob("run_*/final_model"), reverse=True)
        if runs:
            return str(runs[0])

    return None


def _load_model():
    """Load the finetuned model with 4-bit quantization. Graceful degradation on failure."""
    global model, tokenizer, model_path_used

    found_path = _find_model_path()
    if not found_path:
        logger.warning("No finetuned model found. Running in placeholder mode.")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        logger.info(f"Loading model from {found_path}...")

        # Resolve base model
        base_model_name = os.getenv("BASE_MODEL", "").strip()
        if not base_model_name:
            base_model_name = MODEL.get('base_model', 'qwen2.5-7b')
        base_model_hf = AVAILABLE_MODELS.get(base_model_name, base_model_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(found_path, trust_remote_code=True)

        # Quantization config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = None
        if device == "cuda" and MODEL.get('use_4bit', True):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")

        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            base_model_hf,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(base, found_path)
        model.eval()
        model_path_used = found_path

        logger.info(f"Model loaded successfully from {found_path}")

    except Exception as e:
        logger.warning(f"Failed to load model: {e}. Running in placeholder mode.")
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
    confianza: float = Field(0.0, description="Score de confianza global (0-1)")
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

    # Initialize database (non-blocking — if DB is down, API still works)
    await init_db()

    # Load LLM model
    _load_model()

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
    await dispose_engine()


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
        modelo=model_path_used if model else "no_cargado (placeholder)",
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
async def consultar(request: PreguntaRequest, background_tasks: BackgroundTasks, req: Request):
    """
    Answer a question about banking regulation.

    - **pregunta**: Question about banking regulation in Spanish
    - **n_fuentes**: Number of sources to retrieve (1-20)
    - **usar_busqueda_hibrida**: Use hybrid search (semantic + BM25)
    - **peso_semantico**: Weight for semantic search (0-1)
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    t_start = time.time()

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

        # Generate response
        respuesta = _generar_respuesta(request.pregunta, fuentes, contexto)

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

        latencia_ms = int((time.time() - t_start) * 1000)

        response = RespuestaResponse(
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
            confianza=resultado_verificacion['score_confianza'],
            advertencias=advertencias,
            timestamp=datetime.now().isoformat()
        )

        # Log to DB in background (never blocks response)
        try:
            client_ip = req.client.host if req.client else None
            fuentes_log = [
                {
                    "documento": f.documento,
                    "articulo": f.articulo,
                    "relevancia": f.relevancia,
                }
                for f in fuentes_response
            ]
            background_tasks.add_task(
                log_query,
                pregunta=request.pregunta,
                respuesta=respuesta,
                fuentes=fuentes_log,
                score_confianza=resultado_verificacion['score_confianza'],
                nivel_confianza=resultado_verificacion['nivel_confianza'],
                advertencias=advertencias,
                modelo=model_path_used,
                latencia_ms=latencia_ms,
                ip_cliente=client_ip,
            )
        except Exception as e:
            logger.error(f"Error scheduling DB log: {e}")

        return response

    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs", tags=["Admin"])
async def get_logs(limit: int = 50, offset: int = 0):
    """Get paginated query logs."""
    logs = await get_query_logs(limit=limit, offset=offset)
    return {"logs": logs, "limit": limit, "offset": offset}


@app.get("/logs/stats", tags=["Admin"])
async def get_logs_stats():
    """Get aggregate query stats."""
    stats = await get_db_stats()
    return stats


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

def _generar_respuesta(pregunta: str, fuentes: List[Dict], contexto: str) -> str:
    """
    Generate a response using the finetuned LLM if available,
    otherwise fall back to the placeholder.
    """
    if model is None or tokenizer is None:
        return _generar_respuesta_placeholder(pregunta, fuentes)

    try:
        import torch

        system_prompt = INFERENCE.get('system_prompt', '')
        user_content = (
            f"Contexto de documentos regulatorios:\n{contexto}\n\n"
            f"Pregunta: {pregunta}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=INFERENCE.get('max_new_tokens', 300),
                temperature=INFERENCE.get('temperature', 0.7),
                top_p=INFERENCE.get('top_p', 0.95),
                do_sample=INFERENCE.get('do_sample', True),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        respuesta = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not respuesta:
            logger.warning("LLM returned empty response, falling back to placeholder")
            return _generar_respuesta_placeholder(pregunta, fuentes)

        return respuesta

    except Exception as e:
        logger.error(f"LLM generation failed: {e}. Falling back to placeholder.")
        return _generar_respuesta_placeholder(pregunta, fuentes)


def _generar_respuesta_placeholder(pregunta: str, fuentes: List[Dict]) -> str:
    """
    Generate a placeholder response based on sources.
    Used when LLM is not available.
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

def run_server(host: str = None, port: int = None, reload: bool = False):
    """Run the FastAPI server."""
    host = host or os.getenv("API_HOST", "0.0.0.0")
    port = port or int(os.getenv("API_PORT", "8000"))
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
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
