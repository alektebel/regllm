"""RegLLM - Spanish Banking Regulation Language Model."""

__version__ = "0.2.0"
__author__ = "RegLLM Team"
__description__ = "Finetuned LLM for Spanish banking regulation compliance with RAG"

# Import main components
from .rag_system import RegulatoryRAGSystem, create_rag_system, HybridSearch
from .verification import SistemaVerificacion, verificar_respuesta_simple, presentar_respuesta

__all__ = [
    'RegulatoryRAGSystem',
    'create_rag_system',
    'HybridSearch',
    'SistemaVerificacion',
    'verificar_respuesta_simple',
    'presentar_respuesta',
]
