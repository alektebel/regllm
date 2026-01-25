"""RegLLM - Spanish Banking Regulation Language Model."""

__version__ = "0.2.0"
__author__ = "RegLLM Team"
__description__ = "Finetuned LLM for Spanish banking regulation compliance with RAG"

# Lazy imports to avoid loading heavy dependencies when only using agents module
__all__ = [
    'RegulatoryRAGSystem',
    'create_rag_system',
    'HybridSearch',
    'SistemaVerificacion',
    'verificar_respuesta_simple',
    'presentar_respuesta',
]


def __getattr__(name):
    """Lazy import mechanism for heavy dependencies."""
    if name in ('RegulatoryRAGSystem', 'create_rag_system', 'HybridSearch'):
        from .rag_system import RegulatoryRAGSystem, create_rag_system, HybridSearch
        return locals()[name]
    elif name in ('SistemaVerificacion', 'verificar_respuesta_simple', 'presentar_respuesta'):
        from .verification import SistemaVerificacion, verificar_respuesta_simple, presentar_respuesta
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
