"""
Documents Module for RegLLM

Handles document uploads, processing, and comparison functionality.
"""

from .attachment_handler import AttachmentHandler
from .document_store import DocumentStore

__all__ = ["AttachmentHandler", "DocumentStore"]
