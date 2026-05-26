"""
Admin endpoints — only accessible with ADMIN_TOKEN env var.
POST /admin/ingest  — ingest a document into the RAG using the already-loaded model.
"""

import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
_bearer = HTTPBearer()


def _require_admin(creds: HTTPAuthorizationCredentials = Depends(_bearer)):
    token = os.getenv("ADMIN_TOKEN", "")
    # If no ADMIN_TOKEN configured, any bearer value passes (local dev mode)
    if token and creds.credentials != token:
        raise HTTPException(status_code=403, detail="Forbidden")
    return creds


class IngestRequest(BaseModel):
    documents: List[Dict[str, Any]]
    embed_batch_size: int = 16


@router.post("/ingest")
def ingest(req: IngestRequest, _=Depends(_require_admin)):
    from api.main import get_rag_instance

    rag = get_rag_instance()
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialised")

    n = rag.procesar_documentos(
        req.documents,
        embed_batch_size=req.embed_batch_size,
    )
    return {"chunks_upserted": n}
