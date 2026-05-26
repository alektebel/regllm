"""Compliance checker API — file upload + SSE run endpoint.

Endpoints:
  POST /compliance/upload   — multipart file upload, returns file_id
  POST /compliance/run      — SSE stream: run data validator or SAS annotator
  GET  /compliance/sample   — download generated sample data as zip
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["compliance"])

# ── In-memory upload store ─────────────────────────────────────────────────────
# Maps file_id → temp file path. Cleared on server restart (fine for dev).
_uploads: dict[str, str] = {}

_ALLOWED_EXTENSIONS = {".csv", ".parquet", ".sas", ".egp", ".txt", ".pdf"}


# ── Models ─────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    file_id: str
    mode: str = "data"          # "data" | "annotate"
    entity_type: str = "ciclos" # "ciclos" | "contratos" | "titularidades" | "generic"
    no_rag: bool = False
    tiered: bool = True
    grain: str = ""             # auto-derived from entity_type if blank
    backend: str = "ollama"
    ollama_model: str = "phi3:mini"
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


_ENTITY_GRAINS = {
    "ciclos":        "ciclo de recuperación IRB — período de provisión y LGD",
    "contratos":     "contrato de crédito — segmentación, producto y ventana de observación",
    "titularidades": "titularidad/garantía — avalistas, garantes y protección crediticia",
    "generic":       "registro de datos de riesgo de crédito",
}


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}")

    file_id = str(uuid.uuid4())
    tmp_path = os.path.join(tempfile.gettempdir(), f"regllm_{file_id}{suffix}")

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    _uploads[file_id] = tmp_path
    logger.info("Uploaded %s → %s (%d bytes)", file.filename, tmp_path, len(content))

    # Detect mode: .sas/.egp → annotate; .txt → sniff for SAS keywords
    if suffix in (".sas", ".egp"):
        detected_mode = "annotate"
    elif suffix == ".txt":
        _SAS_SNIFF = re.compile(r"\b(PROC|DATA\s|%MACRO|LIBNAME|RUN;|QUIT;)\b", re.IGNORECASE)
        detected_mode = "annotate" if _SAS_SNIFF.search(content.decode("utf-8", errors="replace")) else "data"
    else:
        detected_mode = "data"

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "detected_mode": detected_mode,
    }


# ── Run ────────────────────────────────────────────────────────────────────────

@router.post("/run")
async def run_compliance(body: RunRequest):
    if body.file_id not in _uploads:
        raise HTTPException(404, "File not found — upload it first")

    return StreamingResponse(
        _stream_run(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_run(body: RunRequest) -> AsyncGenerator[str, None]:
    def emit(event_type: str, payload: dict) -> str:
        return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"

    path = _uploads[body.file_id]
    suffix = Path(path).suffix.lower()

    try:
        # ── SAS annotation mode ───────────────────────────────────────────────
        if body.mode == "annotate" or suffix in (".sas", ".egp"):
            yield emit("progress", {"message": f"Parsing {Path(path).suffix.upper()} file…"})
            await asyncio.sleep(0)

            result = await asyncio.to_thread(_run_sas_annotation, path, body)
            yield emit("annotated", result)

        # ── Data validation mode ──────────────────────────────────────────────
        else:
            yield emit("progress", {"message": "Loading data file…"})
            await asyncio.sleep(0)

            row_count = await asyncio.to_thread(_count_rows, path)
            yield emit("progress", {"message": f"Loaded {row_count} rows — running compliance check…"})
            await asyncio.sleep(0)

            report = await asyncio.to_thread(_run_data_check, path, body)
            yield emit("result", {"report": report})

        yield emit("done", {})

    except Exception as e:
        logger.exception("Compliance run failed")
        yield emit("error", {"message": str(e)})


# ── Blocking workers (run in thread pool) ─────────────────────────────────────

def _build_rag(body: RunRequest):
    if body.no_rag:
        return None
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from dotenv import load_dotenv
        load_dotenv()
        from src.rag_system import RegulatoryRAGSystem
        return RegulatoryRAGSystem()
    except Exception as e:
        logger.warning("RAG unavailable: %s", e)
        return None


def _build_llm_fn(body: RunRequest):
    if body.no_rag:
        return None
    if body.backend == "ollama":
        def llm_fn(messages):
            import ollama as _ol  # type: ignore
            r = _ol.Client(host=body.ollama_host).chat(
                model=body.ollama_model,
                messages=messages,
                stream=False,
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            return r["message"]["content"]
        return llm_fn
    elif body.backend == "groq":
        from groq import Groq  # type: ignore
        gm = "llama-3.3-70b-versatile"
        def llm_fn(messages):
            return Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
                model=gm, messages=messages, temperature=0.1, max_tokens=256
            ).choices[0].message.content
        return llm_fn
    return None


def _run_sas_annotation(path: str, body: RunRequest) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.sas_parser import SASParser
    from src.compliance.sas_annotator import SASAnnotator

    blocks = SASParser().parse(path)
    rag = _build_rag(body)
    llm_fn = _build_llm_fn(body)

    annotator = SASAnnotator(rag_system=rag, llm_fn=llm_fn)
    annotated = annotator.annotate(blocks)
    full_code = annotator.render(annotated)

    warnings = []
    for ab in annotated:
        for w in ab.warnings:
            warnings.append({"task": ab.task_name, "text": w[:200]})

    return {
        "code": full_code,
        "blocks": len(blocks),
        "warnings": warnings,
    }


def _run_data_check(path: str, body: RunRequest) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import pandas as pd
    from src.compliance import ComplianceChecker

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    rag = _build_rag(body)

    grain = body.grain or _ENTITY_GRAINS.get(body.entity_type, _ENTITY_GRAINS["generic"])

    checker = ComplianceChecker(
        rag_system=rag,
        llm_backend=body.backend,
        ollama_model=body.ollama_model,
        ollama_host=body.ollama_host,
        grain_description=grain,
    )

    if body.tiered:
        report = checker.run_tiered(df, table_name=Path(path).stem)
    else:
        report = checker.run(df, table_name=Path(path).stem)

    return report


def _count_rows(path: str) -> int:
    try:
        import pandas as pd
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        return len(df)
    except Exception:
        return 0


# ── Sample data download ───────────────────────────────────────────────────────

@router.get("/sample")
async def download_sample():
    """Stream the generated sample data as a zip file."""
    sample_dir = Path(__file__).parent.parent.parent / "data" / "samples"
    if not sample_dir.exists():
        raise HTTPException(404, "Sample data not found — run generate_sample_data.py first")

    def iter_zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(sample_dir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(sample_dir))
        yield buf.getvalue()

    return StreamingResponse(
        iter_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=regllm_samples.zip"},
    )
