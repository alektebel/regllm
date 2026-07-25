"""DQC API schemas — the wire contract, free of behaviour.

Split out of ``dqc.py`` so the transport shapes live in one place: the route
handlers, the persistence layer (``dqc_store``) and the upload helpers
(``dqc_uploads``) all speak in terms of these models without importing each
other. Pure Pydantic — no I/O, no database, no LLM.
"""

from __future__ import annotations

from pydantic import BaseModel


class DQCItem(BaseModel):
    """One generated data-quality check, as the LLM emits it."""

    dqc_id: str = ""
    prev_id: str = ""              # previous/official id given by the user
    variable: str = ""
    descripcion: str = ""
    tipo: str = ""
    severidad: str = ""
    regla_sql: str = ""
    condicion_error: str = ""
    campos_entrada: list[str] = []
    referencia_regulatoria: str = ""
    umbral: str = ""
    periodicidad: str = ""
    justificacion: str = ""


class GenerateResponse(BaseModel):
    dqcs: list[DQCItem]
    dictionary_fields: int
    context_summary: str = ""
    sheet_used: str = ""
    mapping_source: str = ""       # llm | heuristic | user
    formats_inferred: int = 0
    agents_used: int = 0           # stateless LLM calls spent on this run


class SheetSummary(BaseModel):
    name: str
    rows: int
    headers: list[str]
    score: int


class InspectResponse(BaseModel):
    sheets: list[SheetSummary]
    proposed_sheet: str | None = None
    column_mapping: dict[str, str | None] = {}
    confidence: float = 0.0
    source: str = "heuristic"      # llm | heuristic
    question: str | None = None    # non-null ⇒ the UI should ask the user
    options: list[str] = []


class CheckRecord(BaseModel):
    """A stored check. ``sql`` is null and ``motivo`` set for the
    ambiguous/errored items that never produced a query."""

    check_id: str
    rule_id: str | None = None
    name: str
    description: str = ""
    severity: str
    category: str
    sql: str | None = None
    visible: bool = True
    status: str = "pending"
    reward: float | None = None
    motivo: str | None = None
    variable: str | None = None
    tipo: str | None = None
    condicion_error: str | None = None
    campos_entrada: list[str] = []
    referencia_regulatoria: str | None = None
    umbral: str | None = None
    periodicidad: str | None = None
    justificacion: str | None = None
    created_at: str | None = None
    validated_at: str | None = None


class EvalResult(BaseModel):
    check_id: str
    name: str
    prev_id: str | None = None
    descripcion: str = ""
    condicion_error: str = ""      # human explanation of why a row is flagged
    ok: bool
    error: str = ""
    n_casos: int = 0
    columnas: list[str] = []
    ejemplos: list[dict] = []
    precision: float | None = None
    recall: float | None = None
    esperados: int | None = None


class EvaluateResponse(BaseModel):
    casos: int
    resultados: list[EvalResult]
    evaluados: int                 # queries that executed successfully
    fallidos: int                  # queries that errored on the cases
    precision_media: float | None = None
    recall_medio: float | None = None


class StatusUpdate(BaseModel):
    status: str  # "validated" | "rejected"


class DashboardResponse(BaseModel):
    ready: bool
    pending_visible: int
    validated: int
    rejected: int
    oculto: int
    sql: str | None = None
    checks: list[CheckRecord] = []
