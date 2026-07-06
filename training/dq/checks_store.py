"""Storage-agnostic persistence for DQ checks (the validation pipeline).

The API router talks to a :class:`ChecksStore`, not to a concrete database.
Two backends are provided and selected by the ``CHECKS_BACKEND`` env var:

  * ``sqlite`` (default) — local, offline, zero-config. Wraps the existing
    :mod:`training.dq.checks_db` SQLite layer. Used for local development.
  * ``dynamodb`` — a DynamoDB table named by ``CHECKS_TABLE``. Used in AWS
    (the CDK stack creates the table and injects the env vars).

Both backends expose the same method set and return the same row dicts, so
``api/routers/dqc.py`` is identical regardless of backend. The dashboard
UNION-ALL query is storage-agnostic and built by
:func:`dashboard_query_from_rows`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from training.dq.checks_db import _SEVERITY_CANON
from training.dq.coherence_rules import PK_COLUMN

# Canonical column/attribute set shared by both backends — used to normalise
# a DynamoDB item back into the same shape SQLite's ``SELECT *`` returns.
FIELDS = (
    "check_id", "rule_id", "name", "description", "severity", "category",
    "sql", "visible", "status", "reward", "variable", "tipo",
    "condicion_error", "campos_entrada", "referencia_regulatoria", "umbral",
    "periodicidad", "justificacion", "created_at", "validated_at",
)


def dashboard_query_from_rows(rows: list[dict]) -> str | None:
    """UNION ALL of every check's per-PK violations (storage-agnostic).

    Contract: each check's ``sql`` must ``SELECT`` :data:`PK_COLUMN`
    (``ID_CONTR_CICLO_LGD``); the wrapper projects it as ``pk`` so the
    dashboard rowset is uniform. Returns ``None`` for an empty input.
    """
    if not rows:
        return None
    parts: list[str] = []
    for r in rows:
        inner = r["sql"].strip().rstrip(";").strip()
        alias = "_" + r["check_id"].replace("-", "_")
        sev = _SEVERITY_CANON.get(r["severity"], r["severity"])
        name_lit = r["name"].replace("'", "''")
        parts.append(
            f"SELECT '{r['check_id']}' AS check_id, "
            f"'{name_lit}' AS check_name, "
            f"'{sev}' AS severity, "
            f"{alias}.\"{PK_COLUMN}\" AS pk "
            f"FROM ({inner}) AS {alias}"
        )
    return "\nUNION ALL\n".join(parts)


def _counts_from_rows(rows: list[dict]) -> dict[str, int]:
    """Status × visibility breakdown + ``dashboard_ready`` flag."""
    out = {"pending_visible": 0, "validated": 0, "rejected": 0, "oculto": 0}
    for r in rows:
        s, v = r["status"], bool(r["visible"])
        if s == "pending" and v:
            out["pending_visible"] += 1
        if s == "validated":
            out["validated"] += 1
            if not v:
                out["oculto"] += 1
        if s == "rejected":
            out["rejected"] += 1
    out["dashboard_ready"] = out["pending_visible"] == 0 and out["validated"] > 0
    return out


# ── SQLite backend ──────────────────────────────────────────────────────────

class SqliteStore:
    """Local backend — a thin adapter over :mod:`training.dq.checks_db`.

    Opens a fresh connection per call (SQLite connections are thread-bound
    and the API is served by a threadpool).
    """

    def __init__(self) -> None:
        from training.dq import checks_db
        self._db = checks_db

    def insert_check(self, **kwargs: Any) -> str:
        return self._db.insert_check(self._db.connect(), **kwargs)

    def set_status(self, check_id: str, status: str) -> bool:
        return self._db.set_status(self._db.connect(), check_id, status)

    def list_checks(self, *, status: str | None = None,
                    visible: bool | None = None) -> list[dict]:
        return self._db.list_checks(self._db.connect(), status=status, visible=visible)

    def get_check(self, check_id: str) -> dict | None:
        return self._db.get_check(self._db.connect(), check_id)

    def counts(self) -> dict[str, int]:
        return self._db.counts(self._db.connect())

    def delete_check(self, check_id: str) -> bool:
        conn = self._db.connect()
        cur = conn.execute("DELETE FROM checks WHERE check_id=?", (check_id,))
        conn.commit()
        return cur.rowcount > 0

    def build_dashboard_query(self, *, status: str = "validated") -> str | None:
        return dashboard_query_from_rows(self.list_checks(status=status))

    def export_validated(self) -> list[dict]:
        return self.list_checks(status="validated")


# ── DynamoDB backend ────────────────────────────────────────────────────────

class DynamoStore:
    """AWS backend — one item per check in a single DynamoDB table.

    Partition key: ``check_id`` (String). Volumes are small (a human
    validates each check), so ``list``/``counts`` use ``scan`` + in-Python
    filtering — no GSI required.
    """

    def __init__(self, table_name: str | None = None,
                 region: str | None = None) -> None:
        import boto3
        table_name = table_name or os.getenv("CHECKS_TABLE")
        if not table_name:
            raise RuntimeError(
                "CHECKS_BACKEND=dynamodb requires CHECKS_TABLE to be set"
            )
        region = (region or os.getenv("AWS_REGION")
                  or os.getenv("BEDROCK_REGION") or os.getenv("AWS_DEFAULT_REGION"))
        kwargs = {"region_name": region} if region else {}
        self._table = boto3.resource("dynamodb", **kwargs).Table(table_name)

    # -- helpers --
    @staticmethod
    def _to_item(d: dict) -> dict:
        """Python dict → DynamoDB item: drop Nones, floats → Decimal."""
        from decimal import Decimal
        item: dict[str, Any] = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, float):
                v = Decimal(str(v))
            item[k] = v
        return item

    @staticmethod
    def _normalise(item: dict | None) -> dict | None:
        """DynamoDB item → the canonical row dict both backends return."""
        if item is None:
            return None
        from decimal import Decimal
        out: dict[str, Any] = {}
        for f in FIELDS:
            v = item.get(f)
            if isinstance(v, Decimal):
                v = float(v)
            out[f] = v
        out["visible"] = bool(out.get("visible"))
        if out.get("campos_entrada") is None:
            out["campos_entrada"] = []
        return out

    def insert_check(self, *, name: str, description: str, severity: str,
                     category: str, sql: str, check_id: str | None = None,
                     rule_id: str | None = None, visible: bool = True,
                     reward: float | None = None, status: str = "pending",
                     variable: str | None = None, tipo: str | None = None,
                     condicion_error: str | None = None,
                     campos_entrada: list[str] | None = None,
                     referencia_regulatoria: str | None = None,
                     umbral: str | None = None, periodicidad: str | None = None,
                     justificacion: str | None = None) -> str:
        import uuid
        from datetime import datetime
        from boto3.dynamodb.conditions import Attr

        # Idempotent on (rule_id, sql) — mirror the SQLite dedup for RL rows.
        if rule_id:
            hit = self._table.scan(
                FilterExpression=Attr("rule_id").eq(rule_id) & Attr("sql").eq(sql),
                ProjectionExpression="check_id",
            ).get("Items", [])
            if hit:
                return hit[0]["check_id"]

        cid = check_id or f"chk_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._table.put_item(Item=self._to_item({
            "check_id": cid, "rule_id": rule_id, "name": name,
            "description": description, "severity": severity, "category": category,
            "sql": sql, "visible": visible, "status": status, "reward": reward,
            "variable": variable, "tipo": tipo, "condicion_error": condicion_error,
            "campos_entrada": campos_entrada or None,
            "referencia_regulatoria": referencia_regulatoria, "umbral": umbral,
            "periodicidad": periodicidad, "justificacion": justificacion,
            "created_at": now,
        }))
        return cid

    def set_status(self, check_id: str, status: str) -> bool:
        from datetime import datetime
        from botocore.exceptions import ClientError
        if status not in ("pending", "validated", "rejected"):
            raise ValueError(f"invalid status: {status}")
        expr = "SET #s = :status"
        values: dict[str, Any] = {":status": status}
        if status == "validated":
            expr += ", validated_at = if_not_exists(validated_at, :now)"
            values[":now"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        try:
            self._table.update_item(
                Key={"check_id": check_id},
                UpdateExpression=expr,
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues=values,
                ConditionExpression="attribute_exists(check_id)",
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def list_checks(self, *, status: str | None = None,
                    visible: bool | None = None) -> list[dict]:
        rows = [self._normalise(i) for i in self._scan_all()]
        if status is not None:
            rows = [r for r in rows if r["status"] == status]
        if visible is not None:
            rows = [r for r in rows if bool(r["visible"]) == visible]
        rows.sort(key=lambda r: (r["severity"] or "", r["created_at"] or ""),
                  reverse=True)
        return rows

    def get_check(self, check_id: str) -> dict | None:
        return self._normalise(
            self._table.get_item(Key={"check_id": check_id}).get("Item")
        )

    def counts(self) -> dict[str, int]:
        return _counts_from_rows([self._normalise(i) for i in self._scan_all()])

    def delete_check(self, check_id: str) -> bool:
        res = self._table.delete_item(
            Key={"check_id": check_id}, ReturnValues="ALL_OLD"
        )
        return "Attributes" in res

    def build_dashboard_query(self, *, status: str = "validated") -> str | None:
        return dashboard_query_from_rows(self.list_checks(status=status))

    def export_validated(self) -> list[dict]:
        return self.list_checks(status="validated")

    def _scan_all(self) -> list[dict]:
        items: list[dict] = []
        kwargs: dict[str, Any] = {}
        while True:
            resp = self._table.scan(**kwargs)
            items.extend(resp.get("Items", []))
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                return items
            kwargs["ExclusiveStartKey"] = lek


# ── Factory ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_store() -> SqliteStore | DynamoStore:
    """Return the configured checks store (memoised).

    ``CHECKS_BACKEND=dynamodb`` selects DynamoDB; anything else (default)
    selects local SQLite.
    """
    backend = os.getenv("CHECKS_BACKEND", "sqlite").lower()
    if backend == "dynamodb":
        return DynamoStore()
    return SqliteStore()
