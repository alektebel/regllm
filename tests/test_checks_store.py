"""Parity tests for the two ChecksStore backends (SQLite + DynamoDB).

The same sequence of operations is run against both stores and must yield
the same observable behaviour, so ``api/routers/dqc.py`` is backend-agnostic.
The DynamoDB backend is exercised against a moto mock (no AWS account).
"""

from __future__ import annotations

import pytest

from training.dq.checks_store import DynamoStore, SqliteStore
from training.dq.coherence_rules import PK_COLUMN

SQL1 = f'SELECT "{PK_COLUMN}" FROM lgd WHERE lgd_realizada < 0'
SQL2 = f'SELECT "{PK_COLUMN}" FROM lgd WHERE ead IS NULL'


def _exercise(store) -> None:
    # -- insert two visible pending checks --
    c1 = store.insert_check(
        name="lgd_neg", description="LGD negativa", severity="HIGH",
        category="rango", sql=SQL1, variable="LGD_REALIZADA",
        campos_entrada=["LGD_REALIZADA", "EAD"],
    )
    c2 = store.insert_check(
        name="ead_null", description="EAD nulo", severity="MED",
        category="completitud", sql=SQL2,
    )
    assert c1 != c2

    # -- campos_entrada round-trips as a list --
    got = store.get_check(c1)
    assert got["campos_entrada"] == ["LGD_REALIZADA", "EAD"]
    assert got["status"] == "pending" and bool(got["visible"]) is True

    # -- listing + counts before validation --
    assert len(store.list_checks(status="pending", visible=True)) == 2
    pre = store.counts()
    assert pre["pending_visible"] == 2 and pre["dashboard_ready"] is False

    # -- validate one, reject the other --
    assert store.set_status(c1, "validated") is True
    assert store.set_status(c2, "rejected") is True
    assert store.set_status("chk_missing", "validated") is False  # 404 path

    post = store.counts()
    assert post == {
        "pending_visible": 0, "validated": 1, "rejected": 1,
        "oculto": 0, "dashboard_ready": True,
    }
    assert store.get_check(c1)["validated_at"] is not None

    # -- dashboard query (single validated check → no UNION) --
    q = store.build_dashboard_query(status="validated")
    assert q is not None and c1 in q and "UNION ALL" not in q
    assert f'"{PK_COLUMN}" AS pk' in q
    assert [r["check_id"] for r in store.export_validated()] == [c1]

    # -- delete is idempotent-safe (returns False the second time) --
    assert store.delete_check(c2) is True
    assert store.delete_check(c2) is False

    # -- rule_id insert dedups on (rule_id, sql) --
    r1 = store.insert_check(name="r", description="", severity="LOW",
                            category="formula", sql=SQL1, rule_id="R1")
    r2 = store.insert_check(name="r", description="", severity="LOW",
                            category="formula", sql=SQL1, rule_id="R1")
    assert r1 == r2


def test_sqlite_store(monkeypatch, tmp_path):
    import training.dq.checks_db as cdb
    db = tmp_path / "checks.db"
    # Point every SqliteStore connection at an isolated temp DB.
    monkeypatch.setattr(cdb, "connect", lambda *a, **k: _connect_at(db, cdb._SCHEMA))
    _exercise(SqliteStore())


def _connect_at(db, schema):
    import sqlite3
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.executescript(schema)
    return conn


def test_dynamo_store():
    moto = pytest.importorskip("moto")
    import boto3
    with moto.mock_aws():
        boto3.resource("dynamodb", region_name="eu-west-1").create_table(
            TableName="regllm-dqc-checks",
            KeySchema=[{"AttributeName": "check_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "check_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        _exercise(DynamoStore(table_name="regllm-dqc-checks", region="eu-west-1"))
