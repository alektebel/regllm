"""Tests for ``training.dq.checks_db.build_dashboard_query``.

Guards the UNION ALL builder against SQL injection via ``check_id``/``name``
(both are attacker-shaped strings from an LLM-generated ``dqc_id`` in
practice) and confirms the generated SQL still executes correctly.
"""

from __future__ import annotations

import pytest

from training.dq import checks_db


@pytest.fixture()
def conn(tmp_path):
    c = checks_db.connect(tmp_path / "checks.db")
    yield c
    c.close()


def test_returns_none_when_no_checks(conn):
    assert checks_db.build_dashboard_query(conn) is None


def test_builds_valid_union_all_sql(conn):
    checks_db.insert_check(
        conn, name="chk1", description="d", severity="HIGH", category="rango",
        sql=f'SELECT "{checks_db.PK_COLUMN}" FROM t WHERE x > 1',
        status="validated",
    )
    checks_db.insert_check(
        conn, name="chk2", description="d", severity="MED", category="rango",
        sql=f'SELECT "{checks_db.PK_COLUMN}" FROM t WHERE x > 2',
        status="validated",
    )
    conn.execute(f'CREATE TABLE t ("{checks_db.PK_COLUMN}" TEXT, x INTEGER)')
    conn.execute(f'INSERT INTO t VALUES (\'CIC_1\', 3)')
    conn.commit()

    sql = checks_db.build_dashboard_query(conn)
    assert "UNION ALL" in sql
    rows = conn.execute(sql).fetchall()
    # x=3 satisfies both x>1 and x>2, so both checks fire for CIC_1
    assert len(rows) == 2


def test_check_id_with_sql_metacharacters_is_escaped_not_executed(conn):
    """check_id is always internally generated (chk_<hex>) today, but the
    builder must not trust that — a quote/semicolon in check_id must not
    break out of the string literal or the UNION ALL structure."""
    malicious_id = "chk_evil'; DROP TABLE checks; --"
    checks_db.insert_check(
        conn, name="chk", description="d", severity="HIGH", category="rango",
        sql=f'SELECT "{checks_db.PK_COLUMN}" FROM t',
        status="validated", check_id=malicious_id,
    )
    conn.execute(f'CREATE TABLE t ("{checks_db.PK_COLUMN}" TEXT)')
    conn.execute("INSERT INTO t VALUES ('CIC_1')")
    conn.commit()

    sql = checks_db.build_dashboard_query(conn)
    rows = conn.execute(sql).fetchall()  # must not raise / must not drop the table
    assert len(rows) == 1
    assert rows[0]["check_id"] == malicious_id
    # Table must still exist — no injection occurred.
    assert conn.execute("SELECT * FROM checks").fetchall()


def test_name_with_quote_is_escaped(conn):
    checks_db.insert_check(
        conn, name="O'Brien's check", description="d", severity="HIGH",
        category="rango", sql=f'SELECT "{checks_db.PK_COLUMN}" FROM t',
        status="validated",
    )
    conn.execute(f'CREATE TABLE t ("{checks_db.PK_COLUMN}" TEXT)')
    conn.execute("INSERT INTO t VALUES ('CIC_1')")
    conn.commit()

    sql = checks_db.build_dashboard_query(conn)
    rows = conn.execute(sql).fetchall()
    assert rows[0]["check_name"] == "O'Brien's check"
