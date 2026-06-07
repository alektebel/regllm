"""Unit tests for src/agent/code_diff.py."""

from __future__ import annotations

from src.agent.code_diff import compare


_BASE_SAS = """\
DATA work.a;
    SET mylib.input;
    X = A + B;
RUN;

DATA work.b;
    SET work.a;
    Y = X * 2;
RUN;
"""


def test_identical_versions_have_no_changes() -> None:
    res = compare(target="Y", sas_v2=_BASE_SAS, sas_v3=_BASE_SAS)
    assert res.v2_present and res.v3_present
    assert not res.added_steps
    assert not res.removed_steps
    assert not res.modified_steps
    assert "WORK.A" in res.unchanged_steps
    assert "WORK.B" in res.unchanged_steps


def test_modified_step_detected() -> None:
    v3 = _BASE_SAS.replace("Y = X * 2;", "Y = X * 3;")
    res = compare(target="Y", sas_v2=_BASE_SAS, sas_v3=v3)
    assert not res.added_steps
    assert not res.removed_steps
    mods = [m["output"] for m in res.modified_steps]
    assert "WORK.B" in mods, f"expected WORK.B modified; got {mods}"
    # The unrelated step must remain unchanged
    assert "WORK.A" in res.unchanged_steps


def test_added_step_detected() -> None:
    v3 = _BASE_SAS + """
DATA work.c;
    SET work.b;
    Z = Y + 1;
RUN;
"""
    res = compare(target="Z", sas_v2=_BASE_SAS, sas_v3=v3)
    added = [s["output"] for s in res.added_steps]
    assert "WORK.C" in added


def test_removed_step_detected() -> None:
    v2 = _BASE_SAS + """
DATA work.legacy;
    SET work.b;
    LEGACY_FLAG = 1;
RUN;
"""
    res = compare(target="LEGACY_FLAG", sas_v2=v2, sas_v3=_BASE_SAS)
    removed = [s["output"] for s in res.removed_steps]
    assert "WORK.LEGACY" in removed


def test_target_scoping_filters_unrelated_steps() -> None:
    """A step that doesn't feed the target must not appear in the diff."""
    v2 = _BASE_SAS + """
DATA work.unrelated;
    SET mylib.other;
    Q = 7;
RUN;
"""
    v3 = v2.replace("Q = 7;", "Q = 9;")
    # Target Y has nothing to do with WORK.UNRELATED; it must be filtered out
    res = compare(target="Y", sas_v2=v2, sas_v3=v3)
    mods = [m["output"] for m in res.modified_steps]
    assert "WORK.UNRELATED" not in mods


def test_unified_diff_present_when_changes_exist() -> None:
    v3 = _BASE_SAS.replace("Y = X * 2;", "Y = X * 5;")
    res = compare(target="Y", sas_v2=_BASE_SAS, sas_v3=v3)
    assert "X * 2" in res.unified_diff or "X * 5" in res.unified_diff
