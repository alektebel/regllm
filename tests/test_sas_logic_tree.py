"""Tests for the SAS logic tree parser and simulator.

No external dependencies — just the parser and evaluator against
known SAS snippets matching the real IRB/IFRS9 patterns in data/samples/.
"""

from pathlib import Path

import pytest
from src.sas_logic_tree import (
    SASLogicTree,
    AssignNode, IfNode, FilterNode, DataStepNode, ProcNode,
    trace_field_ancestors,
)

tree = SASLogicTree()


# ── Parsing ───────────────────────────────────────────────────────────────────

class TestParsing:

    def test_simple_assignment(self):
        nodes = tree.parse("DATA work.out; SET work.in; ECL = PD * LGD * EAD; RUN;")
        ds = nodes[0]
        assert isinstance(ds, DataStepNode)
        assign = ds.body[0]
        assert isinstance(assign, AssignNode)
        assert assign.var == "ECL"
        assert "PD" in assign.expr

    def test_where_filter(self):
        nodes = tree.parse("DATA work.out; SET work.in; WHERE X >= 9; RUN;")
        body = nodes[0].body
        assert isinstance(body[0], FilterNode)
        assert "X" in body[0].condition

    def test_if_then_single_line(self):
        code = "DATA w.out; SET w.in; IF PD < 0.0003 THEN PD = 0.0003; RUN;"
        nodes = tree.parse(code)
        if_node = nodes[0].body[0]
        assert isinstance(if_node, IfNode)
        assert "PD" in if_node.condition
        assert isinstance(if_node.then_branch[0], AssignNode)
        assert if_node.else_branch == []

    def test_if_then_do_block(self):
        code = """
        DATA w.out; SET w.in;
        IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
            IF LGD < 0.30 THEN LGD = 0.30;
        END;
        RUN;
        """
        nodes = tree.parse(code)
        outer_if = nodes[0].body[0]
        assert isinstance(outer_if, IfNode)
        assert outer_if.condition == "COLATERAL_TIPO = 'HIPOTECA'"
        inner_if = outer_if.then_branch[0]
        assert isinstance(inner_if, IfNode)
        assert isinstance(inner_if.then_branch[0], AssignNode)

    def test_if_then_else(self):
        code = """
        DATA w.out; SET w.in;
        IF STAGE = 1 THEN PERIOD = 12;
        ELSE PERIOD = 24;
        RUN;
        """
        nodes = tree.parse(code)
        if_node = nodes[0].body[0]
        assert isinstance(if_node, IfNode)
        assert len(if_node.then_branch) == 1
        assert len(if_node.else_branch) == 1
        assert if_node.else_branch[0].expr == "24"

    def test_multiple_data_steps(self):
        code = """
        DATA w.a; SET w.x; A = 1; RUN;
        DATA w.b; SET w.a; B = 2; RUN;
        """
        nodes = tree.parse(code)
        assert len(nodes) == 2
        assert nodes[0].output_dataset == "w.a"
        assert nodes[1].output_dataset == "w.b"

    def test_proc_captured(self):
        code = "PROC MEANS DATA=work.out N MEAN; VAR PD; RUN;"
        nodes = tree.parse(code)
        assert isinstance(nodes[0], ProcNode)
        assert nodes[0].kind == "MEANS"
        assert nodes[0].data == "work.out"

    def test_block_comments_stripped(self):
        code = """
        DATA w.out; SET w.in;
        /* This is a comment; with semicolons; inside */
        ECL = PD * LGD * EAD;
        RUN;
        """
        nodes = tree.parse(code)
        assert nodes[0].body[0].var == "ECL"

    def test_set_dataset_captured(self):
        code = "DATA w.out; SET mylib.ciclos_recuperacion; RUN;"
        ds = tree.parse(code)[0]
        assert ds.input_dataset == "mylib.ciclos_recuperacion"

    def test_real_lgd_snippet(self):
        """Parse a representative slice of the real LGD sample file."""
        code = """
        DATA work.lgd_con_suelos;
            SET work.ciclos;
            IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
                IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
            END;
            IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
        RUN;

        DATA work.ecl_calculo;
            SET work.lgd_con_suelos;
            IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
            ECL = PD_ESTIMADA * LGD_ESTIMADA * EAD;
        RUN;
        """
        nodes = tree.parse(code)
        assert len(nodes) == 2
        assert isinstance(nodes[0], DataStepNode)
        assert isinstance(nodes[1], DataStepNode)
        # First step has two IF nodes
        assert sum(1 for n in nodes[0].body if isinstance(n, IfNode)) == 2
        # Second step ends with an ECL assignment
        last = nodes[1].body[-1]
        assert isinstance(last, AssignNode)
        assert last.var == "ECL"


# ── Display and serialization ─────────────────────────────────────────────────

class TestDisplay:

    def test_display_returns_string(self):
        nodes = tree.parse("DATA w.out; SET w.in; X = 1; RUN;")
        s = tree.display(nodes)
        assert "DATA w.out" in s
        assert "X = 1" in s

    def test_to_dict_structure(self):
        code = "DATA w.out; SET w.in; IF A = 1 THEN B = 2; RUN;"
        nodes = tree.parse(code)
        d = tree.to_dict(nodes)
        assert d[0]["type"] == "data_step"
        assert d[0]["output"] == "w.out"
        if_d = d[0]["body"][0]
        assert if_d["type"] == "if"
        assert "A" in if_d["condition"]
        assert if_d["then"][0]["type"] == "assign"

    def test_to_dict_proc(self):
        nodes = tree.parse("PROC MEANS DATA=work.t; RUN;")
        d = tree.to_dict(nodes)
        assert d[0]["type"] == "proc"
        assert d[0]["kind"] == "MEANS"


# ── Evaluation ────────────────────────────────────────────────────────────────

class TestEvaluation:

    def _parse_eval(self, code, values):
        nodes = tree.parse(code)
        return tree.evaluate(nodes, values)

    def test_assignment_updates_value(self):
        trace = self._parse_eval(
            "DATA w.o; SET w.i; ECL = PD * LGD * EAD; RUN;",
            {"PD": 0.01, "LGD": 0.45, "EAD": 100_000},
        )
        assert abs(trace.final["ECL"] - 450.0) < 1e-6

    def test_if_branch_taken_when_true(self):
        trace = self._parse_eval(
            "DATA w.o; SET w.i; IF PD < 0.0003 THEN PD = 0.0003; RUN;",
            {"PD": 0.0001},
        )
        assert trace.final["PD"] == pytest.approx(0.0003)
        assert any(s.kind == "if_taken" for s in trace.steps)

    def test_if_branch_skipped_when_false(self):
        trace = self._parse_eval(
            "DATA w.o; SET w.i; IF PD < 0.0003 THEN PD = 0.0003; RUN;",
            {"PD": 0.01},
        )
        assert trace.final["PD"] == pytest.approx(0.01)  # unchanged
        assert any(s.kind == "if_skipped" for s in trace.steps)

    def test_else_branch_taken(self):
        trace = self._parse_eval(
            "DATA w.o; SET w.i; IF STAGE = 1 THEN PERIOD = 12; ELSE PERIOD = 36; RUN;",
            {"STAGE": 2},
        )
        assert trace.final["PERIOD"] == 36

    def test_where_passes(self):
        trace = self._parse_eval(
            "DATA w.o; SET w.i; WHERE PROVISION_PERIOD_MONTHS >= 9; RUN;",
            {"PROVISION_PERIOD_MONTHS": 12},
        )
        assert trace.row_passes_filter is True

    def test_where_blocks(self):
        trace = self._parse_eval(
            "DATA w.o; SET w.i; WHERE PROVISION_PERIOD_MONTHS >= 9; RUN;",
            {"PROVISION_PERIOD_MONTHS": 6},
        )
        assert trace.row_passes_filter is False

    def test_nested_if_do_block(self):
        code = """
        DATA w.o; SET w.i;
        IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
            IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
        END;
        RUN;
        """
        trace = self._parse_eval(code, {"COLATERAL_TIPO": "HIPOTECA", "LGD_ESTIMADA": 0.20})
        assert trace.final["LGD_ESTIMADA"] == pytest.approx(0.30)

    def test_nested_if_outer_skipped(self):
        """Outer IF false → inner assignment not reached."""
        code = """
        DATA w.o; SET w.i;
        IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
            IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
        END;
        RUN;
        """
        trace = self._parse_eval(code, {"COLATERAL_TIPO": "CORP", "LGD_ESTIMADA": 0.20})
        assert trace.final["LGD_ESTIMADA"] == pytest.approx(0.20)  # unchanged

    def test_full_lgd_ecl_simulation(self):
        """End-to-end simulation matching the real sample_lgd.sas logic."""
        code = """
        DATA work.lgd_con_suelos;
            SET work.ciclos;
            WHERE PROVISION_PERIOD_MONTHS >= 9;
            IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
                IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
            END;
            IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
        RUN;

        DATA work.ecl_calculo;
            SET work.lgd_con_suelos;
            IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
            ECL = PD_ESTIMADA * LGD_ESTIMADA * EAD;
            IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
                STAGE_IFRS9 = 2;
            END;
        RUN;
        """
        # Row: CORP exposure with PD below floor and DPD triggering Stage 2
        values = {
            "PROVISION_PERIOD_MONTHS": 12,
            "COLATERAL_TIPO": "NINGUNA",
            "SEGMENTO": "CORP",
            "LGD_ESTIMADA": 0.30,   # below 45% corp floor
            "PD_ESTIMADA": 0.0001,  # below 0.03% floor
            "EAD": 200_000,
            "DPDS": 45,
            "STAGE_IFRS9": 1,       # should be reclassified
        }
        trace = self._parse_eval(code, values)

        # LGD should be raised to corp floor
        assert trace.final["LGD_ESTIMADA"] == pytest.approx(0.45)
        # PD should be raised to regulatory floor
        assert trace.final["PD_ESTIMADA"] == pytest.approx(0.0003)
        # ECL = 0.0003 * 0.45 * 200_000
        assert trace.final["ECL"] == pytest.approx(0.0003 * 0.45 * 200_000)
        # Stage should be reclassified
        assert trace.final["STAGE_IFRS9"] == 2
        assert trace.row_passes_filter is True

    def test_trace_summary_contains_key_info(self):
        trace = self._parse_eval(
            "DATA w.o; SET w.i; IF PD < 0.0003 THEN PD = 0.0003; RUN;",
            {"PD": 0.0001},
        )
        summary = trace.summary()
        assert "PD" in summary
        assert "0.0001" in summary
        assert "0.0003" in summary


class TestTraceFieldAncestors:

    def test_ecl_backward_deps(self):
        code = """
        DATA work.ecl_calc;
            SET work.inputs;
            ECL = PD_ESTIMADA * LGD_ESTIMADA * EAD;
        RUN;
        """
        nodes = tree.parse(code)
        lg = tree.lineage(nodes)
        trace = trace_field_ancestors(lg, "ECL")
        assert trace["found"] is True
        assert {"PD_ESTIMADA", "LGD_ESTIMADA", "EAD"}.issubset(set(trace["ancestors"]))
        assert trace["depth"]["ECL"] == 0
        assert trace["depth"]["EAD"] == 1
        assert len(trace["layers"]) >= 2

    def test_intermediate_field_trace(self):
        code = """
        DATA work.step1;
            SET work.inputs;
            X = A + B;
        RUN;
        DATA work.step2;
            SET work.step1;
            Y = X * C;
        RUN;
        """
        nodes = tree.parse(code)
        lg = tree.lineage(nodes)
        trace = trace_field_ancestors(lg, "Y")
        assert "X" in trace["ancestors"]
        assert {"A", "B", "C"}.issubset(set(trace["ancestors"]))
        assert len(trace["nodes"]) == len(trace["depth"])

    def test_api_lineage_with_target(self):
        from fastapi.testclient import TestClient
        from api.main import app
        code = "DATA w.o; SET w.i; ECL = PD * LGD * EAD; RUN;"
        client = TestClient(app)
        r = client.post("/sas/lineage", json={
            "code": code,
            "target": "ECL",
            "ancestors_only": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["target"] == "ECL"
        assert {"PD", "LGD", "EAD"}.issubset(set(data["ancestors"]))


class TestRealSasPrograms:

    def test_or_conced_program_compiles_lineage(self):
        code = (
            Path(__file__).resolve().parents[1]
            / "data/sas/v3/L14_P51_OR_CONCED.sas"
        ).read_text(encoding="utf-8")

        nodes = tree.parse(code)
        lg = tree.lineage(nodes)
        field_ids = {node["id"] for node in lg.nodes}

        assert nodes
        assert {"OR_CONCED", "OR_CONCED_01", "CAMP_RECUPERAT"}.issubset(field_ids)

        or_conced_01 = trace_field_ancestors(lg, "OR_CONCED_01")
        assert or_conced_01["found"] is True
        assert any(
            edge["source"] == "OR_CONCED"
            for edge in or_conced_01["direct_predecessors"]
        )

        camp_recuperat = trace_field_ancestors(lg, "CAMP_RECUPERAT")
        assert camp_recuperat["found"] is True
        assert {
            "OR_CONCED_01_PRI",
            "OR_CONCED_02_PRI",
            "OR_CONCED_03_PRI",
            "OR_CONCED_04_PRI",
            "OR_CONCED_05_PRI",
            "OR_CONCED_06_PRI",
        }.issubset(set(camp_recuperat["ancestors"]))

    def test_all_v3_sas_files_compile(self):
        v3_root = Path(__file__).resolve().parents[1] / "data/sas/v3"
        files = sorted(v3_root.rglob("*.sas"))
        assert files, "expected at least one v3 SAS file"
        for f in files:
            code = f.read_text(encoding="utf-8")
            nodes = tree.parse(code)
            lg = tree.lineage(nodes)
            assert nodes, f"{f} produced no AST nodes"
            assert lg.nodes, f"{f} produced no lineage fields"

    def test_trace_sample_endpoint_uses_full_v3(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        sample = client.get("/sas/sample")
        assert sample.status_code == 200
        code = sample.json()["code"]

        # The combined v3 program spans every v3 file, so OR_CONCED (defined in
        # L14_P51_OR_CONCED.sas) must be traceable through the lineage endpoint.
        lineage_res = client.post("/sas/lineage", json={"code": code})
        assert lineage_res.status_code == 200
        field_ids = {n["id"] for n in lineage_res.json()["nodes"]}
        assert "OR_CONCED" in field_ids
        assert "CAMP_RECUPERAT" in field_ids
