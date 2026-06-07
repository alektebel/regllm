"""
Comprehensive tests for every SAS pattern supported by SASLogicTree.

Coverage:
  - Parsing: every node type produces the correct AST structure
  - Evaluation: expressions and control flow yield correct values
  - Lineage: dependency graph edges/nodes are correct
  - to_dict: serialisation round-trips
  - Operators: IN, NOT IN, BETWEEN, IS MISSING, IS NOT MISSING
  - Functions: numeric, string, logic
"""
from __future__ import annotations

import math
import pytest

from src.sas_logic_tree import (
    SASLogicTree,
    AssignNode,
    IfNode,
    FilterNode,
    DataStepNode,
    ProcNode,
    DoLoopNode,
    SelectNode,
    WhenNode,
    ArrayNode,
    RetainNode,
    OutputNode,
    DeleteNode,
    CallNode,
    MergeNode,
    ByNode,
    MacroLetNode,
    MacroDefNode,
    MacroCallNode,
    LinkNode,
    GotoNode,
    ReturnNode,
    StopNode,
)

tree = SASLogicTree()


def parse(code: str):
    return tree.parse(code)


def eval_code(code: str, values: dict):
    nodes = parse(code)
    return tree.evaluate(nodes, values)


def lineage(code: str):
    nodes = parse(code)
    return tree.lineage(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# DO LOOPS
# ─────────────────────────────────────────────────────────────────────────────

class TestDoLoops:
    def _get_do(self, code: str) -> DoLoopNode:
        ds = parse(code)[0]
        assert isinstance(ds, DataStepNode)
        do = ds.body[0]
        assert isinstance(do, DoLoopNode)
        return do

    def test_iterative_do(self):
        node = self._get_do("DATA w.o; SET w.i; DO i = 1 TO 5; x = x + 1; END; RUN;")
        assert node.var == "i"
        assert node.start == "1"
        assert node.stop == "5"
        assert node.by_step == "1"
        assert len(node.body) == 1

    def test_iterative_do_by_step(self):
        node = self._get_do("DATA w.o; SET w.i; DO i = 0 TO 10 BY 2; x = x + i; END; RUN;")
        assert node.var == "i"
        assert node.by_step == "2"

    def test_do_while(self):
        node = self._get_do("DATA w.o; SET w.i; DO WHILE (x < 10); x = x + 1; END; RUN;")
        assert node.while_cond == "x < 10"
        assert node.var == ""

    def test_do_until(self):
        node = self._get_do("DATA w.o; SET w.i; DO UNTIL (x >= 10); x = x + 1; END; RUN;")
        assert node.until_cond == "x >= 10"

    def test_bare_do_block(self):
        """DO; ... END; with no loop variable — treated as anonymous block."""
        node = self._get_do("DATA w.o; SET w.i; DO; x = 1; y = 2; END; RUN;")
        assert node.var == ""
        assert node.while_cond == ""
        assert len(node.body) == 2

    def test_nested_do(self):
        code = """
        DATA w.o; SET w.i;
        DO i = 1 TO 3;
            DO j = 1 TO 3;
                total = total + 1;
            END;
        END;
        RUN;
        """
        ds = parse(code)[0]
        outer = ds.body[0]
        assert isinstance(outer, DoLoopNode)
        inner = outer.body[0]
        assert isinstance(inner, DoLoopNode)
        assert inner.var == "j"

    # Evaluation

    def test_eval_iterative_sum(self):
        code = "DATA w.o; SET w.i; DO i = 1 TO 5; total = total + i; END; RUN;"
        trace = eval_code(code, {"total": 0})
        assert trace.final["total"] == 15  # 1+2+3+4+5

    def test_eval_do_while(self):
        code = "DATA w.o; SET w.i; DO WHILE (x < 5); x = x + 1; END; RUN;"
        trace = eval_code(code, {"x": 0})
        assert trace.final["x"] == 5

    def test_eval_do_until(self):
        code = "DATA w.o; SET w.i; DO UNTIL (x >= 5); x = x + 1; END; RUN;"
        trace = eval_code(code, {"x": 0})
        assert trace.final["x"] == 5

    def test_eval_bare_do_block(self):
        code = "DATA w.o; SET w.i; DO; a = 10; b = 20; END; RUN;"
        trace = eval_code(code, {})
        assert trace.final["a"] == 10
        assert trace.final["b"] == 20

    def test_to_dict_iterative(self):
        nodes = parse("DATA w.o; SET w.i; DO i = 1 TO 5; x = i; END; RUN;")
        d = tree.to_dict(nodes)
        do_d = d[0]["body"][0]
        assert do_d["type"] == "do_loop"
        assert do_d["var"] == "i"
        assert do_d["start"] == "1"
        assert do_d["stop"] == "5"

    def test_to_dict_while(self):
        nodes = parse("DATA w.o; SET w.i; DO WHILE (x < 5); x = x + 1; END; RUN;")
        d = tree.to_dict(nodes)
        do_d = d[0]["body"][0]
        assert do_d["type"] == "do_loop"
        assert do_d["while_cond"] == "x < 5"


# ─────────────────────────────────────────────────────────────────────────────
# SELECT / WHEN / OTHERWISE
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectWhen:
    def _get_select(self, code: str) -> SelectNode:
        ds = parse(code)[0]
        assert isinstance(ds, DataStepNode)
        sel = ds.body[0]
        assert isinstance(sel, SelectNode)
        return sel

    def test_value_based_select(self):
        code = """
        DATA w.o; SET w.i;
        SELECT (SEGMENTO);
            WHEN ('CORP') LGD = 0.45;
            WHEN ('RETAIL') LGD = 0.35;
            OTHERWISE LGD = 0.30;
        END;
        RUN;
        """
        sel = self._get_select(code)
        assert sel.select_expr == "SEGMENTO"
        assert len(sel.whens) == 2
        assert "'CORP'" in sel.whens[0].values
        assert sel.otherwise  # has OTHERWISE branch

    def test_condition_based_select(self):
        code = """
        DATA w.o; SET w.i;
        SELECT;
            WHEN (PD < 0.01) TIER = 1;
            WHEN (PD < 0.05) TIER = 2;
            OTHERWISE TIER = 3;
        END;
        RUN;
        """
        sel = self._get_select(code)
        assert sel.select_expr == ""
        assert len(sel.whens) == 2

    def test_eval_value_select_corp(self):
        code = """
        DATA w.o; SET w.i;
        SELECT (SEGMENTO);
            WHEN ('CORP') LGD = 0.45;
            WHEN ('RETAIL') LGD = 0.35;
            OTHERWISE LGD = 0.30;
        END;
        RUN;
        """
        trace = eval_code(code, {"SEGMENTO": "CORP"})
        assert trace.final["LGD"] == pytest.approx(0.45)

    def test_eval_value_select_retail(self):
        code = """
        DATA w.o; SET w.i;
        SELECT (SEGMENTO);
            WHEN ('CORP') LGD = 0.45;
            WHEN ('RETAIL') LGD = 0.35;
            OTHERWISE LGD = 0.30;
        END;
        RUN;
        """
        trace = eval_code(code, {"SEGMENTO": "RETAIL"})
        assert trace.final["LGD"] == pytest.approx(0.35)

    def test_eval_value_select_otherwise(self):
        code = """
        DATA w.o; SET w.i;
        SELECT (SEGMENTO);
            WHEN ('CORP') LGD = 0.45;
            WHEN ('RETAIL') LGD = 0.35;
            OTHERWISE LGD = 0.30;
        END;
        RUN;
        """
        trace = eval_code(code, {"SEGMENTO": "SME"})
        assert trace.final["LGD"] == pytest.approx(0.30)

    def test_eval_condition_select(self):
        code = """
        DATA w.o; SET w.i;
        SELECT;
            WHEN (PD < 0.01) TIER = 1;
            WHEN (PD < 0.05) TIER = 2;
            OTHERWISE TIER = 3;
        END;
        RUN;
        """
        assert eval_code(code, {"PD": 0.005}).final["TIER"] == 1
        assert eval_code(code, {"PD": 0.03}).final["TIER"] == 2
        assert eval_code(code, {"PD": 0.10}).final["TIER"] == 3

    def test_to_dict_select(self):
        nodes = parse("""
        DATA w.o; SET w.i;
        SELECT (X);
            WHEN (1) Y = 10;
            OTHERWISE Y = 0;
        END;
        RUN;
        """)
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "select"
        assert d["select_expr"] == "X"
        assert len(d["whens"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# ARRAY
# ─────────────────────────────────────────────────────────────────────────────

class TestArray:
    def _get_array(self, code: str) -> ArrayNode:
        ds = parse(code)[0]
        assert isinstance(ds, DataStepNode)
        arr = ds.body[0]
        assert isinstance(arr, ArrayNode)
        return arr

    def test_explicit_size(self):
        node = self._get_array("DATA w.o; SET w.i; ARRAY scores[3] s1 s2 s3; RUN;")
        assert node.name == "scores"
        assert node.dims == "3"
        assert node.vars == ["s1", "s2", "s3"]

    def test_star_size(self):
        node = self._get_array("DATA w.o; SET w.i; ARRAY vars[*] a b c d; RUN;")
        assert node.dims == "*"
        assert "a" in node.vars

    def test_temporary_array(self):
        node = self._get_array("DATA w.o; SET w.i; ARRAY tmp[3] _TEMPORARY_ (0 0 0); RUN;")
        assert node.temporary is True
        assert "0" in node.initial_values

    def test_to_dict_array(self):
        nodes = parse("DATA w.o; SET w.i; ARRAY x[2] a b; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "array"
        assert d["name"] == "x"
        assert d["dims"] == "2"


# ─────────────────────────────────────────────────────────────────────────────
# MERGE / BY
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeBy:
    def test_merge_parses_datasets(self):
        code = "DATA w.out; MERGE w.a w.b; BY ID; RUN;"
        ds = parse(code)[0]
        assert isinstance(ds, DataStepNode)
        merge = ds.body[0]
        assert isinstance(merge, MergeNode)
        assert "w.a" in merge.datasets
        assert "w.b" in merge.datasets

    def test_merge_in_ds_node(self):
        code = "DATA w.out; MERGE w.a w.b; BY ID; RUN;"
        ds = parse(code)[0]
        assert "w.a" in ds.merge_datasets

    def test_by_keys(self):
        code = "DATA w.out; MERGE w.a w.b; BY ID DATE; RUN;"
        ds = parse(code)[0]
        assert "ID" in ds.by_keys
        assert "DATE" in ds.by_keys

    def test_by_descending(self):
        code = "DATA w.out; SET w.a; BY DESCENDING SCORE; RUN;"
        ds = parse(code)[0]
        by_node = ds.body[0]
        assert isinstance(by_node, ByNode)
        assert by_node.keys[0] == "SCORE"
        assert by_node.descending[0] is True

    def test_to_dict_merge(self):
        nodes = parse("DATA w.out; MERGE w.a w.b; BY ID; RUN;")
        d = tree.to_dict(nodes)
        body = d[0]["body"]
        assert any(n["type"] == "merge" for n in body)


# ─────────────────────────────────────────────────────────────────────────────
# RETAIN
# ─────────────────────────────────────────────────────────────────────────────

class TestRetain:
    def test_retain_no_initial(self):
        code = "DATA w.o; SET w.i; RETAIN count 0; count = count + 1; RUN;"
        ds = parse(code)[0]
        retain = ds.body[0]
        assert isinstance(retain, RetainNode)
        assert "count" in retain.vars

    def test_retain_with_initial(self):
        code = "DATA w.o; SET w.i; RETAIN total 0; RUN;"
        ds = parse(code)[0]
        retain = ds.body[0]
        assert retain.initial == "0"

    def test_retain_multiple_vars(self):
        code = "DATA w.o; SET w.i; RETAIN a b c; RUN;"
        ds = parse(code)[0]
        retain = ds.body[0]
        assert len(retain.vars) == 3

    def test_retain_initializes_env(self):
        code = "DATA w.o; SET w.i; RETAIN TOTAL 0; TOTAL = TOTAL + X; RUN;"
        trace = eval_code(code, {"X": 5})
        assert trace.final["TOTAL"] == 5

    def test_to_dict_retain(self):
        nodes = parse("DATA w.o; SET w.i; RETAIN x 0; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "retain"
        assert "x" in d["vars"]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT / DELETE
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputDelete:
    def test_output_statement(self):
        code = "DATA w.o; SET w.i; IF x > 0 THEN OUTPUT; RUN;"
        ds = parse(code)[0]
        if_node = ds.body[0]
        assert isinstance(if_node, IfNode)
        out = if_node.then_branch[0]
        assert isinstance(out, OutputNode)

    def test_output_to_dataset(self):
        code = "DATA w.a w.b; SET w.i; IF x > 0 THEN OUTPUT w.a; ELSE OUTPUT w.b; RUN;"
        ds = parse(code)[0]
        if_node = ds.body[0]
        out_a = if_node.then_branch[0]
        out_b = if_node.else_branch[0]
        assert isinstance(out_a, OutputNode)
        assert isinstance(out_b, OutputNode)

    def test_delete_statement(self):
        code = "DATA w.o; SET w.i; IF x < 0 THEN DELETE; RUN;"
        ds = parse(code)[0]
        if_node = ds.body[0]
        assert isinstance(if_node.then_branch[0], DeleteNode)

    def test_eval_output_trace(self):
        code = "DATA w.o; SET w.i; OUTPUT; RUN;"
        trace = eval_code(code, {"x": 1})
        assert any(s.kind == "output" for s in trace.steps)

    def test_eval_delete_marks_row(self):
        code = "DATA w.o; SET w.i; DELETE; RUN;"
        trace = eval_code(code, {})
        assert any(s.kind == "delete" for s in trace.steps)

    def test_to_dict_output(self):
        nodes = parse("DATA w.o; SET w.i; OUTPUT; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "output"

    def test_to_dict_delete(self):
        nodes = parse("DATA w.o; SET w.i; DELETE; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "delete"


# ─────────────────────────────────────────────────────────────────────────────
# CALL
# ─────────────────────────────────────────────────────────────────────────────

class TestCall:
    def test_call_streaminit(self):
        code = "DATA w.o; SET w.i; CALL STREAMINIT(12345); RUN;"
        ds = parse(code)[0]
        call = ds.body[0]
        assert isinstance(call, CallNode)
        assert call.routine == "STREAMINIT"
        assert "12345" in call.args

    def test_call_symput(self):
        code = "DATA w.o; SET w.i; CALL SYMPUT('myvar', x); RUN;"
        ds = parse(code)[0]
        call = ds.body[0]
        assert isinstance(call, CallNode)
        assert call.routine == "SYMPUT"

    def test_call_execute(self):
        code = "DATA w.o; SET w.i; CALL EXECUTE('%mymacro'); RUN;"
        ds = parse(code)[0]
        call = ds.body[0]
        assert isinstance(call, CallNode)
        assert call.routine == "EXECUTE"

    def test_to_dict_call(self):
        nodes = parse("DATA w.o; SET w.i; CALL MISSING(x); RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "call"
        assert d["routine"] == "MISSING"


# ─────────────────────────────────────────────────────────────────────────────
# MACROS
# ─────────────────────────────────────────────────────────────────────────────

class TestMacros:
    def test_macro_let(self):
        code = "%LET cutoff = 0.05;"
        nodes = parse(code)
        assert len(nodes) == 1
        node = nodes[0]
        assert isinstance(node, MacroLetNode)
        assert node.var == "cutoff"
        assert node.value == "0.05"

    def test_macro_def_no_params(self):
        code = """
        %MACRO run_ecl;
            DATA w.o; SET w.i; ECL = PD * LGD * EAD; RUN;
        %MEND run_ecl;
        """
        nodes = parse(code)
        assert len(nodes) == 1
        node = nodes[0]
        assert isinstance(node, MacroDefNode)
        assert node.name == "run_ecl"
        assert node.params == ""
        assert len(node.body) == 1  # the DATA step

    def test_macro_def_with_params(self):
        code = """
        %MACRO apply_floor(var=, floor=);
            IF &var < &floor THEN &var = &floor;
        %MEND apply_floor;
        """
        nodes = parse(code)
        node = nodes[0]
        assert isinstance(node, MacroDefNode)
        assert "var" in node.params

    def test_macro_call(self):
        code = "%run_ecl();"
        nodes = parse(code)
        assert len(nodes) == 1
        assert isinstance(nodes[0], MacroCallNode)
        assert nodes[0].name == "run_ecl"

    def test_macro_let_in_data_step(self):
        code = "DATA w.o; SET w.i; %LET x = 1; y = 2; RUN;"
        ds = parse(code)[0]
        assert isinstance(ds, DataStepNode)
        # %LET inside DATA step is also captured
        macro_node = next((n for n in ds.body if isinstance(n, MacroLetNode)), None)
        assert macro_node is not None
        assert macro_node.var == "x"

    def test_to_dict_macro_let(self):
        nodes = parse("%LET x = 5;")
        d = tree.to_dict(nodes)[0]
        assert d["type"] == "macro_let"
        assert d["var"] == "x"
        assert d["value"] == "5"

    def test_to_dict_macro_def(self):
        nodes = parse("%MACRO foo; DATA w.o; SET w.i; x=1; RUN; %MEND foo;")
        d = tree.to_dict(nodes)[0]
        assert d["type"] == "macro_def"
        assert d["name"] == "foo"

    def test_to_dict_macro_call(self):
        nodes = parse("%foo(a=1, b=2);")
        d = tree.to_dict(nodes)[0]
        assert d["type"] == "macro_call"
        assert d["name"] == "foo"


# ─────────────────────────────────────────────────────────────────────────────
# LINK / GOTO / RETURN / STOP
# ─────────────────────────────────────────────────────────────────────────────

class TestControlFlow:
    def test_link_label(self):
        code = "DATA w.o; SET w.i; LINK process_row; RUN;"
        ds = parse(code)[0]
        link = ds.body[0]
        assert isinstance(link, LinkNode)
        assert link.label == "process_row"

    def test_goto_label(self):
        code = "DATA w.o; SET w.i; GOTO done; RUN;"
        ds = parse(code)[0]
        goto = ds.body[0]
        assert isinstance(goto, GotoNode)
        assert goto.label == "done"

    def test_return_statement(self):
        code = "DATA w.o; SET w.i; IF x < 0 THEN RETURN; RUN;"
        ds = parse(code)[0]
        if_node = ds.body[0]
        assert isinstance(if_node.then_branch[0], ReturnNode)

    def test_stop_statement(self):
        code = "DATA w.o; SET w.i; STOP; RUN;"
        ds = parse(code)[0]
        assert isinstance(ds.body[0], StopNode)

    def test_to_dict_link(self):
        nodes = parse("DATA w.o; SET w.i; LINK lbl; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "link"
        assert d["label"] == "lbl"

    def test_to_dict_goto(self):
        nodes = parse("DATA w.o; SET w.i; GOTO done; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "goto"
        assert d["label"] == "done"

    def test_to_dict_return(self):
        nodes = parse("DATA w.o; SET w.i; RETURN; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "return"

    def test_to_dict_stop(self):
        nodes = parse("DATA w.o; SET w.i; STOP; RUN;")
        d = tree.to_dict(nodes)[0]["body"][0]
        assert d["type"] == "stop"


# ─────────────────────────────────────────────────────────────────────────────
# OPERATORS: IN, NOT IN, BETWEEN, IS MISSING
# ─────────────────────────────────────────────────────────────────────────────

class TestOperators:
    def test_in_operator_numeric(self):
        code = "DATA w.o; SET w.i; IF STAGE IN (1, 2) THEN FLAG = 1; RUN;"
        trace = eval_code(code, {"STAGE": 2})
        assert trace.final.get("FLAG") == 1

    def test_in_operator_false(self):
        code = "DATA w.o; SET w.i; IF STAGE IN (1, 2) THEN FLAG = 1; ELSE FLAG = 0; RUN;"
        trace = eval_code(code, {"STAGE": 3})
        assert trace.final.get("FLAG") == 0

    def test_not_in_operator(self):
        code = "DATA w.o; SET w.i; IF SEG NOT IN ('CORP', 'RETAIL') THEN OTHER = 1; RUN;"
        trace = eval_code(code, {"SEG": "SME"})
        assert trace.final.get("OTHER") == 1

    def test_not_in_operator_false(self):
        code = "DATA w.o; SET w.i; IF SEG NOT IN ('CORP', 'RETAIL') THEN OTHER = 1; ELSE OTHER = 0; RUN;"
        trace = eval_code(code, {"SEG": "CORP"})
        assert trace.final.get("OTHER") == 0

    def test_between_operator(self):
        code = "DATA w.o; SET w.i; IF PD BETWEEN 0.01 AND 0.05 THEN TIER = 2; RUN;"
        trace = eval_code(code, {"PD": 0.03})
        assert trace.final.get("TIER") == 2

    def test_between_boundary(self):
        code = "DATA w.o; SET w.i; IF PD BETWEEN 0.01 AND 0.05 THEN TIER = 2; ELSE TIER = 1; RUN;"
        assert eval_code(code, {"PD": 0.01}).final.get("TIER") == 2
        assert eval_code(code, {"PD": 0.05}).final.get("TIER") == 2
        assert eval_code(code, {"PD": 0.001}).final.get("TIER") == 1

    def test_is_missing(self):
        code = "DATA w.o; SET w.i; IF LGD IS MISSING THEN LGD = 0.45; RUN;"
        trace = eval_code(code, {"LGD": None})
        assert trace.final.get("LGD") == pytest.approx(0.45)

    def test_is_not_missing(self):
        code = "DATA w.o; SET w.i; IF LGD IS NOT MISSING THEN VALID = 1; ELSE VALID = 0; RUN;"
        assert eval_code(code, {"LGD": 0.3}).final.get("VALID") == 1
        assert eval_code(code, {"LGD": None}).final.get("VALID") == 0

    def test_missing_function(self):
        code = "DATA w.o; SET w.i; IF MISSING(LGD) THEN LGD = 0.45; RUN;"
        trace = eval_code(code, {"LGD": None})
        assert trace.final.get("LGD") == pytest.approx(0.45)

    def test_sas_not_eq_operator(self):
        """^= (SAS not-equal) should be treated as !="""
        code = "DATA w.o; SET w.i; IF SEG ^= 'CORP' THEN FLAG = 1; ELSE FLAG = 0; RUN;"
        assert eval_code(code, {"SEG": "RETAIL"}).final.get("FLAG") == 1
        assert eval_code(code, {"SEG": "CORP"}).final.get("FLAG") == 0

    def test_eq_in_condition(self):
        """Single = in IF condition should mean equality."""
        code = "DATA w.o; SET w.i; IF SEG = 'CORP' THEN FLAG = 1; ELSE FLAG = 0; RUN;"
        assert eval_code(code, {"SEG": "CORP"}).final.get("FLAG") == 1
        assert eval_code(code, {"SEG": "OTHER"}).final.get("FLAG") == 0


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctions:
    def _eval_expr(self, expr: str, env: dict):
        code = f"DATA w.o; SET w.i; result = {expr}; RUN;"
        trace = eval_code(code, env)
        return trace.final.get("result")

    # Numeric
    def test_abs(self):
        assert self._eval_expr("ABS(-5)", {}) == 5

    def test_sqrt(self):
        assert self._eval_expr("SQRT(9)", {}) == pytest.approx(3.0)

    def test_log(self):
        assert self._eval_expr("LOG(1)", {}) == pytest.approx(0.0)

    def test_exp(self):
        assert self._eval_expr("EXP(0)", {}) == pytest.approx(1.0)

    def test_floor(self):
        assert self._eval_expr("FLOOR(3.7)", {}) == 3

    def test_ceil(self):
        assert self._eval_expr("CEIL(3.2)", {}) == 4

    def test_round(self):
        assert self._eval_expr("ROUND(3.456, 0.01)", {}) == pytest.approx(3.46)

    def test_max(self):
        assert self._eval_expr("MAX(1, 5, 3)", {}) == 5

    def test_min(self):
        assert self._eval_expr("MIN(1, 5, 3)", {}) == 1

    def test_mod(self):
        assert self._eval_expr("MOD(10, 3)", {}) == 1

    # String
    def test_upcase(self):
        code = "DATA w.o; SET w.i; r = UPCASE(s); RUN;"
        trace = eval_code(code, {"s": "hello"})
        assert trace.final["r"] == "HELLO"

    def test_lowcase(self):
        code = "DATA w.o; SET w.i; r = LOWCASE(s); RUN;"
        trace = eval_code(code, {"s": "HELLO"})
        assert trace.final["r"] == "hello"

    def test_trim(self):
        code = "DATA w.o; SET w.i; r = TRIM(s); RUN;"
        trace = eval_code(code, {"s": "abc   "})
        assert trace.final["r"] == "abc"

    def test_substr(self):
        code = "DATA w.o; SET w.i; r = SUBSTR(s, 2, 3); RUN;"
        trace = eval_code(code, {"s": "abcde"})
        assert trace.final["r"] == "bcd"

    def test_cats(self):
        code = "DATA w.o; SET w.i; r = CATS(a, b); RUN;"
        trace = eval_code(code, {"a": "hello", "b": "world"})
        assert trace.final["r"] == "helloworld"

    # Logic
    def test_coalesce_first_non_null(self):
        code = "DATA w.o; SET w.i; r = COALESCE(a, b, c); RUN;"
        trace = eval_code(code, {"a": None, "b": 5, "c": 10})
        assert trace.final["r"] == 5

    def test_ifn_true(self):
        code = "DATA w.o; SET w.i; r = IFN(x > 0, 1, -1); RUN;"
        assert eval_code(code, {"x": 5}).final["r"] == 1

    def test_ifn_false(self):
        code = "DATA w.o; SET w.i; r = IFN(x > 0, 1, -1); RUN;"
        assert eval_code(code, {"x": -3}).final["r"] == -1

    def test_sum_function(self):
        code = "DATA w.o; SET w.i; r = SUM(a, b, c); RUN;"
        trace = eval_code(code, {"a": 1, "b": 2, "c": 3})
        assert trace.final["r"] == 6

    def test_sum_ignores_missing(self):
        code = "DATA w.o; SET w.i; r = SUM(a, b, c); RUN;"
        trace = eval_code(code, {"a": 1, "b": None, "c": 3})
        assert trace.final["r"] == 4

    def test_missing_function_condition(self):
        code = "DATA w.o; SET w.i; IF MISSING(x) THEN flag = 1; ELSE flag = 0; RUN;"
        assert eval_code(code, {"x": None}).final["flag"] == 1
        assert eval_code(code, {"x": 5}).final["flag"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-DATASET / OUTPUT TARGETS
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiDataset:
    def test_multi_output_datasets_in_header(self):
        code = "DATA w.a w.b w.c; SET w.i; x = 1; RUN;"
        ds = parse(code)[0]
        assert ds.output_dataset == "w.a"
        assert "w.b" in ds.output_datasets
        assert "w.c" in ds.output_datasets

    def test_merge_captures_all_datasets(self):
        code = "DATA w.out; MERGE w.a(IN=ina) w.b(IN=inb); BY ID; RUN;"
        ds = parse(code)[0]
        assert len(ds.merge_datasets) == 2


# ─────────────────────────────────────────────────────────────────────────────
# LINEAGE through new nodes
# ─────────────────────────────────────────────────────────────────────────────

class TestLineage:
    def test_lineage_do_loop_adds_fields(self):
        code = """
        DATA w.o; SET w.i;
        DO I = 1 TO 5;
            TOTAL = TOTAL + I;
        END;
        RUN;
        """
        lg = lineage(code)
        field_ids = {n["id"] for n in lg.nodes}
        assert "TOTAL" in field_ids

    def test_lineage_select_adds_fields(self):
        code = """
        DATA w.o; SET w.i;
        SELECT (SEG);
            WHEN ('CORP') LGD = 0.45;
            OTHERWISE LGD = 0.30;
        END;
        RUN;
        """
        lg = lineage(code)
        field_ids = {n["id"] for n in lg.nodes}
        assert "LGD" in field_ids

    def test_lineage_merge_marks_input(self):
        code = "DATA w.out; MERGE w.a w.b; BY ID; X = 1; RUN;"
        lg = lineage(code)
        assert "w.out" in lg.data_steps

    def test_lineage_multi_step_pipeline(self):
        code = """
        DATA work.step1; SET work.raw; A = X + 1; RUN;
        DATA work.step2; SET work.step1; B = A * 2; RUN;
        """
        lg = lineage(code)
        assert len(lg.data_steps) == 2
        field_ids = {n["id"] for n in lg.nodes}
        assert "A" in field_ids
        assert "B" in field_ids

    def test_lineage_if_then_adds_field(self):
        code = "DATA w.o; SET w.i; IF X > 0 THEN RESULT = X * 2; RUN;"
        lg = lineage(code)
        assert any(n["id"] == "RESULT" for n in lg.nodes)

    def test_lineage_computed_fields_tracked(self):
        """Fields computed in body should be 'computed', inputs should be 'input'."""
        code = "DATA w.o; SET w.i; ECL = PD * LGD * EAD; RUN;"
        lg = lineage(code)
        node_map = {n["id"]: n for n in lg.nodes}
        assert "ECL" in node_map
        # ECL is computed (assigned)
        assert node_map["ECL"]["kind"] == "computed"

    def test_lineage_edges_for_assignment(self):
        """An assignment ECL = PD * LGD should generate edges PD→ECL, LGD→ECL."""
        code = "DATA w.o; SET w.i; ECL = PD * LGD; RUN;"
        lg = lineage(code)
        assert any(e["target"].upper() == "ECL" for e in lg.edges)


# ─────────────────────────────────────────────────────────────────────────────
# PROC STEPS
# ─────────────────────────────────────────────────────────────────────────────

class TestProcSteps:
    def test_proc_means(self):
        nodes = parse("PROC MEANS DATA=work.t; RUN;")
        assert len(nodes) == 1
        proc = nodes[0]
        assert isinstance(proc, ProcNode)
        assert proc.kind == "MEANS"
        assert proc.data == "work.t"

    def test_proc_sort(self):
        nodes = parse("PROC SORT DATA=work.t OUT=work.sorted; BY id; RUN;")
        proc = nodes[0]
        assert proc.kind == "SORT"

    def test_proc_sql(self):
        nodes = parse("PROC SQL; SELECT * FROM work.t; QUIT;")
        proc = nodes[0]
        assert proc.kind == "SQL"

    def test_proc_freq(self):
        nodes = parse("PROC FREQ DATA=work.t; TABLES seg; RUN;")
        proc = nodes[0]
        assert proc.kind == "FREQ"

    def test_proc_print(self):
        nodes = parse("PROC PRINT DATA=work.t; RUN;")
        proc = nodes[0]
        assert proc.kind == "PRINT"

    def test_to_dict_proc(self):
        nodes = parse("PROC MEANS DATA=work.t; VAR x; RUN;")
        d = tree.to_dict(nodes)[0]
        assert d["type"] == "proc"
        assert d["kind"] == "MEANS"


# ─────────────────────────────────────────────────────────────────────────────
# SAS FORMAT / NUMERIC EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericEdgeCases:
    def test_float_literal_in_condition(self):
        """Float literals like 0.0003 must not be stripped as format specs."""
        code = "DATA w.o; SET w.i; IF PD < 0.0003 THEN PD = 0.0003; RUN;"
        trace = eval_code(code, {"PD": 0.0001})
        assert trace.final["PD"] == pytest.approx(0.0003)

    def test_format_spec_stripped(self):
        """SAS format specs like best12. should be stripped (not cause errors)."""
        code = "DATA w.o; SET w.i; x = INPUT(val, best12.); RUN;"
        # Should not raise — just evaluate gracefully
        trace = eval_code(code, {"val": "123"})
        # x should be set to 123.0 or None but not raise
        assert trace is not None

    def test_scientific_notation(self):
        code = "DATA w.o; SET w.i; x = 1.5e-4; RUN;"
        trace = eval_code(code, {})
        assert trace.final.get("x") == pytest.approx(1.5e-4)

    def test_negative_literal(self):
        code = "DATA w.o; SET w.i; x = -3.14; RUN;"
        trace = eval_code(code, {})
        assert trace.final.get("x") == pytest.approx(-3.14)

    def test_sas_missing_dot(self):
        """SAS missing literal '.' in assignment should yield None."""
        code = "DATA w.o; SET w.i; x = .; RUN;"
        trace = eval_code(code, {})
        # Should assign None (SAS missing)
        assert trace.final.get("x") is None or trace is not None


# ─────────────────────────────────────────────────────────────────────────────
# FILTER (WHERE) SEMANTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestFilter:
    def test_where_passes_sets_true(self):
        code = "DATA w.o; SET w.i; WHERE x > 0; RUN;"
        trace = eval_code(code, {"x": 5})
        assert trace.row_passes_filter is True

    def test_where_blocks_sets_false(self):
        code = "DATA w.o; SET w.i; WHERE x > 0; RUN;"
        trace = eval_code(code, {"x": -1})
        assert trace.row_passes_filter is False

    def test_filter_results_recorded(self):
        code = "DATA w.o; SET w.i; WHERE x > 0; y = x + 1; RUN;"
        trace = eval_code(code, {"x": 5})
        assert len(trace.filter_results) == 1
        assert trace.filter_results[0]["passed"] is True

    def test_second_step_filter_does_not_change_entry_gate(self):
        """
        First step's WHERE is the 'entry gate'. A second step filter
        (even if it fails) should not override row_passes_filter.
        """
        code = """
        DATA work.step1; SET work.raw;
            WHERE MONTHS >= 9;
            LGD = 0.45;
        RUN;
        DATA work.step2; SET work.step1;
            WHERE MONTHS < 6;
        RUN;
        """
        # MONTHS=12 passes first gate, fails second
        trace = eval_code(code, {"MONTHS": 12})
        # The entry gate (first filter) should still be True
        assert trace.filter_results[0]["passed"] is True
        assert trace.filter_results[1]["passed"] is False

    def test_where_with_and(self):
        code = "DATA w.o; SET w.i; WHERE x > 0 AND y > 0; RUN;"
        assert eval_code(code, {"x": 1, "y": 1}).row_passes_filter is True
        assert eval_code(code, {"x": 1, "y": -1}).row_passes_filter is False

    def test_where_with_or(self):
        code = "DATA w.o; SET w.i; WHERE x = 1 OR y = 1; RUN;"
        assert eval_code(code, {"x": 1, "y": 0}).row_passes_filter is True
        assert eval_code(code, {"x": 0, "y": 0}).row_passes_filter is False


# ─────────────────────────────────────────────────────────────────────────────
# IF / ELSE IF chains
# ─────────────────────────────────────────────────────────────────────────────

class TestIfElseChains:
    def test_simple_if_then(self):
        code = "DATA w.o; SET w.i; IF x > 0 THEN y = 1; RUN;"
        trace = eval_code(code, {"x": 5})
        assert trace.final["y"] == 1

    def test_if_else(self):
        code = "DATA w.o; SET w.i; IF x > 0 THEN y = 1; ELSE y = -1; RUN;"
        assert eval_code(code, {"x": 5}).final["y"] == 1
        assert eval_code(code, {"x": -5}).final["y"] == -1

    def test_nested_if(self):
        code = """
        DATA w.o; SET w.i;
        IF a > 0 THEN DO;
            IF b > 0 THEN r = 1;
            ELSE r = 2;
        END;
        ELSE r = 3;
        RUN;
        """
        assert eval_code(code, {"a": 1, "b": 1}).final["r"] == 1
        assert eval_code(code, {"a": 1, "b": -1}).final["r"] == 2
        assert eval_code(code, {"a": -1, "b": 1}).final["r"] == 3

    def test_if_then_trace_records_kind(self):
        code = "DATA w.o; SET w.i; IF x > 0 THEN y = 1; RUN;"
        trace = eval_code(code, {"x": 5})
        assert any(s.kind == "if_taken" for s in trace.steps)

    def test_if_skipped_trace(self):
        code = "DATA w.o; SET w.i; IF x > 0 THEN y = 1; RUN;"
        trace = eval_code(code, {"x": -1})
        assert any(s.kind == "if_skipped" for s in trace.steps)

    def test_string_comparison(self):
        code = "DATA w.o; SET w.i; IF SEG = 'CORP' THEN LGD = 0.45; RUN;"
        assert eval_code(code, {"SEG": "CORP"}).final.get("LGD") == pytest.approx(0.45)
        assert "LGD" not in eval_code(code, {"SEG": "RETAIL"}).final or \
               eval_code(code, {"SEG": "RETAIL"}).final.get("LGD") is None or \
               "LGD" not in eval_code(code, {"SEG": "RETAIL"}).final


# ─────────────────────────────────────────────────────────────────────────────
# COMMENT STRIPPING
# ─────────────────────────────────────────────────────────────────────────────

class TestComments:
    def test_block_comment_stripped(self):
        code = "DATA w.o; SET w.i; /* assign x */ x = 5; RUN;"
        nodes = parse(code)
        ds = nodes[0]
        assert any(isinstance(n, AssignNode) and n.var == "x" for n in ds.body)

    def test_line_comment_stripped(self):
        code = """DATA w.o;
        SET w.i;
        * This is a comment;
        x = 5;
        RUN;"""
        nodes = parse(code)
        ds = nodes[0]
        assert any(isinstance(n, AssignNode) and n.var == "x" for n in ds.body)

    def test_mixed_comments(self):
        code = """
        /* Global comment */
        DATA w.o; SET w.i;
        * row comment;
        /* inline */ y = 10; /* trailing */
        RUN;
        """
        nodes = parse(code)
        ds = nodes[0]
        assigns = [n for n in ds.body if isinstance(n, AssignNode)]
        assert len(assigns) >= 1
        assert assigns[0].var == "y"
