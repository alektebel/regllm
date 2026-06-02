"""SAS logic tree — parser and simulator for IRB/IFRS9 DATA steps.

Parses SAS source code into an AST of business logic nodes, then walks
that tree with example variable values to show how a given row would
be calculated step by step.

Supported SAS constructs
------------------------
- DATA ... RUN blocks (multiple, executed in sequence)
- SET and WHERE statements
- IF/THEN with optional DO...END blocks (nested)
- Assignment statements (var = expr)
- PROC blocks (captured for display, not simulated)

Unsupported (silently skipped)
------------------------------
- ARRAY / DO loops / SELECT
- Macro definitions (%MACRO) and calls (%name)
- LABEL / ATTRIB / FORMAT / KEEP / DROP

Usage
-----
    tree = SASLogicTree()
    nodes = tree.parse(sas_code)

    # Print the tree
    print(tree.display(nodes))

    # Simulate with example values
    trace = tree.evaluate(nodes, {
        "PD_ESTIMADA": 0.0001,
        "LGD_ESTIMADA": 0.25,
        "EAD": 100_000,
        "DPDS": 45,
        "STAGE_IFRS9": 1,
        "COLATERAL_TIPO": "HIPOTECA",
        "SEGMENTO": "CORP",
    })
    print(trace.summary())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Union


# ── AST node types ────────────────────────────────────────────────────────────

@dataclass
class AssignNode:
    var: str
    expr: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"{self.var} = {self.expr}"

    def to_dict(self) -> dict:
        return {"type": "assign", "var": self.var, "expr": self.expr}


@dataclass
class IfNode:
    condition: str
    then_branch: list["AnyNode"]
    else_branch: list["AnyNode"] = field(default_factory=list)

    def display(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [pad + f"IF {self.condition}"]
        for n in self.then_branch:
            lines.append(n.display(indent + 1))
        if self.else_branch:
            lines.append(pad + "ELSE")
            for n in self.else_branch:
                lines.append(n.display(indent + 1))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "type": "if",
            "condition": self.condition,
            "then": [n.to_dict() for n in self.then_branch],
            "else": [n.to_dict() for n in self.else_branch],
        }


@dataclass
class FilterNode:
    condition: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"WHERE {self.condition}"

    def to_dict(self) -> dict:
        return {"type": "filter", "condition": self.condition}


@dataclass
class DataStepNode:
    output_dataset: str
    input_dataset: str
    body: list["AnyNode"]

    def display(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [pad + f"DATA {self.output_dataset}  (SET {self.input_dataset})"]
        for n in self.body:
            lines.append(n.display(indent + 1))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "type": "data_step",
            "output": self.output_dataset,
            "input": self.input_dataset,
            "body": [n.to_dict() for n in self.body],
        }


@dataclass
class ProcNode:
    kind: str
    data: str
    raw: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"PROC {self.kind}  DATA={self.data}"

    def to_dict(self) -> dict:
        return {"type": "proc", "kind": self.kind, "data": self.data}


AnyNode = Union[AssignNode, IfNode, FilterNode, DataStepNode, ProcNode]


# ── Evaluation trace ──────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    kind: str          # "assign" | "if_taken" | "if_skipped" | "filter_pass" | "filter_block"
    label: str
    var: str | None = None
    old_val: Any = None
    new_val: Any = None


@dataclass
class EvalTrace:
    initial: dict[str, Any]
    final: dict[str, Any]
    steps: list[TraceStep]
    row_passes_filter: bool = True

    def summary(self) -> str:
        lines = ["── Input ──────────────────────────────────────────"]
        for k, v in self.initial.items():
            lines.append(f"  {k} = {v!r}")

        lines.append("── Trace ──────────────────────────────────────────")
        for s in self.steps:
            if s.kind == "assign":
                changed = " ← changed" if s.old_val != s.new_val else ""
                lines.append(f"  ASSIGN  {s.label}")
                if s.old_val != s.new_val:
                    lines.append(f"          {s.var}: {s.old_val!r} → {s.new_val!r}{changed}")
            elif s.kind == "if_taken":
                lines.append(f"  IF  ✓   {s.label}")
            elif s.kind == "if_skipped":
                lines.append(f"  IF  ✗   {s.label}")
            elif s.kind == "filter_pass":
                lines.append(f"  WHERE ✓ {s.label}")
            elif s.kind == "filter_block":
                lines.append(f"  WHERE ✗ {s.label}  ← row excluded")

        lines.append("── Output ─────────────────────────────────────────")
        for k, v in self.final.items():
            orig = self.initial.get(k)
            tag = f"  (was {orig!r})" if k in self.initial and orig != v else ""
            lines.append(f"  {k} = {v!r}{tag}")
        if not self.row_passes_filter:
            lines.append("  ⚠  Row excluded by WHERE condition")
        return "\n".join(lines)


# ── Expression translation ────────────────────────────────────────────────────

_KEYWORD_OPS = [
    (r"\bGE\b", ">="),
    (r"\bLE\b", "<="),
    (r"\bGT\b", ">"),
    (r"\bLT\b", "<"),
    (r"\bNE\b", "!="),
    (r"\bEQ\b", "=="),
    (r"\bAND\b", " and "),
    (r"\bOR\b",  " or "),
    (r"\bNOT\b", " not "),
]

_SAFE_GLOBALS = {"__builtins__": None}
_SAFE_LOCALS: dict[str, Any] = {
    "max": max, "min": min, "abs": abs, "round": round, "len": len,
    "True": True, "False": False, "None": None,
}


def _to_python(expr: str, is_condition: bool = False) -> str:
    """Translate a SAS expression to a Python-evaluable string."""
    result = expr.strip()
    for pattern, replacement in _KEYWORD_OPS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    # ^= ~= → !=
    result = re.sub(r"[\^~]=", "!=", result)
    # SAS missing value literal → None
    result = re.sub(r"(?<!\w)\.(?!\w)", "None", result)
    # SAS function names
    result = re.sub(r"\bMAX\s*\(", "max(", result, flags=re.IGNORECASE)
    result = re.sub(r"\bMIN\s*\(", "min(", result, flags=re.IGNORECASE)
    result = re.sub(r"\bABS\s*\(", "abs(", result, flags=re.IGNORECASE)
    result = re.sub(r"\bROUND\s*\(", "round(", result, flags=re.IGNORECASE)
    # In conditions, standalone = means equality
    if is_condition:
        result = re.sub(r"(?<![<>!=^~])=(?!=)", "==", result)
    return result


def _eval_expr(expr: str, env: dict[str, Any], is_condition: bool = False) -> Any:
    """Safely evaluate a translated SAS expression. Returns None/False on error."""
    py = _to_python(expr, is_condition=is_condition)
    try:
        return eval(py, _SAFE_GLOBALS, {**_SAFE_LOCALS, **env})  # noqa: S307
    except Exception:
        return False if is_condition else None


# ── Parser ────────────────────────────────────────────────────────────────────

_SKIP_KEYWORDS = re.compile(
    r"^(LIBNAME|FILENAME|OPTIONS|TITLE|FOOTNOTE|ODS|RUN|QUIT|"
    r"LABEL|ATTRIB|FORMAT|INFORMAT|LENGTH|KEEP|DROP|RENAME|RETAIN|"
    r"ARRAY|%MACRO|%MEND|%LET|%IF|%DO|%END)\b",
    re.IGNORECASE,
)


def _strip_comments(code: str) -> str:
    """Remove SAS block comments /* ... */."""
    return re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)


def _split_statements(code: str) -> list[str]:
    """Split SAS code by ';' and return non-empty stripped statements."""
    return [s.strip() for s in code.split(";") if s.strip()]


class _Parser:
    def __init__(self, stmts: list[str]):
        self._s = stmts
        self._pos = 0

    # ── Cursor helpers ────────────────────────────────────────────────────────

    def _peek(self) -> str | None:
        return self._s[self._pos] if self._pos < len(self._s) else None

    def _consume(self) -> str:
        s = self._s[self._pos]
        self._pos += 1
        return s

    def _upper(self) -> str:
        p = self._peek()
        return p.upper().strip() if p else ""

    # ── Top-level parse ───────────────────────────────────────────────────────

    def parse(self) -> list[AnyNode]:
        nodes: list[AnyNode] = []
        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^DATA\s+\w", u):
                nodes.append(self._parse_data_step())
            elif re.match(r"^PROC\s+\w", u):
                nodes.append(self._parse_proc())
            else:
                self._consume()
        return nodes

    # ── DATA step ─────────────────────────────────────────────────────────────

    def _parse_data_step(self) -> DataStepNode:
        header = self._consume()
        m = re.match(r"DATA\s+([\w.]+)", header, re.IGNORECASE)
        output_ds = m.group(1) if m else "work.unknown"
        input_ds = ""
        body: list[AnyNode] = []

        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^(RUN|QUIT)\s*$", u):
                self._consume()
                break
            node_or_ds = self._parse_body_stmt()
            if isinstance(node_or_ds, str):
                input_ds = node_or_ds
            elif node_or_ds is not None:
                body.append(node_or_ds)

        return DataStepNode(output_ds, input_ds, body)

    def _parse_body_stmt(self) -> "AnyNode | str | None":
        """Parse one statement inside a DATA step. Returns str for SET dataset."""
        stmt = self._peek()
        if stmt is None:
            return None
        u = stmt.upper().strip()

        if re.match(r"^SET\s+", u):
            self._consume()
            m = re.match(r"SET\s+([\w.]+)", stmt, re.IGNORECASE)
            return m.group(1) if m else ""

        if re.match(r"^WHERE\s+", u):
            self._consume()
            cond = re.sub(r"^WHERE\s+", "", stmt, flags=re.IGNORECASE).strip()
            return FilterNode(cond)

        if re.match(r"^IF\s+", u):
            return self._parse_if()

        if _SKIP_KEYWORDS.match(u):
            self._consume()
            return None

        if re.match(r"^[\w]+\s*=", u):
            self._consume()
            return self._stmt_to_assign(stmt)

        self._consume()
        return None

    # ── IF ────────────────────────────────────────────────────────────────────

    def _parse_if(self) -> IfNode | None:
        stmt = self._consume()
        m = re.match(r"IF\s+(.*?)\s+THEN\s+(.*)", stmt, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        condition = m.group(1).strip()
        then_raw = m.group(2).strip()

        # THEN DO → multi-statement block
        if re.match(r"^DO\s*$", then_raw, re.IGNORECASE):
            then_branch = self._parse_do_block()
        else:
            node = self._inline_stmt(then_raw)
            then_branch = [node] if node else []

        else_branch: list[AnyNode] = []
        if self._peek() and re.match(r"^ELSE\b", self._upper()):
            else_stmt = self._consume()
            else_body = re.sub(r"^ELSE\s*", "", else_stmt, flags=re.IGNORECASE).strip()
            if re.match(r"^DO\s*$", else_body, re.IGNORECASE):
                else_branch = self._parse_do_block()
            elif else_body:
                node = self._inline_stmt(else_body)
                else_branch = [node] if node else []

        return IfNode(condition, then_branch, else_branch)

    def _parse_do_block(self) -> list[AnyNode]:
        nodes: list[AnyNode] = []
        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^END\s*$", u):
                self._consume()
                break
            if re.match(r"^IF\s+", u):
                node = self._parse_if()
            elif re.match(r"^[\w]+\s*=", u):
                stmt = self._consume()
                node = self._stmt_to_assign(stmt)
            else:
                self._consume()
                node = None
            if node is not None:
                nodes.append(node)
        return nodes

    # ── Statement helpers ─────────────────────────────────────────────────────

    def _stmt_to_assign(self, stmt: str) -> AssignNode | None:
        m = re.match(r"^([\w]+)\s*=\s*(.+)$", stmt.strip(), re.DOTALL)
        if not m:
            return None
        return AssignNode(m.group(1).strip(), m.group(2).strip())

    def _inline_stmt(self, stmt: str) -> AnyNode | None:
        """Parse a single inline statement (right side of THEN/ELSE)."""
        u = stmt.upper().strip()
        if re.match(r"^IF\s+", u):
            # nested inline IF (e.g. ELSE IF ...)
            m = re.match(r"IF\s+(.*?)\s+THEN\s+(.*)", stmt, re.IGNORECASE | re.DOTALL)
            if m:
                inner = self._inline_stmt(m.group(2).strip())
                return IfNode(m.group(1).strip(), [inner] if inner else [], [])
        if re.match(r"^[\w]+\s*=", u):
            return self._stmt_to_assign(stmt)
        return None

    # ── PROC ──────────────────────────────────────────────────────────────────

    def _parse_proc(self) -> ProcNode:
        header = self._consume()
        m = re.match(r"PROC\s+(\w+)(?:\s+DATA\s*=\s*([\w.]+))?", header, re.IGNORECASE)
        kind = m.group(1).upper() if m else "UNKNOWN"
        data = m.group(2) if (m and m.group(2)) else ""
        raw_lines = [header]
        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^(RUN|QUIT)\s*$", u):
                self._consume()
                break
            raw_lines.append(self._consume())
        return ProcNode(kind, data, "; ".join(raw_lines))


# ── Evaluator ─────────────────────────────────────────────────────────────────

class _Evaluator:
    def __init__(self):
        self.steps: list[TraceStep] = []
        self.row_passes_filter = True

    def run(self, nodes: list[AnyNode], env: dict[str, Any]) -> None:
        for node in nodes:
            if isinstance(node, DataStepNode):
                self._eval_body(node.body, env)
            elif isinstance(node, ProcNode):
                pass  # PROC steps not simulated
            else:
                self._eval_node(node, env)

    def _eval_body(self, nodes: list[AnyNode], env: dict[str, Any]) -> None:
        for node in nodes:
            if not self.row_passes_filter:
                break
            self._eval_node(node, env)

    def _eval_node(self, node: AnyNode, env: dict[str, Any]) -> None:
        if isinstance(node, AssignNode):
            self._eval_assign(node, env)
        elif isinstance(node, IfNode):
            self._eval_if(node, env)
        elif isinstance(node, FilterNode):
            self._eval_filter(node, env)

    def _eval_assign(self, node: AssignNode, env: dict[str, Any]) -> None:
        old = env.get(node.var)
        new = _eval_expr(node.expr, env, is_condition=False)
        if new is not None:
            env[node.var] = new
        self.steps.append(TraceStep(
            kind="assign",
            label=f"{node.var} = {node.expr}",
            var=node.var,
            old_val=old,
            new_val=env.get(node.var),
        ))

    def _eval_if(self, node: IfNode, env: dict[str, Any]) -> None:
        result = _eval_expr(node.condition, env, is_condition=True)
        taken = bool(result)
        self.steps.append(TraceStep(
            kind="if_taken" if taken else "if_skipped",
            label=node.condition,
        ))
        branch = node.then_branch if taken else node.else_branch
        self._eval_body(branch, env)

    def _eval_filter(self, node: FilterNode, env: dict[str, Any]) -> None:
        result = _eval_expr(node.condition, env, is_condition=True)
        passes = bool(result)
        self.steps.append(TraceStep(
            kind="filter_pass" if passes else "filter_block",
            label=node.condition,
        ))
        if not passes:
            self.row_passes_filter = False


# ── Field lineage extraction ──────────────────────────────────────────────────

_LINEAGE_SKIP = {
    # Operators / keywords
    'AND', 'OR', 'NOT', 'EQ', 'NE', 'GT', 'LT', 'GE', 'LE', 'IN',
    'IF', 'THEN', 'ELSE', 'DO', 'END', 'WHERE', 'SET', 'DATA',
    'TRUE', 'FALSE', 'NULL', 'MISSING', 'TO', 'BY',
    # Common SAS functions
    'MAX', 'MIN', 'ABS', 'ROUND', 'INT', 'FLOOR', 'CEIL', 'SQRT',
    'LOG', 'EXP', 'MOD', 'LENGTH', 'SUBSTR', 'TRIM', 'STRIP',
    'UPCASE', 'LOWCASE', 'COMPRESS', 'CAT', 'CATS', 'CATX',
    'PUT', 'INPUT', 'SCAN', 'INDEX', 'TRANWRD',
    'SUM', 'MEAN', 'STD', 'NMISS', 'LAG', 'DIF', 'N',
    'DATEPART', 'TODAY', 'DATE', 'YEAR', 'MONTH', 'DAY',
    'COALESCE', 'IFN', 'IFC',
    # Library/dataset name prefixes
    'WORK', 'MYLIB',
}


def _vars_in_expr(expr: str) -> set[str]:
    """Extract SAS variable names from an expression, ignoring keywords/literals."""
    cleaned = re.sub(r"'[^']*'", " ", expr)
    cleaned = re.sub(r'"[^"]*"', " ", cleaned)
    tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', cleaned)
    return {t.upper() for t in tokens if t.upper() not in _LINEAGE_SKIP}


@dataclass
class FieldLineage:
    """Field dependency graph extracted from a SAS AST."""
    nodes: list[dict]      # {id, label, kind: "input"|"modified"|"computed", layer, data_steps}
    edges: list[dict]      # {source, target, kind: "assigns"|"conditions", data_step, expr}
    data_steps: list[str]  # output dataset names in parse order


class _LineageWalker:
    def __init__(self) -> None:
        self._written: set[str] = set()
        self._read: set[str] = set()
        self._edges: list[dict] = []
        self._seen_edges: set[tuple] = set()
        self._step_order: list[str] = []
        self._field_step: dict[str, str] = {}   # field → first writing step
        self._current_step: str = ""
        self._cond_stack: list[set[str]] = []   # nested condition variables

    def walk(self, nodes: list[AnyNode]) -> FieldLineage:
        for node in nodes:
            if isinstance(node, DataStepNode):
                self._current_step = node.output_dataset
                if node.output_dataset not in self._step_order:
                    self._step_order.append(node.output_dataset)
                self._walk_body(node.body)
        return self._build()

    def _walk_body(self, body: list[AnyNode]) -> None:
        for node in body:
            if isinstance(node, AssignNode):
                self._on_assign(node)
            elif isinstance(node, IfNode):
                self._on_if(node)
            elif isinstance(node, FilterNode):
                self._on_filter(node)

    def _on_assign(self, node: AssignNode) -> None:
        target = node.var.upper()
        sources = _vars_in_expr(node.expr)
        self._written.add(target)
        self._read.update(sources)
        if target not in self._field_step:
            self._field_step[target] = self._current_step
        for src in sources:
            self._add_edge(src, target, "assigns", node.expr)
        # Condition context: every variable that guards this assignment influences the target
        for cond_vars in self._cond_stack:
            for cv in cond_vars:
                if cv != target:   # skip self-loops
                    self._add_edge(cv, target, "conditions", "")

    def _on_if(self, node: IfNode) -> None:
        cond_vars = _vars_in_expr(node.condition)
        self._read.update(cond_vars)
        self._cond_stack.append(cond_vars)
        self._walk_body(node.then_branch)
        self._walk_body(node.else_branch)
        self._cond_stack.pop()

    def _on_filter(self, node: FilterNode) -> None:
        self._read.update(_vars_in_expr(node.condition))

    def _add_edge(self, source: str, target: str, kind: str, expr: str) -> None:
        key = (source, target, kind, self._current_step)
        if key in self._seen_edges:
            return
        self._seen_edges.add(key)
        self._edges.append({
            "source": source,
            "target": target,
            "kind": kind,
            "data_step": self._current_step,
            "expr": expr[:120] if expr else "",
        })

    def _build(self) -> FieldLineage:
        all_fields = self._written | self._read
        step_idx = {s: i for i, s in enumerate(self._step_order)}

        # Compute contiguous layer numbers (skip empty layers)
        raw_layers: dict[str, int] = {}
        for f in all_fields:
            if f in self._written:
                step = self._field_step.get(f, "")
                raw_layers[f] = step_idx.get(step, 0) + 1
            else:
                raw_layers[f] = 0
        used = sorted(set(raw_layers.values()))
        remap = {old: new for new, old in enumerate(used)}

        nodes = []
        for f in sorted(all_fields):
            if f in self._written:
                kind = "modified" if f in self._read else "computed"
                step = self._field_step.get(f, "")
                steps = [step] if step else []
            else:
                kind = "input"
                steps = []
            nodes.append({
                "id": f,
                "label": f,
                "kind": kind,
                "layer": remap[raw_layers[f]],
                "data_steps": steps,
            })

        return FieldLineage(nodes=nodes, edges=self._edges, data_steps=self._step_order)


# ── Public API ────────────────────────────────────────────────────────────────

class SASLogicTree:
    """Parse SAS DATA step code into a logic tree and simulate it with example values."""

    def parse(self, code: str) -> list[AnyNode]:
        """Return the AST for the given SAS source code."""
        clean = _strip_comments(code)
        stmts = _split_statements(clean)
        return _Parser(stmts).parse()

    def display(self, nodes: list[AnyNode]) -> str:
        """Human-readable indented tree."""
        return "\n".join(n.display() for n in nodes)

    def to_dict(self, nodes: list[AnyNode]) -> list[dict]:
        """JSON-serializable tree."""
        return [n.to_dict() for n in nodes]

    def lineage(self, nodes: list[AnyNode]) -> FieldLineage:
        """Extract field dependency graph from the AST."""
        return _LineageWalker().walk(nodes)

    def evaluate(
        self,
        nodes: list[AnyNode],
        values: dict[str, Any],
    ) -> EvalTrace:
        """Walk the tree with *values* as the starting variable state.

        Returns an EvalTrace with:
        - initial / final variable snapshots
        - step-by-step execution record
        - whether the row passed all WHERE filters
        """
        env = dict(values)
        evaluator = _Evaluator()
        evaluator.run(nodes, env)
        return EvalTrace(
            initial=dict(values),
            final=env,
            steps=evaluator.steps,
            row_passes_filter=evaluator.row_passes_filter,
        )
