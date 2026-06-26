"""Toy SAS compiler — executes parsed SAS AST against in-memory datasets.

Usage:
    compiler = SASCompiler()
    datasets = compiler.run(nodes)
    # datasets["work.final"] = [{"ECL": 315.0, ...}, ...]
"""

from __future__ import annotations

import re
from typing import Any

from src.sas_logic_tree import (
    AnyNode,
    AssignNode,
    DataStepNode,
    DoLoopNode,
    FilterNode,
    IfNode,
    MergeNode,
    OutputNode,
    ProcNode,
    RetainNode,
    SelectNode,
    _eval_expr,
)


def _parse_input_vars_from_ds(ds: DataStepNode) -> list[tuple[str, bool]]:
    """Reconstruct input vars from the first datalines row if possible."""
    if not ds.datalines_data:
        return []
    row = ds.datalines_data[0]
    return [(k, isinstance(v, str)) for k, v in row.items()]


def _parse_datalines(text: str, vars_: list[tuple[str, bool]]) -> list[dict]:
    """Parse DATALINES text into rows using input vars spec."""
    from src.sas_logic_tree import SASLogicTree
    return SASLogicTree()._parse_datalines(text, vars_)  # noqa: SLF001


class SASCompiler:
    """Execute a compiled SAS AST against in-memory row-oriented datasets."""

    def __init__(self) -> None:
        self.datasets: dict[str, list[dict]] = {}

    def run(
        self,
        nodes: list[AnyNode],
        initial_data: dict[str, list[dict]] | None = None,
    ) -> dict[str, list[dict]]:
        """Run the pipeline. Returns {dataset_name: [{col: val}, ...]}."""
        self.datasets = dict(initial_data or {})
        for node in nodes:
            if isinstance(node, DataStepNode):
                self._run_data_step(node)
            elif isinstance(node, ProcNode):
                self._run_proc(node)
        return self.datasets

    # ── DATA step ─────────────────────────────────────────────────────────

    def _run_data_step(self, node: DataStepNode) -> None:
        if node.datalines_data:
            existing = self.datasets.get(node.output_dataset, [])
            if existing:
                # Supplement existing rows with DATALINES fields.
                # If fewer existing rows, broadcast extra fields from first row.
                extra = {}
                if len(existing) < len(node.datalines_data):
                    extra = {k: v for k, v in existing[0].items()
                             if k not in node.datalines_data[0]}
                merged = []
                for i, row in enumerate(node.datalines_data):
                    merged_row = dict(existing[i]) if i < len(existing) else {}
                    merged_row.update(row)
                    if extra:
                        for k, v in extra.items():
                            merged_row.setdefault(k, v)
                    merged.append(merged_row)
                self.datasets[node.output_dataset] = merged
            else:
                self.datasets[node.output_dataset] = list(node.datalines_data)
            return

        input_rows: list[dict] = []
        if node.merge_datasets and node.by_keys:
            input_rows = self._merge_rows(node)
        elif node.input_dataset:
            input_rows = list(self.datasets.get(node.input_dataset, []))

        output_rows: list[dict] = []
        for row in input_rows:
            env = dict(row)
            ev = _RowEvaluator()
            result = ev.eval_data_step(node, env)
            if result is not None:
                output_rows.append(result)

        self.datasets[node.output_dataset] = output_rows

    def _merge_rows(self, node: DataStepNode) -> list[dict]:
        """Simple sorted MERGE by first BY key."""
        if not node.by_keys:
            return []
        by_key = node.by_keys[0]

        # Collect and sort each merge dataset
        sorted_dss: list[list[dict]] = []
        in_vars_map: dict[str, int] = {}
        for i, ds_name in enumerate(node.merge_datasets):
            rows = list(self.datasets.get(ds_name, []))
            rows.sort(key=lambda r: str(r.get(by_key, "")))
            sorted_dss.append(rows)
            # Extract IN= variable names from the MergeNode in body
            for bn in node.body:
                if isinstance(bn, MergeNode) and i < len(bn.in_vars):
                    in_vars_map[bn.in_vars[i]] = i

        # Sorted merge (one pass)
        result: list[dict] = []
        idx = [0] * len(sorted_dss)
        n = [len(ds) for ds in sorted_dss]

        # Collect all unique keys across all datasets
        all_keys: set[str] = set()
        for ds in sorted_dss:
            for row in ds:
                all_keys.add(str(row.get(by_key, "")))
        all_keys.discard("None")
        all_keys.discard("")

        for key in sorted(all_keys):
            merged: dict = {}
            in_flags: dict[int, bool] = {}
            for i, ds in enumerate(sorted_dss):
                # Find all rows with this key in dataset i
                key_rows = [r for r in ds if str(r.get(by_key, "")) == key]
                # Use the LAST row for this key (SAS behavior)
                row = key_rows[-1] if key_rows else {}
                # Set IN= flag variables
                in_flags[i] = bool(key_rows)
                # Merge columns (overwrite in order: later datasets win)
                merged.update(row)

            # Set IN= variables
            for bn in node.body:
                if isinstance(bn, MergeNode):
                    for var_name in bn.in_vars:
                        # find which dataset index this IN= var corresponds to
                        pass  # set below

            # Set IN= vars on the merged row
            for i, ds_rows in enumerate(sorted_dss):
                for bn in node.body:
                    if isinstance(bn, MergeNode):
                        for var_name in bn.in_vars:
                            merged[var_name] = 1 if in_flags.get(i) else 0

            result.append(merged)

        return result

    # ── PROC SQL ──────────────────────────────────────────────────────────

    def _run_proc(self, node: ProcNode) -> None:
        kind = (node.kind or "").upper()
        if kind == "SQL":
            self._run_sql(node)

    def _run_sql(self, node: ProcNode) -> None:
        raw = (node.raw or node.data or "").strip()
        if not raw:
            return
        # Strip PROC SQL; prefix if present
        cleaned = re.sub(
            r"^PROC\s+SQL\s*;\s*", "", raw, flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        self._exec_sql(cleaned)

    def _exec_sql(self, sql: str) -> None:
        """Execute a single SQL statement (CREATE TABLE ... SELECT ...)."""
        # Match: CREATE TABLE name AS SELECT ...
        ct_m = re.match(
            r"CREATE\s+TABLE\s+(\S+)\s+AS\s+(.+)",
            sql, re.IGNORECASE | re.DOTALL,
        )
        if ct_m:
            out_table = ct_m.group(1)
            select_body = ct_m.group(2).strip().rstrip(";").strip()
            rows = self._exec_select(select_body)
            self.datasets[out_table] = rows
            return

        # Bare SELECT (no CREATE TABLE) — just evaluate, no storage
        self._exec_select(sql.rstrip(";").strip())

    def _exec_select(self, select_body: str) -> list[dict]:
        """Execute SELECT [fields] FROM table [JOIN ...] [WHERE ...] [GROUP BY ...] [ORDER BY ...]."""
        body = select_body.strip()

        # Extract FROM clause
        from_m = re.search(r"\bFROM\s+(\S+)(?:\s+AS\s+\w+)?", body, re.IGNORECASE)
        if not from_m:
            return []
        main_table = from_m.group(1)

        # Get base rows
        rows = list(self.datasets.get(main_table, []))

        # Handle LEFT JOIN
        join_m = re.search(
            r"\bLEFT\s+JOIN\s+(\S+)(?:\s+AS\s+\w+)?\s+ON\s+(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|$)",
            body, re.IGNORECASE | re.DOTALL,
        )
        join_table: str | None = None
        join_on: str | None = None
        if join_m:
            join_table = join_m.group(1)
            join_on = join_m.group(2).strip()

        # Extract WHERE clause
        where_m = re.search(
            r"\bWHERE\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bHAVING\b|$)",
            body, re.IGNORECASE | re.DOTALL,
        )
        where_cond = where_m.group(1).strip() if where_m else None

        # Extract GROUP BY clause
        group_m = re.search(
            r"\bGROUP\s+BY\s+(.+?)(?:\bORDER\b|\bHAVING\b|$)",
            body, re.IGNORECASE | re.DOTALL,
        )
        group_by_raw = group_m.group(1).strip() if group_m else None

        # Extract ORDER BY clause
        order_m = re.search(
            r"\bORDER\s+BY\s+(.+?)$",
            body, re.IGNORECASE | re.DOTALL,
        )
        order_by_raw = order_m.group(1).strip() if order_m else None

        # Extract SELECT fields
        select_m = re.match(
            r"\bSELECT\s+(.+?)\bFROM\b",
            body, re.IGNORECASE | re.DOTALL,
        )
        if not select_m:
            return []
        select_raw = select_m.group(1).strip()

        # Parse select expressions
        select_exprs = self._parse_select_exprs(select_raw)

        # JOIN
        if join_table and join_on:
            join_rows = self.datasets.get(join_table, [])
            rows = self._left_join(rows, join_rows, join_on)

        # WHERE
        if where_cond:
            rows = [r for r in rows if self._eval_sql_cond(where_cond, r)]

        # GROUP BY
        if group_by_raw:
            group_keys = [g.strip() for g in group_by_raw.split(",")]
            rows = self._sql_group_by(rows, group_keys, select_exprs)

        # ORDER BY
        if order_by_raw:
            order_fields = [g.strip() for g in order_by_raw.split(",")]
            for f in reversed(order_fields):
                desc = f.upper().endswith(" DESC")
                fname = re.sub(r"\s+(ASC|DESC)\s*$", "", f, flags=re.IGNORECASE).strip()
                rows.sort(key=lambda r, fn=fname: str(r.get(fn, "")), reverse=desc)

        # Project to selected fields
        result = self._sql_project(rows, select_exprs)

        return result

    def _parse_select_exprs(self, raw: str) -> list[tuple[str, str]]:
        """Parse SELECT expressions into list of (expr, alias).
        Each entry: (raw_expression, alias_name)
        For 'MAX(field) AS alias' → ('MAX(field)', 'alias')
        For 't1.*' → ('t1.*', '*')
        For 'field' → ('field', 'field')
        """
        if raw.upper().strip() == "*":
            return [("*", "*")]
        exprs: list[tuple[str, str]] = []
        # Split by commas respecting parentheses
        parts = self._split_sql_exprs(raw)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Check for AS alias
            as_m = re.search(r"\bAS\s+(\w+)\s*$", part, re.IGNORECASE)
            if as_m:
                expr = part[: as_m.start()].strip()
                alias = as_m.group(1)
            else:
                expr = part
                # Derive alias from expression
                if "." in expr and not expr.startswith("*"):
                    alias = expr.split(".")[-1].strip()
                elif "(" in expr:
                    fn_m = re.match(r"\w+\s*\((.+)\)", expr, re.DOTALL)
                    alias = (fn_m.group(1).strip().split(",")[0].strip()
                             if fn_m else expr)
                else:
                    alias = expr.strip()
                # Strip table prefix for alias
                if "." in alias:
                    alias = alias.split(".")[-1].strip()
            exprs.append((expr, alias))
        return exprs

    def _split_sql_exprs(self, text: str) -> list[str]:
        """Split SQL expression list on commas, respecting parentheses."""
        parts = []
        depth = 0
        current: list[str] = []
        for ch in text:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    def _left_join(
        self,
        left: list[dict],
        right: list[dict],
        on_cond: str,
    ) -> list[dict]:
        """Execute LEFT JOIN ... ON condition."""
        result = []
        for lrow in left:
            matches = 0
            for rrow in right:
                # Build combined env for condition evaluation
                env = {**lrow, **rrow}
                # Prefix disambiguation (t1.X, t2.X)
                for k, v in lrow.items():
                    env[f"t1.{k}"] = v
                    env[f"__t1_{k}"] = v
                for k, v in rrow.items():
                    env[f"t2.{k}"] = v
                    env[f"__t2_{k}"] = v
                if self._eval_sql_cond(on_cond, env):
                    merged = {**lrow, **rrow}
                    result.append(merged)
                    matches += 1
            if matches == 0:
                result.append(dict(lrow))
        return result

    def _eval_sql_cond(self, cond: str, env: dict) -> bool:
        """Evaluate a SQL WHERE/ON condition."""
        c = cond
        # Collapse whitespace (multiline ON conditions)
        c = re.sub(r"\s+", " ", c)
        # Build two candidate expressions:
        #  - mangled: t1.X → __t1_X (for JOIN ON with same-named columns)
        c_m = re.sub(r"\b(\w+)\.(\w+)", r"__\1_\2", c)
        #  - bare: t1.X → X (for WHERE on merged rows, or JOIN ON without conflict)
        c_b = re.sub(r"\b\w+\.(\w+)", r"\1", c)
        for c in set([c_m, c_b]):
            expr = c
            expr = re.sub(r"\bAND\b", "and", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bOR\b", "or", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bNOT\b", "not", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bIS\s+NOT\s+MISSING\b", "is not None", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bIS\s+MISSING\b", "is None", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bIS\s+NOT\s+NULL\b", "is not None", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bIS\s+NULL\b", "is None", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bNE\b", "!=", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bEQ\b", "==", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bGT\b", ">", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bLT\b", "<", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bGE\b", ">=", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bLE\b", "<=", expr, flags=re.IGNORECASE)
            # = → == (standalone = not followed by =)
            expr = re.sub(r"(?<![!<>])=(?!=)", "==", expr)
            # 'string' → "string"
            expr = re.sub(r"'([^']*)'", r'"\1"', expr)
            try:
                return bool(eval(expr, {"__builtins__": {}}, env))  # noqa: S307
            except (NameError, KeyError):
                continue
            except Exception:
                return False
        return False

    def _sql_group_by(
        self,
        rows: list[dict],
        group_keys: list[str],
        select_exprs: list[tuple[str, str]],
    ) -> list[dict]:
        """Group rows and aggregate."""
        groups: dict[str, list[dict]] = {}
        for row in rows:
            key_parts = []
            for gk in group_keys:
                key_parts.append(str(row.get(gk.strip(), "")))
            key = "|".join(key_parts)
            groups.setdefault(key, []).append(row)

        result = []
        for key, group_rows in groups.items():
            out = dict(group_rows[0])  # start with first row's values
            # Apply aggregations
            for expr, alias in select_exprs:
                up = expr.upper().strip()
                # Detect aggregation functions
                agg_m = re.match(
                    r"(MAX|MIN|SUM|MEAN|AVG|COUNT|N)\s*\((.+)\)\s*$",
                    up, re.IGNORECASE,
                )
                if agg_m:
                    fn = agg_m.group(1).upper()
                    field = agg_m.group(2).strip()
                    values = []
                    for gr in group_rows:
                        v = gr.get(field)
                        if v is not None:
                            try:
                                values.append(float(v))
                            except (ValueError, TypeError):
                                values.append(v)
                    if fn in ("MAX",):
                        out[alias] = max(values) if values else None
                    elif fn in ("MIN",):
                        out[alias] = min(values) if values else None
                    elif fn in ("SUM",):
                        out[alias] = sum(values) if values else None
                    elif fn in ("MEAN", "AVG"):
                        out[alias] = (sum(values) / len(values)
                                      if values else None)
                    elif fn in ("COUNT", "N"):
                        out[alias] = len(group_rows)
                elif up == "*":
                    pass  # all fields already included
            result.append(out)
        return result

    def _sql_project(
        self,
        rows: list[dict],
        select_exprs: list[tuple[str, str]],
    ) -> list[dict]:
        """Project rows to only selected fields/aliases."""
        if not select_exprs:
            return rows
        if len(select_exprs) == 1 and select_exprs[0][0] == "*":
            return rows

        result = []
        for row in rows:
            out = {}
            has_wildcard = False
            for expr, alias in select_exprs:
                up = expr.upper().strip()
                # Wildcard: t1.* or * — include all already-present fields
                if up.endswith(".*") or up == "*":
                    has_wildcard = True
                    continue
                # Aggregation — already computed in group by
                agg_m = re.match(
                    r"(MAX|MIN|SUM|MEAN|AVG|COUNT|N)\s*\(.+\)",
                    up, re.IGNORECASE,
                )
                if agg_m:
                    if alias in row:
                        out[alias] = row[alias]
                    continue
                # Plain field reference
                if "." in expr:
                    field = expr.split(".")[-1].strip()
                else:
                    field = expr.strip()
                if field in row:
                    out[alias] = row[field]
            if has_wildcard:
                out = {**row, **out}
            result.append(out)
        return result


class _RowEvaluator:
    """Evaluates a DATA step body for one row, returning output env or None."""

    def __init__(self) -> None:
        self._step_passes = True
        self._row_deleted = False
        self._first_filter_done = False
        self._first_filter_passed = True
        self._current_step = ""

    def eval_data_step(
        self, node: DataStepNode, env: dict,
    ) -> dict | None:
        """Process one row through a DATA step body. Returns output or None."""
        self._current_step = node.output_dataset
        self._step_passes = True
        self._row_deleted = False
        self._first_filter_done = False
        self._first_filter_passed = True

        # Set MERGE IN= variables
        for bn in node.body:
            if isinstance(bn, MergeNode):
                for v in bn.in_vars:
                    env[v] = 1

        # Collect all body nodes excluding MergeNode (metadata)
        body = [n for n in node.body if not isinstance(n, MergeNode)]
        self._eval_body(body, env)

        if self._row_deleted or not self._step_passes:
            return None
        return dict(env)

    def _eval_body(self, body: list[AnyNode], env: dict) -> None:
        for n in body:
            if not self._step_passes or self._row_deleted:
                break
            self._eval_node(n, env)

    def _eval_node(self, node: AnyNode, env: dict) -> None:
        if isinstance(node, AssignNode):
            self._eval_assign(node, env)
        elif isinstance(node, IfNode):
            self._eval_if(node, env)
        elif isinstance(node, FilterNode):
            self._eval_filter(node, env)
        elif isinstance(node, DoLoopNode):
            self._eval_do_loop(node, env)
        elif isinstance(node, SelectNode):
            self._eval_select(node, env)
        elif isinstance(node, RetainNode):
            self._eval_retain(node, env)
        elif isinstance(node, OutputNode):
            pass  # implicit output at end of step
        # MergeNode, ByNode, etc. — skip

    def _eval_assign(self, node: AssignNode, env: dict) -> None:
        new = _eval_expr(node.expr, env)
        if new is not None:
            env[node.var] = new

    def _eval_if(self, node: IfNode, env: dict) -> None:
        result = _eval_expr(node.condition, env, is_condition=True)
        taken = bool(result)
        self._eval_body(node.then_branch if taken else node.else_branch, env)

    def _eval_filter(self, node: FilterNode, env: dict) -> None:
        cond = _eval_expr(node.condition, env, is_condition=True)
        passes = bool(cond)
        if not self._first_filter_done:
            self._first_filter_done = True
            self._first_filter_passed = passes
        if not passes:
            self._step_passes = False

    def _eval_do_loop(self, node: DoLoopNode, env: dict) -> None:
        iters = 0
        if node.var:
            start = _eval_expr(node.start, env) or 0
            stop = _eval_expr(node.stop, env) or 0
            step = _eval_expr(node.by_step, env) or 1
            val = float(start)
            while val <= float(stop) and iters < 1000:
                env[node.var] = val
                self._eval_body(node.body, env)
                if self._row_deleted or not self._step_passes:
                    break
                val += float(step)
                iters += 1
        elif node.while_cond:
            while self._eval_expr_bool(node.while_cond, env) and iters < 1000:
                self._eval_body(node.body, env)
                if self._row_deleted or not self._step_passes:
                    break
                iters += 1
        elif node.until_cond:
            while iters < 1000:
                self._eval_body(node.body, env)
                if self._row_deleted or not self._step_passes:
                    break
                iters += 1
                if self._eval_expr_bool(node.until_cond, env):
                    break
        else:
            # Bare DO/END block
            self._eval_body(node.body, env)

    def _eval_select(self, node: SelectNode, env: dict) -> None:
        value = None
        if node.expression:
            value = _eval_expr(node.expression, env)
        for when_node in node.when_clauses:
            if node.expression and when_node.condition:
                # Value-based: WHEN (value) ...
                wv = _eval_expr(when_node.condition, env)
                if value is not None and value == wv:
                    self._eval_body(when_node.body, env)
                    return
            elif when_node.condition:
                # Condition-based: WHEN (condition) ...
                cond = _eval_expr(when_node.condition, env, is_condition=True)
                if bool(cond):
                    self._eval_body(when_node.body, env)
                    return
        if node.otherwise_body:
            self._eval_body(node.otherwise_body, env)

    def _eval_retain(self, node: RetainNode, env: dict) -> None:
        if node.var not in env:
            if node.initial_value:
                env[node.var] = _eval_expr(node.initial_value, env)
            else:
                env[node.var] = 0 if isinstance(node.var, str) else None

    def _eval_expr_bool(self, expr: str, env: dict) -> bool:
        return bool(_eval_expr(expr, env, is_condition=True))
