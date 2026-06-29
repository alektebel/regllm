"""AST mutation engine for generating unlimited verifiable SAS bugs.

Takes a parsed SAS AST + input row, mutates one AST node, verifies the
mutation changes the output, and returns structured metadata for RL training.

Mutation types:
  - swap_vars: swap two variables in an expression
  - wrong_op: change a comparison operator
  - wrong_literal: change a numeric literal
  - remove_guard: remove a COALESCE or IF guard
  - wrong_agg: change an aggregation function (SUM↔MAX, etc.)

Usage:
    from training.bug_generator import BugGenerator
    gen = BugGenerator(sas_code, input_rows)
    bug = gen.generate()  # returns MutatedBug or None
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sas_logic_tree import (
    SASLogicTree,
    AssignNode,
    IfNode,
    FilterNode,
    DataStepNode,
    ProcNode,
    DoLoopNode,
    SelectNode,
    AnyNode,
)


# ── Mutation metadata ──────────────────────────────────────────────────

@dataclass
class MutationMeta:
    """Describes what was mutated and where."""
    mutation_type: str          # swap_vars, wrong_op, wrong_literal, etc.
    step_name: str              # DATA step or PROC where mutation lives
    node_index: int             # index within the step body
    original_expr: str          # expression before mutation
    mutated_expr: str           # expression after mutation
    target_var: str             # variable affected (for AssignNode/IfNode)
    description: str            # human-readable explanation


@dataclass
class MutatedBug:
    """A complete mutated bug instance ready for RL training."""
    original_code: str
    mutated_code: str
    mutation: MutationMeta
    correct_values: dict[str, Any]     # evaluator output on original
    buggy_values: dict[str, Any]       # evaluator output on mutated
    changed_fields: list[str]          # fields whose values differ
    depth: int                         # how many steps between mutation and output
    expected_tools: list[str] = field(default_factory=list)
    task_kind: str = ""                # lineage | regulation | docs | schema | debug
    source: str = "procedural"         # procedural | curated_level | tool_task
    level_id: str = ""
    # The exact input row whose evaluation produced correct_values/buggy_values.
    # Required so the reward function can re-evaluate a proposed fix without
    # needing the generator that produced this bug.
    input_row: dict[str, Any] = field(default_factory=dict)
    # BFS path symptom → mutation root (for trace_path reward).
    gold_trace: dict[str, Any] = field(default_factory=dict)


# ── Mutation operators ─────────────────────────────────────────────────

_COMPARISON_OPS = ['>=', '<=', '>', '<', '=', 'NE']
_ARITH_OPS = ['+', '-', '*', '/']
_AGG_FUNCS = ['SUM', 'MAX', 'MIN', 'MEAN', 'COUNT']


def _find_variables_in_expr(expr: str) -> list[str]:
    """Extract SAS variable names from an expression (non-keyword)."""
    _SKIP = {
        'AND', 'OR', 'NOT', 'IF', 'THEN', 'ELSE', 'DO', 'END',
        'MAX', 'MIN', 'ABS', 'ROUND', 'SUM', 'MEAN', 'COALESCE',
        'INT', 'FLOOR', 'CEIL', 'SQRT', 'LOG', 'EXP', 'MOD',
        'UPCASE', 'LOWCASE', 'TRIM', 'STRIP', 'COMPRESS',
        'PUT', 'INPUT', 'SUBSTR', 'LENGTH', 'SCAN', 'INDEX',
        'WORK', 'TRUE', 'FALSE', 'NULL', 'MISSING',
        'EQ', 'NE', 'GT', 'LT', 'GE', 'LE', 'IN', 'BETWEEN',
        'WHERE', 'SET', 'DATA', 'TO', 'BY', 'LIKE',
        'IFN', 'IFC', 'COALESCEC', 'COUNT', 'DISTINCT',
        'TEMPORARY', 'INITIAL', 'N', 'I', 'J', 'K',
    }
    cleaned = re.sub(r"'[^']*'", " ", expr)
    cleaned = re.sub(r'"[^"]*"', " ", cleaned)
    tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', cleaned)
    return [t for t in tokens if t.upper() not in _SKIP]


def _find_numeric_literals(expr: str) -> list[tuple[int, int, str]]:
    """Find numeric literals in an expression. Returns (start, end, value)."""
    results = []
    for m in re.finditer(r'(?<![A-Za-z_])(\d+\.?\d*)', expr):
        results.append((m.start(), m.end(), m.group()))
    return results


def _find_comparison_ops(expr: str) -> list[tuple[int, int, str]]:
    """Find comparison operators. Returns (start, end, op)."""
    results = []
    for m in re.finditer(r'(>=|<=|>(?!=)|<(?!=)|(?<![<>])=(?!=))', expr):
        results.append((m.start(), m.end(), m.group()))
    return results


# ── Individual mutation functions ──────────────────────────────────────

def _mutate_swap_vars(expr: str, rng: random.Random) -> tuple[str, str] | None:
    """Swap two variable names in an expression."""
    variables = _find_variables_in_expr(expr)
    unique_vars = list(set(variables))
    if len(unique_vars) < 2:
        return None
    v1, v2 = rng.sample(unique_vars, 2)
    # Use a placeholder to avoid collision
    new_expr = re.sub(rf'\b{re.escape(v1)}\b', '__PLACEHOLDER__', expr)
    new_expr = re.sub(rf'\b{re.escape(v2)}\b', v1, new_expr)
    new_expr = new_expr.replace('__PLACEHOLDER__', v2)
    if new_expr == expr:
        return None
    return new_expr, f"swapped {v1} ↔ {v2}"


def _mutate_wrong_op(expr: str, rng: random.Random) -> tuple[str, str] | None:
    """Change a comparison operator."""
    ops = _find_comparison_ops(expr)
    if not ops:
        return None
    start, end, old_op = rng.choice(ops)
    alternatives = [op for op in _COMPARISON_OPS if op != old_op]
    new_op = rng.choice(alternatives)
    new_expr = expr[:start] + new_op + expr[end:]
    return new_expr, f"changed operator {old_op} → {new_op}"


def _mutate_wrong_literal(expr: str, rng: random.Random) -> tuple[str, str] | None:
    """Change a numeric literal to a nearby value."""
    literals = _find_numeric_literals(expr)
    if not literals:
        return None
    start, end, old_val = rng.choice(literals)
    try:
        num = float(old_val)
    except ValueError:
        return None
    # Perturbation strategies
    if num == 0:
        new_num = rng.choice([1, 0.01, -1])
    elif abs(num) < 1:
        # Small number: shift by 0.05–0.20
        delta = rng.uniform(0.05, 0.20) * rng.choice([-1, 1])
        new_num = round(num + delta, 4)
    elif abs(num) < 100:
        # Medium: shift by 1–5 or multiply
        new_num = num + rng.choice([-3, -1, 1, 3])
    else:
        # Large: scale by 0.5 or 2.0
        new_num = num * rng.choice([0.5, 2.0])
    # Format like original
    if '.' in old_val:
        new_str = f"{new_num:.{len(old_val.split('.')[1])}f}"
    else:
        new_str = str(int(new_num))
    new_expr = expr[:start] + new_str + expr[end:]
    if new_expr == expr:
        return None
    return new_expr, f"changed literal {old_val} → {new_str}"


def _mutate_remove_guard(expr: str, rng: random.Random) -> tuple[str, str] | None:
    """Remove a COALESCE wrapper, leaving just the first argument."""
    m = re.search(r'\bCOALESCE\s*\(', expr, re.IGNORECASE)
    if not m:
        return None
    # Find matching paren
    start = m.start()
    paren_start = m.end() - 1
    depth = 0
    i = paren_start
    while i < len(expr):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        return None
    inner = expr[paren_start + 1:i]
    # Take first argument only
    args = []
    d = 0
    buf = []
    for ch in inner:
        if ch == '(':
            d += 1
        elif ch == ')':
            d -= 1
        elif ch == ',' and d == 0:
            args.append(''.join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    args.append(''.join(buf).strip())
    if len(args) < 2:
        return None
    first_arg = args[0]
    new_expr = expr[:start] + first_arg + expr[i + 1:]
    return new_expr, f"removed COALESCE guard (kept only {first_arg})"


def _mutate_wrong_agg_sql(raw_sql: str, rng: random.Random) -> tuple[str, str] | None:
    """Change an aggregation function in PROC SQL."""
    pattern = r'\b(SUM|MAX|MIN|MEAN|COUNT)\s*\('
    matches = list(re.finditer(pattern, raw_sql, re.IGNORECASE))
    if not matches:
        return None
    m = rng.choice(matches)
    old_agg = m.group(1).upper()
    alternatives = [a for a in _AGG_FUNCS if a != old_agg]
    new_agg = rng.choice(alternatives)
    new_sql = raw_sql[:m.start(1)] + new_agg + raw_sql[m.end(1):]
    return new_sql, f"changed aggregation {old_agg} → {new_agg}"


# ── Mutation target collector ──────────────────────────────────────────

@dataclass
class MutationTarget:
    """A candidate location for mutation."""
    step_name: str
    step_index: int         # index in the top-level node list
    node_path: list[int]    # path within body to reach the node
    node: AnyNode
    depth_from_output: int  # 0 = last step, 1 = second-to-last, etc.


def _collect_targets(
    nodes: list[AnyNode],
    step_name: str,
    step_index: int,
    depth: int,
    body: list[AnyNode],
    path: list[int] | None = None,
) -> list[MutationTarget]:
    """Recursively collect mutable nodes from a step body."""
    if path is None:
        path = []
    targets = []
    for i, node in enumerate(body):
        current_path = path + [i]
        if isinstance(node, AssignNode):
            targets.append(MutationTarget(step_name, step_index, current_path, node, depth))
        elif isinstance(node, IfNode):
            targets.append(MutationTarget(step_name, step_index, current_path, node, depth))
            targets.extend(_collect_targets(nodes, step_name, step_index, depth, node.then_branch, current_path + [0]))
            targets.extend(_collect_targets(nodes, step_name, step_index, depth, node.else_branch, current_path + [1]))
        elif isinstance(node, FilterNode):
            targets.append(MutationTarget(step_name, step_index, current_path, node, depth))
        elif isinstance(node, DoLoopNode):
            targets.extend(_collect_targets(nodes, step_name, step_index, depth, node.body, current_path))
        elif isinstance(node, SelectNode):
            for wi, when in enumerate(node.whens):
                targets.extend(_collect_targets(nodes, step_name, step_index, depth, when.body, current_path + [wi]))
    return targets


# ── Main generator ─────────────────────────────────────────────────────

_MUTATION_FUNCS = {
    'swap_vars': _mutate_swap_vars,
    'wrong_op': _mutate_wrong_op,
    'wrong_literal': _mutate_wrong_literal,
    'remove_guard': _mutate_remove_guard,
}


class BugGenerator:
    """Generate mutated SAS bugs with verifiable ground truth.

    Args:
        sas_code: Original (correct) SAS source code.
        input_rows: List of dicts, each a row to evaluate through the pipeline.
        seed: Random seed for reproducibility.
        max_depth: Maximum depth from final output to allow mutations.
                   0 = only last step, 1 = last two steps, None = any step.
    """

    def __init__(
        self,
        sas_code: str,
        input_rows: list[dict[str, Any]],
        seed: int | None = None,
        max_depth: int | None = None,
    ):
        self.sas_code = sas_code
        self.input_rows = input_rows
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self.tree = SASLogicTree()
        self.original_nodes = self.tree.parse(sas_code)

        # Compute correct baseline values for each row
        self.correct_outputs: list[dict[str, Any]] = []
        for row in input_rows:
            trace = self.tree.evaluate(self.original_nodes, dict(row))
            self.correct_outputs.append(trace.final)

        # Collect mutation targets with depth info
        self._targets = self._collect_all_targets()

    def _collect_all_targets(self) -> list[MutationTarget]:
        """Collect all mutable nodes across all DATA steps."""
        # Count DATA steps to compute depth from output
        data_steps = [
            (i, n) for i, n in enumerate(self.original_nodes)
            if isinstance(n, DataStepNode)
        ]
        num_steps = len(data_steps)
        targets = []
        for rank, (idx, step) in enumerate(data_steps):
            depth = num_steps - 1 - rank  # 0 = last step
            if self.max_depth is not None and depth > self.max_depth:
                continue
            targets.extend(
                _collect_targets(self.original_nodes, step.output_dataset, idx, depth, step.body)
            )
        # Also collect from PROC SQL nodes
        for idx, node in enumerate(self.original_nodes):
            if isinstance(node, ProcNode) and node.kind == "SQL" and node.raw:
                targets.append(MutationTarget(
                    step_name=node.output_table or f"PROC_SQL_{idx}",
                    step_index=idx,
                    node_path=[],
                    node=node,
                    depth_from_output=0,  # treated as shallow
                ))
        return targets

    def generate(self, max_attempts: int = 50) -> MutatedBug | None:
        """Try to generate a valid mutated bug.

        Returns None if no mutation produces a different output.
        """
        if not self._targets:
            return None

        for _ in range(max_attempts):
            target = self.rng.choice(self._targets)
            result = self._try_mutate(target)
            if result is not None:
                return result
        return None

    def generate_batch(self, n: int, max_attempts_per: int = 50) -> list[MutatedBug]:
        """Generate n unique bugs."""
        bugs = []
        seen_exprs: set[str] = set()
        for _ in range(n * 3):  # try harder than n
            if len(bugs) >= n:
                break
            bug = self.generate(max_attempts_per)
            if bug and bug.mutation.mutated_expr not in seen_exprs:
                seen_exprs.add(bug.mutation.mutated_expr)
                bugs.append(bug)
        return bugs

    def _try_mutate(self, target: MutationTarget) -> MutatedBug | None:
        """Attempt a single mutation on the target node."""
        node = target.node

        # Choose mutation type based on node type
        if isinstance(node, ProcNode):
            # PROC SQL: try wrong_agg
            result = _mutate_wrong_agg_sql(node.raw, self.rng)
            if result is None:
                return None
            new_raw, desc = result
            mutation_type = "wrong_agg"
            original_expr = node.raw[:120]
            mutated_expr = new_raw[:120]
            target_var = node.output_table

            # Apply mutation to code text
            mutated_code = self.sas_code.replace(node.raw, new_raw, 1)
            if mutated_code == self.sas_code:
                return None

        elif isinstance(node, AssignNode):
            mutation_type, result = self._pick_mutation_for_expr(node.expr)
            if result is None:
                return None
            new_expr, desc = result
            original_expr = node.expr
            mutated_expr = new_expr
            target_var = node.var

            # Apply mutation to code — replace the assignment in source
            mutated_code = self._apply_assign_mutation(node.var, node.expr, new_expr)
            if mutated_code is None:
                return None

        elif isinstance(node, IfNode):
            mutation_type, result = self._pick_mutation_for_expr(node.condition)
            if result is None:
                return None
            new_expr, desc = result
            original_expr = node.condition
            mutated_expr = new_expr
            target_var = f"IF_condition"

            # Apply mutation to source
            mutated_code = self._apply_condition_mutation(node.condition, new_expr)
            if mutated_code is None:
                return None

        elif isinstance(node, FilterNode):
            mutation_type, result = self._pick_mutation_for_expr(node.condition)
            if result is None:
                return None
            new_expr, desc = result
            original_expr = node.condition
            mutated_expr = new_expr
            target_var = f"WHERE_filter"

            mutated_code = self._apply_condition_mutation(node.condition, new_expr)
            if mutated_code is None:
                return None

        else:
            return None

        # Parse and evaluate the mutated code
        try:
            mutated_nodes = self.tree.parse(mutated_code)
        except Exception:
            return None

        # Check if mutation actually changes output for any row
        changed_fields_all: set[str] = set()
        buggy_outputs: list[dict[str, Any]] = []
        for i, row in enumerate(self.input_rows):
            try:
                trace = self.tree.evaluate(mutated_nodes, dict(row))
            except Exception:
                return None
            buggy_outputs.append(trace.final)
            correct = self.correct_outputs[i]
            for k in set(correct.keys()) | set(trace.final.keys()):
                cv = correct.get(k)
                bv = trace.final.get(k)
                if cv != bv:
                    # Skip string-only fields for "changed" detection
                    if isinstance(cv, str) and isinstance(bv, str):
                        continue
                    changed_fields_all.add(k)

        if not changed_fields_all:
            return None  # mutation is dead code

        # Use first row that shows a difference for the output
        diff_row_idx = 0
        for i, (c, b) in enumerate(zip(self.correct_outputs, buggy_outputs)):
            for k in changed_fields_all:
                if c.get(k) != b.get(k):
                    diff_row_idx = i
                    break

        mutation = MutationMeta(
            mutation_type=mutation_type,
            step_name=target.step_name,
            node_index=target.step_index,
            original_expr=original_expr,
            mutated_expr=mutated_expr,
            target_var=target_var,
            description=desc,
        )

        from training.gold_trace import attach_gold_trace
        return attach_gold_trace(MutatedBug(
            original_code=self.sas_code,
            mutated_code=mutated_code,
            mutation=mutation,
            correct_values=self.correct_outputs[diff_row_idx],
            buggy_values=buggy_outputs[diff_row_idx],
            changed_fields=sorted(changed_fields_all),
            depth=target.depth_from_output,
            input_row=dict(self.input_rows[diff_row_idx]),
        ))

    def _pick_mutation_for_expr(self, expr: str) -> tuple[str, tuple[str, str] | None]:
        """Pick a random applicable mutation for an expression."""
        applicable = []
        for name, func in _MUTATION_FUNCS.items():
            result = func(expr, self.rng)
            if result is not None:
                applicable.append((name, result))
        if not applicable:
            return ("none", None)
        name, result = self.rng.choice(applicable)
        return (name, result)

    def _apply_assign_mutation(self, var: str, old_expr: str, new_expr: str) -> str | None:
        """Replace an assignment expression in the source code."""
        # Match: VAR = old_expr ;
        # Normalize whitespace for matching
        old_norm = ' '.join(old_expr.split())
        pattern = re.compile(
            rf'\b{re.escape(var)}\s*=\s*' + re.escape(old_norm).replace(r'\ ', r'\s+'),
            re.IGNORECASE,
        )
        m = pattern.search(self.sas_code)
        if m:
            return self.sas_code[:m.start()] + f"{var} = {new_expr}" + self.sas_code[m.end():]

        # Fallback: direct string replacement
        old_stmt = f"{var} = {old_expr}"
        if old_stmt in self.sas_code:
            return self.sas_code.replace(old_stmt, f"{var} = {new_expr}", 1)
        return None

    def _apply_condition_mutation(self, old_cond: str, new_cond: str) -> str | None:
        """Replace a condition in the source code (IF or WHERE)."""
        # Try direct replacement
        if old_cond in self.sas_code:
            return self.sas_code.replace(old_cond, new_cond, 1)

        # Try normalized whitespace
        old_norm = ' '.join(old_cond.split())
        code_norm = ' '.join(self.sas_code.split())
        if old_norm in code_norm:
            # Reconstruct by replacing in normalized then undoing
            # Simpler: just try fuzzy match
            pattern = re.escape(old_norm).replace(r'\ ', r'\s+')
            m = re.search(pattern, self.sas_code, re.IGNORECASE)
            if m:
                return self.sas_code[:m.start()] + new_cond + self.sas_code[m.end():]
        return None


# ── Convenience: generate from toy_lgd correct code ───────────────────

_TOY_LGD = PROJECT_ROOT / "data" / "sas" / "toy_lgd"

# Standard input rows for the toy LGD pipeline
TOY_INPUT_ROWS = [
    {
        "ID_CONTR_CICLO_LGD": "CIC_001", "ID_CONTRATO": "CONT_001",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "NINGUNA",
        "PD_ESTIMADA": 0.010, "LGD_ESTIMADA": 0.40, "EAD": 100000,
        "DPDS": 0, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
    },
    {
        "ID_CONTR_CICLO_LGD": "CIC_003", "ID_CONTRATO": "CONT_003",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "HIPOTECA",
        "PD_ESTIMADA": 0.010, "LGD_ESTIMADA": 0.20, "EAD": 150000,
        "DPDS": 0, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
    },
    {
        "ID_CONTR_CICLO_LGD": "CIC_007", "ID_CONTRATO": "CONT_007",
        "SEGMENTO": "RETAIL", "COLATERAL_TIPO": "PERSONAL",
        "PD_ESTIMADA": 0.030, "LGD_ESTIMADA": 0.35, "EAD": 80000,
        "DPDS": 30, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
    },
    {
        "ID_CONTR_CICLO_LGD": "CIC_009", "ID_CONTRATO": "CONT_009",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "NINGUNA",
        "PD_ESTIMADA": 0.010, "LGD_ESTIMADA": 0.45, "EAD": 120000,
        "DPDS": 5, "STAGE_IFRS9": 1, "CURE_FLAG": 1,
        "PROVISION_PERIOD_MONTHS": 12,
    },
]


def make_toy_generator(
    seed: int | None = None,
    max_depth: int | None = None,
    correct_code_path: str | Path | None = None,
) -> BugGenerator:
    """Create a BugGenerator from the toy_lgd correct SAS code."""
    if correct_code_path is None:
        correct_code_path = _TOY_LGD / "toy_lgd_correct.sas"
    code = Path(correct_code_path).read_text(encoding="utf-8")
    return BugGenerator(code, TOY_INPUT_ROWS, seed=seed, max_depth=max_depth)


if __name__ == "__main__":
    gen = make_toy_generator(seed=42)
    print(f"Targets: {len(gen._targets)}")
    print(f"Input rows: {len(gen.input_rows)}")

    bugs = gen.generate_batch(10)
    print(f"\nGenerated {len(bugs)} unique bugs:\n")
    for i, bug in enumerate(bugs):
        print(f"  [{i}] type={bug.mutation.mutation_type}  step={bug.mutation.step_name}  depth={bug.depth}")
        print(f"      {bug.mutation.description}")
        print(f"      changed: {bug.changed_fields}")
        cv = {k: bug.correct_values.get(k) for k in bug.changed_fields[:3]}
        bv = {k: bug.buggy_values.get(k) for k in bug.changed_fields[:3]}
        print(f"      correct: {cv}")
        print(f"      buggy:   {bv}")
        print()
