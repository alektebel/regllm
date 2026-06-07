#!/usr/bin/env python3
"""
RegLLM · SAS Compiler — Terminal Demo

Shows the three pillars of the SAS compiler:
  1. AST — parsed abstract syntax tree from SAS DATA steps
  2. Lineage — field dependency graph showing which table produces each field
  3. Simulation — trace a real cycle row through the DATA step logic

Run:
    python demo/sas_compiler_demo.py                      # sample_lgd.sas + first 3 cycles
    python demo/sas_compiler_demo.py --sas path/to/f.sas  # custom SAS file
    python demo/sas_compiler_demo.py --cycle CIC_00005    # specific cycle
    python demo/sas_compiler_demo.py --all-cycles         # simulate every cycle
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from src.sas_logic_tree import SASLogicTree, AnyNode, DataStepNode, IfNode, AssignNode, FilterNode, ProcNode

console = Console()

BLUE   = "bright_blue"
GREEN  = "bright_green"
RED    = "bright_red"
AMBER  = "yellow"
MUTED  = "bright_black"
WHITE  = "bright_white"
INDIGO = "magenta"

DATA_DIR    = ROOT / "data" / "samples"
SAMPLE_SAS  = DATA_DIR / "sample_lgd.sas"
CYCLES_CSV  = DATA_DIR / "cycles_sample.csv"
CYCLES_V2   = DATA_DIR / "cycles_v2.csv"
CYCLES_V3   = DATA_DIR / "cycles_v3.csv"

_NUMERIC = {
    "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9",
    "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS",
    "VENTANA_CALIBRACION_YEARS", "RATING_GRADO", "ECL",
    "LGD_FLOOR_APLICADO", "MOC", "CURE_FLAG", "RWA",
}
_SIM_EXCLUDE = {
    "CICLO_ID", "CONTRATO", "ECL", "LGD_FLOOR_APLICADO",
    "FECHA_INCUMPLIMIENTO", "CURE_FLAG", "PERIODO_OBSERVACION", "RWA", "MOC",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_cycles(path: Path = CYCLES_CSV) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if k.upper() in _NUMERIC:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _cycle_values(row: dict) -> dict:
    return {
        k.upper(): v
        for k, v in row.items()
        if k.upper() not in _SIM_EXCLUDE and v != "" and v is not None
    }


# ── AST rendering ──────────────────────────────────────────────────────────────

def _add_ast_nodes(tree_node, nodes: list[AnyNode]) -> None:
    for node in nodes:
        if isinstance(node, DataStepNode):
            branch = tree_node.add(
                Text.assemble(
                    ("[DATA STEP] ", f"bold {INDIGO}"),
                    (node.output_dataset, f"bold {WHITE}"),
                    ("  ←  SET ", MUTED),
                    (node.input_dataset or "(none)", AMBER),
                )
            )
            _add_ast_nodes(branch, node.body)

        elif isinstance(node, IfNode):
            branch = tree_node.add(
                Text.assemble(
                    ("[IF]  ", f"bold {AMBER}"),
                    (node.condition, WHITE),
                )
            )
            if node.then_branch:
                then_b = branch.add(Text("[then]", style=GREEN))
                _add_ast_nodes(then_b, node.then_branch)
            if node.else_branch:
                else_b = branch.add(Text("[else]", style=MUTED))
                _add_ast_nodes(else_b, node.else_branch)

        elif isinstance(node, AssignNode):
            tree_node.add(
                Text.assemble(
                    ("[ASSIGN]  ", f"bold {GREEN}"),
                    (node.var, f"bold {WHITE}"),
                    (" = ", MUTED),
                    (node.expr, WHITE),
                )
            )

        elif isinstance(node, FilterNode):
            cond = " ".join(node.condition.split())
            tree_node.add(
                Text.assemble(
                    ("[WHERE]  ", f"bold {BLUE}"),
                    (cond[:120], WHITE),
                )
            )

        elif isinstance(node, ProcNode):
            tree_node.add(
                Text.assemble(
                    ("[PROC]  ", f"bold {MUTED}"),
                    (node.kind, WHITE),
                    ("  DATA=", MUTED),
                    (node.data or "(none)", WHITE),
                )
            )


def show_ast(sas_code: str, tree: SASLogicTree) -> None:
    nodes = tree.parse(sas_code)
    root = Tree(
        Text.assemble(("SAS Abstract Syntax Tree", f"bold {BLUE}"), ("  ·  ", MUTED),
                      (f"{len(nodes)} top-level nodes", MUTED)),
        guide_style=MUTED,
    )
    _add_ast_nodes(root, nodes)
    console.print(Panel(root, title="[bold]AST[/bold]", border_style=BLUE, box=box.ROUNDED))


# ── Lineage rendering ──────────────────────────────────────────────────────────

def show_lineage(sas_code: str, tree: SASLogicTree) -> None:
    nodes = tree.parse(sas_code)
    lg = tree.lineage(nodes)

    # Pipeline table
    pipe = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {MUTED}", expand=False)
    pipe.add_column("#",    style=MUTED, width=3, no_wrap=True)
    pipe.add_column("DATA step",   style=f"bold {WHITE}", no_wrap=True)
    for i, step in enumerate(lg.data_steps):
        pipe.add_row(str(i + 1), step)

    # Fields table
    fields = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {MUTED}", expand=True)
    fields.add_column("Field",      style=f"bold {WHITE}", no_wrap=True)
    fields.add_column("Kind",       no_wrap=True, width=10)
    fields.add_column("Layer",      style=MUTED, width=6, no_wrap=True)
    fields.add_column("Produced in", style=MUTED)

    kind_color = {"input": BLUE, "modified": AMBER, "computed": GREEN}
    for n in sorted(lg.nodes, key=lambda x: (x["layer"], x["id"])):
        step = ", ".join(n["data_steps"]) if n["data_steps"] else "—"
        kc = kind_color.get(n["kind"], MUTED)
        fields.add_row(
            n["id"],
            f"[{kc}]{n['kind']}[/{kc}]",
            str(n["layer"]),
            step,
        )

    # Edges table
    edges = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {MUTED}", expand=True)
    edges.add_column("Source",  style=BLUE, no_wrap=True)
    edges.add_column("→ Target", style=GREEN, no_wrap=True)
    edges.add_column("Kind",    style=MUTED, width=11, no_wrap=True)
    edges.add_column("Expr",    style=MUTED)

    for e in lg.edges:
        kind_style = AMBER if e["kind"] == "conditions" else MUTED
        edges.add_row(
            e["source"],
            e["target"],
            f"[{kind_style}]{e['kind']}[/{kind_style}]",
            e["expr"][:60] if e["expr"] else "—",
        )

    console.print(Panel(
        Padding(pipe, (0, 1)),
        title="[bold]DATA step pipeline[/bold]",
        border_style=BLUE, box=box.ROUNDED,
    ))
    console.print(Panel(
        Columns([
            Padding(fields, (0, 1)),
            Padding(edges, (0, 1)),
        ]),
        title=f"[bold]Field Lineage[/bold]  [{MUTED}]({len(lg.nodes)} fields · {len(lg.edges)} edges)[/{MUTED}]",
        border_style=GREEN, box=box.ROUNDED,
    ))


# ── Simulation rendering ───────────────────────────────────────────────────────

def show_simulation(sas_code: str, tree: SASLogicTree, cycles: list[dict], cycle_ids: list[str]) -> None:
    nodes = tree.parse(sas_code)
    cycle_map = {r.get("CICLO_ID", r.get("ciclo_id", "")): r for r in cycles}

    for cid in cycle_ids:
        row = cycle_map.get(cid)
        if row is None:
            console.print(f"[{RED}]Cycle {cid!r} not found.[/{RED}]")
            continue

        values = _cycle_values(row)
        trace = tree.evaluate(nodes, values)

        # Header for this cycle
        meta = Table.grid(padding=(0, 3))
        meta.add_column(style=MUTED, no_wrap=True)
        meta.add_column(style=WHITE, no_wrap=True)
        for k in ["CICLO_ID", "SEGMENTO", "COLATERAL_TIPO"]:
            meta.add_row(k, str(row.get(k, "—")))

        # Input values
        inp = Table(box=box.SIMPLE_HEAD, show_header=False, expand=False)
        inp.add_column(style=MUTED, no_wrap=True)
        inp.add_column(style=WHITE, no_wrap=True)
        for k, v in values.items():
            inp.add_row(k, str(v))

        # Trace steps
        trace_t = Table(box=box.SIMPLE_HEAD, show_header=True,
                        header_style=f"bold {MUTED}", expand=True, padding=(0, 1))
        trace_t.add_column("DATA step", style=MUTED, no_wrap=True, width=22)
        trace_t.add_column("Op",        style=WHITE, width=9,  no_wrap=True)
        trace_t.add_column("Detail",    style=WHITE)

        cur_step = ""
        for s in trace.steps:
            step_label = s.data_step.replace("work.", "") if s.data_step != cur_step else ""
            if s.data_step != cur_step:
                cur_step = s.data_step
            if s.kind == "assign":
                changed = s.old_val != s.new_val
                op = f"[{AMBER}]ASSIGN[/{AMBER}]" if changed else f"[{MUTED}]assign[/{MUTED}]"
                detail = f"[{WHITE}]{s.var}[/{WHITE}] = [{AMBER if changed else MUTED}]{s.new_val}[/]"
                if changed:
                    detail += f"  [{MUTED}](was {s.old_val})[/{MUTED}]"
                trace_t.add_row(step_label, op, detail)
            elif s.kind == "if_taken":
                trace_t.add_row(step_label, f"[{GREEN}]IF ✓[/{GREEN}]",
                                f"[{GREEN}]{s.label[:80]}[/{GREEN}]")
            elif s.kind == "if_skipped":
                trace_t.add_row(step_label, f"[{MUTED}]IF ✗[/{MUTED}]",
                                f"[{MUTED}]{s.label[:80]}[/{MUTED}]")
            elif s.kind == "filter_pass":
                trace_t.add_row(step_label, f"[{BLUE}]WHERE ✓[/{BLUE}]",
                                f"[{BLUE}]{s.label[:80]}[/{BLUE}]")
            elif s.kind == "filter_block":
                trace_t.add_row(step_label, f"[{MUTED}]WHERE ✗[/{MUTED}]",
                                f"[{MUTED}]{s.label[:80]}[/{MUTED}]")

        # Changed fields summary
        changed_fields = [(k, trace.initial.get(k), v)
                          for k, v in trace.final.items()
                          if trace.initial.get(k) != v]

        summary_lines: list[Text] = []
        gate_icon = f"[{GREEN}]✓[/{GREEN}]" if trace.row_passes_filter else f"[{RED}]✗[/{RED}]"
        summary_lines.append(Text.from_markup(
            f"  Entry filter: {gate_icon}  "
            f"({'row qualifies' if trace.row_passes_filter else 'row excluded'})"
        ))
        for fr in trace.filter_results:
            icon = f"[{BLUE}]✓[/{BLUE}]" if fr["passed"] else f"[{MUTED}]○[/{MUTED}]"
            step_name = fr["data_step"].replace("work.", "")
            cond = fr["condition"][:70]
            summary_lines.append(Text.from_markup(
                f"  {icon}  [{MUTED}][{step_name}][/{MUTED}]  {cond}"
            ))
        if changed_fields:
            summary_lines.append(Text(""))
            for k, old, new in changed_fields:
                summary_lines.append(Text.from_markup(
                    f"  [{AMBER}]{k}[/{AMBER}]  "
                    f"[{MUTED}]{old}[/{MUTED}] → [{WHITE}]{new}[/{WHITE}]"
                ))

        from rich.console import Group
        body = Group(
            Padding(Columns([meta, inp]), (0, 1)),
            Padding(trace_t, (1, 1)),
            Padding(Group(*summary_lines), (0, 1)),
        )
        entry_c = GREEN if trace.row_passes_filter else RED
        console.print(Panel(
            body,
            title=f"[bold]Simulation[/bold]  [{MUTED}]{cid}[/{MUTED}]  "
                  f"[{entry_c}]{'✓ qualifies' if trace.row_passes_filter else '✗ excluded'}[/{entry_c}]",
            border_style=entry_c,
            box=box.ROUNDED,
        ))


# ── V2/V3 diff explainer ──────────────────────────────────────────────────────

def show_diff(sas_code: str, ciclo_id: str, target: str = "ECL", use_gemma: bool = False) -> None:
    """Run the gradient + Shapley explainer on one V2/V3 row pair."""
    from src.sas_diff import explain_field_diff

    v2_rows = _load_cycles(CYCLES_V2)
    v3_rows = _load_cycles(CYCLES_V3)
    if not v2_rows or not v3_rows:
        console.print(
            f"[{RED}]V2/V3 sample CSVs not found. Run scripts/generate_v3.py first.[/{RED}]"
        )
        return
    by_pk = {r.get("CICLO_ID"): r for r in v3_rows}
    r2 = next((r for r in v2_rows if r.get("CICLO_ID") == ciclo_id), None)
    r3 = by_pk.get(ciclo_id)
    if not r2 or not r3:
        console.print(f"[{RED}]Cycle {ciclo_id!r} not found in V2/V3[/{RED}]")
        return

    v2 = {k.upper(): v for k, v in r2.items() if v != "" and v is not None}
    v3 = {k.upper(): v for k, v in r3.items() if v != "" and v is not None}

    report = explain_field_diff(
        sas_code=sas_code, row_v2=v2, row_v3=v3, target=target, method="both",
    )

    # Header
    delta = report.delta_y if report.delta_y is not None else float("nan")
    console.print(Panel(
        Text.assemble(
            (f"{target}", f"bold {WHITE}"),
            ("  V2 = ", MUTED),
            (f"{report.y_v2:.4f}" if report.y_v2 is not None else "—", AMBER),
            ("  V3 = ", MUTED),
            (f"{report.y_v3:.4f}" if report.y_v3 is not None else "—", GREEN),
            ("   Δ = ", MUTED),
            (f"{delta:+.4f}" if report.delta_y is not None else "—",
             GREEN if (report.delta_y or 0) >= 0 else RED),
        ),
        title=f"[bold]V2 → V3 diff[/bold]  [{MUTED}]{ciclo_id}[/{MUTED}]",
        border_style=BLUE,
        box=box.ROUNDED,
    ))

    # Field deltas table
    if report.field_deltas:
        ft = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {MUTED}")
        ft.add_column("Field", style=f"bold {WHITE}")
        ft.add_column("V2", style=BLUE, justify="right")
        ft.add_column("V3", style=AMBER, justify="right")
        for d in report.field_deltas:
            ft.add_row(d.field, str(d.v2), str(d.v3))
        console.print(Panel(ft, title="[bold]Input field changes[/bold]",
                            border_style=AMBER, box=box.ROUNDED))

    # Attribution table
    if report.gradient or report.shapley:
        at = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {MUTED}")
        at.add_column("Field", style=f"bold {WHITE}")
        at.add_column("Gradient φ", justify="right")
        at.add_column("Shapley φ", justify="right")
        grad_map = {a.field: a.contribution for a in (report.gradient.attributions if report.gradient else [])}
        shap_map = {a.field: a.contribution for a in (report.shapley.attributions if report.shapley else [])}
        all_fields = sorted(set(grad_map) | set(shap_map),
                            key=lambda k: -(abs(grad_map.get(k, 0)) + abs(shap_map.get(k, 0))))
        for f in all_fields:
            g = grad_map.get(f)
            s = shap_map.get(f)
            at.add_row(
                f,
                f"{g:+.4f}" if g is not None else "—",
                f"{s:+.4f}" if s is not None else "—",
            )
        console.print(Panel(at, title="[bold]Field attribution to ΔY[/bold]",
                            border_style=GREEN, box=box.ROUNDED))

    # Branch flips
    if report.branch_flips:
        bt = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {MUTED}")
        bt.add_column("Kind", style=AMBER)
        bt.add_column("Step", style=MUTED)
        bt.add_column("Condition", style=WHITE)
        bt.add_column("V2 → V3", style=AMBER, justify="center")
        for fl in report.branch_flips:
            arrow = f"{'T' if fl.v2_taken else 'F'} → {'T' if fl.v3_taken else 'F'}"
            bt.add_row(fl.kind, fl.data_step, fl.condition[:60], arrow)
        console.print(Panel(bt, title="[bold]Branch flips[/bold]",
                            border_style=AMBER, box=box.ROUNDED))

    # Optional Gemma justification
    if use_gemma:
        try:
            from src.knowledge import GraphRAG
            attrs = [a.to_dict() for a in (report.shapley.attributions if report.shapley else [])]
            kb_report = GraphRAG().justify_report(target=target, attributions=attrs, top_k=5)
            body = kb_report.overall + "\n\n" + "\n".join(
                f"[{GREEN if pf.justified else RED}]{'✓' if pf.justified else '✗'}[/]  "
                f"{pf.field}  conf {pf.confidence:.2f}\n    {pf.rationale}"
                for pf in kb_report.per_field
            )
            console.print(Panel(
                Text.from_markup(body),
                title=f"[bold]Gemma + GraphRAG[/bold]  [{MUTED}]{kb_report.backend}/{kb_report.model}[/{MUTED}]",
                border_style=INDIGO, box=box.ROUNDED,
            ))
        except Exception as e:
            console.print(f"[{AMBER}]GraphRAG unavailable: {e}[/{AMBER}]")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="RegLLM SAS Compiler Demo")
    p.add_argument("--sas",        default=str(SAMPLE_SAS), help="SAS file to compile")
    p.add_argument("--cycle",      action="append", metavar="ID", help="Cycle ID(s) to simulate (repeatable)")
    p.add_argument("--all-cycles", action="store_true", help="Simulate all cycles in cycles_sample.csv")
    p.add_argument("--no-ast",     action="store_true", help="Skip AST output")
    p.add_argument("--no-lineage", action="store_true", help="Skip lineage output")
    p.add_argument("--no-sim",     action="store_true", help="Skip simulation")
    p.add_argument("--diff",       metavar="ID",
                   help="V2/V3 diff-explain mode: pass a cycle ID (e.g. CIC_00100)")
    p.add_argument("--target",     default="ECL", help="Target field for --diff (default ECL)")
    p.add_argument("--ask-gemma",  action="store_true",
                   help="With --diff, also ask Gemma to justify each suspect field")
    args = p.parse_args()

    sas_path = Path(args.sas)
    if not sas_path.exists():
        console.print(f"[{RED}]SAS file not found: {sas_path}[/{RED}]")
        sys.exit(1)

    sas_code = sas_path.read_text(encoding="utf-8")
    tree = SASLogicTree()

    console.print()
    console.rule(f"[bold {BLUE}]RegLLM SAS Compiler[/bold {BLUE}]  ·  [{MUTED}]{sas_path.name}[/{MUTED}]")
    console.print()

    if args.diff:
        show_diff(sas_code, args.diff, target=args.target, use_gemma=args.ask_gemma)
        console.print()
        console.rule(f"[{MUTED}]Done[/{MUTED}]")
        console.print()
        return

    if not args.no_ast:
        show_ast(sas_code, tree)
        console.print()

    if not args.no_lineage:
        show_lineage(sas_code, tree)
        console.print()

    if not args.no_sim:
        cycles = _load_cycles()
        if not cycles:
            console.print(f"[{AMBER}]No cycles found at {CYCLES_CSV}[/{AMBER}]")
        else:
            if args.all_cycles:
                cycle_ids = [r.get("CICLO_ID", "") for r in cycles]
            elif args.cycle:
                cycle_ids = args.cycle
            else:
                # Default: first 3 cycles
                cycle_ids = [r.get("CICLO_ID", "") for r in cycles[:3]]

            show_simulation(sas_code, tree, cycles, cycle_ids)
            console.print()

    console.rule(f"[{MUTED}]Done[/{MUTED}]")
    console.print()


if __name__ == "__main__":
    main()
