"""Audit pack: the versioned evidence bundle a certification run emits.

JSON for machines (re-verification, diffing between runs), HTML for the
humans in the room — supervisor, external auditor, internal validation.
Everything on the pack's face is reproducible from the pinned inputs:
manifest + register + seed.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .defects import CLASS_SLUGS
from .harness import Certification


def to_dict(cert: Certification) -> dict:
    return {
        "apdq_audit_pack": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": cert.manifest_name,
        "seed": cert.seed,
        "trap_k": cert.k,
        "certified": cert.certified,
        "gates": {
            "lineage_ok": cert.lineage_ok,
            "register_ok": cert.register_ok,
            "specificity_ok": cert.specificity_ok,
            "recall_ok": cert.recall_ok,
        },
        "obligations": {
            "summary": cert.obligation_summary,
            "missing": [
                {"node": o.node, "kind": o.kind, "detail": o.detail}
                for o in cert.obligations if o.status == "missing"],
            "waived": [
                {"node": o.node, "kind": o.kind, "reason": o.detail}
                for o in cert.obligations if o.status == "waived"],
        },
        "register": {
            "summary": cert.register_summary,
            "unsigned": cert.unsigned_requirements,
            "unbound": cert.unbound_requirements,
        },
        "per_class": {
            str(cls): stats for cls, stats in sorted(cert.per_class().items())},
        "defects": [
            {
                "id": r.defect_id, "class": r.dq_class,
                "class_slug": CLASS_SLUGS[r.dq_class],
                "table": r.table, "description": r.description,
                "clean_rows": r.clean_rows,
                "planted": r.planted, "caught": r.caught,
                "aggregate": r.aggregate, "recall": r.recall,
                "error": r.error, "oracle_sql": r.oracle_sql,
            }
            for r in cert.defect_results],
        "overlap": cert.overlap,
        "overbroad": cert.overbroad,
        "declared_limits": [
            "External-evidence defects (values consistently wrong at origin) "
            "are out of scope — no auditor detects them from the data either.",
            "Class 11 (distributional) findings are advisory and never count "
            "toward the certified set.",
            "Class 10 (panel) and class 12 (semantic drift) are covered by "
            "schema-specific tooling in this MVP; see apdq/README.md.",
            "Waived obligations are enumerated on the face of this pack.",
        ],
    }


def write_json(cert: Certification, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(cert), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _badge(ok: bool, label: str) -> str:
    color = "#1a7f37" if ok else "#b42318"
    text = "PASS" if ok else "FAIL"
    return (f'<span style="display:inline-block;margin:0 8px 8px 0;'
            f'padding:4px 10px;border-radius:6px;color:#fff;'
            f'background:{color};font-weight:600">{label}: {text}</span>')


def write_html(cert: Certification, path: Path) -> Path:
    data = to_dict(cert)
    esc = html.escape
    rows = []
    for d in data["defects"]:
        status = "OK" if (d["clean_rows"] == 0 and d["recall"] == 1.0
                          and not d["error"]) else "FAIL"
        caught_cell = "fired" if d["aggregate"] else \
            f"{d['caught']}/{d['planted']}"
        rows.append(
            f"<tr><td>{esc(d['id'])}</td><td>{d['class']} "
            f"{esc(d['class_slug'])}</td><td>{esc(d['description'])}</td>"
            f"<td>{d['clean_rows']}</td><td>{caught_cell}</td>"
            f"<td>{d['recall']:.2f}</td><td>{status}</td></tr>")
    class_rows = [
        f"<tr><td>{cls}</td><td>{esc(stats['slug'])}</td>"
        f"<td>{stats['defects']}</td><td>{stats['recall_min']:.2f}</td></tr>"
        for cls, stats in data["per_class"].items()]
    waived = [
        f"<li><code>{esc(w['node'])}</code> [{esc(w['kind'])}]: "
        f"{esc(w['reason'])}</li>"
        for w in data["obligations"]["waived"]]
    limits = "".join(f"<li>{esc(t)}</li>" for t in data["declared_limits"])
    verdict = ("CERTIFIED" if data["certified"] else "NOT CERTIFIED")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>APDQ audit pack — {esc(data['manifest'])}</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto;
        max-width: 70rem; padding: 0 1rem; color: #1f2328; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left;
          font-size: 13px; }}
 th {{ background: #f6f8fa; }}
 h1 span {{ font-size: 0.6em; padding: 4px 12px; border-radius: 8px;
           color: #fff; vertical-align: middle;
           background: {'#1a7f37' if data['certified'] else '#b42318'}; }}
 code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }}
</style></head><body>
<h1>APDQ audit pack — {esc(data['manifest'])} <span>{verdict}</span></h1>
<p>Generated {esc(data['generated_utc'])} · seed {data['seed']} ·
k={data['trap_k']} mutations per defect · reproducible from the pinned
manifest + register + seed.</p>
<div>
{_badge(data['gates']['lineage_ok'], 'Lineage obligations')}
{_badge(data['gates']['register_ok'], 'Requirement register')}
{_badge(data['gates']['specificity_ok'], 'Specificity (clean twin)')}
{_badge(data['gates']['recall_ok'], 'Recall (planted mutations)')}
</div>
<h2>Coverage by defect class</h2>
<table><tr><th>Class</th><th>Name</th><th>Defects</th><th>Min recall</th></tr>
{''.join(class_rows)}</table>
<h2>Defects ({len(data['defects'])})</h2>
<table><tr><th>Defect</th><th>Class</th><th>Description</th>
<th>Clean rows</th><th>Caught</th><th>Recall</th><th>Status</th></tr>
{''.join(rows)}</table>
<h2>Waived obligations ({len(waived)})</h2>
<ul>{''.join(waived) or '<li>none</li>'}</ul>
<h2>Declared limits</h2>
<ul>{limits}</ul>
</body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
