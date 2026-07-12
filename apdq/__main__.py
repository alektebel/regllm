"""APDQ command-line interface.

    python -m apdq validate  MANIFEST.yaml
    python -m apdq lineage   MANIFEST.yaml
    python -m apdq defects   MANIFEST.yaml
    python -m apdq twin      MANIFEST.yaml -o twin.db [--seed N]
    python -m apdq certify   MANIFEST.yaml [--register REGISTER.yaml]
                             [--out DIR] [--seed N] [--k N] [--json]

``certify`` exits non-zero unless every gate passes — wire it straight
into CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audit_pack, lineage
from .defects import generate_defects
from .harness import TRAP_K, certify
from .manifest import ManifestError, load_manifest
from .register import RegisterError, load_register
from .twin import build_twin


def _load(path: str):
    try:
        return load_manifest(path)
    except ManifestError as exc:
        sys.exit(f"manifest error: {exc}")


def cmd_validate(args) -> int:
    manifest = _load(args.manifest)
    n_cols = sum(len(t.columns) for t in manifest.tables)
    print(f"OK: {manifest.name} — {len(manifest.tables)} tables, "
          f"{n_cols} columns, {len(manifest.bound_concepts())} concepts bound")
    return 0


def cmd_lineage(args) -> int:
    manifest = _load(args.manifest)
    obligations = lineage.check_obligations(manifest)
    for o in obligations:
        mark = {"ok": "✓", "waived": "w", "missing": "✗"}[o.status]
        print(f" {mark} {o.node:<40} {o.kind:<10} {o.detail}")
    stats = lineage.summary(obligations)
    print(f"\n{stats['ok']} ok, {stats['waived']} waived, "
          f"{stats['missing']} missing of {stats['total']}")
    return 1 if stats["missing"] else 0


def cmd_defects(args) -> int:
    manifest = _load(args.manifest)
    defects = generate_defects(manifest)
    if getattr(args, "sql", None):
        def comment(text: str) -> str:
            # keep ';' out of comments so naive split-on-';' runners work
            return "-- " + text.replace(";", ",")

        lines = [
            comment("APDQ generated data-quality check suite"),
            comment(f"manifest: {manifest.name} · {len(defects)} checks"),
            comment("Each query returns VIOLATING rows (empty result = pass)."),
            comment("SQLite dialect. Aggregate checks return metric names."),
            ""]
        for d in defects:
            refs = f" · {', '.join(d.regulation_refs)}" if d.regulation_refs else ""
            lines.append(comment(f"{d.defect_id} [{d.class_slug}]{refs}"))
            lines.append(comment(d.description))
            lines.append(d.oracle_sql.strip() + ";")
            lines.append("")
        Path(args.sql).write_text("\n".join(lines), encoding="utf-8")
        print(f"{len(defects)} checks written to {args.sql}")
        return 0
    for d in defects:
        print(f" {d.defect_id:<45} [{d.class_slug}] {d.description}")
    print(f"\n{len(defects)} defects across "
          f"{len({d.dq_class for d in defects})} classes")
    return 0


def cmd_twin(args) -> int:
    import sqlite3
    manifest = _load(args.manifest)
    out = Path(args.out)
    out.unlink(missing_ok=True)
    disk = sqlite3.connect(out)
    build_twin(manifest, seed=args.seed, conn=disk)
    disk.close()
    print(f"clean twin written to {out}")
    return 0


def cmd_propose(args) -> int:
    from .assist import propose_manifest
    draft = propose_manifest(args.database, args.name)
    Path(args.out).write_text(draft, encoding="utf-8")
    print(f"draft manifest written to {args.out} — every TODO is a human "
          f"review decision (optionally LLM-prefilled)")
    return 0


def cmd_certify(args) -> int:
    manifest = _load(args.manifest)
    register = None
    if args.register:
        try:
            register = load_register(args.register, manifest.concepts)
        except RegisterError as exc:
            sys.exit(f"register error: {exc}")

    cert = certify(manifest, register, seed=args.seed, k=args.k)

    out_dir = Path(args.out) if args.out else Path("apdq_audit")
    json_path = audit_pack.write_json(
        cert, out_dir / f"{manifest.name}_audit_pack.json")
    html_path = audit_pack.write_html(
        cert, out_dir / f"{manifest.name}_audit_pack.html")

    if args.json:
        print(json.dumps(audit_pack.to_dict(cert), indent=2,
                         ensure_ascii=False))
    else:
        per_class = cert.per_class()
        print(f"APDQ certification — {manifest.name}")
        print(f"  lineage obligations : "
              f"{'PASS' if cert.lineage_ok else 'FAIL'} "
              f"({cert.obligation_summary})")
        if register is not None:
            print(f"  requirement register: "
                  f"{'PASS' if cert.register_ok else 'FAIL'} "
                  f"({cert.register_summary}, "
                  f"unbound={len(cert.unbound_requirements)})")
        print(f"  specificity (clean) : "
              f"{'PASS' if cert.specificity_ok else 'FAIL'}")
        print(f"  recall (mutations)  : "
              f"{'PASS' if cert.recall_ok else 'FAIL'}")
        for cls, stats in sorted(per_class.items()):
            print(f"    class {cls:>2} {stats['slug']:<26} "
                  f"{stats['defects']:>3} defects  "
                  f"min recall {stats['recall_min']:.2f}")
        for res in cert.defect_results:
            if not res.clean_ok or res.recall < 1.0:
                why = res.error or (
                    f"clean_rows={res.clean_rows}, "
                    f"caught={res.caught}/{res.planted}")
                print(f"    FAIL {res.defect_id}: {why}")
        if cert.overbroad:
            print(f"  overbroad oracles   : {cert.overbroad}")
        print(f"  audit pack          : {json_path}  {html_path}")
        print(f"  VERDICT             : "
              f"{'CERTIFIED' if cert.certified else 'NOT CERTIFIED'}")
    return 0 if cert.certified else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apdq")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in (("validate", cmd_validate), ("lineage", cmd_lineage)):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        p.set_defaults(fn=fn)

    p = sub.add_parser("defects")
    p.add_argument("manifest")
    p.add_argument("--sql", metavar="FILE",
                   help="write the generated checks as a runnable .sql file")
    p.set_defaults(fn=cmd_defects)

    p = sub.add_parser("twin")
    p.add_argument("manifest")
    p.add_argument("-o", "--out", default="apdq_twin.db")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_twin)

    p = sub.add_parser("propose")
    p.add_argument("database")
    p.add_argument("--name", default="draft")
    p.add_argument("-o", "--out", default="draft_manifest.yaml")
    p.set_defaults(fn=cmd_propose)

    p = sub.add_parser("certify")
    p.add_argument("manifest")
    p.add_argument("--register")
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=TRAP_K)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_certify)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
