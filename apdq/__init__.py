"""APDQ — Auditor-Parity Data Quality: reference implementation (MVP).

Implements the standard described in ``docs/AUDITOR_PARITY_STANDARD.md``:
a binding manifest describes a bank's reporting schema (columns bound to
canonical concepts, derivation formulas, reconciliation surfaces, control
totals); from that single artifact the package derives

* a clean-by-construction synthetic **twin** database (``twin.py``),
* **mutation generators + oracles** per normative defect class
  (``defects.py``),
* the **lineage-obligation check** (``lineage.py``),
* a hash-pinned atomic **requirement register** (``register.py``),
* the **certification harness** and **audit pack** (``harness.py``,
  ``audit_pack.py``).

Everything is deterministic and self-contained (SQLite, no LLM in the
trust chain). See ``apdq/README.md`` for formats and expansion points.
"""

__version__ = "0.1.0"
