# Variable Mapping Guide

How to populate, review, and maintain `data/audit/mapping.json`.

---

## What the mapping is

`mapping.json` is the bridge between regulation language and the database.
It answers: *"When the regulation says 'período de dotación', which column in
`mylib.ciclos_recuperacion` holds that value, and how is it computed in the SAS code?"*

Every DQC query in `data/audit/dqc/` must reference only column names that
exist in `mapping.json` with `confidence >= 0.70`. Lower-confidence entries are
flagged `"needs_review": true` and must be resolved before being used in a DQC query.

---

## Generating the mapping automatically

```bash
# Full auto-build (requires Ollama running with qwen3:32b)
python -m src.audit.variable_mapper --sas-version v3

# Print a summary of what is already in the file
python -m src.audit.variable_mapper --report
```

The auto-build:
1. Parses all SAS code in `data/sas/v3/` via `SASLogicTree`.
2. Extracts field mentions from `data/regulation/*.md`.
3. Calls the LLM once per column to propose descriptions.
4. Writes the result to `data/audit/mapping.json`.

**Important:** the auto-build is a *first draft*. Always review
entries with `confidence < 0.85` before approving DQC queries that reference them.

---

## Entry schema

```json
{
  "db_column": "PROVISION_PERIOD_MONTHS",
  "regulation_concept": "período de dotación de provisiones en meses",
  "regulation_variable": "PROVISION_PERIOD_MONTHS",
  "articles": ["art_12_periodos_dotacion", "art_23_liberacion_provisiones"],
  "sas_variable": "PROVISION_PERIOD_MONTHS",
  "sas_lineage": ["FECHA_INCUMPLIMIENTO"],
  "computation_description": "...",
  "regulation_description": "...",
  "confidence": 0.97,
  "needs_review": false,
  "notes": ""
}
```

| Field | Required | Description |
|---|---|---|
| `db_column` | Yes | Exact column name in `mylib.ciclos_recuperacion` (uppercase) |
| `regulation_concept` | Yes | Natural language name in Spanish as the regulation uses it |
| `regulation_variable` | Yes | Formal variable name in the regulation (often = `db_column`) |
| `articles` | Yes | Article file stems that reference this variable (can be empty) |
| `sas_variable` | Yes | Variable name in SAS source code (may differ from `db_column`) |
| `sas_lineage` | Yes | Direct ancestor fields from `trace_field_ancestors()` (can be empty) |
| `computation_description` | Yes | 1–2 sentences: how SAS computes this field |
| `regulation_description` | Yes | 1–2 sentences: what the regulation means by this concept |
| `confidence` | Yes | Float 0–1. Auto-set to < 0.70 triggers `needs_review: true` |
| `needs_review` | Auto | Set `true` if confidence < 0.70 or if a human has a question |
| `notes` | No | Free text for reviewers |

---

## Confidence levels

| Range | Meaning | DQC status |
|---|---|---|
| 0.85 – 1.00 | High confidence. Column name unambiguous, regulation reference clear. | ✅ Approved |
| 0.70 – 0.84 | Medium. Column name matches but description may be incomplete. | ⚠️ Usable, review recommended |
| 0.00 – 0.69 | Low. Mapping is uncertain — SAS name differs, or no regulation mention found. | ❌ Blocked until reviewed |

To mark a low-confidence entry as reviewed and approved, set `"needs_review": false`
and add a note explaining why the mapping is correct despite low LLM confidence.

---

## Adding a new column manually

1. Add a new entry to the `"mappings"` dict in `mapping.json`.
2. Follow the schema above.
3. Set `"confidence"` honestly:
   - Is the column name unambiguous and in the regulation text? → 0.90+
   - Is the column found by lineage but not in regulation text? → 0.60–0.80
   - Is this an internal/derived field with no regulatory counterpart? → 0.40–0.60
4. If the column has no regulatory meaning (identifiers, audit keys), set
   `"articles": []` and `"confidence": 0.95` with a note: "Identifier only."

---

## Adding a new regulation article

When a new regulation article is added to `data/regulation/`:

1. Run `python -m src.audit.variable_mapper --sas-version v3` to regenerate.
2. Review entries whose `"articles"` list changed — their descriptions may need updating.
3. Check that existing DQC queries in `data/audit/dqc/` are still valid against the new article.

---

## Columns requiring special attention

### Version-dependent columns

Some columns have different semantics or thresholds in V2 vs V3:

| Column | V2 | V3 | Impact |
|---|---|---|---|
| `PD_ESTIMADA` | Floor 0.0003 (0.03%) | Floor 0.0005 (0.05%) | Art. 15 DQC threshold |
| `LGD_FLOOR_APLICADO` | CORP floor 0.45 | CORP floor 0.50 | Art. 15 DQC threshold |
| `MoC` | Not adjusted | CURE_FLAG adjustment | ECL_AJUSTADO differs |

DQC queries **must be version-parameterised** or have separate V2/V3 files.

### Mixed-case column names

`MoC` appears as mixed case in the CSV but as `MOC` in SAS code. DQC queries
should use `UPPER(column_name)` or the confirmed case from the database DDL.

### Identifier-only columns

`CICLO_ID`, `CONTRATO`, and `PERIODO_OBSERVACION` have no regulatory threshold.
They appear in the mapping for completeness and audit traceability, but no DQC
rules apply to them directly.

### Low-confidence entries (require review)

As of 2026-06-15, the following entries need human review before any DQC query
references them:

| Column | Reason |
|---|---|
| `PERIODO_OBSERVACION` | Not found in regulation article text. Confirm regulatory mapping. |

---

## Relationship to other files

```
data/audit/mapping.json          ← this file (the bridge)
    ↑ built from:
        data/regulation/*.md      regulation articles
        data/sas/v3/*.sas         SAS source code
        data/docs/regulation/*.md DB context documentation

    ↓ used by:
        data/audit/dqc/*.sql      DQC queries (reference db_column names)
        src/audit/variable_mapper.py  build_mapping() / load_mapping()
        src/audit/finding_generator.py  look up regulation_description
        src/agent/tools.py        search_regulation tool
```

---

## Validation checklist (before approving a DQC query)

For each column referenced in a DQC query:

- [ ] `db_column` matches exact column name in `mylib.ciclos_recuperacion` DDL
- [ ] `confidence >= 0.70`
- [ ] `needs_review: false`
- [ ] `articles` list includes the article the DQC query is checking
- [ ] `regulation_description` correctly describes the rule being checked
- [ ] Version sensitivity noted in `notes` if applicable
