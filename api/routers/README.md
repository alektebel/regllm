# DQC backend — module layout

The DQC feature is split by responsibility so each layer can be read, changed
and unit-tested on its own. Dependencies point one way (top depends on bottom);
nothing below imports the routes.

```
dqc.py            HTTP routes + generation orchestration (the ReAct loop driver)
  ├── dqc_react.py       the agents: sufficiency, SAS generation, judge,
  │                      static validation, cases→SQLite execution
  ├── dqc_dictionary.py  dictionary workbook: sheet scoring + column mapping
  ├── dqc_store.py       persistence (the only module that opens SQLite)
  ├── dqc_uploads.py     upload validation + user-facing workbook errors
  └── dqc_models.py      Pydantic wire contract (pure data)
```

| Module | Owns | Never does |
|---|---|---|
| `dqc_models.py` | Request/response schemas | I/O of any kind |
| `dqc_uploads.py` | "Is this a usable upload?", error messages | DB, LLM |
| `dqc_store.py` | `checks` + `check_eval_cases` tables | HTTP, LLM |
| `dqc_dictionary.py` | Parsing the dictionary workbook | Persistence |
| `dqc_react.py` | LLM agents + validating checks against the cases | HTTP, persistence |
| `dqc.py` | Routes, orchestration, SSE streaming | Direct SQL |

## Why the split

`dqc.py` had grown to ~1,170 lines mixing schemas, SQL, upload validation and
route handlers, so every change touched the same file and little of it could be
tested without spinning up an HTTP client. The layers above are behaviour-
preserving extractions — `dqc.py` keeps thin aliases (`_db`,
`_persist_dqc_items`, …) so existing call sites and tests read unchanged.

## Two invariants worth knowing

1. **The generated check is SAS PROC SQL, but it is validated in SQLite.**
   `dqc_react._sas_to_sqlite()` rewrites SAS-only idioms (`^=`, `= .`,
   `MISSING()`, `UPCASE`, …) for the validation run only — what is stored and
   shown to the user stays SAS.
2. **Column names from the cases Excel are sanitised** before the SQLite table
   is built (`dqc_react._dedupe_headers()`): SQLite is case-insensitive and
   rejects blank/duplicate names, which an otherwise-fine workbook can contain
   (e.g. an empty trailing column).

## Tests

| File | Covers |
|---|---|
| `tests/test_dqc_uploads.py` | upload validation, workbook error hints |
| `tests/test_dqc_store.py` | persistence, failed items, schema migration |
| `tests/test_dqc_cases_sqlite.py` | header sanitation, SAS→SQLite, query execution |
| `tests/test_dqc_dictionary.py` | sheet/column-mapping resolution |
| `tests/test_dqc_generate_stream.py` | the SSE generation endpoint |
| `tests/test_dqc_react_unit.py` | the agent functions |

Run them with `python -m pytest tests/ -q`.
