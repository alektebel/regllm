# Branch Map

Audit of every branch on `origin` (updated 2026-07-12 after the merge),
grouped by feature, with a recommended action per branch. Actions beyond
creating branches and merging to `main` (which the owner requested) are
documented rather than executed.

## MERGED: APDQ / data-quality standard → `main`

The whole data-quality line — research memos, the APDQ standard spec,
handbook, setup runbook, and the full `apdq/` implementation with tests
— was fast-forwarded into **`main`** (commit `dfc197e`, 68 tests green).

Now-redundant branches whose content is fully contained in `main`
(**delete via the GitHub UI** — this environment's git proxy does not
support remote branch deletion):

- `claude/data-quality-research-15s3by` (the old integration branch)
- `claude/feature/dq-research-docs`, `claude/feature/apdq-mvp`
  (pre-merge review splits)
- `feature/sas-embeddings-explorer` (a briefly-created duplicate name
  for the embedding-visualizer line; superseded below)

## Feature: SAS field-diff explainer

| Branch | State | Content |
|---|---|---|
| `feature/sas-diff-explainer` | working branch, currently equal to `main` | The home for future SAS compiler / field-diff work. The compiler itself (`src/sas_logic_tree.py` etc.) is in `main`; the unmerged `.sas/.egp` upload commit (`955dec8`) lives on the embedding-visualizer line and conflicts with `main`'s newer agent router — rebase it here when picking that work up. |

## Feature: embedding-space visualizer

| Branch | State | Content |
|---|---|---|
| `feature/embedding-visualizer` | preserved, unmerged (5 commits, base 27 behind `main`) | Spreadsheet-driven embedding visualizer with evolving-register support, `.sas/.egp` upload to the diff page, no-Docker Windows launcher, two robustness fixes. Same line as `claude/embedding-space-visualizer-j99u00` (which can be deleted as a duplicate name). Rebase onto `main` before continuing. |

## Feature: DQC product slimming & persistence

| Branch | Ahead/behind | Last commit | Content & recommendation |
|---|---|---|---|
| `feat/dqc-slim-setup` | 10 / 11 | 2026-07-07 | Pluggable checks store (local SQLite or DynamoDB), CDK DynamoDB table, DQC-focused README + Windows/AWS setup docs, and a "slim repo to the DQC product" commit. **Unmerged feature work** — but the slimming commit deletes non-DQC code, so merging it into `main` would remove the SAS differ and APDQ. Recommendation: cherry-pick the persistence + docs commits (`9cb5d83`, `874fe23`, `8556534`, `36e2420`) into `main`; keep the slimming commit only if a separate slim distribution repo is actually wanted. |
| `dqc-slim` | 0 / 5 | 2026-07-07 | Fully merged into `main`. **Delete.** |
| `feat/dqc-copy-all-button` | 5 / 11 | 2026-07-06 | Mixed bag: "copy all" UI button, Gemini backend + SAS prompts + batch tests, self-contained CDK deploy, two frontend fixes. **Unmerged.** Recommendation: rebase onto `main` and split into UI (`e18f0bd`, `ad8a8da`, `dc7a73f`) vs backend/deploy (`594bc88`, `14a3087`) PRs — they are independent concerns. |

## Merged / stale (safe to delete)

| Branch | State |
|---|---|
| `claude/project-overview-aws-eval-au64l7` | 0 ahead — fully merged (project overview + AWS eval work now in `main`). |
| `dqc-gguf-regulation-rag` | 0 ahead — merged (GGUF local-LLM + regulation RAG line, now in `main`). |
| `dqc-slim` | 0 ahead — merged (see above). |

## Legacy divergent lines (decide, then archive)

| Branch | Ahead/behind | Last commit | Content & recommendation |
|---|---|---|---|
| `master` | 33 / 52 | 2026-05-18 | Pre-`main` history: HF Inference Endpoint switch, JWT/frontend fixes, test repairs. Diverged before the DQC pivot. Recommendation: confirm nothing on it is still wanted (the HF inference work was superseded by the Bedrock/GGUF backends), then delete or tag as `archive/master-hf-endpoint`. Keeping both `master` and `main` invites wrong-branch pushes. |
| `update-model` | 22 / 52 | 2026-03-08 | Oldest line: Gradio 6 core engine, Modal deployment, postgres/vector storage, adversarial alignment tests. Superseded by the current FastAPI/Next.js architecture. Recommendation: tag as `archive/update-model-gradio` and delete the branch; the Modal script may be worth extracting if Modal deployment is ever revisited. |

## Suggested end state

- `main` — the only long-lived branch.
- Short-lived feature branches per concern (as split above), merged via
  PR and deleted after merge.
- Legacy lines tagged `archive/*` and removed from the branch list.
