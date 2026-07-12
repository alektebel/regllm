# Branch Map

Audit of every branch on `origin` (snapshot 2026-07-12), grouped by
feature, with ahead/behind counts against `origin/main` and a
recommended action. Recommendations are documented here rather than
executed — deleting or rewriting remote branches is the repo owner's
call.

## Feature: APDQ / data-quality standard (this line of work)

| Branch | State | Content |
|---|---|---|
| `claude/data-quality-research-15s3by` | **integration branch** for this feature line | Research memos (industry SOTA, Spanish-bank niche), the APDQ standard spec + MVP roadmap, and the full `apdq/` implementation with tests. |
| `claude/feature/dq-research-docs` | split from the above | Docs only: the four research/spec commits, for reviewing the strategy documents independently of code. |
| `claude/feature/apdq-mvp` | split from the above | Implementation only: the `apdq/` package + tests + gap-fill work, cherry-picked onto `main`, for a code-only PR. |

Recommended merge order: `dq-research-docs` first (docs, zero risk),
then `apdq-mvp` (self-contained new package; only shared file touched is
`README.md`). The integration branch can then be deleted or kept as the
running workspace.

## Feature: DQC product slimming & persistence

| Branch | Ahead/behind | Last commit | Content & recommendation |
|---|---|---|---|
| `feat/dqc-slim-setup` | 10 / 11 | 2026-07-07 | Pluggable checks store (local SQLite or DynamoDB), CDK DynamoDB table, DQC-focused README + Windows/AWS setup docs, and a "slim repo to the DQC product" commit. **Unmerged feature work** — but the slimming commit deletes non-DQC code, so merging it into `main` would remove the SAS differ and APDQ. Recommendation: cherry-pick the persistence + docs commits (`9cb5d83`, `874fe23`, `8556534`, `36e2420`) into `main`; keep the slimming commit only if a separate slim distribution repo is actually wanted. |
| `dqc-slim` | 0 / 5 | 2026-07-07 | Fully merged into `main`. **Delete.** |
| `feat/dqc-copy-all-button` | 5 / 11 | 2026-07-06 | Mixed bag: "copy all" UI button, Gemini backend + SAS prompts + batch tests, self-contained CDK deploy, two frontend fixes. **Unmerged.** Recommendation: rebase onto `main` and split into UI (`e18f0bd`, `ad8a8da`, `dc7a73f`) vs backend/deploy (`594bc88`, `14a3087`) PRs — they are independent concerns. |

## Feature: embedding-space visualizer (SAS differ side)

| Branch | Ahead/behind | Last commit | Content & recommendation |
|---|---|---|---|
| `claude/embedding-space-visualizer-j99u00` | 5 / 27 | 2026-06-30 | Spreadsheet-driven embedding visualizer with evolving-register support, .sas/.egp upload, no-Docker Windows launcher, two robustness fixes. **Unmerged, coherent feature.** Recommendation: rebase onto `main` (27 behind) and open a single PR; the five commits form one feature. |

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
