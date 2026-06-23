# Audit System — v2 Ideas

Ideas explored during design but deferred from MVP. Each section documents the concept, why it was deferred, and what needs to be true before it's worth building.

---

## Cycle narration pipeline

**Idea:** Convert a tabular cycle row to natural language ("Ciclo CORP en FASE_CONTRACCION con 10 meses de dotación...") using the variable md files and EvalTrace from the SAS parser. The narration becomes the input for embedding and LLM explanation.

**Why deferred:** the DQC engine already catches violations without narration. Narration adds quality to the *explanation* of a finding, not to its *detection*. Build after the DQC layer is validated.

**Prerequisites:** variable mapping complete, EvalTrace working on production SAS code.

---

## Shared embedding space (articles ↔ cycles)

**Idea:** Embed narrated cycle descriptions and regulation article sections with the same sentence transformer (e.g. `paraphrase-multilingual-mpnet-base-v2`). Use nearest-article retrieval to identify which article a suspicious cycle is most likely violating.

**Why deferred:** requires the narration pipeline. Also: without fine-tuning, nearest-article = topically similar article, not violated article. Useful for retrieval-augmented explanation, not for detection.

**The fine-tuning path:** with labeled findings (cycle → article), build positive/negative pairs and fine-tune a bi-encoder with contrastive loss. Result: nearest article = violated article for similar cycles. Requires ~1000+ labeled pairs per article to be reliable.

**Prerequisites:** narration pipeline, audit findings labeled at article granularity.

---

## TabBERT / SCARF embeddings for anomaly detection

**Idea:** Train a masked-column-prediction model (TabBERT or SCARF) on 1M cycles to produce tabular embeddings. Use these to find structurally anomalous cycles that no DQC rule catches — the "unknown unknown" audit signal.

**Why deferred:** answers a different question than the audit finding generator. DQC catches known rule violations. Embeddings catch unknown anomalies. Both are valuable but independent; build DQC first.

**Data requirements:** 1M cycles confirmed available. If cycles have multiple temporal snapshots per `CICLO_ID`, a sequence model (time-series transformer) is more appropriate than vanilla TabBERT. The temporal structure question must be answered before architecture selection.

**Relationship to findings dataset:** audit findings provide weak supervision — cycles with known findings are contrastive negatives for "normal" cycles during pretraining.

---

## UMAP + DBSCAN visualisation

**Idea:** Project cycle embeddings to 2D/3D with UMAP (parametric, generalises to new points). Colour by `TERMINACION`, `CALIBRACION_SEGMENT_DEF_NEW_TOT`, or any user-selected field. Run DBSCAN on the embedding space to score anomalies.

**Why UMAP over t-SNE:** UMAP is parametric — new cycles can be embedded into the existing projection without recomputation. t-SNE is not.

**Colouring variable:** should be configurable at render time. Suggested default: `TERMINACION`. Candidate alternates: `STAGE_IFRS9`, `SEGMENTO`, `COLATERAL_TIPO`, `CALIBRACION_SEGMENT_DEF_NEW_TOT`.

**Prerequisites:** TabBERT embeddings.

---

## Code audit (SAS logic vs regulation)

**Idea:** Compare the SAS AST structure against regulation requirements. Example: verify that the CORP LGD floor in the SAS code is `>= 0.50` as required by Art. 15, not just that the data values satisfy the floor.

**Why deferred:** this is a different audit type — it flags implementation errors in the SAS code, not data violations in specific cycles. Valuable for release validation (V2 → V3 transitions) but out of scope for the data audit finding generator.

**Tools available:** `SASLogicTree.evaluate()`, `compare_sas_versions()`, `inspect_lineage()` — all already built.

---

## ChatRAG interface

**Idea:** Conversational interface over findings and regulation. "What are the most common Art. 12 violations this quarter?" / "Show me cycles similar to CIC_00031 that also have findings."

**Why deferred:** requires findings to be generated and stored first. A chat interface with no findings to query is not useful.

**Architecture when ready:** the existing `SASDiffAgent` loop + new `run_dqc` and `list_findings` tools is almost sufficient. The narration pipeline would improve the quality of per-cycle explanations in chat.

---

## Dynamic classifier for article attribution

**Idea:** Multi-label classifier trained on audit findings (cycle features → violated articles). Would predict which articles a new cycle is likely violating without running the full DQC battery.

**Why deferred:** requires a large, article-labeled findings dataset. Current findings are partially free-text; article attribution requires an NLP preprocessing step first. Also, this is useful for *triage* (which cycles to inspect), not for *evidence* (what exactly is wrong) — the DQC still needs to run to produce a defensible finding.

**Prerequisites:** findings dataset labeled at article granularity (either directly or via NLP extraction from free-text descriptions).
