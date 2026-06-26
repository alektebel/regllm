# Training Status — Phase 1 (Tool-Calling + Investigation RL)

## Current state (2026-06-25)

### SFT (completed iterations)

| Iteration | Adapter | Accuracy | Lineage | RAG | Both | Examples | Notes |
|-----------|---------|----------|---------|-----|------|----------|-------|
| 0 | phase2-sft | 78.4% | 70.0% | 97.4% | 0.0% | 74 | Initial SFT |
| 1 | tool-sft-iter1 | 85.8% | 78.2% | 97.9% | 0.0% | 254 | +failure replay |
| antihalluc | tool-sft-antihalluc | ? | ? | ? | ? | 281 | Running (checkpoint-45, 5 epochs) |

### Key fixes applied
- **Format mismatch fix**: Training data now includes `tools=_tool_defs` in `apply_chat_template()` so model sees the same tool schema JSON blocks that Ollama injects at inference.
- **GGUF export pipeline**: CPU bf16 merge → llama.cpp `convert_hf_to_gguf.py` → `ollama create`. Avoids the transformers 5.5.0 `reverse_op` bug.
- **Best GGUF so far**: `training/output/tool-sft-iter1/gguf/regllm-4b-f16.gguf` (8.0 GB)

### Files
- `training/sft_run.py` — SFT runner (used by iterate loop)
- `training/finetune.py` — Two-phase pipeline (phase1 pretrain, phase2 SFT, export)
- `training/train_antihalluc.py` — Anti-hallucination SFT (merges tool + investigation data)
- `training/iterate_tool_calling.py` — Automatic SFT iteration loop
- `training/tool_utils.py` — Eval helpers (load_adapter, predict_tools, eval_adapter)
- `training/rl_reward.py` — Verifiable reward function for RL
- `training/build_rl_dataset.py` — Builds RL prompts from toy LGD levels
- `training/train_rl.py` — GRPO training script (needs the procedural bug generator first)

---

## Next: RL for Investigation (GRPO)

### Goal
Teach the 4B model to **investigate** SAS pipeline bugs: backtrace through the computation graph, inspect intermediate values, identify the broken step.

### Why RL not more SFT
SFT teaches tool selection (which tool to call). RL teaches investigation strategy (what to look at, in what order, how to reason about what you find). The toy LGD curriculum has verifiable ground truth — no LLM judge needed.

### Architecture: AST-Mutated Procedural Bugs

The key insight: **we don't need hand-crafted bugs. We can generate unlimited verifiable problems automatically.**

#### How it works

1. **Baseline run**: Take the toy LGD pipeline (or any SAS session). Run a cycle through `SASLogicTree.evaluate()` and record the correct final values (e.g., `ECL=472.5`, `LGD_ESTIMADA=0.30`).

2. **AST mutation**: Parse the SAS into AST nodes. Randomly mutate one node:
   - Swap two variables in an expression (`EAD` ↔ `LGD_ESTIMADA`)
   - Change an operator (`>=` → `>`, `<` → `<=`, `+` → `-`)
   - Change a numeric literal (`0.30` → `0.45`, `9` → `12`)
   - Remove a COALESCE/default check
   - Change a GROUP BY / MERGE BY column
   - Change a JOIN key
   - Change an aggregation function (`SUM` → `MAX`)
   - Remove an IF condition

3. **Verify the mutation matters**: Re-run the same cycle through the mutated AST. If the final values change, this is a valid bug. If they don't change (mutation is dead code), discard and try again.

4. **Ground truth**: The mutation location (step name, line, original vs mutated) is the solution. The agent must identify:
   - **Where**: which DATA step / PROC SQL the bug is in
   - **What**: which variable/operator/value is wrong
   - **Why**: what value changed and how it propagates

5. **Reward**: The agent outputs a structured diagnosis. We check:
   - Did it identify the correct step? (match against mutation metadata)
   - Did it identify the correct variable/expression? 
   - Did it use trace_dependencies / inspect_lineage?
   - Does its proposed fix restore the original correct values? (run the fix through the evaluator)

   The strongest signal: **run the agent's proposed fix through the SAS evaluator and compare to the correct baseline**. If values match → reward=1.0. This is functionally verifiable, ungameable.

#### Why this is powerful

- **Unlimited training data**: Each SAS file × each row × each AST node = thousands of unique problems
- **Auto-verifiable**: The evaluator is the judge. No keywords, no LLM judge.
- **Progressive difficulty**: Start with single mutations (easy), progress to double mutations (compound bugs), then mutations in upstream steps that propagate through multiple tables.
- **Domain-realistic**: The mutations mirror real bugs found in production SAS code (wrong variables, boundary errors, aggregation mistakes).

### Implementation — VibeThinker-style Curriculum (DONE)

#### Architecture: Two Phases

**Phase A — Generalized debugging** (Stages 1-3): Teach the model to find bugs across multiple SAS pipelines. Uses AST mutation on all project SAS files, not just toy_lgd. Reward = functional verification (evaluator is the judge).

**Phase B — Experience-accelerated debugging** (Stages 4-5): Once the model can find bugs reliably, teach it to find them *faster* by leveraging the experience KB (55 past bug/insight records in `data/experience/`). Reward = same correctness + fewer tool calls + using `search_experience` first.

#### Files

| File | Purpose |
|------|---------|
| `training/bug_generator.py` | AST mutation engine — 5 operators (swap_vars, wrong_op, wrong_literal, remove_guard, wrong_agg) |
| `training/sas_corpus.py` | Multi-pipeline corpus — loads 4 SAS pipelines (128 mutation targets total) |
| `training/curriculum.py` | 5-stage curriculum (3 Phase A + 2 Phase B) with progression tracking |
| `training/rl_env.py` | RL env with 5 simulated tools + experience KB + efficiency/experience rewards |
| `training/rl_reward.py` | Functional verification reward + legacy keyword reward |
| `training/train_rl.py` | Curricular GRPO + self-distillation + legacy mode |

#### Curriculum stages

| # | Stage | Phase | Depth | Key Reward Signal | Advance |
|---|-------|-------|-------|-------------------|---------|
| 1 | tool_selection | A | 0 | 70% tool_call + 30% correct_tool | >0.6 / 20 eps |
| 2 | single_depth | A | 0 | step + var + fix identification | >0.5 / 30 eps |
| 3 | multi_depth | A | ≥1 | propagation trace + functional fix | >0.5 / 40 eps |
| 4 | experience_accelerated | B | any | 30% experience_used + diagnosis | >0.5 / 30 eps |
| 5 | experience_efficiency | B | any | 20% efficiency + 20% experience + fix | >0.5 / 40 eps |

#### Multi-pipeline corpus

| Pipeline | Source | Targets | Steps |
|----------|--------|---------|-------|
| toy_lgd | `toy_lgd_correct.sas` | 19 | 4 DATA |
| sample_lgd_v2 | `data/samples/sample_lgd.sas` | 31 | 6 DATA + PROC SQL |
| sample_lgd_v3 | `data/sas/v3/sample_lgd.sas` | 36 | 7 DATA + PROC SQL |
| debug_lgd_suelos | `sessions/debug_lgd/proj_03*.sas` | 42 | 2 DATA + no_conformes |
| **Total** | | **128** | |

#### Simulated tools (for training without external systems)

| Tool | Simulation |
|------|-----------|
| `trace_dependencies` | `SASLogicTree.trace_lineage()` on mutated code |
| `inspect_lineage` | Lineage edges + data steps from AST |
| `search_docs` | Field definitions (LGD, EAD, ECL, etc.) |
| `search_regulation` | CRR/IFRS 9/BdE articles |
| `search_experience` | Full-text search over 55 experience markdown files |

#### How to run

```bash
# 1. Verify reward function on all 5 stages (no GPU)
.venv/bin/python training/train_rl.py --eval-only

# 2. Self-distillation: collect good traces across all stages
.venv/bin/python training/train_rl.py --distill-only --distill-n 200

# 3. SFT on distilled traces
# .venv/bin/python training/sft_run.py --data training/output/distill-traces/distill_traces.jsonl

# 4. Full curricular GRPO (Phase A → Phase B)
.venv/bin/python training/train_rl.py

# 5. Start from Phase B directly (if Phase A is done)
.venv/bin/python training/train_rl.py --start-stage 3

# 6. Legacy mode (9 fixed toy_lgd levels)
.venv/bin/python training/train_rl.py --legacy
```

#### Eval results (reward function)

| Stage | Good | Bad | Separation |
|-------|------|-----|------------|
| tool_selection | +1.00 | -1.00 | 2.00 |
| single_depth | +1.00 | -0.88 | 1.88 |
| multi_depth | +1.00 | -0.88 | 1.88 |
| experience_accelerated | +0.40* | -0.88 | 1.28 |
| experience_efficiency | +0.60* | -0.88 | 1.48 |

\* Without `search_experience` call. With experience → +1.00.

### Dependencies
- `src/sas_logic_tree.py` — SAS parser + evaluator (exists, tested)
- `src/sas_diff/tensor_evaluator.py` — Tensor-based evaluator (exists)
- `data/sas/toy_lgd/` — 9 levels of hand-crafted bugs (for validation)
- `data/sas/` — Multiple SAS pipelines (for multi-pipeline training)
- `data/experience/` — 55 experience/insight markdown files (for Phase B)
- `src/knowledge/experience_store.py` — Experience KB (KuzuDB, for production)

### Open questions
- What LoRA rank for RL? VibeThinker uses full fine-tuning. QLoRA r=32 might constrain the policy too much.
- When to export to GGUF and test end-to-end via Ollama?
- Multi-turn tool execution during GRPO: current approach is single-turn. For multi-turn, use rejection sampling + SFT (self-distillation) rather than online RL.
- Experience KB growth: how to automatically harvest new confirmed investigations back into `data/experience/` for the next training cycle?
