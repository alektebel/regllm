# Resume RL Training — Stage 3 (experience_intro)

## Quick Resume

```bash
cd /home/diego/dev/regllm

PYTHONPATH=. .venv/bin/python training/train_rl.py \
  --base-adapter training/output/grpo-curriculum/stage_3_experience_intro \
  --start-stage 3 \
  --batch-size 2 \
  --num-generations 2 \
  --epochs-per-stage 1 \
  --max-completion-length 256 \
  --max-prompt-length 1024 \
  --max-minutes 60 \
  --session-name stage3_intro_$(date +%Y%m%d_%H%M) \
  --adaptive-bugs
```

## Current State (2026-06-30)

| Key | Value |
|-----|-------|
| Stage | 3 = `experience_intro` (new intermediate stage) |
| Episodes completed | 120 / 30 window |
| Recent 30-ep mean reward | -0.009 |
| Advance threshold | 0.30 over 30 episodes |
| Best single reward | 0.65 |
| Checkpoint | `training/output/grpo-curriculum/stage_3_experience_intro/` |
| Session logs | `training/output/rl_sessions/stage3_intro_20260630/` |
| Adaptive difficulty | 0.23 (only `wrong_literal` mutations) |

## What Changed

**Problem**: Stage 3 (`experience_accelerated`) collapsed — 50+ episodes at -0.6 because `experience_used` weight was 0.30 but the model never saw `search_experience` in stages 0-2.

**Fix**: Inserted `experience_intro` as stage 3 (index 3) in `training/curriculum.py`:
- `experience_used` weight = 0.10 (not 0.30)
- Other weights sum to 0.90 so model can score well without calling `search_experience`
- Advance threshold = 0.30 (reachable without the new tool)
- Original `experience_accelerated` shifted to stage 4, `experience_efficiency` to stage 5

**Curriculum state** was reset to stage 3 with clean episode_rewards. Backup at `training/output/curriculum_state_backup_20260630.json`.

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Ti (16.6GB VRAM)
- Autotune: batch=2, gen=2, prompt≤1024, completion≤256
- ~22-34s per training round
- Model: Qwen3-4B in 4-bit QLoRA (3.18% trainable params)

## What to Look For

Check `traces.jsonl` for:
- `tools_found` should include `trace_dependencies` and `inspect_lineage`
- `correct_step` and `correct_var` trending toward 1.0
- Eventually `experience_used` > 0 (model starts calling `search_experience`)

Check `group_stats.jsonl` for:
- `reward_std` > 0.10 (healthy GRPO variance)
- Not all rewards identical (collapse indicator)

## After Stage 3 Advances

Once mean reward over 30 eps exceeds 0.30, the curriculum auto-advances to stage 4 (`experience_accelerated`, `experience_used` weight = 0.30). The model will already know `search_experience` exists from the intro stage.

## Full Pipeline

```bash
# CPU-only: verify reward function
PYTHONPATH=. .venv/bin/python training/train_rl.py --eval-only

# CPU-only: simulate GRPO groups
PYTHONPATH=. .venv/bin/python training/simulate_trace_grpo.py --stage multi_depth --n 6

# GPU: resume training (uses last checkpoint automatically)
PYTHONPATH=. .venv/bin/python training/train_rl.py \
  --base-adapter training/output/grpo-curriculum/stage_3_experience_intro \
  --start-stage 3 \
  --batch-size 2 --num-generations 2 \
  --max-minutes 120 \
  --session-name stage3_continued_$(date +%Y%m%d_%H%M) \
  --adaptive-bugs

# After completion: export to GGUF for Ollama
PYTHONPATH=. .venv/bin/python training/export_sft_ollama.py \
  --adapter training/output/grpo-curriculum/stage_3_experience_intro
```
