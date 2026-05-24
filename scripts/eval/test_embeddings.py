#!/usr/bin/env python3
"""
Embedding-based similarity test suite for the fine-tuned model.

For each question in the test set the script:
  1. Generates an answer with LoRA ENABLED  (fine-tuned)
  2. Generates an answer with LoRA DISABLED (base model)
  3. Embeds all three texts (fine-tuned answer, base answer, expected answer)
     with a multilingual sentence-transformer (no LLM API needed).
  4. Computes cosine similarities:
       • ft_vs_ref  : fine-tuned answer  ↔ expected answer
       • base_vs_ref: base-model answer  ↔ expected answer
       • ft_vs_base : fine-tuned answer  ↔ base-model answer  (drift)

Reports per-sample scores, aggregate statistics, a text histogram, and the
best/worst 3 questions.  Saves a JSON report + CSV for downstream analysis.

Usage:
  python scripts/test_embeddings.py                   # final_model / latest ckpt, 30 samples
  python scripts/test_embeddings.py --sample 60
  python scripts/test_embeddings.py --checkpoint models/finetuned/run_X/final_model
  python scripts/test_embeddings.py --embed-model paraphrase-multilingual-mpnet-base-v2
  python scripts/test_embeddings.py --gt-only        # ground-truth set only
  python scripts/test_embeddings.py --qa-only        # JSONL sample only
  python scripts/test_embeddings.py --no-base        # skip base-model generation (faster)
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Checkpoint discovery (mirrors test_with_judge.py) ───────────────────────

def find_latest_checkpoint() -> Path | None:
    models_dir = PROJECT_ROOT / "models/finetuned"
    if not models_dir.exists():
        return None
    checkpoints = [
        ckpt
        for run in models_dir.iterdir()
        if run.is_dir() and run.name.startswith("run_")
        for ckpt in run.iterdir()
        if ckpt.is_dir()
        and (ckpt.name.startswith("checkpoint-") or ckpt.name == "final_model")
        and (ckpt / "adapter_model.safetensors").exists()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_ground_truth() -> list[dict]:
    path = PROJECT_ROOT / "data/test_ground_truth.json"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping ground-truth set.")
        return []
    data = json.load(open(path))
    return [
        {
            "id":                 e["id"],
            "category":          e["category"],
            "pregunta":          e["pregunta"],
            "respuesta_esperada": e["respuesta_esperada"],
        }
        for e in data.get("entries", [])
    ]


def load_qa_sample(n: int, seed: int = 42) -> list[dict]:
    pool: list[dict] = []
    finetuning_dir = PROJECT_ROOT / "data/finetuning"
    for path in sorted(finetuning_dir.rglob("*.jsonl")):
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                msgs = item.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"),  None)
                asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if user and asst:
                    pool.append({
                        "id":                f"{path.stem}_{i:04d}",
                        "category":          path.stem,
                        "pregunta":          user,
                        "respuesta_esperada": asst,
                    })
    random.seed(seed)
    return random.sample(pool, min(n, len(pool)))


# ─── LLM model ────────────────────────────────────────────────────────────────

BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_STUDENT = (
    "Eres un asistente experto en regulación bancaria y el sector bancario español. "
    "Responde con datos precisos y cita la normativa cuando sea posible."
)


def load_model_and_tokenizer(checkpoint_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"  Base model : {BASE_MODEL}")
    print(f"  Adapter    : {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system: str, user: str,
             max_new_tokens: int = 256, temperature: float = 0.2) -> str:
    import torch
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ─── Embedding similarity ─────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def batch_embed(embed_model, texts: list[str]) -> np.ndarray:
    """Return (N, D) embedding matrix."""
    return embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EmbedResult:
    id:              str
    category:        str
    pregunta:        str
    respuesta_esperada: str
    respuesta_ft:    str        # fine-tuned model answer
    respuesta_base:  str        # base model answer (empty if --no-base)
    ft_vs_ref:       float      # cosine sim: fine-tuned ↔ expected
    base_vs_ref:     float      # cosine sim: base ↔ expected  (-1 if no base)
    ft_vs_base:      float      # cosine sim: fine-tuned ↔ base (-1 if no base)
    ft_gain:         float      # ft_vs_ref - base_vs_ref        (0 if no base)


# ─── Statistics & reporting ───────────────────────────────────────────────────

def _histogram(values: list[float], bins: int = 10, width: int = 30) -> str:
    if not values:
        return ""
    edges = [i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int(v * bins), bins - 1)
        counts[idx] += 1
    max_c = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(c / max_c * width)
        lines.append(f"  [{edges[i]:.1f}-{edges[i+1]:.1f})  {bar:<{width}} {c:3d}")
    return "\n".join(lines)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p * 100))


def print_report(results: list[EmbedResult], title: str, ckpt_label: str,
                 has_base: bool):
    if not results:
        return
    ft_sims   = [r.ft_vs_ref   for r in results]
    base_sims = [r.base_vs_ref for r in results if r.base_vs_ref >= 0]
    drifts    = [r.ft_vs_base  for r in results if r.ft_vs_base  >= 0]
    gains     = [r.ft_gain     for r in results]

    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"  {len(results)} questions  |  checkpoint: {ckpt_label}")
    print(f"{'═'*70}")

    def _row(label, vals):
        if not vals:
            return
        print(f"  {label:<30}  mean={np.mean(vals):.3f}  "
              f"std={np.std(vals):.3f}  "
              f"p25={_pct(vals,.25):.3f}  p50={_pct(vals,.5):.3f}  p75={_pct(vals,.75):.3f}")

    _row("ft_vs_ref  (fine-tuned↔expected)", ft_sims)
    if has_base:
        _row("base_vs_ref (base↔expected)",     base_sims)
        _row("ft_vs_base  (fine-tuned↔base)",   drifts)
        improved = sum(1 for r in results if r.ft_gain > 0.02)
        degraded = sum(1 for r in results if r.ft_gain < -0.02)
        print(f"\n  Fine-tuning improved similarity : {improved}/{len(results)} questions (gain > 0.02)")
        print(f"  Fine-tuning degraded similarity : {degraded}/{len(results)} questions (gain < -0.02)")

    # Histogram of ft_vs_ref
    print(f"\n  Histogram  ft_vs_ref  (0=unrelated  1=identical):")
    print(_histogram(ft_sims))

    # Best / worst 3
    ranked = sorted(results, key=lambda r: r.ft_vs_ref)
    print(f"\n  Worst 3  (lowest fine-tuned↔expected similarity):")
    for r in ranked[:3]:
        gain_str = f"  gain={r.ft_gain:+.3f}" if has_base else ""
        print(f"    [{r.ft_vs_ref:.3f}]{gain_str}  {r.pregunta[:70]}")

    print(f"\n  Best 3  (highest fine-tuned↔expected similarity):")
    for r in ranked[-3:]:
        gain_str = f"  gain={r.ft_gain:+.3f}" if has_base else ""
        print(f"    [{r.ft_vs_ref:.3f}]{gain_str}  {r.pregunta[:70]}")

    print(f"{'═'*70}")


def save_results(results: list[EmbedResult], label: str, ckpt_label: str,
                 embed_model_name: str):
    out_dir = PROJECT_ROOT / "data/exports/eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = label.lower().replace(" ", "_")[:30]

    # JSON
    json_path = out_dir / f"embed_{slug}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "label":       label,
                "checkpoint":  ckpt_label,
                "embed_model": embed_model_name,
                "timestamp":   ts,
                "results":     [asdict(r) for r in results],
            },
            f, ensure_ascii=False, indent=2,
        )

    # CSV
    csv_path = out_dir / f"embed_{slug}_{ts}.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"  Saved JSON: {json_path}")
    print(f"  Saved CSV : {csv_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Embedding-based similarity test suite for the fine-tuned model"
    )
    p.add_argument("--checkpoint",   default=None,
                   help="Path to checkpoint/final_model dir. Default: latest auto-detected.")
    p.add_argument("--sample",       type=int, default=30,
                   help="Number of QA pairs to sample from JSONL files (default 30).")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--embed-model",  default="paraphrase-multilingual-MiniLM-L12-v2",
                   help="Sentence-transformer model name (default: paraphrase-multilingual-MiniLM-L12-v2).")
    p.add_argument("--gt-only",      action="store_true",
                   help="Only run structured ground-truth set.")
    p.add_argument("--qa-only",      action="store_true",
                   help="Only run JSONL sample.")
    p.add_argument("--no-base",      action="store_true",
                   help="Skip base-model generation (faster, no ft_vs_base / gain metrics).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve checkpoint ──
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path is None:
            print("No checkpoint found. Train first with: python scripts/train_combined.py")
            sys.exit(1)
        print(f"Auto-detected checkpoint: {ckpt_path}")

    ckpt_label = str(ckpt_path.relative_to(PROJECT_ROOT))

    # ── Load test data ──
    gt_entries = [] if args.qa_only  else load_ground_truth()
    qa_sample  = [] if args.gt_only  else load_qa_sample(args.sample, args.seed)

    all_entries = gt_entries + qa_sample
    total = len(all_entries)
    print(f"\nTest sets:")
    print(f"  Ground truth : {len(gt_entries)} entries")
    print(f"  QA sample    : {len(qa_sample)} entries  (--sample {args.sample})")
    print(f"  Total        : {total}")

    if total == 0:
        print("No test data found. Exiting.")
        sys.exit(1)

    # ── Load LLM ──
    print(f"\nLoading generative model ...")
    model, tokenizer = load_model_and_tokenizer(str(ckpt_path))

    # ── Load embedding model ──
    print(f"\nLoading embedding model: {args.embed_model} ...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model)
    embed_dim   = embed_model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embed_dim}")

    # ── Generate answers ──
    print(f"\nGenerating answers ({'with base comparison' if not args.no_base else 'fine-tuned only'}) ...")
    results: list[EmbedResult] = []

    for i, entry in enumerate(all_entries, 1):
        print(f"  [{i:3d}/{total}] {entry['pregunta'][:65]}...", end="", flush=True)

        # Fine-tuned answer
        model.enable_adapter_layers()
        ans_ft = generate(model, tokenizer, SYSTEM_STUDENT, entry["pregunta"])

        # Base model answer
        ans_base = ""
        if not args.no_base:
            model.disable_adapter_layers()
            ans_base = generate(model, tokenizer, SYSTEM_STUDENT, entry["pregunta"])

        # Embed
        texts_to_embed = [ans_ft, entry["respuesta_esperada"]]
        if not args.no_base:
            texts_to_embed.append(ans_base)

        embs = batch_embed(embed_model, texts_to_embed)
        e_ft  = embs[0]
        e_ref = embs[1]
        e_base = embs[2] if not args.no_base else None

        ft_vs_ref   = cosine_similarity(e_ft,   e_ref)
        base_vs_ref = cosine_similarity(e_base, e_ref) if e_base is not None else -1.0
        ft_vs_base  = cosine_similarity(e_ft,   e_base) if e_base is not None else -1.0
        ft_gain     = (ft_vs_ref - base_vs_ref) if e_base is not None else 0.0

        results.append(EmbedResult(
            id=entry["id"],
            category=entry["category"],
            pregunta=entry["pregunta"],
            respuesta_esperada=entry["respuesta_esperada"],
            respuesta_ft=ans_ft,
            respuesta_base=ans_base,
            ft_vs_ref=round(ft_vs_ref, 4),
            base_vs_ref=round(base_vs_ref, 4),
            ft_vs_base=round(ft_vs_base, 4),
            ft_gain=round(ft_gain, 4),
        ))

        # Inline progress indicator
        gain_str = f"  gain={ft_gain:+.3f}" if not args.no_base else ""
        print(f"  sim={ft_vs_ref:.3f}{gain_str}")

    # ── Split back into GT / QA for separate reports ──
    results_gt = results[:len(gt_entries)]
    results_qa = results[len(gt_entries):]
    has_base   = not args.no_base

    if results_gt:
        print_report(results_gt, "Ground Truth  —  Embedding Similarity", ckpt_label, has_base)
        save_results(results_gt, "ground_truth", ckpt_label, args.embed_model)

    if results_qa:
        print_report(results_qa, "QA Sample  —  Embedding Similarity", ckpt_label, has_base)
        save_results(results_qa, "qa_sample", ckpt_label, args.embed_model)

    # ── Overall summary ──
    if len(results) > 0:
        all_ft  = [r.ft_vs_ref   for r in results]
        all_base = [r.base_vs_ref for r in results if r.base_vs_ref >= 0]
        all_gain = [r.ft_gain     for r in results]

        print(f"\n{'═'*70}")
        print(f"  OVERALL  ({len(results)} questions)  checkpoint: {ckpt_label}")
        print(f"  Embed model  : {args.embed_model}")
        print(f"  ft_vs_ref    mean={np.mean(all_ft):.3f}  std={np.std(all_ft):.3f}  "
              f"p25={float(np.percentile(all_ft,.25)):.3f}  "
              f"p75={float(np.percentile(all_ft,.75)):.3f}")
        if has_base and all_base:
            print(f"  base_vs_ref  mean={np.mean(all_base):.3f}  std={np.std(all_base):.3f}  "
                  f"p25={float(np.percentile(all_base,.25)):.3f}  "
                  f"p75={float(np.percentile(all_base,.75)):.3f}")
            print(f"  avg ft_gain  {np.mean(all_gain):+.3f}  "
                  f"({'fine-tuning helps' if np.mean(all_gain) > 0 else 'fine-tuning hurts or neutral'})")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
