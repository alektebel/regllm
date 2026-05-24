#!/usr/bin/env python3
"""
Evaluate the fine-tuned model against QA pairs.

Two test sets are used:
  1. data/test_ground_truth.json  — 50 structured entries with expected key facts
     (used for precision scoring: are the key data points present?)
  2. data/finetuning/*.jsonl      — QA pairs used for training; a random sample
     is drawn so we can check how well the model has retained the knowledge
     (keyword F1 against the reference answer)

Usage:
  python scripts/eval_qa.py                          # auto-detect latest model
  python scripts/eval_qa.py --model models/.../final_model
  python scripts/eval_qa.py --sample 50              # sample size from JSONL
  python scripts/eval_qa.py --no-model               # dry-run: just inspect the test sets
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Model finder ─────────────────────────────────────────────────────────────

def find_latest_model() -> Path | None:
    models_dir = PROJECT_ROOT / "models/finetuned"
    if not models_dir.exists():
        return None
    candidates = sorted(
        (p / "final_model" for p in models_dir.iterdir()
         if p.is_dir() and (p / "final_model" / "adapter_config.json").exists()),
        key=lambda p: p.parent.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ─── Scoring ──────────────────────────────────────────────────────────────────

def keyword_hit_rate(response: str, keywords: list[str]) -> float:
    """Fraction of keywords that appear in the response (case-insensitive)."""
    if not keywords:
        return 1.0
    resp_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return hits / len(keywords)


def token_f1(reference: str, hypothesis: str) -> float:
    """Token-level F1 between reference and hypothesis (unigrams, Spanish-aware)."""
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+", text.lower())

    ref_tokens  = tokenize(reference)
    hyp_tokens  = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1
    hyp_set = {}
    for t in hyp_tokens:
        hyp_set[t] = hyp_set.get(t, 0) + 1

    common = sum(min(ref_set.get(t, 0), hyp_set.get(t, 0)) for t in hyp_set)
    precision = common / sum(hyp_set.values())
    recall    = common / sum(ref_set.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_ground_truth() -> list[dict]:
    path = PROJECT_ROOT / "data/test_ground_truth.json"
    if not path.exists():
        print(f"Warning: {path} not found, skipping ground-truth set.")
        return []
    data = json.load(open(path))
    return data.get("entries", [])


def load_qa_sample(n: int, seed: int = 42) -> list[dict]:
    """Sample n QA pairs from all finetuning JSONL files."""
    all_items = []
    finetuning_dir = PROJECT_ROOT / "data/finetuning"
    for path in sorted(finetuning_dir.rglob("*.jsonl")):
        with open(path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Extract question (user) and reference answer (assistant)
                    msgs = item.get("messages", [])
                    user = next((m["content"] for m in msgs if m["role"] == "user"), None)
                    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                    if user and asst:
                        all_items.append({
                            "id": f"qa_{len(all_items):04d}",
                            "category": "qa_pair",
                            "pregunta": user,
                            "respuesta_esperada": asst,
                            "datos_clave": [],          # derived below
                            "source_file": path.name,
                        })

    # Derive key numbers / short phrases as pseudo datos_clave
    for item in all_items:
        ref = item["respuesta_esperada"]
        nums = re.findall(r"\d[\d.,]*(?:\s*%|\s*millones)?", ref)
        item["datos_clave"] = list(dict.fromkeys(nums))[:6]  # unique, capped

    random.seed(seed)
    sample = random.sample(all_items, min(n, len(all_items)))
    return sample


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model + LoRA adapter from {model_path}...")
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
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return model, tokenizer


def infer(model, tokenizer, question: str, system: str) -> str:
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


SYSTEM = (
    "Eres un asistente experto en regulación bancaria y el sector bancario español. "
    "Responde con datos precisos y cita la normativa cuando sea posible."
)


# ─── Report ───────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value*100:5.1f}%"


def print_report(results: list[dict], title: str):
    if not results:
        return

    print(f"\n{'='*65}")
    print(f"  {title}  ({len(results)} questions)")
    print(f"{'='*65}")

    hit_rates = [r["keyword_hit_rate"] for r in results]
    f1_scores  = [r["token_f1"]        for r in results]

    avg_hit = sum(hit_rates) / len(hit_rates)
    avg_f1  = sum(f1_scores)  / len(f1_scores)

    print(f"  Keyword hit rate  {_bar(avg_hit)}")
    print(f"  Token F1          {_bar(avg_f1)}")
    print()

    # Per-category breakdown
    cats = {}
    for r in results:
        c = r.get("category", "?")
        cats.setdefault(c, []).append(r)
    if len(cats) > 1:
        print("  By category:")
        for cat, items in sorted(cats.items()):
            avg = sum(i["keyword_hit_rate"] for i in items) / len(items)
            print(f"    {cat:<20} {_bar(avg, 15)}  (n={len(items)})")
        print()

    # Worst cases
    worst = sorted(results, key=lambda r: r["keyword_hit_rate"])[:3]
    print("  Lowest scoring questions:")
    for r in worst:
        print(f"    [{r['keyword_hit_rate']*100:4.0f}%] {r['pregunta'][:70]}")
    print(f"{'='*65}")


def save_report(results: list[dict], title: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = title.lower().replace(" ", "_")[:30]
    out_path = output_dir / f"eval_{slug}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"title": title, "timestamp": ts, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"  Report saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned model on QA pairs")
    p.add_argument("--model",    default=None,
                   help="Path to final_model dir. Defaults to latest run.")
    p.add_argument("--sample",   type=int, default=100,
                   help="Number of QA pairs to sample for the JSONL eval (default 100).")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--no-model", action="store_true",
                   help="Skip inference — just show dataset stats.")
    p.add_argument("--gt-only",  action="store_true",
                   help="Only run the ground-truth set, skip JSONL sampling.")
    p.add_argument("--qa-only",  action="store_true",
                   help="Only run the JSONL sample, skip ground-truth set.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve model path ──
    model_path = None
    if not args.no_model:
        if args.model:
            model_path = args.model
        else:
            latest = find_latest_model()
            if latest:
                model_path = str(latest)
                print(f"Auto-detected model: {model_path}")
            else:
                print("No trained model found. Run train_combined.py first.")
                print("Use --no-model to inspect test sets without inference.")
                sys.exit(1)

    # ── Load test sets ──
    gt_entries = [] if args.qa_only  else load_ground_truth()
    qa_sample  = [] if args.gt_only  else load_qa_sample(args.sample, args.seed)

    print(f"\nTest sets ready:")
    print(f"  Ground truth entries : {len(gt_entries)}")
    print(f"  QA pair sample       : {len(qa_sample)}")

    if args.no_model:
        print("\n--no-model flag set, exiting without inference.")
        return

    # ── Load model ──
    model, tokenizer = load_model(model_path)

    results_gt = []
    results_qa = []
    total = len(gt_entries) + len(qa_sample)
    done  = 0

    # ── Evaluate ground truth ──
    if gt_entries:
        print(f"\nRunning ground-truth evaluation ({len(gt_entries)} questions)...")
        for entry in gt_entries:
            done += 1
            print(f"  [{done}/{total}] {entry['pregunta'][:60]}...", end="", flush=True)

            response = infer(model, tokenizer, entry["pregunta"], SYSTEM)

            hit  = keyword_hit_rate(response, entry.get("datos_clave", []))
            f1   = token_f1(entry["respuesta_esperada"], response)
            pass_ = hit >= entry.get("umbral_confianza", 0.6)

            print(f"  hit={hit*100:.0f}%  f1={f1*100:.0f}%  {'✓' if pass_ else '✗'}")

            results_gt.append({
                "id":                entry["id"],
                "category":          entry["category"],
                "pregunta":          entry["pregunta"],
                "respuesta_esperada": entry["respuesta_esperada"],
                "respuesta_modelo":  response,
                "datos_clave":       entry.get("datos_clave", []),
                "keyword_hit_rate":  hit,
                "token_f1":          f1,
                "passed":            pass_,
            })

    # ── Evaluate QA sample ──
    if qa_sample:
        print(f"\nRunning QA-pair evaluation ({len(qa_sample)} questions)...")
        for entry in qa_sample:
            done += 1
            print(f"  [{done}/{total}] {entry['pregunta'][:60]}...", end="", flush=True)

            response = infer(model, tokenizer, entry["pregunta"], SYSTEM)

            hit = keyword_hit_rate(response, entry.get("datos_clave", []))
            f1  = token_f1(entry["respuesta_esperada"], response)

            print(f"  hit={hit*100:.0f}%  f1={f1*100:.0f}%")

            results_qa.append({
                "id":                entry["id"],
                "category":          entry.get("source_file", "qa_pair"),
                "pregunta":          entry["pregunta"],
                "respuesta_esperada": entry["respuesta_esperada"],
                "respuesta_modelo":  response,
                "datos_clave":       entry.get("datos_clave", []),
                "keyword_hit_rate":  hit,
                "token_f1":          f1,
            })

    # ── Reports ──
    reports_dir = PROJECT_ROOT / "data/exports/eval_reports"

    if results_gt:
        print_report(results_gt, "Ground Truth Evaluation")
        save_report(results_gt, "Ground Truth Evaluation", reports_dir)

    if results_qa:
        print_report(results_qa, "QA Pair Sample Evaluation")
        save_report(results_qa, "QA Pair Sample Evaluation", reports_dir)

    # ── Combined pass/fail summary ──
    all_results = results_gt + results_qa
    if all_results:
        avg_hit = sum(r["keyword_hit_rate"] for r in all_results) / len(all_results)
        avg_f1  = sum(r["token_f1"]         for r in all_results) / len(all_results)
        print(f"\n{'='*65}")
        print(f"  OVERALL  ({len(all_results)} questions)")
        print(f"  Avg keyword hit rate : {avg_hit*100:.1f}%")
        print(f"  Avg token F1         : {avg_f1*100:.1f}%")
        if results_gt:
            passed = sum(1 for r in results_gt if r.get("passed"))
            print(f"  Ground-truth pass    : {passed}/{len(results_gt)}")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
