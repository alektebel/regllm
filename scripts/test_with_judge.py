#!/usr/bin/env python3
"""
LLM-judge QA evaluation.

Loads the latest checkpoint (or a specified one), then for each sampled QA pair:
  1. Generates an answer with LoRA ENABLED  (fine-tuned model)
  2. Evaluates it with LoRA DISABLED         (base model as judge)

The two roles run on the exact same in-memory model — no double load needed.

Usage:
  python scripts/test_with_judge.py                  # latest checkpoint, 50 samples
  python scripts/test_with_judge.py --sample 100
  python scripts/test_with_judge.py --checkpoint models/finetuned/run_X/checkpoint-Y
  python scripts/test_with_judge.py --gt-only        # ground-truth set only
  python scripts/test_with_judge.py --qa-only        # JSONL sample only
"""

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Checkpoint / model discovery ─────────────────────────────────────────────

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
        print(f"  Warning: {path} not found, skipping.")
        return []
    data = json.load(open(path))
    entries = data.get("entries", [])
    # Normalise to internal format
    return [
        {
            "id":                 e["id"],
            "category":          e["category"],
            "pregunta":          e["pregunta"],
            "respuesta_esperada": e["respuesta_esperada"],
            "datos_clave":       e.get("datos_clave", []),
            "umbral_confianza":  e.get("umbral_confianza", 0.6),
        }
        for e in entries
    ]


def _extract_datos_clave(text: str) -> list[str]:
    """Pull key phrases from a reference answer to use as judge datos_clave."""
    clave = []
    # Numbers with units: 11.076 millones, 12.8%, etc.
    clave += re.findall(r"[\d.,]+\s*(?:%|millones(?:\s*EUR)?|EUR)?", text)
    # Key regulatory terms
    clave += re.findall(
        r"\b(?:IRB|PD|LGD|EAD|MoC|CRR|CET1|Tier\s*1|morosidad|margen de conservadurismo"
        r"|probabilidad de default|pérdida en caso de impago)\b",
        text, re.IGNORECASE,
    )
    # Short quoted phrases (if any)
    clave += re.findall(r'"([^"]{5,60})"', text)

    # Deduplicate, keep non-empty, cap at 8
    seen: set[str] = set()
    out: list[str] = []
    for k in clave:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out[:8]


def load_qa_sample(n: int, seed: int = 42) -> list[dict]:
    """Random sample of n pairs from all data/finetuning JSONL files."""
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
                        "datos_clave":       _extract_datos_clave(asst),
                        "umbral_confianza":  0.5,
                    })

    random.seed(seed)
    return random.sample(pool, min(n, len(pool)))


# ─── Model ────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_STUDENT = (
    "Eres un asistente experto en regulación bancaria y el sector bancario español. "
    "Responde con datos precisos y cita la normativa cuando sea posible."
)
SYSTEM_JUDGE = "Eres un evaluador preciso y objetivo de respuestas sobre regulación bancaria."

JUDGE_PROMPT = """\
Evalúa si la RESPUESTA GENERADA es adecuada comparada con la RESPUESTA ESPERADA.

PREGUNTA: {pregunta}

RESPUESTA ESPERADA: {respuesta_esperada}

RESPUESTA GENERADA: {respuesta_generada}

DATOS CLAVE que deben estar presentes: {datos_clave}

Considera:
1. ¿Contiene los datos clave esperados?
2. ¿Es factualmente correcta respecto a la respuesta esperada?
3. ¿Cubre los puntos principales?

Responde EXACTAMENTE en este formato:
ADECUADA: [SI/NO]
PUNTUACION: [0.0-1.0]
DATOS_PRESENTES: [lista separada por comas, o "ninguno"]
DATOS_AUSENTES: [lista separada por comas, o "ninguno"]
EXPLICACION: [una línea]\
"""


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
        llm_int8_enable_fp32_cpu_offload=True,
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


def _generate(model, tokenizer, system: str, user: str,
              max_new_tokens: int = 250, temperature: float = 0.2) -> str:
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


# ─── Judge result ─────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    is_adequate:       bool
    score:             float
    explanation:       str
    facts_present:     list[str] = field(default_factory=list)
    facts_missing:     list[str] = field(default_factory=list)
    raw_output:        str = ""
    used_fallback:     bool = False


def _parse_judge_output(output: str, datos_clave: list[str],
                        response: str) -> JudgeResult:
    """Parse structured judge output; fall back to keyword heuristic."""
    try:
        adq  = re.search(r"ADECUADA:\s*(SI|NO|Sí|No|si|no)", output, re.IGNORECASE)
        pun  = re.search(r"PUNTUACION:\s*([\d.]+)", output)
        pres = re.search(r"DATOS_PRESENTES:\s*(.+?)(?:\n|$)", output)
        aus  = re.search(r"DATOS_AUSENTES:\s*(.+?)(?:\n|$)", output)
        expl = re.search(r"EXPLICACION:\s*(.+?)(?:\n|$)", output)

        if adq and pun:
            score = max(0.0, min(1.0, float(pun.group(1))))

            def _split(m):
                if not m:
                    return []
                return [s.strip() for s in m.group(1).split(",")
                        if s.strip() and s.strip().lower() not in ("ninguno", "none", "n/a")]

            return JudgeResult(
                is_adequate=adq.group(1).upper() in ("SI", "SÍ"),
                score=score,
                explanation=(expl.group(1).strip() if expl else ""),
                facts_present=_split(pres),
                facts_missing=_split(aus),
                raw_output=output,
            )
    except (ValueError, AttributeError):
        pass

    # Fallback: keyword hit rate
    resp_lower = response.lower()
    present = [k for k in datos_clave if k.lower() in resp_lower]
    missing = [k for k in datos_clave if k.lower() not in resp_lower]
    score   = len(present) / len(datos_clave) if datos_clave else 0.5
    return JudgeResult(
        is_adequate=score >= 0.5,
        score=score,
        explanation=f"Fallback heurístico: {len(present)}/{len(datos_clave)} datos clave presentes",
        facts_present=present,
        facts_missing=missing,
        raw_output=output,
        used_fallback=True,
    )


# ─── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate_entry(model, tokenizer, entry: dict) -> dict:
    # 1. Fine-tuned model generates answer
    model.enable_adapter_layers()
    student_answer = _generate(model, tokenizer, SYSTEM_STUDENT, entry["pregunta"])

    # 2. Base model judges the answer
    model.disable_adapter_layers()
    judge_prompt = JUDGE_PROMPT.format(
        pregunta=entry["pregunta"],
        respuesta_esperada=entry["respuesta_esperada"],
        respuesta_generada=student_answer,
        datos_clave=", ".join(entry["datos_clave"]) if entry["datos_clave"] else "N/A",
    )
    judge_raw = _generate(
        model, tokenizer, SYSTEM_JUDGE, judge_prompt,
        max_new_tokens=300, temperature=0.1,
    )

    result = _parse_judge_output(judge_raw, entry["datos_clave"], student_answer)
    passed = result.is_adequate and result.score >= entry.get("umbral_confianza", 0.5)

    return {
        "id":                entry["id"],
        "category":          entry["category"],
        "pregunta":          entry["pregunta"],
        "respuesta_esperada": entry["respuesta_esperada"],
        "respuesta_modelo":  student_answer,
        "judge_score":       result.score,
        "judge_adequate":    result.is_adequate,
        "judge_explanation": result.explanation,
        "facts_present":     result.facts_present,
        "facts_missing":     result.facts_missing,
        "passed":            passed,
        "used_fallback":     result.used_fallback,
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

def _bar(v: float, w: int = 22) -> str:
    f = int(round(v * w))
    return "[" + "█" * f + "░" * (w - f) + f"] {v*100:5.1f}%"


def print_summary(results: list[dict], title: str):
    if not results:
        return
    scores   = [r["judge_score"] for r in results]
    passed   = [r for r in results if r["passed"]]
    adequate = [r for r in results if r["judge_adequate"]]
    fallback = sum(1 for r in results if r["used_fallback"])

    avg = sum(scores) / len(scores)

    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"  {len(results)} questions | checkpoint: {_checkpoint_label}")
    print(f"{'═'*65}")
    print(f"  Avg judge score   {_bar(avg)}")
    print(f"  Adequate (SI)     {len(adequate):3d} / {len(results)}  ({len(adequate)/len(results)*100:.0f}%)")
    print(f"  Passed threshold  {len(passed):3d} / {len(results)}  ({len(passed)/len(results)*100:.0f}%)")
    if fallback:
        print(f"  Fallback heuristic used: {fallback} times")
    print()

    # Per-category
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    if len(cats) > 1:
        print("  By category:")
        for cat, items in sorted(cats.items()):
            avg_c = sum(r["judge_score"] for r in items) / len(items)
            print(f"    {cat:<28} {_bar(avg_c, 14)}  n={len(items)}")
        print()

    # Worst 3
    worst = sorted(results, key=lambda r: r["judge_score"])[:3]
    print("  Lowest-scoring questions:")
    for r in worst:
        mark = "✓" if r["passed"] else "✗"
        print(f"    {mark} [{r['judge_score']*100:4.0f}%] {r['pregunta'][:65]}")
        if r["facts_missing"]:
            print(f"           Missing: {', '.join(r['facts_missing'][:4])}")
        print(f"           Judge: {r['judge_explanation'][:80]}")
    print(f"{'═'*65}")


def save_results(results: list[dict], label: str):
    out_dir = PROJECT_ROOT / "data/exports/eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = label.lower().replace(" ", "_")[:30]
    path = out_dir / f"judge_{slug}_{ts}.json"
    with open(path, "w") as f:
        json.dump(
            {"label": label, "checkpoint": _checkpoint_label,
             "timestamp": ts, "results": results},
            f, ensure_ascii=False, indent=2,
        )
    print(f"  Saved: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

_checkpoint_label = "unknown"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate fine-tuned checkpoint with LLM judge (LoRA toggle)"
    )
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint dir. Default: latest found automatically.")
    p.add_argument("--sample", type=int, default=50,
                   help="QA pairs to sample from JSONL files (default 50).")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--gt-only",  action="store_true",
                   help="Only run structured ground-truth set.")
    p.add_argument("--qa-only",  action="store_true",
                   help="Only run JSONL sample.")
    return p.parse_args()


def main():
    global _checkpoint_label
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

    _checkpoint_label = str(ckpt_path.relative_to(PROJECT_ROOT))

    # ── Load test data ──
    gt_entries = [] if args.qa_only  else load_ground_truth()
    qa_sample  = [] if args.gt_only  else load_qa_sample(args.sample, args.seed)
    total = len(gt_entries) + len(qa_sample)

    print(f"\nTest sets:")
    print(f"  Ground truth : {len(gt_entries)} entries")
    print(f"  QA sample    : {len(qa_sample)} entries  (--sample {args.sample})")
    print(f"  Total        : {total}\n")

    # ── Load model ──
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(str(ckpt_path))

    # ── Evaluate ──
    results_gt: list[dict] = []
    results_qa: list[dict] = []
    done = 0

    def run_set(entries: list[dict], label: str, out: list[dict]):
        nonlocal done
        if not entries:
            return
        print(f"\n── {label} ({'─'*(55-len(label))})")
        for entry in entries:
            done += 1
            print(f"  [{done:3d}/{total}] {entry['pregunta'][:62]}...", end="", flush=True)
            result = evaluate_entry(model, tokenizer, entry)
            mark = "✓" if result["passed"] else "✗"
            print(f"  {mark} score={result['judge_score']*100:.0f}%")
            out.append(result)

    run_set(gt_entries, "Ground Truth", results_gt)
    run_set(qa_sample,  "QA Pair Sample", results_qa)

    # ── Reports ──
    if results_gt:
        print_summary(results_gt, "Ground Truth Evaluation")
        save_results(results_gt, "ground_truth")

    if results_qa:
        print_summary(results_qa, "QA Pair Sample Evaluation")
        save_results(results_qa, "qa_sample")

    # ── Overall ──
    all_r = results_gt + results_qa
    if all_r:
        avg = sum(r["judge_score"] for r in all_r) / len(all_r)
        passed = sum(1 for r in all_r if r["passed"])
        print(f"\n{'═'*65}")
        print(f"  OVERALL  ({len(all_r)} questions, checkpoint: {_checkpoint_label})")
        print(f"  Avg judge score  : {avg*100:.1f}%")
        print(f"  Passed           : {passed}/{len(all_r)}")
        print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
