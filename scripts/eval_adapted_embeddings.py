#!/usr/bin/env python3
"""
eval_adapted_embeddings.py
--------------------------
Contrast-test regllm model answers against the ground-truth QA dataset using a
domain-calibrated embedding metric: multilingual-E5-large + MetricAdapter.

The MetricAdapter is a small MLP (1024→512→512, LayerNorm) trained with
CosineEmbeddingLoss on positive/negative banking QA pairs (from embed_dist).
It produces a 512-dim space where correct Q-A pairs are closer than random ones.

Pipeline per question
---------------------
1. Generate fine-tuned answer (LoRA on) and base answer (LoRA off)
   — OR load pre-generated answers via --answers-file
2. Embed question, ft_answer, base_answer, ref_answer through E5 + adapter
3. Compute adapted cosine similarities:
     ft_vs_ref   : fine-tuned ↔ reference
     base_vs_ref : base model ↔ reference
     ft_gain     : ft_vs_ref − base_vs_ref
4. Also compute raw cosine (no adapter) for comparison
5. Aggregate stats + save JSON/CSV report

Modes
-----
  # Full eval — generate answers with LoRA, then score (GPU recommended)
  python scripts/eval_adapted_embeddings.py

  # Pre-generated answers (embedding only, no GPU needed)
  python scripts/eval_adapted_embeddings.py --answers-file data/exports/answers.jsonl

  # Train a regllm-specific adapter first, then eval
  python scripts/eval_adapted_embeddings.py --train-adapter

  # Quick smoke-test (3 samples, no model generation)
  python scripts/eval_adapted_embeddings.py --mock
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_ADAPTER   = PROJECT_ROOT / "models" / "adapted_metric" / "metric_adapter.pt"
REGLLM_ADAPTER    = PROJECT_ROOT / "models" / "adapted_metric" / "metric_adapter_regllm.pt"
GT_PATH           = PROJECT_ROOT / "data" / "test_ground_truth.json"
FINETUNE_DIR      = PROJECT_ROOT / "data" / "finetuning"
REPORT_DIR        = PROJECT_ROOT / "data" / "exports" / "eval_reports"
E5_MODEL_NAME     = "intfloat/multilingual-e5-large"
E5_DIM            = 1024
ADAPTER_OUT_DIM   = 512
BASE_LLM          = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "Eres un asistente especializado en regulación bancaria y riesgo de crédito. "
    "Responde en español, de forma técnica y precisa."
)


# ─── MetricAdapter ────────────────────────────────────────────────────────────

class MetricAdapter(nn.Module):
    """
    Same architecture as embed_dist/banking_qa_project/train_adapter.py.
    Maps E5-large 1024-dim embeddings → 512-dim domain-calibrated space.
    """
    def __init__(self, input_dim: int = E5_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_adapter(path: Path, device: str = "cpu") -> MetricAdapter:
    adapter = MetricAdapter()
    state = torch.load(path, map_location=device)
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter


# ─── Embedding ────────────────────────────────────────────────────────────────

def _load_e5(local_path: str | None) -> object:
    from sentence_transformers import SentenceTransformer
    model_name = local_path if (local_path and Path(local_path).exists()) else E5_MODEL_NAME
    print(f"  Loading E5 model: {model_name}")
    return SentenceTransformer(model_name, device="cpu")


def embed_raw(texts: list[str], st_model, batch_size: int = 16) -> np.ndarray:
    """Raw E5 embeddings, L2-normalised (uses 'query:' prefix as in embed_dist)."""
    prefixed = [f"query: {t}" for t in texts]
    embs = st_model.encode(prefixed, batch_size=batch_size,
                           show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-9)


def embed_adapted(texts: list[str], st_model, adapter: MetricAdapter,
                  batch_size: int = 16) -> np.ndarray:
    """E5 → MetricAdapter → L2-normalised 512-dim vectors."""
    raw = embed_raw(texts, st_model, batch_size)
    with torch.no_grad():
        proj = adapter(torch.tensor(raw, dtype=torch.float32)).numpy()
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    return proj / np.maximum(norms, 1e-9)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised vectors."""
    return float(np.dot(a, b))


# ─── Ground truth + QA data ───────────────────────────────────────────────────

def load_ground_truth() -> list[dict]:
    if not GT_PATH.exists():
        print(f"  WARNING: {GT_PATH} not found.")
        return []
    data = json.loads(GT_PATH.read_text())
    return [
        {
            "id":       e["id"],
            "category": e["category"],
            "pregunta": e["pregunta"],
            "ref":      e["respuesta_esperada"],
        }
        for e in data.get("entries", [])
    ]


def load_finetuning_qa(max_items: int = 500) -> list[dict]:
    """Load Q-A pairs from the finetuning JSONL files."""
    pool = []
    for path in sorted(FINETUNE_DIR.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            msgs = item.get("messages", [])
            q = next((m["content"] for m in msgs if m["role"] == "user"), None)
            a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if q and a:
                pool.append({"pregunta": q, "ref": a})
            if len(pool) >= max_items:
                break
        if len(pool) >= max_items:
            break
    return pool


# ─── Separation dataset (for adapter training) ───────────────────────────────

def build_separation_pairs(entries: list[dict], seed: int = 42) -> list[dict]:
    """
    Build positive / negative pairs from ground-truth + finetuning entries,
    mirroring the embed_dist generate_data.py approach.
    """
    random.seed(seed)
    pairs = []
    n = len(entries)

    for i, e in enumerate(entries):
        # Positive: question ↔ correct answer (low distance)
        pairs.append({"type": "positive",
                      "text1": e["pregunta"], "text2": e["ref"],
                      "expected": "low_distance"})
        # Negative: question ↔ a different entry's answer (high distance)
        j = (i + random.randint(1, n - 1)) % n
        pairs.append({"type": "negative",
                      "text1": e["pregunta"], "text2": entries[j]["ref"],
                      "expected": "high_distance"})

    # Guardrail mismatch: banking question ↔ off-topic refusal
    refusals = [
        "Lo siento, no puedo responder eso.",
        "Mi ámbito se limita a la regulación bancaria.",
        "Esa consulta está fuera de mi ámbito.",
    ]
    for i in range(min(10, n)):
        pairs.append({"type": "guardrail_mismatch",
                      "text1": entries[i]["pregunta"],
                      "text2": random.choice(refusals),
                      "expected": "very_high_distance"})

    return pairs


# ─── Adapter training ─────────────────────────────────────────────────────────

class _PairDataset(Dataset):
    def __init__(self, pairs: list[dict], st_model):
        unique = list({p["text1"] for p in pairs} | {p["text2"] for p in pairs})
        print(f"  Pre-computing E5 embeddings for {len(unique)} unique texts …")
        raw = st_model.encode([f"query: {t}" for t in unique],
                              batch_size=16, show_progress_bar=True,
                              convert_to_numpy=True)
        t2e = {t: e for t, e in zip(unique, raw)}
        self.samples = [
            (t2e[p["text1"]], t2e[p["text2"]],
             1.0 if p["type"] == "positive" else 0.0)
            for p in pairs
        ]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        e1, e2, lbl = self.samples[idx]
        return torch.tensor(e1), torch.tensor(e2), torch.tensor(lbl)


def train_adapter(
    pairs: list[dict],
    st_model,
    start_from: Path | None = None,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 64,
    save_path: Path = REGLLM_ADAPTER,
) -> MetricAdapter:
    print(f"\n  Training MetricAdapter — {len(pairs)} pairs, {epochs} epochs")
    adapter = MetricAdapter()
    if start_from and start_from.exists():
        print(f"  Starting from pre-trained weights: {start_from}")
        adapter.load_state_dict(torch.load(start_from, map_location="cpu"))

    dataset = _PairDataset(pairs, st_model)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt     = optim.Adam(adapter.parameters(), lr=lr)
    loss_fn = nn.CosineEmbeddingLoss(margin=0.3)

    adapter.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for e1, e2, lbl in loader:
            t = (lbl * 2 - 1)          # 0/1 → -1/+1
            opt.zero_grad()
            loss = loss_fn(adapter(e1), adapter(e2), t)
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={total/len(loader):.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), save_path)
    print(f"  Adapter saved → {save_path}")
    adapter.eval()
    return adapter


# ─── LLM generation ───────────────────────────────────────────────────────────

def find_latest_checkpoint() -> Path | None:
    models_dir = PROJECT_ROOT / "models" / "finetuned"
    if not models_dir.exists():
        return None
    ckpts = [
        c for run in models_dir.iterdir()
        if run.is_dir() and run.name.startswith("run_")
        for c in run.iterdir()
        if c.is_dir()
        and (c.name.startswith("checkpoint-") or c.name == "final_model")
        and (c / "adapter_model.safetensors").exists()
    ]
    return max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None


def _generate_answer(model, tokenizer, question: str, lora_on: bool,
                     max_new_tokens: int = 256) -> str:
    from peft import PeftModel
    import torch

    if lora_on and hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()
    elif hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    out = ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def generate_all_answers(entries: list[dict], checkpoint: str,
                         max_new_tokens: int) -> list[dict]:
    """Load LoRA model and generate ft + base answers for every entry."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n  Loading base model: {BASE_LLM}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(BASE_LLM, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_LLM, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    print(f"  Loading LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()

    results = []
    for i, e in enumerate(entries, 1):
        print(f"  [{i}/{len(entries)}] {e['id']} …", end=" ", flush=True)
        ft_ans   = _generate_answer(model, tok, e["pregunta"], lora_on=True,
                                    max_new_tokens=max_new_tokens)
        base_ans = _generate_answer(model, tok, e["pregunta"], lora_on=False,
                                    max_new_tokens=max_new_tokens)
        results.append({**e, "ft_answer": ft_ans, "base_answer": base_ans})
        print("done")

    return results


# ─── Score dataclass ──────────────────────────────────────────────────────────

@dataclass
class Score:
    id:              str
    category:        str
    # Adapted metric (E5 + MetricAdapter)
    adapted_ft_ref:   float
    adapted_base_ref: float
    adapted_ft_gain:  float
    # Raw E5 cosine (no adapter) — for comparison
    raw_ft_ref:       float
    raw_base_ref:     float
    raw_ft_gain:      float


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(entries_with_answers: list[dict], st_model, adapter: MetricAdapter,
             batch_size: int = 16) -> list[Score]:
    print("\n  Embedding all texts …")
    all_texts = []
    for e in entries_with_answers:
        all_texts += [e["pregunta"], e["ref"], e["ft_answer"], e["base_answer"]]

    print("  → adapted embeddings")
    adapted = embed_adapted(all_texts, st_model, adapter, batch_size)
    print("  → raw E5 embeddings")
    raw     = embed_raw(all_texts, st_model, batch_size)

    scores = []
    for i, e in enumerate(entries_with_answers):
        base_i = i * 4
        # adapted
        a_q,  a_ref, a_ft, a_base = (adapted[base_i + k] for k in range(4))
        aft_r  = cos(a_ft,   a_ref)
        abas_r = cos(a_base, a_ref)
        # raw
        r_q,  r_ref, r_ft, r_base = (raw[base_i + k] for k in range(4))
        rft_r  = cos(r_ft,   r_ref)
        rbas_r = cos(r_base, r_ref)

        scores.append(Score(
            id=e["id"], category=e["category"],
            adapted_ft_ref=round(aft_r,  4),
            adapted_base_ref=round(abas_r, 4),
            adapted_ft_gain=round(aft_r - abas_r, 4),
            raw_ft_ref=round(rft_r,  4),
            raw_base_ref=round(rbas_r, 4),
            raw_ft_gain=round(rft_r - rbas_r, 4),
        ))

    return scores


# ─── Reporting ────────────────────────────────────────────────────────────────

def _agg(vals: list[float]) -> dict:
    a = np.array(vals)
    return {
        "mean": round(float(a.mean()), 4),
        "std":  round(float(a.std()),  4),
        "p25":  round(float(np.percentile(a, 25)), 4),
        "p50":  round(float(np.percentile(a, 50)), 4),
        "p75":  round(float(np.percentile(a, 75)), 4),
    }


def report(scores: list[Score], adapter_path: Path, n_trained_on: int = 0):
    print("\n" + "═" * 65)
    print("  RegLLM — Adapted Embedding Eval")
    print("  Adapter :", adapter_path.name)
    print(f"  Samples : {len(scores)}")
    print("═" * 65)

    # ── Overall stats ──
    for label, ft_key, base_key, gain_key in [
        ("ADAPTED (E5 + MetricAdapter)", "adapted_ft_ref", "adapted_base_ref", "adapted_ft_gain"),
        ("RAW     (E5, no adapter)     ", "raw_ft_ref",     "raw_base_ref",     "raw_ft_gain"),
    ]:
        ft_vals   = [getattr(s, ft_key)   for s in scores]
        base_vals = [getattr(s, base_key) for s in scores]
        gain_vals = [getattr(s, gain_key) for s in scores]
        improved  = sum(1 for g in gain_vals if g > 0.02)

        print(f"\n  {label}")
        print(f"  ft_vs_ref   mean={np.mean(ft_vals):.3f}  std={np.std(ft_vals):.3f}")
        print(f"  base_vs_ref mean={np.mean(base_vals):.3f}  std={np.std(base_vals):.3f}")
        print(f"  ft_gain     mean={np.mean(gain_vals):+.3f}  improved={improved}/{len(scores)}")

    # ── Per-category (adapted) ──
    cats = sorted({s.category for s in scores})
    if len(cats) > 1:
        print("\n  Per-category (adapted)")
        print(f"  {'Category':20} {'ft_ref':>8} {'base_ref':>9} {'gain':>8} {'n':>4}")
        print("  " + "─" * 54)
        for cat in cats:
            sub = [s for s in scores if s.category == cat]
            print(f"  {cat:20} "
                  f"{np.mean([s.adapted_ft_ref for s in sub]):8.3f} "
                  f"{np.mean([s.adapted_base_ref for s in sub]):9.3f} "
                  f"{np.mean([s.adapted_ft_gain for s in sub]):+8.3f} "
                  f"{len(sub):4d}")

    # ── Best / worst (adapted ft_gain) ──
    ranked = sorted(scores, key=lambda s: s.adapted_ft_gain, reverse=True)
    for label, subset in [("Best 3", ranked[:3]), ("Worst 3", ranked[-3:])]:
        print(f"\n  {label} (adapted ft_gain)")
        for s in subset:
            print(f"    [{s.id}] gain={s.adapted_ft_gain:+.3f}  "
                  f"ft={s.adapted_ft_ref:.3f}  base={s.adapted_base_ref:.3f}")

    print("\n" + "═" * 65)


def save_report(scores: list[Score], adapter_path: Path,
                entries_with_answers: list[dict] | None = None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── JSON ──
    payload = {
        "generated_at": ts,
        "adapter":      str(adapter_path),
        "n_samples":    len(scores),
        "aggregate": {
            "adapted": {
                "ft_vs_ref":   _agg([s.adapted_ft_ref   for s in scores]),
                "base_vs_ref": _agg([s.adapted_base_ref for s in scores]),
                "ft_gain":     _agg([s.adapted_ft_gain  for s in scores]),
            },
            "raw": {
                "ft_vs_ref":   _agg([s.raw_ft_ref   for s in scores]),
                "base_vs_ref": _agg([s.raw_base_ref for s in scores]),
                "ft_gain":     _agg([s.raw_ft_gain  for s in scores]),
            },
        },
        "scores": [asdict(s) for s in scores],
    }
    json_path = REPORT_DIR / f"adapted_embed_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  JSON  → {json_path}")

    # ── CSV ──
    csv_path = REPORT_DIR / f"adapted_embed_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(scores[0]).keys()))
        w.writeheader()
        w.writerows(asdict(s) for s in scores)
    print(f"  CSV   → {csv_path}")

    # ── Answers JSONL (if available) ──
    if entries_with_answers:
        ans_path = REPORT_DIR / f"adapted_answers_{ts}.jsonl"
        with open(ans_path, "w", encoding="utf-8") as f:
            for e in entries_with_answers:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  Answers → {ans_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Eval regllm answers vs ground truth using E5 + MetricAdapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", default=None,
                   help="LoRA adapter dir (default: latest run_*/final_model)")
    p.add_argument("--answers-file", default=None, metavar="JSONL",
                   help="Pre-generated answers JSONL (skips model generation).")
    p.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER),
                   help=f"MetricAdapter .pt weights (default: {DEFAULT_ADAPTER})")
    p.add_argument("--e5-local", default=None,
                   help="Local path to multilingual-e5-large (skips HF download).")
    p.add_argument("--train-adapter", action="store_true",
                   help="Build separation pairs from regllm QA data and fine-tune adapter.")
    p.add_argument("--adapter-epochs", type=int, default=30)
    p.add_argument("--sample", type=int, default=None,
                   help="Limit ground-truth entries to N (default: all 50).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--mock", action="store_true",
                   help="Use 3 hardcoded mock entries — no model, no GPU needed.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Mock mode ──────────────────────────────────────────────────────────────
    if args.mock:
        print("  [MOCK] Using 3 hardcoded entries — no generation or GPU needed.")
        entries_with_answers = [
            {
                "id": "mock_001", "category": "regulatory",
                "pregunta": "¿Qué establece el artículo 92 del CRR sobre capital?",
                "ref": "El artículo 92 del CRR establece los requisitos de fondos propios: CET1 ≥ 4.5%, Tier 1 ≥ 6%, Total capital ≥ 8%.",
                "ft_answer": "El artículo 92 del CRR establece que las entidades deben mantener un ratio CET1 mínimo del 4.5%, Tier 1 del 6% y capital total del 8%.",
                "base_answer": "El CRR establece requisitos de capital para los bancos europeos en términos de fondos propios.",
            },
            {
                "id": "mock_002", "category": "financial",
                "pregunta": "¿Cuál es el ratio de capital de Santander en 2023?",
                "ref": "El ratio de capital CET1 de Banco Santander en 2023 fue del 12.3% fully loaded.",
                "ft_answer": "El ratio CET1 de Santander fue aproximadamente del 12% en 2023.",
                "base_answer": "Santander mantiene ratios de capital por encima de los mínimos regulatorios.",
            },
            {
                "id": "mock_003", "category": "regulatory",
                "pregunta": "¿Cómo se calcula el ratio de apalancamiento?",
                "ref": "El ratio de apalancamiento = Tier 1 / Exposición total. Mínimo regulatorio: 3% según CRR.",
                "ft_answer": "El ratio de apalancamiento se calcula dividiendo el capital Tier 1 entre la exposición total. El mínimo es del 3%.",
                "base_answer": "El ratio de apalancamiento mide la relación entre capital y activos totales.",
            },
        ]
        gt_entries = entries_with_answers   # same structure

    # ── Load ground truth ──────────────────────────────────────────────────────
    else:
        gt_entries = load_ground_truth()
        if not gt_entries:
            print("ERROR: No ground truth found at", GT_PATH)
            sys.exit(1)
        if args.sample:
            gt_entries = gt_entries[: args.sample]
        print(f"  Ground truth: {len(gt_entries)} entries")

    # ── Load / train adapter ───────────────────────────────────────────────────
    print("\n  Loading E5 sentence-transformer …")
    st_model = _load_e5(args.e5_local)

    adapter_path = Path(args.adapter_path)

    if args.train_adapter:
        print("\n  Building separation pairs from regllm QA data …")
        ft_qa = load_finetuning_qa(max_items=500)
        all_entries = gt_entries + ft_qa
        pairs = build_separation_pairs(all_entries)
        print(f"  {len(pairs)} pairs built "
              f"({sum(1 for p in pairs if p['type']=='positive')} pos, "
              f"{sum(1 for p in pairs if p['type']=='negative')} neg, "
              f"{sum(1 for p in pairs if p['type']=='guardrail_mismatch')} guardrail)")
        adapter = train_adapter(
            pairs, st_model,
            start_from=DEFAULT_ADAPTER if DEFAULT_ADAPTER.exists() else None,
            epochs=args.adapter_epochs,
            save_path=REGLLM_ADAPTER,
        )
        adapter_path = REGLLM_ADAPTER
    else:
        if not adapter_path.exists():
            print(f"ERROR: Adapter not found at {adapter_path}")
            print("  Run with --train-adapter to build one, or check the path.")
            sys.exit(1)
        print(f"  Loading adapter: {adapter_path}")
        adapter = load_adapter(adapter_path)

    # ── Generate / load answers ────────────────────────────────────────────────
    if args.mock:
        pass  # entries_with_answers already set above

    elif args.answers_file:
        print(f"\n  Loading pre-generated answers from {args.answers_file} …")
        answers_map = {}
        for line in Path(args.answers_file).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                answers_map[rec["id"]] = rec

        entries_with_answers = []
        for e in gt_entries:
            rec = answers_map.get(e["id"])
            if rec is None:
                print(f"  WARNING: no answer for id={e['id']}, skipping.")
                continue
            entries_with_answers.append({
                **e,
                "ft_answer":   rec.get("ft_answer", ""),
                "base_answer": rec.get("base_answer", ""),
            })

    else:
        ckpt = args.checkpoint or str(find_latest_checkpoint() or "")
        if not ckpt:
            print("ERROR: No LoRA checkpoint found. Use --checkpoint or --answers-file.")
            sys.exit(1)
        print(f"\n  Generating answers with checkpoint: {ckpt}")
        entries_with_answers = generate_all_answers(
            gt_entries, ckpt, args.max_new_tokens
        )

    # ── Score ──────────────────────────────────────────────────────────────────
    scores = evaluate(entries_with_answers, st_model, adapter)

    # ── Report + save ──────────────────────────────────────────────────────────
    report(scores, adapter_path)
    save_report(scores, adapter_path,
                entries_with_answers=entries_with_answers if not args.mock else None)


if __name__ == "__main__":
    main()
