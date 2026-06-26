#!/usr/bin/env python3
"""Chat with the KB sleep fine-tuned adapter (PEFT — no Ollama export needed).

Usage:
    .venv/bin/python autoresearch/chat_kb.py --mode organize --prompt "..."
    .venv/bin/python autoresearch/chat_kb.py --interactive
    .venv/bin/python autoresearch/chat_kb.py --serve --port 8766
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ADAPTER = PROJECT_ROOT / "autoresearch/output/kb-sleep-sft"

SYSTEM_SLEEP_LEARN = """Eres RegLLM en MODO SUEÑO. Durante el sueño, procesas las sesiones del día para decidir qué merece ser recordado.

## Tareas
1. Revisa la sesión y extrae INSIGHTS: lecciones aprendidas, bugs, descubrimientos.
2. Decide QUÉ merece persistirse (no todo es importante — filtra ruido).
3. Decide QUÉ NO merece persistirse (información trivial, obvia, redundante).
4. Para cada insight: usa save_insight con id, label, summary, tags, priority, fields, articles.
5. Para cada quirk: usa save_quirk con id, label, description, severity.
6. Para feedback: usa save_feedback con type, original_claim, corrected_understanding.

Criterios de granularidad:
- NO guardes cada detalle. Un bug es UN insight, no múltiples.
- Prioriza: bugs y feedback > detalles técnicos > observaciones triviales
- Si algo ya existe en el grafo, omítelo (el sistema MERGE automáticamente).

Responde con una lista de acciones concretas (una por línea, prefijo '-')."""

SYSTEM_SLEEP_ORGANIZE = """Eres RegLLM en MODO SUEÑO — fase de ORGANIZACIÓN. Revisas el grafo de conocimiento existente para mantenerlo limpio y coherente.

## Tareas
1. DEDUPLICACIÓN: Identifica nodos duplicados (mismo insight expresado distinto). Fusiona usando merge_nodes.
2. TAGGING: Asigna tags consistentes a nodos sin etiquetar. Usa add_tags.
3. CROSS-LINKING: Conecta insights relacionados que no estén linkeados. Usa link_nodes.
4. CONFLICTOS: Identifica insights contradictorios. Señálalos con flag_conflict.
5. PRIORIDADES: Ajusta prioridades de insights sub/sobre-valorados.
6. LIMPIEZA: Elimina nodos huérfanos o irrelevantes con delete_node.

Responde con una lista de acciones concretas (una por línea, prefijo '-')."""

SYSTEM_SLEEP_CONSOLIDATE = """Eres RegLLM en MODO SUEÑO — fase de CONSOLIDACIÓN (EverMemOS-style).

## Tareas
1. DETECCIÓN: Identifica clusters de insights relacionados (mismo campo/tema, distintos segmentos).
2. ABSTRACCIÓN: Crea un Concept con create_concept(label, definition, covers=[ids], links_to=[campos]).
3. COMPRESIÓN: Solo consolida cuando ≥2 insights se comprimen — evita sobre-abstracción.
4. ENLACE: Usa link_to_concept para insights nuevos que encajen en un concepto existente.

Responde con una lista de acciones concretas (una por línea, prefijo '-')."""

SYSTEM_SLEEP_RETRIEVE = """Eres RegLLM en MODO SUEÑO — fase de RETRIEVAL. Durante el sueño, también puedes recuperar información del grafo para consolidar conocimiento.

## Tareas
1. RECUPERACIÓN PRECISA: Cuando te pregunten por un hecho específico, recupera EXACTAMENTE ese hecho, no uno similar.
2. Si hay múltiples experiencias similares con valores diferentes, discrimina por el criterio correcto (segmento, fecha, campo, etc.).
3. Si la respuesta exacta no existe en el grafo, dilo explícitamente.
4. Usa search_experience con queries precisas para encontrar lo que buscas.
5. Si hay feedback que corrige un insight anterior, usa la versión corregida, no la original.

Responde con acciones concretas o la respuesta recuperada."""

MODES = {
    "learn": SYSTEM_SLEEP_LEARN,
    "organize": SYSTEM_SLEEP_ORGANIZE,
    "consolidate": SYSTEM_SLEEP_CONSOLIDATE,
    "retrieve": SYSTEM_SLEEP_RETRIEVE,
}

KB_TOOLS = [
    "search_experience", "save_insight", "save_feedback", "save_quirk",
    "merge_nodes", "add_tags", "link_nodes", "flag_conflict", "delete_node",
    "create_concept", "link_to_concept",
]

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: str = Field(default="organize")
    max_new_tokens: int = 512
    temperature: float = 0.1

class ChatResponse(BaseModel):
    mode: str
    response: str
    adapter: str

_model = None
_tokenizer = None


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
    except Exception:
        pass


def load_model(adapter: Path = DEFAULT_ADAPTER):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _free_gpu()
    from unsloth import FastLanguageModel

    adapter = Path(adapter)
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=1536,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    _model, _tokenizer = model, tokenizer
    return model, tokenizer


def _strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|redacted_im_end\|>", "", text)
    return text.strip()


def _inject_no_think(messages: list[dict]) -> list[dict]:
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {**out[i], "content": "/no_think\n" + out[i]["content"]}
            break
    return out


def chat(
    prompt: str,
    mode: str = "organize",
    adapter: Path = DEFAULT_ADAPTER,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    model, tokenizer = load_model(adapter)
    system = MODES.get(mode, MODES["organize"])
    messages = _inject_no_think([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ])
    # Sleep-phase training uses plain-text action lists, not tool_call blocks.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_cache=True,
        do_sample=temperature > 0,
    )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return _strip_think_tags(raw)


def serve(port: int, adapter: Path) -> None:
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="RegLLM KB Sleep Chat", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "adapter": str(adapter)}

    @app.post("/chat", response_model=ChatResponse)
    def chat_endpoint(payload: ChatRequest):
        text = chat(
            payload.prompt,
            mode=payload.mode,
            adapter=adapter,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        )
        return ChatResponse(mode=payload.mode, response=text, adapter=str(adapter))

    print(f"Loading adapter {adapter}...", flush=True)
    load_model(adapter)
    print(f"KB chat server ready → http://localhost:{port}", flush=True)
    print(f"  POST /chat  {{\"prompt\": \"...\", \"mode\": \"organize|learn|retrieve\"}}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def interactive(adapter: Path, mode: str) -> None:
    print(f"KB sleep chat — mode={mode}  adapter={adapter}")
    print("Type 'quit' to exit, '/mode learn|organize|retrieve' to switch.\n")
    load_model(adapter)
    current_mode = mode
    while True:
        try:
            user = input(f"[{current_mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            break
        if user.startswith("/mode "):
            m = user.split(maxsplit=1)[1].strip()
            if m in MODES:
                current_mode = m
                print(f"  → mode={current_mode}")
            continue
        print(chat(user, mode=current_mode, adapter=adapter), flush=True)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with KB sleep PEFT adapter")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--mode", choices=list(MODES), default="organize")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    adapter = Path(args.adapter)
    if args.serve:
        serve(args.port, adapter)
    elif args.interactive:
        interactive(adapter, args.mode)
    elif args.prompt:
        print(chat(args.prompt, mode=args.mode, adapter=adapter))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
