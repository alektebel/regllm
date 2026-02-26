#!/usr/bin/env python3
"""
RegLLM — Unified Gradio UI  (local / groq / ollama backends)

Usage:
    python app.py --backend local   [--adapter PATH] [--port N] [--share]
    python app.py --backend groq    [--model MODEL]  [--port N] [--share]
    python app.py --backend ollama  [--model MODEL]  [--port N] [--share]

Backend-specific imports are deferred inside _init_* functions so that
this module is importable without torch / groq / ollama installed.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Generator

import gradio as gr
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RegulatoryRAGSystem
from src.citation_rag import CitationRAG
from src.chat_engine import ChatEngine, CSS, EXAMPLES, _content_to_text
from src.db import init_db, run_db_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Globals ──────────────────────────────────────────────────────────────────

engine: ChatEngine = None
_backend: str = None

# Local backend
_local_model = None
_local_tokenizer = None

# Groq backend
_groq_client = None
_groq_model = "llama-3.3-70b-versatile"

# Ollama backend
_ollama_model = "regllm"
_ollama_host = "http://localhost:11434"


# ─── Backend: local ───────────────────────────────────────────────────────────

def _find_latest_adapter() -> str:
    root = Path(__file__).parent / "models/finetuned"
    candidates = sorted(
        [
            p for p in root.glob("run_*/final_model")
            if (p / "adapter_model.safetensors").exists()
        ]
        + [
            p for p in root.glob("run_*/checkpoint-*")
            if (p / "adapter_model.safetensors").exists()
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No trained adapter found under models/finetuned/. "
            "Run: python scripts/train_combined.py"
        )
    return str(candidates[-1].relative_to(Path(__file__).parent))


def _init_local(adapter: str) -> None:
    global _local_model, _local_tokenizer

    import torch

    # Patch caching_allocator_warmup before importing transformers
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **kw: None

    # Patch peft/accelerate: get_balanced_memory fails with non-hashable set
    import accelerate.utils.modeling as _am
    _orig_gbm = _am.get_balanced_memory

    def _patched_gbm(model, max_memory=None, no_split_module_classes=None, **kwargs):
        if isinstance(no_split_module_classes, (set, list)):
            no_split_module_classes = list(no_split_module_classes)
        return _orig_gbm(
            model, max_memory=max_memory,
            no_split_module_classes=no_split_module_classes, **kwargs
        )

    _am.get_balanced_memory = _patched_gbm

    import peft.peft_model as _pm
    import peft.utils.other as _po
    _pm.get_balanced_memory = _patched_gbm
    try:
        _po.get_balanced_memory = _patched_gbm
    except AttributeError:
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = Path(adapter)
    if not adapter_path.is_absolute():
        adapter_path = Path(__file__).parent / adapter_path

    logger.info(f"Loading tokenizer from {adapter_path}")
    _local_tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    logger.info(f"Loading base model: {BASE_MODEL} (4-bit nf4 QLoRA)")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info(f"Applying LoRA adapter from {adapter_path}")
    _local_model = PeftModel.from_pretrained(base, str(adapter_path))
    _local_model.eval()
    logger.info("Local model ready")


def _generate_local(messages: list) -> str:
    """Run QLoRA inference; return raw model output string."""
    import torch

    text = _local_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _local_tokenizer(text, return_tensors="pt").to(_local_model.device)
    with torch.no_grad():
        output_ids = _local_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True,
            pad_token_id=_local_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── Backend: groq ────────────────────────────────────────────────────────────

def _init_groq(model: str = None) -> None:
    global _groq_client, _groq_model

    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to .env or export GROQ_API_KEY=gsk_..."
        )
    if model:
        _groq_model = model
    _groq_client = Groq(api_key=api_key)
    logger.info(f"Groq client ready — model: {_groq_model}")


def _generate_groq(messages: list) -> str:
    response = _groq_client.chat.completions.create(
        model=_groq_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── Backend: ollama ──────────────────────────────────────────────────────────

def _init_ollama(model: str = None) -> None:
    global _ollama_model, _ollama_host

    import ollama as _ollama_lib

    if model:
        _ollama_model = model
    _ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        client = _ollama_lib.Client(host=_ollama_host)
        models = [m["name"] for m in client.list()["models"]]
        model_names = [m.split(":")[0] for m in models]
        if _ollama_model not in model_names:
            raise RuntimeError(
                f"Model '{_ollama_model}' not found in Ollama.\n"
                f"Available: {models}\n"
                f"Build it: ./scripts/build_ollama_model.sh"
            )
        logger.info(f"Ollama model '{_ollama_model}' ready")
    except _ollama_lib.ResponseError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {_ollama_host}: {e}\n"
            "Start it with: ollama serve"
        ) from e


def _generate_ollama_stream(messages: list) -> Generator[str, None, None]:
    """Yield partial text chunks from Ollama streaming response."""
    import ollama as _ollama_lib

    client = _ollama_lib.Client(host=_ollama_host)
    stream = client.chat(
        model=_ollama_model,
        messages=messages,
        stream=True,
        options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    )
    for chunk in stream:
        yield chunk["message"]["content"]


# ─── Shared init ──────────────────────────────────────────────────────────────

def init(backend: str, **kwargs) -> int:
    """
    Initialize the selected backend + shared RAG + DB.
    Returns doc_count.
    """
    global engine, _backend

    _backend = backend

    logger.info("Initializing RAG system (CPU embedder)...")
    rag_system = RegulatoryRAGSystem()
    doc_count = rag_system.collection.count()
    logger.info(f"RAG ready — {doc_count} documents indexed")

    try:
        citation_rag = CitationRAG()
        logger.info(f"Citation RAG ready — {citation_rag.count()} citation vectors")
    except Exception as e:
        logger.warning(f"Citation RAG unavailable: {e}")
        citation_rag = None

    try:
        run_db_sync(init_db())
    except Exception as e:
        logger.warning(f"DB init skipped: {e}")

    if backend == "local":
        adapter = kwargs.get("adapter") or _find_latest_adapter()
        _init_local(adapter)
    elif backend == "groq":
        _init_groq(model=kwargs.get("model"))
    elif backend == "ollama":
        _init_ollama(model=kwargs.get("model"))
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose: local, groq, ollama")

    engine = ChatEngine(rag_system, citation_rag=citation_rag, db_source=f"app_{backend}")
    return doc_count


# ─── Query handlers ───────────────────────────────────────────────────────────

def ask(
    question: str,
    history: list,
    n_sources: int = 5,
    hybrid: bool = True,
) -> tuple:
    """Non-streaming handler for local and groq backends."""
    if not question.strip():
        return "", history

    if not engine:
        return "Sistema no inicializado.", history

    t0 = time.time()

    context, _ = engine.build_context(question, n_sources=n_sources, hybrid=hybrid)
    messages = engine.build_messages(question, context, history)

    try:
        if _backend == "local":
            raw = _generate_local(messages)
        elif _backend == "groq":
            raw = _generate_groq(messages)
        else:
            raw = f"Backend {_backend!r} does not support non-streaming ask()."
    except Exception as e:
        logger.error(f"Generation error ({_backend}): {e}", exc_info=True)
        raw = f"Error durante la generación: {str(e)}"

    parsed = engine.parse_response(raw)
    parsed = engine.enrich_references(parsed, question)
    rendered = engine.render_message(parsed)

    latency_ms = int((time.time() - t0) * 1000)
    engine.log(question, rendered, latency_ms=latency_ms)

    # Normalise history to dict format (handles both tuple and dict styles)
    normalised = [
        {
            "role": t["role"] if isinstance(t, dict) else t[0],
            "content": _content_to_text(t["content"] if isinstance(t, dict) else t[1]),
        }
        for t in history
    ]
    normalised.append({"role": "user", "content": question})
    normalised.append({"role": "assistant", "content": rendered})
    return "", normalised


def ask_stream(
    question: str,
    history: list,
    n_sources: int = 5,
    hybrid: bool = True,
) -> Generator:
    """Streaming handler for ollama backend."""
    if not question.strip():
        yield "", history
        return

    if not engine:
        yield "Sistema no inicializado. Reinicia la app.", history
        return

    t0 = time.time()

    context, _ = engine.build_context(question, n_sources=n_sources, hybrid=hybrid)
    messages = engine.build_messages(question, context, history)

    partial = ""
    try:
        for chunk in _generate_ollama_stream(messages):
            partial += chunk
            yield "", [
                *history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": partial},
            ]
    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        partial = f"Error al consultar el modelo: {str(e)}"
        yield "", [
            *history,
            {"role": "user", "content": question},
            {"role": "assistant", "content": partial},
        ]
        return

    parsed = engine.parse_response(partial)
    parsed = engine.enrich_references(parsed, question)
    rendered = engine.render_message(parsed)

    latency_ms = int((time.time() - t0) * 1000)
    engine.log(question, rendered, latency_ms=latency_ms)

    yield "", [
        *history,
        {"role": "user", "content": question},
        {"role": "assistant", "content": rendered},
    ]


# ─── UI ───────────────────────────────────────────────────────────────────────

def build_ui(backend: str, doc_count: int, label: str) -> gr.Blocks:
    title_map = {
        "local": "RegLLM — Local",
        "groq": "RegLLM — Groq",
        "ollama": "RegLLM — Ollama",
    }
    fn = ask_stream if backend == "ollama" else ask

    with gr.Blocks(title=title_map.get(backend, "RegLLM")) as demo:

        gr.HTML(f"""
        <div class="regllm-header">
            <div style="flex:1">
                <h1>RegLLM</h1>
                <p class="subtitle">Asistente de regulación bancaria · EBA · CRR/CRD · Basilea III/IV</p>
            </div>
            <span class="badge">{label}</span>
            <span class="badge badge-green">{doc_count} docs</span>
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes="chatbot-wrap"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="",
                    height=540,
                    render_markdown=True,
                    show_label=False,
                )
                with gr.Row(elem_classes="input-row"):
                    txt = gr.Textbox(
                        placeholder="Escribe tu pregunta sobre regulación bancaria...",
                        show_label=False,
                        scale=5,
                        lines=1,
                        container=False,
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=1, elem_id="send-btn")
                gr.Examples(examples=EXAMPLES, inputs=txt)

            with gr.Column(scale=1, min_width=220):
                with gr.Accordion("Configuracion", open=False, elem_classes="settings-panel"):
                    n_sources = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Fragmentos RAG",
                    )
                    hybrid_search = gr.Checkbox(
                        value=True, label="Busqueda hibrida (semantica + BM25)"
                    )
                    clear_btn = gr.Button("Limpiar conversacion", variant="secondary", size="sm")

        send_btn.click(
            fn=fn,
            inputs=[txt, chatbot, n_sources, hybrid_search],
            outputs=[txt, chatbot],
        )
        txt.submit(
            fn=fn,
            inputs=[txt, chatbot, n_sources, hybrid_search],
            outputs=[txt, chatbot],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, txt])

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RegLLM — Unified Gradio UI")
    parser.add_argument(
        "--backend", choices=["local", "groq", "ollama"], default="groq",
        help="Inference backend: local (QLoRA), groq (API), ollama (GGUF)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    # Backend-specific flags
    parser.add_argument(
        "--adapter", default=None,
        help="[local] Path to LoRA adapter (relative to project root)",
    )
    parser.add_argument(
        "--model", default=None,
        help="[groq|ollama] Model name override",
    )
    args = parser.parse_args()

    doc_count = init(args.backend, adapter=args.adapter, model=args.model)

    if args.backend == "local":
        lbl = f"Qwen2.5-7B · {args.adapter or 'auto'}"
    elif args.backend == "groq":
        lbl = f"Groq · {_groq_model}"
    else:
        lbl = f"Ollama · {_ollama_model}"

    demo = build_ui(args.backend, doc_count, lbl)

    print(f"\n{'='*60}")
    print(f"  RegLLM — backend: {args.backend}")
    print(f"  Label   : {lbl}")
    print(f"  Docs    : {doc_count}")
    print(f"  Local   : http://localhost:{args.port}")
    print(f"{'='*60}\n")

    if args.backend == "ollama":
        demo.queue()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        prevent_thread_lock=False,
        css=CSS,
    )
