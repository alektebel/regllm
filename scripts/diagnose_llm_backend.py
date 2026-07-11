#!/usr/bin/env python3
"""Diagnose why LocalLLMClient isn't picking the backend you expect.

Runs the EXACT detection code api/main.py's /health endpoint uses, and
prints every intermediate check so you can see precisely which one fails —
instead of guessing between "wrong path", "llama-cpp-python not importable
in this Python", "env var not actually set in this shell", "config.yaml
not found/parsed", or "auto mode picked something else first".

Run it in the SAME terminal/environment you intend to start uvicorn from —
that's the whole point: it must see the same env vars and the same Python
interpreter (a very common cause of "it's not detecting it" on Windows is
`pip install llama-cpp-python` landing in a different Python than the one
`uvicorn` runs from).

Primarily focused on the standalone GGUF backend (the numbered [1]-[7]
checks), with a shorter [B1]-[B4] section at the end covering the same
class of problem for Amazon Bedrock (boto3 installed?, credentials visible
to this process?, model/region resolved?).

    python scripts/diagnose_llm_backend.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _bool_str(b: bool) -> str:
    return "YES" if b else "NO  <-- problem"


def main() -> None:
    print(f"Python interpreter : {sys.executable}")
    print(f"Working directory  : {Path.cwd()}")
    print()

    # 1. llama-cpp-python importable in *this* interpreter?
    try:
        import llama_cpp  # noqa: F401
        print(f"[1] llama-cpp-python importable : YES ({llama_cpp.__file__})")
    except ImportError as e:
        print(f"[1] llama-cpp-python importable : NO  <-- problem")
        print(f"      ImportError: {e}")
        print(f"      Fix: pip install llama-cpp-python   (using THIS interpreter:")
        print(f"           {sys.executable} -m pip install llama-cpp-python)")
    print()

    from src.knowledge.llm_client import LocalLLMClient, _Llama, _LLM_CFG, _yaml

    # 2. Was config.yaml found and parsed at all?
    cfg_path_root = ROOT / "config.yaml"
    print(f"[2] config.yaml found at {cfg_path_root} : {_bool_str(cfg_path_root.is_file())}")
    print(f"      pyyaml installed : {_bool_str(_yaml is not None)}")
    print(f"      llm._LLM_CFG as parsed : {_LLM_CFG}")
    print()

    # 3. Raw env vars, exactly as os.getenv sees them in THIS process
    print("[3] Environment variables in this process:")
    for var in ("REGLLM_LLM", "GGUF_MODEL_PATH", "GGUF_N_CTX", "GGUF_N_GPU_LAYERS"):
        print(f"      {var} = {os.getenv(var)!r}")
    print()

    # 4. What the client actually resolved from env + config.yaml
    client = LocalLLMClient()
    print("[4] LocalLLMClient resolved config:")
    print(f"      prefer (backend selector) = {client.prefer!r}")
    print(f"      gguf_model_path           = {client.gguf_model_path!r}")
    print(f"      gguf_n_ctx                = {client.gguf_n_ctx!r}")
    print(f"      gguf_n_gpu_layers         = {client.gguf_n_gpu_layers!r}")
    if client.prefer not in ("auto", "gguf"):
        print(f"      <-- problem: prefer is {client.prefer!r}, not 'auto' or 'gguf', so")
        print(f"          the GGUF backend is never even attempted. Check REGLLM_LLM.")
    print()

    # 5. Does the path actually resolve to a file, from here?
    print("[5] GGUF weight file check:")
    if not client.gguf_model_path:
        print("      <-- problem: gguf_model_path is empty (GGUF_MODEL_PATH not set")
        print("          in this process, and config.yaml's gguf_model_path is unset).")
    else:
        p = Path(client.gguf_model_path).expanduser()
        print(f"      resolved path : {p}")
        print(f"      exists        : {_bool_str(p.exists())}")
        print(f"      is a file     : {_bool_str(p.is_file())}")
        if p.exists() and not p.is_file():
            print("      <-- problem: this path is a DIRECTORY, not the .gguf file itself.")
        if not p.exists():
            print("      <-- problem: no file at this exact path. On Windows, check for a")
            print("          stray trailing backslash-escape in config.yaml (use / not \\),")
            print("          or a typo, or that the drive/path is spelled exactly as ls/dir shows.")
    print()

    # 6. Would litert/ollama get picked first in "auto" mode, masking gguf?
    if client.prefer == "auto":
        print("[6] 'auto' mode tries litert -> ollama -> gguf, in that order:")
        litert_ok = client._probe_litert()
        print(f"      litert reachable at {client.litert_url} : {_bool_str(litert_ok)}"
              f"{'  <-- would be picked BEFORE gguf' if litert_ok else ''}")
        ollama_ok = client._probe_ollama()
        ollama_model_ok = ollama_ok and client._ollama_has_model(client.ollama_model)
        print(f"      ollama reachable at {client.ollama_url}   : {_bool_str(ollama_ok)}")
        if ollama_ok:
            print(f"      ollama has model {client.ollama_model!r} : {_bool_str(ollama_model_ok)}"
                  f"{'  <-- would be picked BEFORE gguf' if ollama_model_ok else ''}")
        print("      If either of those is YES, set REGLLM_LLM=gguf explicitly (not 'auto')")
        print("      to force GGUF regardless of what else is running.")
        print()

    # 7. The actual verdict
    print("=" * 60)
    backend = client.detect_backend()
    print(f"detect_backend() -> {backend!r}")
    print("=" * 60)
    if backend != "gguf":
        print("\nNot using GGUF. Re-check the numbered problem(s) flagged above.")
    else:
        print("\nGGUF backend correctly detected. If /health still shows something")
        print("else, restart uvicorn — it may be running with stale env vars from")
        print("before you set them.")

    print()
    _check_bedrock(client)


def _check_bedrock(client: "LocalLLMClient") -> None:
    """Bedrock has the same class of 'why isn't it detecting it' failure
    modes as GGUF (dependency missing, credentials not visible to this
    process, wrong region/model), so this mirrors the same numbered-check
    approach. Runs even if you're not using Bedrock — it's cheap and makes
    no network calls."""
    from src.knowledge.llm_client import _boto3

    print("=" * 60)
    print("Bonus: Amazon Bedrock backend (REGLLM_LLM=bedrock)")
    print("=" * 60)
    print(f"[B1] boto3 importable : {_bool_str(_boto3 is not None)}")
    if _boto3 is None:
        print("      Fix: pip install boto3")
        return
    print(f"[B2] bedrock_model_id (BEDROCK_MODEL_ID / config.yaml) = {client.bedrock_model_id!r}")
    print(f"[B3] bedrock_region   (BEDROCK_REGION / config.yaml)   = {client.bedrock_region!r}")
    try:
        creds = _boto3.Session().get_credentials()
        if creds is None:
            print("[B4] AWS credentials visible to this process : NO  <-- problem")
            print("      Fix: set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, run `aws configure`,")
            print("      or (in ECS/EC2) attach an IAM role/instance profile with")
            print("      bedrock:InvokeModel on the model ARN.")
        else:
            frozen = creds.get_frozen_credentials()
            masked = (frozen.access_key or "")[:4] + "…" if frozen.access_key else "?"
            print(f"[B4] AWS credentials visible to this process : YES "
                  f"(access key {masked}, source: {creds.method})")
    except Exception as e:
        print(f"[B4] AWS credentials visible to this process : could not check ({e})")
    print()
    print("Note: 'auto' mode never picks Bedrock — it only tries")
    print("litert -> ollama -> gguf -> stub, in that order. Set REGLLM_LLM=bedrock")
    print("explicitly to use it (this is by design: Bedrock is a paid managed")
    print("service, so it should never be silently auto-selected).")


if __name__ == "__main__":
    main()
