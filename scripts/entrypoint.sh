#!/usr/bin/env bash
# Container entrypoint for RegLLM.
# Handles LoRA adapter download (only for local backend), then starts the app.
set -euo pipefail

BACKEND="${REGLLM_BACKEND:-groq}"
ADAPTER_DIR="${ADAPTER_DIR:-/app/models/finetuned/current}"

echo "[entrypoint] Starting RegLLM (backend=${BACKEND})"

# ── Pull LoRA adapter weights (only needed for local backend) ─────────────────
if [ "$BACKEND" = "local" ]; then
    echo "[entrypoint] Backend=local — fetching LoRA adapter..."

    # Strategy 1: MLflow Model Registry (preferred when MLflow server is available)
    if [ -n "${MLFLOW_TRACKING_URI:-}" ] && [ -n "${MLFLOW_MODEL_NAME:-}" ]; then
        echo "[entrypoint] Pulling from MLflow registry: ${MLFLOW_MODEL_NAME}"
        python - <<'PYEOF'
import os, sys
from pathlib import Path

try:
    import mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model_name  = os.environ.get("MLFLOW_MODEL_NAME", "regllm-lora-adapter")
    model_stage = os.environ.get("MLFLOW_MODEL_STAGE", "Production")
    dest        = Path(os.environ.get("ADAPTER_DIR", "/app/models/finetuned/current"))
    dest.mkdir(parents=True, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(
        f"models:/{model_name}/{model_stage}",
        dst_path=str(dest),
    )
    print(f"[entrypoint] Adapter downloaded to {local_path}")
except Exception as e:
    print(f"[entrypoint] MLflow pull failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    # Strategy 2: S3 direct sync (fallback / CI use)
    elif [ -n "${MODEL_S3_URI:-}" ]; then
        echo "[entrypoint] Pulling from S3: ${MODEL_S3_URI}"
        mkdir -p "${ADAPTER_DIR}"
        aws s3 sync "${MODEL_S3_URI}" "${ADAPTER_DIR}" --no-progress
        echo "[entrypoint] Adapter synced from S3 to ${ADAPTER_DIR}"

    # Strategy 3: Volume mount or pre-baked path (local dev / docker-compose)
    else
        echo "[entrypoint] No MLflow/S3 config — expecting adapter via volume mount."
    fi
fi

# ── Start the application ─────────────────────────────────────────────────────
echo "[entrypoint] Launching app.py --backend ${BACKEND} $*"
exec python app.py --backend "${BACKEND}" "$@"
