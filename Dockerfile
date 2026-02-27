# ── Base: plain Python (CPU-only) ─────────────────────────────────────────────
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch CPU-only (separate layer — rarely changes, good cache hit) ─────────
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# ── Application dependencies ───────────────────────────────────────────────────
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

# ── Application source ────────────────────────────────────────────────────────
# models/finetuned, vector_db are excluded by .dockerignore — mount as volumes
COPY app.py .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

CMD ["python", "app.py", "--backend", "local", "--port", "7860"]
