# ── Base: CUDA 12.4 runtime (compatible with PyTorch cu124 stable wheels) ──────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        git \
        build-essential \
        curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch (separate layer — rarely changes, good cache hit) ──────────────────
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

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
