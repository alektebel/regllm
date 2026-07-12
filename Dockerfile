FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    REGLLM_ROUTERS=dqc

# Slim, DQC-only dependency set (no torch/kuzu/chromadb/sklearn/umap — those
# back the SAS diff explainer and knowledge-graph builder, neither of which
# is reachable with REGLLM_ROUTERS=dqc, set below and by the deploy infra).
# See requirements-dqc.txt and docs/DEPLOYMENT.md.
COPY requirements-dqc.txt .
RUN pip install -r requirements-dqc.txt

# Project source — only what the DQC API needs
COPY src/ ./src/
COPY api/ ./api/
COPY training/__init__.py ./training/__init__.py
COPY training/dq/ ./training/dq/
COPY config.yaml ./config.yaml

# Data directory (bind-mount in production for persistence)
COPY data/docs/ ./data/docs/
COPY data/regulation/ ./data/regulation/
RUN mkdir -p data/dq data/sas data/sessions data/knowledge data/samples

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
