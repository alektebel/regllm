FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch — keeps the image small and avoids CUDA dependencies
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

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
