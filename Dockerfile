FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch — keeps the image small and avoids CUDA dependencies
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Project source. Data is also copied so that running the image WITHOUT a
# bind mount still produces a working demo; if the host bind-mounts ./data
# the bundled copy is harmlessly shadowed.
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/ ./data/

COPY docker/api-entrypoint.sh /usr/local/bin/api-entrypoint.sh
RUN chmod +x /usr/local/bin/api-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/api-entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
