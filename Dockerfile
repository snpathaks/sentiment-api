


FROM python:3.11-slim AS builder


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip wheel


COPY requirements.txt .


RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


ARG MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest


ENV TRANSFORMERS_CACHE=/model-cache \
    HF_HOME=/model-cache

RUN python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_name = os.environ["MODEL_NAME"]
cache_dir  = os.environ["HF_HOME"]

print(f"Downloading tokenizer: {model_name}")
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"Downloading model weights: {model_name}")
AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)

print("Download complete.")
EOF


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
   
    TRANSFORMERS_CACHE=/model-cache \
    HF_HOME=/model-cache \
    
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1


RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app


COPY --from=builder /install /usr/local


COPY --from=builder /model-cache /model-cache


COPY app/ ./app/


RUN chown -R appuser:appuser /app /model-cache

USER appuser


EXPOSE 8000


HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"


CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]
