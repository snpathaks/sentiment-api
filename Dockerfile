FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONMALLOC=malloc \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --prefix=/install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

ARG MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest

ENV TRANSFORMERS_CACHE=/model-cache \
    HF_HOME=/model-cache \
    MODEL_NAME=${MODEL_NAME} \
    PYTHONPATH=/install/lib/python3.11/site-packages

RUN python - <<'EOF'
import os, sys
sys.path.insert(0, "/install/lib/python3.11/site-packages")

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_NAME"],
    cache_dir=os.environ["HF_HOME"],
    ignore_patterns=[
        "*.msgpack",         # Flax weights
        "flax_model*",
        "tf_model*",         # TensorFlow weights
        "rust_model*",       # Candle weights
        "pytorch_model.bin", # legacy pickle — safetensors replaces this
        "optimizer*",
        "training_args*",
        "*.ot",
    ],
)
print("Download complete.", flush=True)
EOF

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/model-cache \
    HF_HOME=/model-cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONMALLOC=malloc \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN useradd --no-log-init --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install     /usr/local
COPY --from=builder /model-cache /model-cache
COPY app/                        ./app/

RUN chown -R appuser:appuser /app /model-cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "warning"]