# =============================================================================
# Dockerfile – Sentiment-as-a-Service
# =============================================================================
# Build strategy
# --------------
# Stage 1 (builder) – install all Python deps into an isolated prefix so that
#   only the compiled wheels land in the final image (no build tools, no pip
#   cache, no compiler).
# Stage 2 (runtime) – copy the pre-built site-packages and app source into a
#   slim base image.
#
# Layer-caching order (slowest → fastest to invalidate)
# -------------------------------------------------------
#   1. Base OS + system libs         (changes almost never)
#   2. requirements.txt copy         (changes rarely)
#   3. pip install                   (re-runs only when requirements change)
#   4. HuggingFace model download    (re-runs only when MODEL_NAME changes)
#   5. Application source code       (changes frequently – last layer)
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Keep Python output unbuffered and prevent .pyc files cluttering the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System dependencies required to compile some Python wheels (e.g. tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/wheel in the builder only
RUN pip install --upgrade pip wheel

# ── Layer-cache anchor: copy requirements first ───────────────────────────────
# Docker will reuse the next RUN layer as long as requirements.txt is unchanged.
COPY requirements.txt .

# Install all dependencies into a dedicated prefix so we can COPY them cleanly
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Pre-download the HuggingFace model into the image ────────────────────────
# Baking the weights into the image means:
#   • Zero cold-start latency in production (no network call on first request)
#   • Fully air-gapped deployments work out-of-the-box
# The ARG lets CI override the model without changing the Dockerfile.
ARG MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest

# HuggingFace caches models under TRANSFORMERS_CACHE (legacy) and
# HF_HOME (new standard).  Set both so any version of the library finds them.
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

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Point both env vars at the same baked-in cache directory
    TRANSFORMERS_CACHE=/model-cache \
    HF_HOME=/model-cache \
    # Tell HuggingFace to never attempt a network call at inference time
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# Non-root user for security hardening
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy compiled Python packages from builder
COPY --from=builder /install /usr/local

# Copy the pre-downloaded model weights from builder
COPY --from=builder /model-cache /model-cache

# ── Application source (most volatile layer – copy last) ─────────────────────
COPY app/ ./app/

# Transfer ownership so the non-root user can write temp files if needed
RUN chown -R appuser:appuser /app /model-cache

USER appuser

# Expose the port Uvicorn will bind to
EXPOSE 8000

# Healthcheck – Docker will mark the container unhealthy if /health stops
# returning 200, enabling automatic restarts via restart policies.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command – single worker is fine for CPU inference; add --workers N
# (or switch to Gunicorn) when scaling horizontally behind a load-balancer.
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]