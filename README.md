# Sentiment-as-a-Service (sentiment-api)::

A production-ready, high-performance REST API for sentiment analysis, powered by **FastAPI** and **HuggingFace Transformers**.

This service provides an industrial-grade wrapper around the `cardiffnlp/twitter-roberta-base-sentiment-latest` model, offering multi-stage Docker builds and efficient batch processing.

## 🚀 Key Features::

* **Optimized Inference**: Uses a singleton model pattern with `lru_cache` to ensure the heavy transformer model is loaded exactly once.
* **Batch Processing**: Includes a dedicated `/analyze/batch` endpoint that processes up to 64 texts in a single model forward-pass for significantly higher throughput.
* **Production Readiness**:
    * **Multi-stage Docker Build**: Pre-downloads model weights during the build phase, enabling full offline deployment and faster container startup.
    * **Strict Validation**: Pydantic models enforce character limits (5,000 per text) and validate against blank inputs.
    * **Health Checks**: Integrated liveness probes for Docker/Kubernetes orchestration.

## 🛠 Tech Stack::

* **Framework**: FastAPI
* **Machine Learning**: HuggingFace Transformers, PyTorch
* **Validation**: Pydantic v2
* **Server**: Uvicorn
* **Deployment**: Docker & Docker Compose

## 📋 API Endpoints::

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness probe; checks if the model is loaded and service is healthy. |
| `GET` | `/model/info` | Returns metadata about the current model and its sentiment labels. |
| `POST` | `/analyze` | Analyze a single string; returns label, confidence score, and full distribution. |
| `POST` | `/analyze/batch` | Analyze up to 64 strings simultaneously for maximum efficiency. |

## 🐳 Docker Deployment

The service is designed to be deployed via Docker Compose. It includes a 2GB memory limit and persistent volume mounting for the HuggingFace model cache.

```bash
# Build and start the service
docker-compose up --build
