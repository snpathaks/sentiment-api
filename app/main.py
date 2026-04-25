"""
main.py
-------
FastAPI application entry-point for the Sentiment-as-a-Service API.

Endpoints
---------
GET  /                  → health / welcome
GET  /health            → liveness probe
GET  /model/info        → metadata about the loaded model
POST /analyze           → single-text sentiment analysis
POST /analyze/batch     → batch sentiment analysis (up to 64 items)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.model_loader import get_model, DEFAULT_MODEL



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 64
MAX_TEXT_LENGTH = 5_000 



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the NLP model so the first request is not slow."""
    logger.info("⏳  Warming up sentiment model …")
    get_model(DEFAULT_MODEL)  # caches the singleton
    logger.info("✅  Model ready.  Service is live.")
    yield
    logger.info("🛑  Shutting down Sentiment-as-a-Service.")




app = FastAPI(
    title="Sentiment-as-a-Service",
    description=(
        "Production-ready REST API for sentiment analysis powered by a "
        "HuggingFace transformers pipeline (cardiffnlp/twitter-roberta-base-sentiment-latest)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)





class TextRequest(BaseModel):
    """Request body for a single-text analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        examples=["I absolutely love this product – it changed my life!"],
    )

    @field_validator("text")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain at least one non-whitespace character.")
        return v


class BatchRequest(BaseModel):
    """Request body for a batch of texts."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        examples=[["Great service!", "Absolutely terrible.", "It was okay."]],
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        for i, text in enumerate(v):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"texts[{i}] must be a non-empty, non-blank string."
                )
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"texts[{i}] exceeds the maximum length of {MAX_TEXT_LENGTH} characters."
                )
        return v


class SentimentResult(BaseModel):
    """Sentiment result for a single piece of text."""

    label: str = Field(..., description="Winning sentiment label (positive/neutral/negative).")
    score: float = Field(..., description="Confidence score for the winning label (0–1).")
    raw_label: str = Field(..., description="Original label returned by the model.")
    scores: dict[str, float] = Field(..., description="Full probability distribution.")


class SingleAnalysisResponse(BaseModel):
    text: str
    result: SentimentResult
    processing_time_ms: float


class BatchAnalysisResponse(BaseModel):
    results: list[SentimentResult]
    count: int
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    task: str
    labels: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool




def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1_000, 3)




@app.get(
    "/",
    summary="Welcome",
    response_class=JSONResponse,
    tags=["General"],
)
async def root() -> dict[str, str]:
    return {
        "service": "Sentiment-as-a-Service",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["General"],
)
async def health() -> HealthResponse:
    """Used by Docker / Kubernetes health-checks."""
    try:
        model = get_model()
        return HealthResponse(status="ok", model_loaded=True)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    tags=["Model"],
)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded NLP model."""
    info = get_model().info
    return ModelInfoResponse(**info)


@app.post(
    "/analyze",
    response_model=SingleAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a single text",
    tags=["Sentiment"],
)
async def analyze(body: TextRequest) -> SingleAnalysisResponse:
    """
    Perform sentiment analysis on a single piece of text.

    Returns the predicted label (positive / neutral / negative), the
    confidence score, and the full probability distribution over all classes.
    """
    t0 = time.perf_counter()
    try:
        result = get_model().predict(body.text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during inference.",
        )

    return SingleAnalysisResponse(
        text=body.text,
        result=SentimentResult(**result),
        processing_time_ms=_ms(t0),
    )


@app.post(
    "/analyze/batch",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a batch of texts",
    tags=["Sentiment"],
)
async def analyze_batch(body: BatchRequest) -> BatchAnalysisResponse:
    """
    Perform sentiment analysis on up to **64** texts in a single request.

    All texts are processed in a single model forward-pass for efficiency.
    Results are returned in the **same order** as the input list.
    """
    t0 = time.perf_counter()
    try:
        results = get_model().predict_batch(body.texts)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during batch inference.",
        )

    return BatchAnalysisResponse(
        results=[SentimentResult(**r) for r in results],
        count=len(results),
        processing_time_ms=_ms(t0),
    )
