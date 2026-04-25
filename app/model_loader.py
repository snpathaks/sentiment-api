"""
model_loader.py
---------------
Class-based wrapper around a HuggingFace transformers sentiment-analysis
pipeline. The model is downloaded once at startup and cached so every
request is served from memory.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from transformers import pipeline, Pipeline

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP: dict[str, str] = {
   
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",

    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


# ---------------------------------------------------------------------------
# SentimentModel
# ---------------------------------------------------------------------------


class SentimentModel:
    """
    Thin, thread-safe wrapper around a HuggingFace ``Pipeline``.

    Usage
    -----
    ::

        model = SentimentModel()          # loads model once
        result = model.predict("I love this!")
        # {'label': 'positive', 'score': 0.9821, 'raw_label': 'positive'}

        batch = model.predict_batch(["Great!", "Terrible…"])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._pipeline: Pipeline = self._load_pipeline()

   

    def _load_pipeline(self) -> Pipeline:
        """Download (or load from cache) the model and return a Pipeline."""
        logger.info("Loading sentiment model: %s", self.model_name)
        try:
            nlp = pipeline(
                task="sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
            
                top_k=None,
             
                truncation=True,
                max_length=512,
            )
            logger.info("Model loaded successfully.")
            return nlp
        except Exception as exc:
            logger.exception("Failed to load model '%s': %s", self.model_name, exc)
            raise

    @staticmethod
    def _normalise(raw: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert raw pipeline output for a single input into a clean dict.

        Parameters
        ----------
        raw:
            The list of ``{'label': str, 'score': float}`` dicts returned
            when ``top_k=None``.

        Returns
        -------
        dict with keys:
            ``label``     – winning human-readable sentiment label
            ``score``     – confidence of the winning label (0–1)
            ``raw_label`` – original label string from the model
            ``scores``    – full probability distribution as a dict
        """
        
        sorted_scores = sorted(raw, key=lambda d: d["score"], reverse=True)

        winner = sorted_scores[0]
        raw_label: str = winner["label"]
        human_label: str = LABEL_MAP.get(raw_label, raw_label.lower())

        return {
            "label": human_label,
            "score": round(winner["score"], 6),
            "raw_label": raw_label,
            "scores": {
                LABEL_MAP.get(item["label"], item["label"].lower()): round(
                    item["score"], 6
                )
                for item in sorted_scores
            },
        }


    def predict(self, text: str) -> dict[str, Any]:
        """
        Run sentiment analysis on a single piece of text.

        Parameters
        ----------
        text:
            The input string to classify (will be truncated at 512 tokens).

        Returns
        -------
        A normalised result dict (see ``_normalise``).
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        raw: list[list[dict[str, Any]]] = self._pipeline([text])
        
        return self._normalise(raw[0])

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Run sentiment analysis on a batch of texts in a single forward pass.

        Parameters
        ----------
        texts:
            A non-empty list of strings.  Each will be truncated at 512 tokens.

        Returns
        -------
        A list of normalised result dicts, one per input, in the same order.
        """
        if not texts:
            raise ValueError("texts list must contain at least one item.")

        # Filter out blanks and keep an index map so we can restore order
        indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed:
            raise ValueError("All provided texts are empty or whitespace.")

        indices, clean_texts = zip(*indexed)
        raw_batch: list[list[dict[str, Any]]] = self._pipeline(list(clean_texts))

        results: list[dict[str, Any] | None] = [None] * len(texts)
        for idx, raw in zip(indices, raw_batch):
            results[idx] = self._normalise(raw)

        return results  

    @property
    def info(self) -> dict[str, str]:
        """Return metadata about the loaded model."""
        return {
            "model_name": self.model_name,
            "task": "sentiment-analysis",
            "labels": list(set(LABEL_MAP.values())),
        }





@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_MODEL) -> SentimentModel:
    """
    Return the application-wide ``SentimentModel`` singleton.

    Using ``lru_cache`` with ``maxsize=1`` guarantees the heavy model-loading
    work happens exactly once per process lifetime, and is safe under
    FastAPI's async event loop because the first call is made during the
    synchronous ``startup`` event.
    """
    return SentimentModel(model_name=model_name)
