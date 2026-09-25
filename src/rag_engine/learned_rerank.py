from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from rag_engine.models import ScoredChunk


class CrossEncoderReranker:
    """Learned query/chunk reranking through sentence-transformers CrossEncoder.

    The model is loaded lazily so the default dependency-light RAG engine remains
    usable without PyTorch. A scorer can be injected for testing or custom
    serving backends.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        *,
        scorer: Any | None = None,
        max_length: int = 512,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name is required.")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        self.model_name = model_name
        self.max_length = max_length
        self._scorer = scorer

    def _load(self):
        if self._scorer is not None:
            return self._scorer
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed; install the reranking extra."
            ) from error
        self._scorer = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._scorer

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not candidates:
            return []

        scorer = self._load()
        pairs = [(query, item.chunk.text) for item in candidates]
        raw_scores = scorer.predict(pairs)
        scores = [float(value) for value in raw_scores]
        if len(scores) != len(candidates):
            raise ValueError("Cross-encoder returned the wrong number of scores.")
        if not all(isfinite(score) for score in scores):
            raise ValueError("Cross-encoder scores must be finite.")

        reranked = [
            replace(item, score=score, rerank_score=score)
            for item, score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
