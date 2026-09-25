from __future__ import annotations

from dataclasses import replace
from math import isfinite

from rag_engine.embeddings import EmbeddingProvider, HashEmbeddingProvider
from rag_engine.models import ScoredChunk
from rag_engine.retrieval import cosine_similarity


class MMRDiversityReranker:
    """Maximal-marginal-relevance reranker for relevance + context diversity.

    MMR reduces near-duplicate context by penalizing candidates that are too
    similar to chunks already selected. The embedding provider is pluggable, so
    production callers can use a real embedding endpoint while tests remain
    deterministic with HashEmbeddingProvider.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        lambda_relevance: float = 0.7,
    ) -> None:
        if not isfinite(lambda_relevance) or not 0.0 <= lambda_relevance <= 1.0:
            raise ValueError("lambda_relevance must be finite and between 0 and 1.")
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.lambda_relevance = lambda_relevance

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

        vectors = self.embedding_provider.embed(
            [query, *[item.chunk.text for item in candidates]]
        )
        if len(vectors) != len(candidates) + 1:
            raise ValueError("Embedding provider returned the wrong number of vectors.")
        query_vector, candidate_vectors = vectors[0], vectors[1:]

        relevance = [
            max(-1.0, min(1.0, cosine_similarity(query_vector, vector)))
            for vector in candidate_vectors
        ]
        selected: list[int] = []
        remaining = set(range(len(candidates)))

        while remaining and len(selected) < min(top_k, len(candidates)):
            best_index: int | None = None
            best_score = float("-inf")
            for index in sorted(remaining):
                redundancy = (
                    max(
                        cosine_similarity(candidate_vectors[index], candidate_vectors[chosen])
                        for chosen in selected
                    )
                    if selected
                    else 0.0
                )
                score = (
                    self.lambda_relevance * relevance[index]
                    - (1.0 - self.lambda_relevance) * redundancy
                )
                if score > best_score:
                    best_score = score
                    best_index = index

            assert best_index is not None
            selected.append(best_index)
            remaining.remove(best_index)

        return [
            replace(
                candidates[index],
                score=(
                    self.lambda_relevance * relevance[index]
                    - (
                        (1.0 - self.lambda_relevance)
                        * max(
                            (
                                cosine_similarity(candidate_vectors[index], candidate_vectors[prior])
                                for prior in selected[:position]
                            ),
                            default=0.0,
                        )
                    )
                ),
                rerank_score=(
                    self.lambda_relevance * relevance[index]
                    - (
                        (1.0 - self.lambda_relevance)
                        * max(
                            (
                                cosine_similarity(candidate_vectors[index], candidate_vectors[prior])
                                for prior in selected[:position]
                            ),
                            default=0.0,
                        )
                    )
                ),
            )
            for position, index in enumerate(selected)
        ]
