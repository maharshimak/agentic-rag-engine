from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from math import isfinite, log, sqrt

from rag_engine.embeddings import EmbeddingProvider, HashEmbeddingProvider
from rag_engine.models import Chunk, ScoredChunk


def tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[\w.-]+", text, flags=re.UNICODE)
        if token.strip("._-")
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding dimensions must match and be non-empty.")
    if not all(isfinite(v) for v in [*a, *b]):
        raise ValueError("Embeddings must contain finite values.")
    norm = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0

        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        self.document_frequency = document_frequency

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        total = len(self.chunks)
        scored: list[ScoredChunk] = []

        for chunk, tokens, length in zip(self.chunks, self.tokens, self.lengths):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                score += idf * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=score,
                        lexical_score=score,
                    )
                )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class VectorIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.chunks = chunks
        self.provider = provider or HashEmbeddingProvider()
        self.vectors = self.provider.embed([chunk.text for chunk in chunks])
        if len(self.vectors) != len(chunks):
            raise ValueError("Embedding provider returned the wrong number of vectors.")

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        query_vector = self.provider.embed([query])[0]
        scored: list[ScoredChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = cosine_similarity(query_vector, vector)
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    semantic_score=score,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]],
    k: int = 60,
    *,
    weights: list[float] | None = None,
) -> list[ScoredChunk]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings.")
    if any(not isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("fusion weights must be finite and positive.")

    fused: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    lexical: dict[str, float] = {}
    semantic: dict[str, float] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for position, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + position)
            by_id[chunk_id] = item
            lexical[chunk_id] = max(lexical.get(chunk_id, 0.0), item.lexical_score)
            semantic[chunk_id] = max(semantic.get(chunk_id, 0.0), item.semantic_score)

    return sorted(
        [
            replace(
                by_id[chunk_id],
                score=score,
                lexical_score=lexical.get(chunk_id, 0.0),
                semantic_score=semantic.get(chunk_id, 0.0),
            )
            for chunk_id, score in fused.items()
        ],
        key=lambda item: item.score,
        reverse=True,
    )


def adaptive_hybrid_weights(query: str) -> tuple[float, float]:
    """Choose transparent lexical/semantic weights from query characteristics."""
    tokens = tokenize(query)
    if not tokens:
        return 1.0, 1.0

    quoted = bool(re.search(r'["\'][^"\']+["\']', query))
    identifier_like = any(
        any(character.isdigit() for character in token)
        or "_" in token
        or "." in token
        or "-" in token
        for token in tokens
    )
    if quoted or identifier_like:
        return 1.35, 0.85

    if len(tokens) >= 9:
        return 0.9, 1.2

    return 1.0, 1.0


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider | None = None,
        *,
        adaptive_weights: bool = True,
        lexical_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> None:
        if any(
            not isfinite(weight) or weight <= 0
            for weight in (lexical_weight, semantic_weight)
        ):
            raise ValueError("retrieval weights must be finite and positive.")
        self.lexical = BM25Index(chunks)
        self.semantic = VectorIndex(chunks, provider=embedding_provider)
        self.adaptive_weights = adaptive_weights
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[ScoredChunk]:
        lexical_weight, semantic_weight = (
            adaptive_hybrid_weights(query)
            if self.adaptive_weights
            else (self.lexical_weight, self.semantic_weight)
        )
        fused = reciprocal_rank_fusion(
            [
                self.lexical.search(query, top_k=candidate_k),
                self.semantic.search(query, top_k=candidate_k),
            ],
            weights=[lexical_weight, semantic_weight],
        )
        return fused[:top_k]
