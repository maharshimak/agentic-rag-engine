from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import replace
from heapq import nsmallest
from math import isfinite, log, sqrt

from rag_engine.embeddings import EmbeddingProvider, HashEmbeddingProvider
from rag_engine.models import Chunk, ScoredChunk

_EDGE_PUNCTUATION = "._-"


def tokenize(text: str) -> list[str]:
    """Lexical tokens: runs of word characters, ``.`` and ``-``, casefolded.

    Leading/trailing ``.``, ``-`` and ``_`` are trimmed so sentence-final words
    ("evidence.") match their bare form; internal ones are kept (``v1.2``,
    ``gpt-4o``, ``err_tls_042``). Consequences: ``.env`` -> ``env``, ``now-`` -> ``now``,
    ``__init__`` -> ``init``. Used only for BM25, reranker overlap and query
    heuristics; chunk boundaries and character offsets come from ``chunking``.
    """
    tokens = (
        token.strip(_EDGE_PUNCTUATION)
        for token in re.findall(r"[\w.-]+", text, flags=re.UNICODE)
    )
    return [token.casefold() for token in tokens if token]


def ranking_key(item: ScoredChunk) -> tuple[float, str]:
    """Sort key: descending score, then ascending chunk ID as a deterministic tie-break.

    Without the explicit tie-break, equal scores kept index insertion order, so the
    same corpus could rank differently after a restart (SQLite reloads by ID).
    """
    return (-item.score, item.chunk.id)


def _vector_norm(vector: list[float]) -> float:
    if not vector or not all(isfinite(value) for value in vector):
        raise ValueError("Embeddings must be non-empty and contain finite values.")
    norm = sqrt(sum(value * value for value in vector))
    if not isfinite(norm):
        raise ValueError("Embedding norm exceeds the supported range.")
    return norm


def _cosine(a: list[float], b: list[float], norm: float) -> float:
    if not isfinite(norm):
        raise ValueError("Embedding norm product exceeds the supported range.")
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding dimensions must match and be non-empty.")
    return _cosine(a, b, _vector_norm(a) * _vector_norm(b))


def _top_matches(items: Iterable[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    if top_k < 0:
        return sorted(items, key=ranking_key)[:top_k]
    return nsmallest(top_k, items, key=ranking_key)


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.lengths = []
        for position, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            self.lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                postings[term].append((position, frequency))
        total = len(chunks)
        self.avg_length = sum(self.lengths) / total if total else 1.0
        self.document_frequency = Counter({term: len(entries) for term, entries in postings.items()})
        for term, entries in postings.items():
            df = self.document_frequency[term]
            idf = log(1.0 + (total - df + 0.5) / (df + 0.5))
            for i, (position, frequency) in enumerate(entries):
                denominator = frequency + k1 * (1 - b + b * self.lengths[position] / self.avg_length)
                entries[i] = (position, idf * (frequency * (k1 + 1)) / denominator)
        self._postings = dict(postings)

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        scores: dict[int, float] = defaultdict(float)
        # Preserve query term order, including repetitions, for exact score parity.
        for term in tokenize(query):
            for position, weight in self._postings.get(term, ()):
                scores[position] += weight
        return _top_matches(
            (
                ScoredChunk(chunk=self.chunks[position], score=score, lexical_score=score)
                for position, score in scores.items() if score > 0
            ),
            top_k,
        )


class VectorIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.chunks = chunks
        self.provider = provider or HashEmbeddingProvider()
        self.vectors = [list(vector) for vector in self.provider.embed([chunk.text for chunk in chunks])]
        if len(self.vectors) != len(chunks):
            raise ValueError("Embedding provider returned the wrong number of vectors.")
        self._dimensions = len(self.vectors[0]) if self.vectors else 0
        if any(len(vector) != self._dimensions for vector in self.vectors):
            raise ValueError("Embedding dimensions must match.")
        self._norms = [_vector_norm(vector) for vector in self.vectors]

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        queries = self.provider.embed([query])
        if len(queries) != 1 or len(queries[0]) != self._dimensions:
            raise ValueError("Embedding provider returned an invalid query vector.")
        query_vector = queries[0]
        query_norm = _vector_norm(query_vector)

        def scored() -> Iterable[ScoredChunk]:
            for chunk, vector, norm in zip(self.chunks, self.vectors, self._norms, strict=True):
                score = _cosine(query_vector, vector, query_norm * norm)
                yield ScoredChunk(chunk=chunk, score=score, semantic_score=score)

        return _top_matches(scored(), top_k)


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
        key=ranking_key,
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
