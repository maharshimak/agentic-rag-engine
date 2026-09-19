from dataclasses import replace

from rag_engine.models import ScoredChunk
from rag_engine.retrieval import tokenize


class TransparentReranker:
    """Interpretable reranker that can be replaced by a cross-encoder adapter."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        query_tokens = set(tokenize(query))
        reranked: list[ScoredChunk] = []

        for item in candidates:
            chunk_tokens = set(tokenize(item.chunk.text))
            overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            phrase_bonus = 0.15 if query.lower() in item.chunk.text.lower() else 0.0
            rerank_score = 0.65 * item.score + 0.35 * overlap + phrase_bonus
            reranked.append(
                replace(
                    item,
                    score=rerank_score,
                    rerank_score=rerank_score,
                )
            )

        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
