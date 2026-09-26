from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

from rag_engine.models import ScoredChunk


@dataclass(frozen=True, slots=True)
class RetrievalConfidence:
    score: float
    top_score: float
    score_gap: float
    document_diversity: float
    evidence_count: int

    @property
    def low_confidence(self) -> bool:
        return self.score < 0.45


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def assess_retrieval_confidence(results: list[ScoredChunk]) -> RetrievalConfidence:
    """Estimate whether retrieved evidence is strong enough to support generation.

    The score intentionally combines independent signals instead of trusting a
    single retriever score: absolute top relevance, separation from the runner-up,
    evidence volume, and source-document diversity.
    """
    if not results:
        return RetrievalConfidence(0.0, 0.0, 0.0, 0.0, 0)

    raw_scores = [float(item.score) for item in results]
    if any(not isfinite(value) for value in raw_scores):
        raise ValueError("retrieval scores must be finite")

    ordered = sorted(raw_scores, reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    gap = max(0.0, top - second)

    document_ids = {item.chunk.document_id for item in results}
    diversity = len(document_ids) / len(results)

    # Works for cosine-like, fused, or reranker scores without requiring callers
    # to first normalize each backend to an identical range.
    top_signal = _sigmoid(top)
    gap_signal = 1.0 - exp(-max(0.0, gap))
    volume_signal = min(1.0, len(results) / 4.0)

    score = (
        0.50 * top_signal
        + 0.20 * gap_signal
        + 0.15 * diversity
        + 0.15 * volume_signal
    )
    return RetrievalConfidence(
        score=max(0.0, min(1.0, score)),
        top_score=top,
        score_gap=gap,
        document_diversity=diversity,
        evidence_count=len(results),
    )


def should_abstain(results: list[ScoredChunk], *, threshold: float = 0.45) -> bool:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return assess_retrieval_confidence(results).score < threshold
