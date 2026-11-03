from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log2


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    selected = retrieved[:k]
    if not selected:
        return 0.0
    return len(relevant.intersection(selected)) / len(selected)


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    for position, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def hit_rate_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    return float(bool(relevant.intersection(retrieved[:k])))


def ndcg_at_k(relevance: Mapping[str, float], retrieved: list[str], k: int) -> float:
    """Normalized discounted cumulative gain with linear gain ``rel / log2(rank + 1)``.

    ``relevance`` maps IDs to graded judgments (>= 0; unlisted IDs count as 0).
    Repeated IDs earn gain only at their first rank. Returns 0.0 when no judged
    item has positive relevance, because there is no ideal ranking to normalize by.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    for grade in relevance.values():
        if isinstance(grade, bool) or not isfinite(grade) or grade < 0:
            raise ValueError("relevance grades must be finite and non-negative")
    seen: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in seen:
            continue
        seen.add(item)
        dcg += float(relevance.get(item, 0.0)) / log2(rank + 1)
    ideal = sorted((float(g) for g in relevance.values() if g > 0), reverse=True)[:k]
    idcg = sum(grade / log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True)
class RetrievalEvaluation:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate_at_k: float


def evaluate_retrieval(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        recall_at_k=recall_at_k(relevant, retrieved, k),
        precision_at_k=precision_at_k(relevant, retrieved, k),
        mrr=reciprocal_rank(relevant, retrieved),
        hit_rate_at_k=hit_rate_at_k(relevant, retrieved, k),
    )
