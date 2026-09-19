from dataclasses import dataclass


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
