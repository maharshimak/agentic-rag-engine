from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    citation_coverage: float
    redundancy: float
    evaluated_k: int


def _clean_ids(values: Iterable[str], *, name: str) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain non-empty string identifiers")
        cleaned.append(value.strip())
    return cleaned


def evaluate_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    cited_ids: Iterable[str] = (),
    k: int = 5,
) -> RetrievalDiagnostics:
    """Compute inspectable retrieval/citation diagnostics for one query."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    retrieved = _clean_ids(retrieved_ids, name="retrieved_ids")
    relevant = set(_clean_ids(relevant_ids, name="relevant_ids"))
    citations = set(_clean_ids(cited_ids, name="cited_ids"))
    if not relevant:
        raise ValueError("relevant_ids cannot be empty")

    top = retrieved[:k]
    unique_top = set(top)
    hits = unique_top.intersection(relevant)
    precision = len(hits) / len(top) if top else 0.0
    recall = len(hits) / len(relevant)

    reciprocal_rank = 0.0
    for rank, document_id in enumerate(top, start=1):
        if document_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    citation_coverage = len(citations.intersection(unique_top)) / len(unique_top) if top else 0.0
    redundancy = 1.0 - (len(unique_top) / len(top)) if top else 0.0
    return RetrievalDiagnostics(
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        citation_coverage=citation_coverage,
        redundancy=redundancy,
        evaluated_k=len(top),
    )
