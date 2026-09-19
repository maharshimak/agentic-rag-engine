import pytest

from rag_engine.diagnostics import evaluate_retrieval


def test_retrieval_diagnostics_capture_rank_recall_and_citations() -> None:
    report = evaluate_retrieval(
        ["doc-a", "doc-b", "doc-b", "doc-c"],
        {"doc-b", "doc-z"},
        cited_ids={"doc-b"},
        k=4,
    )
    assert report.precision_at_k == pytest.approx(0.25)
    assert report.recall_at_k == pytest.approx(0.5)
    assert report.reciprocal_rank == pytest.approx(0.5)
    assert report.citation_coverage == pytest.approx(1 / 3)
    assert report.redundancy == pytest.approx(0.25)


def test_retrieval_diagnostics_require_ground_truth() -> None:
    with pytest.raises(ValueError, match="relevant_ids"):
        evaluate_retrieval(["doc-a"], [], k=1)
