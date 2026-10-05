from rag_engine.models import RetrievalTrace
from rag_engine.retrieval import adaptive_hybrid_weights
from rag_engine.service import RAGEngine


def test_adaptive_weights_favor_lexical_for_identifiers() -> None:
    lexical, semantic = adaptive_hybrid_weights('find "ERR_502" in release-2026.10')
    assert lexical > semantic


def test_adaptive_weights_favor_semantics_for_long_natural_language_query() -> None:
    lexical, semantic = adaptive_hybrid_weights(
        "explain how retrieval augmented generation combines evidence across several related sources"
    )
    assert semantic > lexical


def test_answer_abstains_when_retrieval_has_no_evidence(monkeypatch) -> None:
    engine = RAGEngine(abstain_threshold=0.45)
    trace = RetrievalTrace(
        query="unknown",
        planned_queries=["unknown"],
        candidate_count=0,
        returned_count=0,
        latency_ms=0.0,
    )

    monkeypatch.setattr(engine, "retrieve", lambda query, top_k=5: ([], trace))
    result = engine.answer("unknown")

    assert result["abstained"] is True
    assert result["confidence"]["score"] == 0.0
    assert "enough independent evidence" in result["answer"]
    assert result["citations"] == []
