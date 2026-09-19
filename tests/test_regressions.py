import pytest

from rag_engine.evaluation import evaluate_retrieval
from rag_engine.models import Document
from rag_engine.retrieval import cosine_similarity
from rag_engine.service import RAGEngine


def test_cosine_normalizes_external_embeddings():
    assert cosine_similarity([3, 0], [12, 0]) == pytest.approx(1)
    assert cosine_similarity([3, 0], [0, 12]) == 0
    assert cosine_similarity([0, 0], [0, 12]) == 0


@pytest.mark.parametrize("left,right", [([1], [1, 2]), ([float("nan")], [1]), ([], [])])
def test_invalid_embeddings_rejected(left, right):
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


def test_synthetic_retrieval_quality_gate():
    engine = RAGEngine()
    engine.index(
        [
            Document("memory", "SQLite persists conversation memory and session history."),
            Document("sql", "Read-only SQL analytics rejects mutations."),
            Document("media", "Video editing plans include subtitles and color grading."),
        ]
    )
    for query, wanted in [
        ("SQLite conversation memory", "memory"),
        ("SQL analytics", "sql"),
        ("Video subtitles", "media"),
    ]:
        results, _ = engine.retrieve(query, top_k=1)
        metrics = evaluate_retrieval({wanted}, [r.chunk.document_id for r in results], 1)
        assert metrics.recall_at_k == 1 and metrics.mrr == 1
