from rag_engine.core import Document, citation_context, hybrid_retrieve, recall_at_k

DOCS = [
    Document("d1", "RAG combines retrieval with language generation."),
    Document("d2", "MLOps manages machine learning systems in production."),
    Document("d3", "Vector search retrieves semantically related passages."),
]


def test_hybrid_retrieval() -> None:
    ids = hybrid_retrieve("retrieval vector passages", DOCS, top_k=2)
    assert "d3" in ids


def test_citations() -> None:
    assert "[d1]" in citation_context(["d1"], DOCS)


def test_recall() -> None:
    assert recall_at_k({"d1"}, ["d1", "d2"], 1) == 1.0
