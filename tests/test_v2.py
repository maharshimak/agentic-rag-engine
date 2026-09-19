from fastapi.testclient import TestClient

from rag_engine.agent import QueryPlanner
from rag_engine.api import app, engine
from rag_engine.chunking import chunk_document
from rag_engine.evaluation import evaluate_retrieval
from rag_engine.models import Document
from rag_engine.service import RAGEngine

DOCS = [
    Document(
        "rag",
        "Retrieval augmented generation combines search with grounded language generation. "
        "Hybrid retrieval can combine lexical BM25 and semantic vector search.",
        {"topic": "rag"},
    ),
    Document(
        "mlops",
        "MLOps manages model deployment, observability, drift, rollback and production safety.",
        {"topic": "mlops"},
    ),
    Document(
        "vectors",
        "Vector databases store embeddings and retrieve semantically related passages.",
        {"topic": "retrieval"},
    ),
]


def test_chunking_overlap() -> None:
    document = Document("d", "one two three four five six seven")
    chunks = chunk_document(document, chunk_size=4, overlap=2)
    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven",
    ]


def test_agentic_hybrid_retrieval_returns_grounded_chunks() -> None:
    rag = RAGEngine(chunk_size=40, overlap=5)
    count = rag.index(DOCS)
    assert count == 3

    results, trace = rag.retrieve("BM25 and vector retrieval", top_k=2)

    assert results
    assert results[0].chunk.document_id in {"rag", "vectors"}
    assert trace.returned_count == 2
    assert trace.candidate_count >= trace.returned_count
    assert trace.latency_ms >= 0


def test_query_planner_decomposes_multi_part_question() -> None:
    planned = QueryPlanner().plan("Compare RAG and MLOps; explain vector search")
    assert planned[0].startswith("Compare RAG")
    assert len(planned) >= 2


def test_retrieval_metrics() -> None:
    report = evaluate_retrieval({"a", "b"}, ["a", "x", "b"], k=3)
    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 2 / 3
    assert report.mrr == 1.0
    assert report.hit_rate_at_k == 1.0


def test_api_index_retrieve_and_answer() -> None:
    engine.documents.clear()
    engine._agent = None
    client = TestClient(app)

    response = client.post(
        "/v1/documents/index",
        json={
            "documents": [
                {"id": doc.id, "text": doc.text, "metadata": doc.metadata} for doc in DOCS
            ]
        },
    )
    assert response.status_code == 200

    retrieved = client.post(
        "/v1/retrieve",
        json={"query": "semantic vector passages", "top_k": 2},
    )
    assert retrieved.status_code == 200
    assert retrieved.json()["results"]

    answered = client.post(
        "/v1/answer",
        json={"query": "what is vector retrieval?", "top_k": 2},
    )
    assert answered.status_code == 200
    assert answered.json()["citations"]
