from rag_engine.models import Document
from rag_engine.service import RAGEngine
from rag_engine.storage import SQLiteDocumentStore


def test_documents_survive_engine_restart_and_remain_retrievable(tmp_path) -> None:
    store = SQLiteDocumentStore(tmp_path / "rag.db")
    engine = RAGEngine(document_store=store, chunk_size=20, overlap=2)
    engine.index(
        [
            Document(
                id="architecture",
                text="MAKMA uses hybrid retrieval with lexical and vector evidence.",
                metadata={"source": "design.md"},
            )
        ]
    )

    restarted = RAGEngine(document_store=store, chunk_size=20, overlap=2)
    results, _ = restarted.retrieve("hybrid retrieval", top_k=1)

    assert results[0].chunk.document_id == "architecture"
    assert restarted.documents["architecture"].metadata["source"] == "design.md"


def test_delete_removes_document_from_memory_and_durable_store(tmp_path) -> None:
    store = SQLiteDocumentStore(tmp_path / "rag.db")
    engine = RAGEngine(document_store=store)
    engine.index([Document(id="a", text="retrieval evidence")])

    assert engine.delete("a") is True
    assert engine.delete("a") is False

    restarted = RAGEngine(document_store=store)
    assert restarted.documents == {}
