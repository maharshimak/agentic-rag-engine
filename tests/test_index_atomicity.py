import pytest

from rag_engine.embeddings import HashEmbeddingProvider
from rag_engine.models import Document
from rag_engine.service import RAGEngine
from rag_engine.storage import SQLiteDocumentStore


class FailingEmbeddingProvider(HashEmbeddingProvider):
    fail = False

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return super().embed(texts)


def assert_snapshot(engine, store, expected):
    assert engine.documents == {document.id: document for document in expected}
    assert store.list_documents() == sorted(expected, key=lambda document: document.id)
    restarted = RAGEngine(document_store=store)
    assert restarted.documents == engine.documents
    if expected:
        results, _ = engine.retrieve("original evidence", top_k=10)
        assert {item.chunk.document_id for item in results} == {
            document.id for document in expected
        }
        for item in results:
            assert item.chunk.text == engine.documents[item.chunk.document_id].text


def test_invalid_batch_preserves_existing_documents_and_index(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "rag.db")
    engine = RAGEngine(document_store=store)
    original = Document("original", "original evidence")
    engine.index([original])

    with pytest.raises(ValueError, match="non-empty"):
        engine.index([Document("original", "replacement evidence"), Document("bad", " ")])

    assert_snapshot(engine, store, [original])


@pytest.mark.parametrize("operation", ["index", "delete"])
def test_embedding_failure_preserves_documents_and_persisted_snapshot(tmp_path, operation):
    store = SQLiteDocumentStore(tmp_path / "rag.db")
    provider = FailingEmbeddingProvider()
    engine = RAGEngine(document_store=store, embedding_provider=provider)
    originals = [Document("original", "original evidence"), Document("other", "other evidence")]
    engine.index(originals)
    provider.fail = True

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        if operation == "index":
            engine.index([Document("original", "replacement evidence")])
        else:
            engine.delete("original")

    provider.fail = False
    assert_snapshot(engine, store, originals)


@pytest.mark.parametrize("operation", ["index", "delete"])
def test_storage_failure_preserves_documents_and_retrieval(tmp_path, monkeypatch, operation):
    store = SQLiteDocumentStore(tmp_path / "rag.db")
    engine = RAGEngine(document_store=store)
    original = Document("original", "original evidence")
    engine.index([original])

    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(store, "upsert_many" if operation == "index" else "delete", fail)
    with pytest.raises(OSError, match="storage unavailable"):
        if operation == "index":
            engine.index([Document("original", "replacement evidence")])
        else:
            engine.delete("original")

    assert_snapshot(engine, store, [original])


def test_successful_update_publishes_matching_documents_index_and_store(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "rag.db")
    engine = RAGEngine(document_store=store)
    engine.index([Document("original", "original evidence")])
    replacement = Document("original", "replacement evidence")

    assert engine.index([replacement]) == 1
    assert_snapshot(engine, store, [replacement])
    assert engine.delete("original")
    assert_snapshot(engine, store, [])
