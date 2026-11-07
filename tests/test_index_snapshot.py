from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from rag_engine.contracts import run_operation
from rag_engine.embeddings import HashEmbeddingProvider
from rag_engine.models import Chunk, Document
from rag_engine.provenance import sha256_text
from rag_engine.rerank import TransparentReranker
from rag_engine.retrieval import BM25Index, VectorIndex, cosine_similarity
from rag_engine.service import RAGEngine
from rag_engine.storage import SQLiteDocumentStore


class ControlledEmbedding(HashEmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.mode = ""
        self.entered = Event()
        self.release = Event()

    def embed(self, texts):
        if any("replacement" in text for text in texts):
            if self.mode == "failure":
                raise OSError("synthetic provider unavailable")
            if self.mode == "pause":
                self.entered.set()
                if not self.release.wait(5):
                    raise RuntimeError("synthetic test synchronization timeout")
        return super().embed(texts)


def test_failed_embedding_does_not_publish_or_persist_a_replacement(tmp_path):
    provider = ControlledEmbedding()
    store = SQLiteDocumentStore(tmp_path / "docs.db")
    engine = RAGEngine(embedding_provider=provider, document_store=store)
    original = Document("same", "original reference")
    engine.index([original])
    fingerprint = engine.index_fingerprint()
    provider.mode = "failure"

    with pytest.raises(OSError):
        engine.index([Document("same", "replacement reference")])

    assert engine.documents == {"same": original}
    assert store.list_documents() == [original]
    assert engine.index_fingerprint() == fingerprint
    assert engine.retrieve("original")[0][0].chunk.text == original.text


def test_invalid_batch_cannot_leave_a_partial_document(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "docs.db")
    engine = RAGEngine(document_store=store)
    original = Document("keep", "original reference")
    engine.index([original])
    fingerprint = engine.index_fingerprint()
    with pytest.raises(ValueError, match="non-empty"):
        engine.index([Document("stray", "valid text"), Document("invalid", " ")])
    assert engine.documents == {"keep": original}
    assert engine.index_fingerprint() == fingerprint
    assert store.list_documents() == [original]


def test_queries_continue_on_previous_snapshot_while_index_is_built():
    provider = ControlledEmbedding()
    engine = RAGEngine(embedding_provider=provider)
    engine.index([Document("same", "original reference")])
    old_fingerprint = engine.index_fingerprint()
    provider.mode = "pause"
    with ThreadPoolExecutor(max_workers=2) as pool:
        write = pool.submit(engine.index, [Document("same", "replacement reference")])
        try:
            assert provider.entered.wait(3)
            read = pool.submit(engine.retrieve, "original")
            matches, trace = read.result(timeout=2)
            assert matches[0].chunk.text == "original reference"
            assert trace.index_fingerprint == old_fingerprint
        finally:
            provider.release.set()
        assert write.result(timeout=3) == 1
    assert engine.index_fingerprint() != old_fingerprint
    assert engine.retrieve("replacement")[0][0].chunk.text == "replacement reference"


def test_store_failure_does_not_publish_a_new_snapshot(tmp_path, monkeypatch):
    store = SQLiteDocumentStore(tmp_path / "docs.db")
    engine = RAGEngine(document_store=store)
    original = Document("keep", "original reference")
    engine.index([original])

    def fail(documents):
        raise OSError("synthetic store write failure")

    monkeypatch.setattr(store, "upsert_many", fail)
    with pytest.raises(OSError):
        engine.index([Document("stray", "unpublished document")])
    assert engine.documents == {"keep": original}
    assert engine.retrieve("original")[0][0].chunk.text == original.text


def test_chunk_budget_is_checked_before_embedding_or_durable_writes(tmp_path):
    provider = ControlledEmbedding()
    store = SQLiteDocumentStore(tmp_path / "docs.db")
    engine = RAGEngine(embedding_provider=provider, document_store=store, max_index_chunks=1)
    engine.index([Document("keep", "original reference")])
    provider.mode = "failure"
    with pytest.raises(ValueError, match="chunk limit"):
        engine.index([Document("stray", "replacement reference")])
    assert store.list_documents() == [Document("keep", "original reference")]


def test_inflight_contract_provenance_hashes_the_original_source_span():
    entered, release = Event(), Event()

    class GatedReranker(TransparentReranker):
        def rerank(self, query, candidates, top_k):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("synthetic test synchronization timeout")
            return super().rerank(query, candidates, top_k)

    engine = RAGEngine(reranker=GatedReranker())
    original = "original\t reference"
    engine.index([Document("same", original)])
    with ThreadPoolExecutor(max_workers=1) as pool:
        read = pool.submit(run_operation, "retrieve", {"query": "original"}, engine=engine)
        try:
            assert entered.wait(3)
            engine.index([Document("same", "original  reference")])
        finally:
            release.set()
        envelope = read.result(timeout=3)
    assert envelope["status"] == "ok"
    assert envelope["provenance"][0]["sha256"] == sha256_text(original)


def test_failed_delete_rebuild_leaves_durable_and_live_documents(tmp_path):
    provider = ControlledEmbedding()
    store = SQLiteDocumentStore(tmp_path / "docs.db")
    engine = RAGEngine(embedding_provider=provider, document_store=store)
    documents = [Document("keep", "replacement reference"), Document("remove", "retired notes")]
    engine.index(documents)
    provider.mode = "failure"
    with pytest.raises(OSError):
        engine.delete("remove")
    assert store.list_documents() == documents
    assert set(engine.documents) == {"keep", "remove"}



def test_sqlite_reopen_preserves_successful_updates_and_deletions(tmp_path):
    """Live, durable, and reconstructed indexes must agree after writes."""
    db_path = tmp_path / "docs.db"
    store = SQLiteDocumentStore(db_path)
    engine = RAGEngine(document_store=store)
    engine.index(
        [Document("first", "original evidence"), Document("second", "keep evidence")]
    )
    engine.index([Document("first", "replacement evidence")])

    expected = {
        "first": Document("first", "replacement evidence"),
        "second": Document("second", "keep evidence"),
    }
    assert engine.documents == expected
    assert store.list_documents() == [expected["first"], expected["second"]]

    reopened = RAGEngine(document_store=SQLiteDocumentStore(db_path))
    assert reopened.documents == expected
    assert reopened.index_fingerprint() == engine.index_fingerprint()
    assert reopened.retrieve("replacement")[0][0].chunk.document_id == "first"

    assert engine.delete("first")
    reopened_after_delete = RAGEngine(document_store=SQLiteDocumentStore(db_path))
    assert reopened_after_delete.documents == engine.documents
    assert reopened_after_delete.documents == {"second": expected["second"]}
    assert reopened_after_delete.index_fingerprint() == engine.index_fingerprint()
    assert reopened_after_delete.retrieve("keep")[0][0].chunk.document_id == "second"


def test_postings_preserve_repeated_term_scores_and_tie_breaks():
    chunks = [Chunk(name, name, "alpha beta", 0, 2) for name in ("z", "a")]
    index = BM25Index(chunks)
    single = index.search("alpha")
    repeated = index.search("alpha alpha", top_k=1)
    assert [item.chunk.id for item in single] == ["a", "z"]
    assert repeated[0].chunk.id == "a"
    assert repeated[0].score == 2 * single[0].score
    assert index.search("absent") == []


def test_cached_norms_preserve_exact_cosine_scores():
    class Embeddings:
        def embed(self, texts):
            return [
                {"alpha": [3.0, 4.0], "beta": [0.0, 1.0], "query": [2.0, 1.0]}[text]
                for text in texts
            ]

    chunks = [Chunk(name, name, name, 0, 1) for name in ("alpha", "beta")]
    index = VectorIndex(chunks, Embeddings())
    assert [item.score for item in index.search("query")] == [
        cosine_similarity([2.0, 1.0], [3.0, 4.0]),
        cosine_similarity([2.0, 1.0], [0.0, 1.0]),
    ]


def test_invalid_query_vector_count_has_an_explicit_error():
    class MissingQueryVector:
        def embed(self, texts):
            return [] if texts == ["query"] else [[1.0]]

    index = VectorIndex([Chunk("one", "one", "source", 0, 1)], MissingQueryVector())
    with pytest.raises(ValueError, match="invalid query vector"):
        index.search("query")
