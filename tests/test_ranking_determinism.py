"""Regression tests for ranking determinism and candidate-pool sizing.

Both tests in the first section failed on main (47d0119):
- equal-score chunks kept insertion order, so a restarted engine (SQLite reloads by
  ID) ranked the same corpus differently;
- top_k above the fixed 20-candidate pool was silently capped at 20.
"""

from rag_engine.models import Chunk, Document, ScoredChunk
from rag_engine.retrieval import reciprocal_rank_fusion
from rag_engine.service import RAGEngine
from rag_engine.storage import SQLiteDocumentStore


def test_restarted_engine_ranks_tied_chunks_identically(tmp_path) -> None:
    store = SQLiteDocumentStore(tmp_path / "rag.db")
    engine = RAGEngine(document_store=store)
    engine.index([Document("b", "shared evidence text"), Document("a", "shared evidence text")])

    fresh = [item.chunk.id for item in engine.retrieve("shared evidence", top_k=2)[0]]
    restarted = RAGEngine(document_store=store)
    reloaded = [item.chunk.id for item in restarted.retrieve("shared evidence", top_k=2)[0]]

    assert fresh == reloaded == ["a::c0", "b::c0"]
    assert engine.index_fingerprint() == restarted.index_fingerprint()


def test_top_k_larger_than_default_candidate_pool_is_honoured() -> None:
    engine = RAGEngine(chunk_size=5, overlap=0)
    engine.index([Document(f"d{i:02d}", "alpha beta gamma delta epsilon") for i in range(30)])

    results, trace = engine.retrieve("alpha", top_k=30)

    assert len(results) == 30
    assert trace.returned_count == 30
    assert len({item.chunk.id for item in results}) == 30


def _scored(chunk_id: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id, chunk_id, chunk_id, 0, 1),
        score=score,
    )


def test_rrf_breaks_ties_by_chunk_id_independent_of_input_order() -> None:
    # x and y receive identical fused scores (rank 1 in one list, rank 2 in the other).
    first = reciprocal_rank_fusion([[_scored("y"), _scored("x")], [_scored("x"), _scored("y")]])
    second = reciprocal_rank_fusion([[_scored("x"), _scored("y")], [_scored("y"), _scored("x")]])

    assert [item.chunk.id for item in first] == ["x", "y"]
    assert [item.chunk.id for item in second] == ["x", "y"]
    assert first[0].score == first[1].score


def test_rankings_identical_across_repeated_runs_and_insertion_orders() -> None:
    documents = [
        Document(f"doc-{i}", f"retrieval evidence topic {i % 3} shared filler words")
        for i in range(12)
    ]
    forward = RAGEngine()
    forward.index(documents)
    backward = RAGEngine()
    backward.index(list(reversed(documents)))

    for query in ["retrieval evidence", "topic 1", "shared filler", "unrelated words"]:
        runs = [
            [item.chunk.id for item in engine.retrieve(query, top_k=12)[0]]
            for engine in (forward, forward, backward)
        ]
        assert runs[0] == runs[1] == runs[2]
