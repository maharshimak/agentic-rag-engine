from rag_engine.advanced import MMRDiversityReranker
from rag_engine.models import Chunk, ScoredChunk


def _candidate(chunk_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=chunk_id,
            text=text,
            start_token=0,
            end_token=len(text.split()),
        ),
        score=1.0,
    )


def test_mmr_prefers_diverse_context_over_near_duplicates():
    candidates = [
        _candidate("a", "vector retrieval embeddings semantic search"),
        _candidate("b", "vector retrieval embeddings semantic search duplicate"),
        _candidate("c", "BM25 lexical ranking sparse keyword retrieval"),
    ]
    reranked = MMRDiversityReranker(lambda_relevance=0.5).rerank(
        "hybrid retrieval",
        candidates,
        top_k=2,
    )
    ids = [item.chunk.id for item in reranked]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_mmr_validates_configuration():
    try:
        MMRDiversityReranker(lambda_relevance=1.5)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("invalid lambda should fail")
