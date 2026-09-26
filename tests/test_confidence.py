from rag_engine.confidence import assess_retrieval_confidence, should_abstain
from rag_engine.models import Chunk, ScoredChunk


def scored(doc: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"{doc}-c",
            document_id=doc,
            text="evidence",
            start_token=0,
            end_token=1,
            metadata={},
        ),
        score=score,
        lexical_score=score,
        semantic_score=score,
    )


def test_empty_retrieval_abstains():
    confidence = assess_retrieval_confidence([])
    assert confidence.score == 0
    assert should_abstain([])


def test_multiple_strong_sources_increase_confidence():
    results = [scored("a", 3.0), scored("b", 1.5), scored("c", 1.0), scored("d", 0.8)]
    confidence = assess_retrieval_confidence(results)
    assert confidence.score > 0.6
    assert confidence.document_diversity == 1.0
