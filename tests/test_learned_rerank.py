from rag_engine.learned_rerank import CrossEncoderReranker
from rag_engine.models import Chunk, ScoredChunk


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.1 if "lexical" in text else 0.9 for _, text in pairs]


def candidate(chunk_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=chunk_id,
            text=text,
            start_token=0,
            end_token=len(text.split()),
        ),
        score=0.0,
    )


def test_cross_encoder_adapter_uses_model_scores():
    reranker = CrossEncoderReranker(scorer=FakeCrossEncoder())
    rows = reranker.rerank(
        "semantic retrieval",
        [
            candidate("lex", "lexical sparse retrieval"),
            candidate("sem", "semantic dense retrieval"),
        ],
        top_k=2,
    )
    assert [row.chunk.id for row in rows] == ["sem", "lex"]
    assert rows[0].rerank_score == 0.9
