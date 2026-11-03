"""Regression: tokenize kept edge punctuation, so sentence-final words never matched.

On 47d0119 ``tokenize("ranks evidence.")`` returned ``["ranks", "evidence."]`` and a
query for ``evidence`` scored zero BM25 term frequency for that occurrence.
"""

import pytest

from rag_engine.models import Chunk
from rag_engine.retrieval import BM25Index, tokenize


def test_sentence_final_punctuation_is_stripped() -> None:
    assert tokenize("ranks evidence.") == ["ranks", "evidence"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("v1.2", ["v1.2"]),
        ("gpt-4o", ["gpt-4o"]),
        ("rag-engine", ["rag-engine"]),
        ("ERR_TLS_042", ["err_tls_042"]),
        ("release-2026.10.", ["release-2026.10"]),
        ("now- .env _private_ ...", ["now", "env", "private"]),
        ("-- . _ -", []),
    ],
)
def test_internal_punctuation_survives_edges_are_trimmed(text, expected) -> None:
    assert tokenize(text) == expected


def test_bm25_matches_sentence_final_term_like_mid_sentence_term() -> None:
    def chunk(chunk_id: str, text: str) -> Chunk:
        return Chunk(chunk_id, chunk_id, text, 0, len(text.split()))

    index = BM25Index(
        [
            chunk("final", "the ranker returns evidence."),
            chunk("middle", "the evidence ranker returns"),
            chunk("other", "unrelated words only here"),
        ]
    )
    scores = {item.chunk.id: item.score for item in index.search("evidence")}

    assert set(scores) == {"final", "middle"}
    assert scores["final"] == pytest.approx(scores["middle"])
