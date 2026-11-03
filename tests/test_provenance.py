import hashlib
import random

import pytest

from rag_engine.chunking import chunk_document
from rag_engine.embeddings import HashEmbeddingProvider
from rag_engine.models import Chunk, Document
from rag_engine.provenance import (
    describe_embedding_provider,
    index_fingerprint,
    sha256_text,
    source_span,
    verify_citations,
)
from rag_engine.service import RAGEngine


def test_chunk_offsets_map_back_to_source_with_irregular_whitespace() -> None:
    document = Document("d", "  alpha\tbeta\n\ngamma   delta  epsilon zeta ")
    chunks = chunk_document(document, chunk_size=3, overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma",
        "gamma delta epsilon",
        "epsilon zeta",
    ]
    for chunk in chunks:
        span = source_span(chunk, document)
        assert span is not None
        assert span == document.text[chunk.start_char : chunk.end_char]
        assert " ".join(span.split()) == chunk.text
        assert not span[0].isspace() and not span[-1].isspace()


@pytest.mark.parametrize("seed", range(25))
def test_offsets_round_trip_for_random_documents(seed: int) -> None:
    rng = random.Random(seed)
    words = ["rag", "évidence", "chunk", "ID-7", "naïve", "x", "检索", "token"]
    separators = [" ", "  ", "\n", "\t", " \n "]
    text = "".join(rng.choice(words) + rng.choice(separators) for _ in range(rng.randint(1, 60)))
    document = Document("doc", text)
    size = rng.randint(1, 12)
    overlap = rng.randint(0, size - 1)

    chunks = chunk_document(document, chunk_size=size, overlap=overlap)

    assert chunks
    assert chunks[0].start_char == len(text) - len(text.lstrip())
    assert chunks[-1].end_char == len(text.rstrip())
    for chunk in chunks:
        assert source_span(chunk, document) is not None


def test_source_span_rejects_stale_or_foreign_chunks() -> None:
    document = Document("d", "alpha beta gamma")
    (chunk,) = chunk_document(document)
    edited = Document("d", "alpha BETA gamma")

    assert source_span(chunk, edited) is None
    assert source_span(chunk, Document("other", document.text)) is None
    assert source_span(Chunk("x", "d", "alpha", 0, 1), document) is None


def test_sha256_matches_hashlib() -> None:
    assert sha256_text("évidence") == hashlib.sha256("évidence".encode()).hexdigest()


def test_verify_citations_separates_known_and_unknown_ids() -> None:
    answer = "See [a::c0] and [b::c3], again [a::c0]. Ignore [note] and [1]. Bogus [z::c9]."
    check = verify_citations(answer, ["a::c0", "b::c3", "c::c0"])

    assert check.cited_ids == ["a::c0", "b::c3"]
    assert check.unknown_ids == ["z::c9"]
    assert not check.all_citations_known


def test_fingerprint_is_order_independent_and_content_sensitive() -> None:
    first = chunk_document(Document("a", "alpha beta", {"v": 1}))
    second = chunk_document(Document("b", "gamma delta"))
    embedding = describe_embedding_provider(HashEmbeddingProvider())

    base = index_fingerprint(first + second, embedding)
    assert base == index_fingerprint(second + first, embedding)
    assert len(base) == 64
    changed_text = chunk_document(Document("b", "gamma delta!"))
    assert index_fingerprint(first + changed_text, embedding) != base
    changed_meta = chunk_document(Document("a", "alpha beta", {"v": 2}))
    assert index_fingerprint(changed_meta + second, embedding) != base
    other_dims = describe_embedding_provider(HashEmbeddingProvider(dimensions=64))
    assert index_fingerprint(first + second, other_dims) != base


def test_engine_fingerprint_tracks_index_and_deletion() -> None:
    engine = RAGEngine()
    assert engine.index_fingerprint() is None
    engine.index([Document("a", "alpha evidence"), Document("b", "beta evidence")])
    two = engine.index_fingerprint()
    _, trace = engine.retrieve("alpha", top_k=1)
    assert trace.index_fingerprint == two

    engine.delete("b")
    assert engine.index_fingerprint() not in {None, two}
    engine.index([Document("b", "beta evidence")])
    assert engine.index_fingerprint() == two
    engine.delete("a")
    engine.delete("b")
    assert engine.index_fingerprint() is None


def test_deleted_documents_never_retrievable_across_repeated_queries() -> None:
    engine = RAGEngine(chunk_size=4, overlap=1)
    engine.index(
        [
            Document("keep", "retained evidence about backups and restores"),
            Document("drop", "secret legacy evidence about backups and restores to remove"),
        ]
    )
    assert engine.delete("drop") is True

    for _ in range(3):
        for query in ["legacy evidence", "backups restores", "remove secret", "drop"]:
            results, _ = engine.retrieve(query, top_k=50)
            assert all(item.chunk.document_id == "keep" for item in results)
            answer = engine.answer(query, top_k=50)
            assert all(not cid.startswith("drop::") for cid in answer["citations"])
            assert "secret legacy" not in answer["context"]


def test_answer_citations_are_subset_of_retrieved_and_context() -> None:
    engine = RAGEngine(context_tokens=6)
    engine.index(
        [
            Document("a", "alpha evidence about retrieval quality and citations"),
            Document("b", "beta evidence about retrieval latency"),
        ]
    )
    result = engine.answer("retrieval evidence", top_k=2)
    retrieved = {item.chunk.id for item in result["results"]}

    assert set(result["citations"]) <= retrieved
    assert set(result["answer_citations"]) <= set(result["citations"])
    assert result["unsupported_citations"] == []


class HallucinatingGenerator:
    def generate(self, query, context):
        return f"[{context.citations[0]}] plus [ghost::c4]"


def test_unsupported_generator_citations_are_reported() -> None:
    engine = RAGEngine(generator=HallucinatingGenerator(), abstain_threshold=0.0)
    engine.index([Document("a", "alpha evidence")])
    result = engine.answer("alpha", top_k=1)

    assert result["answer_citations"] == ["a::c0"]
    assert result["unsupported_citations"] == ["ghost::c4"]
