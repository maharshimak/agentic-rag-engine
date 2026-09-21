import pytest

from rag_engine.embeddings import HashEmbeddingProvider


@pytest.mark.parametrize("value", [True, 1.5, 0, -1, 100000000])
def test_embedding_dimensions_are_bounded_integers(value):
    with pytest.raises(ValueError):
        HashEmbeddingProvider(value)


def test_hash_vector_is_repeatable_and_normalized():
    provider = HashEmbeddingProvider()
    first = provider.embed(["MAKMA RAG RAG"])[0]
    assert first == provider.embed(["makma rag rag"])[0]
    assert sum(x * x for x in first) == pytest.approx(1)
