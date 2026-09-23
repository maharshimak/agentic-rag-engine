from rag_engine.embeddings import SQLiteCachedEmbeddingProvider


class CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


def test_sqlite_embedding_cache_reuses_unchanged_text(tmp_path) -> None:
    provider = CountingProvider()
    cache = SQLiteCachedEmbeddingProvider(provider, tmp_path / "embeddings.db", namespace="v1")

    first = cache.embed(["alpha", "beta"])
    second = cache.embed(["alpha", "beta"])

    assert first == second
    assert provider.calls == 1


def test_sqlite_embedding_cache_only_generates_missing_values(tmp_path) -> None:
    provider = CountingProvider()
    cache = SQLiteCachedEmbeddingProvider(provider, tmp_path / "embeddings.db", namespace="v1")

    cache.embed(["alpha"])
    cache.embed(["alpha", "beta"])

    assert provider.calls == 2
