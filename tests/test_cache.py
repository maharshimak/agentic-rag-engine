from rag_engine.cache import RetrievalCache, make_cache_key


def test_cache_key_is_canonical() -> None:
    first = make_cache_key(
        "  Revenue   growth ",
        top_k=5,
        filters={"year": 2026, "region": "FR"},
    )
    second = make_cache_key(
        "revenue growth",
        top_k=5,
        filters={"region": "FR", "year": 2026},
    )

    assert first == second


def test_cache_expires_entries_and_tracks_metrics() -> None:
    now = [100.0]
    cache = RetrievalCache(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    key = make_cache_key("risk policy", top_k=3)

    cache.put(key, ["doc-1", "doc-2"])
    assert cache.get(key) == ("doc-1", "doc-2")

    now[0] = 111.0
    assert cache.get(key) is None

    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.size == 0


def test_cache_evicts_least_recently_used_entry() -> None:
    cache = RetrievalCache(max_entries=2, ttl_seconds=30)
    first = make_cache_key("first", top_k=1)
    second = make_cache_key("second", top_k=1)
    third = make_cache_key("third", top_k=1)

    cache.put(first, ["a"])
    cache.put(second, ["b"])
    assert cache.get(first) == ("a",)
    cache.put(third, ["c"])

    assert cache.get(second) is None
    assert cache.stats().evictions == 1
