from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class CacheKey:
    query: str
    top_k: int
    filters: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    size: int


@dataclass(frozen=True, slots=True)
class _Entry:
    document_ids: tuple[str, ...]
    expires_at: float


def make_cache_key(
    query: str,
    *,
    top_k: int,
    filters: Mapping[str, object] | None = None,
) -> CacheKey:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    normalized_filters = tuple(
        sorted((str(key), str(value)) for key, value in (filters or {}).items())
    )
    return CacheKey(
        query=" ".join(query.casefold().split()),
        top_k=top_k,
        filters=normalized_filters,
    )


class RetrievalCache:
    """Small TTL/LRU cache for deterministic retrieval results."""

    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_seconds: float = 60.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise TypeError("ttl_seconds must be a number")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[CacheKey, _Entry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: CacheKey) -> tuple[str, ...] | None:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.expires_at <= self._clock():
            del self._entries[key]
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.document_ids

    def put(self, key: CacheKey, document_ids: Sequence[str]) -> None:
        cleaned = tuple(document_id.strip() for document_id in document_ids)
        if any(not document_id for document_id in cleaned):
            raise ValueError("document_ids must contain non-empty strings")

        self._entries[key] = _Entry(
            document_ids=cleaned,
            expires_at=self._clock() + self.ttl_seconds,
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def invalidate(self, key: CacheKey) -> bool:
        return self._entries.pop(key, None) is not None

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            size=len(self._entries),
        )
