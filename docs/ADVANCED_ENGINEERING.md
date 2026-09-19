# Advanced engineering: retrieval cache

`rag_engine.cache` provides a deterministic TTL/LRU cache for retrieval results.

Cache keys normalize query whitespace/casing and sort filter pairs, so semantically equivalent
requests share a key. Entries expire by TTL, recently used entries are promoted, and capacity
pressure evicts the least-recently-used entry.

The cache exposes hit, miss, eviction and size counters without depending on Redis or another
service. That keeps unit tests fast while leaving a clean seam for a distributed cache adapter in a
larger deployment.

Typical use: build a key from query/top-k/filters, attempt `get`, run retrieval on a miss, then
`put` the returned document identifiers.
