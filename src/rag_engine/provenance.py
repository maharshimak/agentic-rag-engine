"""Evidence provenance: content hashes, source spans, citation checks, index fingerprints.

Everything here is deterministic and offline. Hashes are SHA-256 over UTF-8 bytes,
or over canonical JSON (sorted keys, ``(",", ":")`` separators, UTF-8).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rag_engine.models import Chunk, Document

FINGERPRINT_VERSION = "rag-index-fingerprint.v1"

# Citation-shaped tokens produced by the context builder: ``[<document id>::c<n>]``.
# Restricting to this shape avoids flagging ordinary bracketed prose such as "[note]".
_CITATION_RE = re.compile(r"\[([^\[\]\s]+::c\d+)\]")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    # ``default=str`` only affects non-JSON values (possible in library-supplied metadata);
    # JSON-native input hashes exactly as canonical JSON.
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_span(chunk: Chunk, document: Document) -> str | None:
    """Return the exact source text a chunk was cut from, or ``None`` if unknown/stale."""
    if chunk.start_char is None or chunk.end_char is None:
        return None
    if chunk.document_id != document.id:
        return None
    if not 0 <= chunk.start_char <= chunk.end_char <= len(document.text):
        return None
    span = document.text[chunk.start_char : chunk.end_char]
    # Chunk text is the whitespace-normalized form of its source span.
    if " ".join(span.split()) != chunk.text:
        return None
    return span


@dataclass(frozen=True, slots=True)
class CitationCheck:
    cited_ids: list[str]
    unknown_ids: list[str]

    @property
    def all_citations_known(self) -> bool:
        return not self.unknown_ids


def verify_citations(answer: str, allowed_ids: Iterable[str]) -> CitationCheck:
    """Split citation-shaped tokens in ``answer`` into supported and unsupported IDs.

    ``allowed_ids`` is the set of chunk IDs that were actually placed in the context.
    Order of first appearance is preserved and duplicates are removed.
    """
    allowed = set(allowed_ids)
    cited: list[str] = []
    unknown: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        chunk_id = match.group(1)
        bucket = cited if chunk_id in allowed else unknown
        if chunk_id not in bucket:
            bucket.append(chunk_id)
    return CitationCheck(cited_ids=cited, unknown_ids=unknown)


def describe_embedding_provider(provider: Any) -> dict[str, Any]:
    """Non-secret identity of an embedding provider, used in the index fingerprint."""
    inner = getattr(provider, "provider", None)
    if inner is not None and type(provider).__name__ == "SQLiteCachedEmbeddingProvider":
        return describe_embedding_provider(inner)
    name = type(provider).__name__
    if name == "HashEmbeddingProvider":
        return {"kind": "sha256_token_hash", "dimensions": provider.dimensions}
    if name == "OpenAICompatibleEmbeddingProvider":
        return {"kind": "openai_compatible", "model": provider.model}
    return {"kind": name}


def index_fingerprint(chunks: Sequence[Chunk], embedding: dict[str, Any]) -> str:
    """SHA-256 identifying the exact indexed content and embedding configuration.

    Two engines with the same fingerprint indexed identical chunk IDs, chunk text,
    offsets and metadata with the same embedding provider identity, independent of
    insertion order.
    """
    rows = sorted(
        (
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "text_sha256": sha256_text(chunk.text),
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ),
        key=lambda row: row["chunk_id"],
    )
    return canonical_json_sha256(
        {"version": FINGERPRINT_VERSION, "embedding": embedding, "chunks": rows}
    )
