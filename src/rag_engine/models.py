from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    start_token: int
    end_token: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(frozen=True)
class RetrievalTrace:
    query: str
    planned_queries: list[str]
    candidate_count: int
    returned_count: int
    latency_ms: float
