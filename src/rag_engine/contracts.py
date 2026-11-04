"""MAK'MA shared execution contract ``makma.v1`` for the RAG engine.

``run_operation`` exposes four real engine operations (``ingest``, ``retrieve``,
``answer``, ``delete_document``) inside the shared envelope validated by
``contracts/makma.v1.schema.json``.
"""

from __future__ import annotations

import ipaddress
import re
import secrets
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from time import perf_counter
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_engine.models import Document, ScoredChunk
from rag_engine.provenance import sha256_text
from rag_engine.service import RAGEngine

CONTRACT_VERSION = "makma.v1"
SYSTEM = "agentic-rag-engine"

Status = Literal["ok", "error", "blocked", "pending_approval"]
ExecutionMode = Literal["deterministic_offline", "local_model", "remote_model"]


class ContractError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    message: str = Field(max_length=2000)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["document", "chunk", "artifact", "dataset", "model", "query", "record"]
    id: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    span: tuple[int, int] | None = None
    uri: str | None = None


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["allow", "deny", "requires_approval"]
    reasons: list[str] = Field(default_factory=list)


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["makma.v1"] = CONTRACT_VERSION
    system: Literal["agentic-rag-engine"] = SYSTEM
    operation: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    run_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    status: Status
    execution_mode: ExecutionMode
    data: Any = None
    error: ContractError | None = None
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    policy: Policy | None = None
    started_at: str
    duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def status_invariants(self) -> Envelope:
        if self.status == "ok" and self.error is not None:
            raise ValueError("status ok requires error == null")
        if self.status in {"error", "blocked"} and (self.error is None or self.data is not None):
            raise ValueError("error/blocked status requires an error and null data")
        if self.status == "pending_approval" and (
            self.policy is None or self.policy.decision != "requires_approval"
        ):
            raise ValueError("pending_approval requires policy.decision == requires_approval")
        return self


# --------------------------------------------------------------------------- inputs


class DocumentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=1_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentPayload] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def bounded(self) -> IngestInput:
        if sum(len(item.text) for item in self.documents) > 20_000_000:
            raise ValueError("Index payload exceeds the 20,000,000 character aggregate limit.")
        if len({item.id for item in self.documents}) != len(self.documents):
            raise ValueError("Document IDs must be unique within an ingest batch.")
        for item in self.documents:
            if not item.id.strip() or not item.text.strip():
                raise ValueError("Documents require non-empty IDs and text.")
        return self


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def non_blank(self) -> QueryInput:
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text.")
        return self


class DeleteDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=500)


class _OperationFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------- mode/provenance


def _remote_mode(base_url: str) -> ExecutionMode:
    host = (urlparse(base_url).hostname or "").strip("[]").casefold()
    if host == "localhost" or host.endswith(".localhost"):
        return "local_model"
    try:
        return "local_model" if ipaddress.ip_address(host).is_loopback else "remote_model"
    except ValueError:
        return "remote_model"


_DETERMINISTIC_COMPONENTS = {
    "NoneType",  # engine default -> built-in deterministic component
    "HashEmbeddingProvider",
    "ExtractiveGenerator",
    "QueryPlanner",
    "TransparentReranker",
}


def _component_mode(component: Any) -> ExecutionMode:
    inner = getattr(component, "provider", None)
    if inner is not None and type(component).__name__ == "SQLiteCachedEmbeddingProvider":
        return _component_mode(inner)
    endpoint = getattr(component, "endpoint", None) or getattr(component, "base_url", None)
    if isinstance(endpoint, str) and endpoint:
        return _remote_mode(endpoint)
    name = type(component).__name__
    if name == "MMRDiversityReranker":
        return _component_mode(component.embedding_provider)
    if name in _DETERMINISTIC_COMPONENTS:
        return "deterministic_offline"
    if name == "CrossEncoderReranker":
        return "local_model"
    # Unknown injected component: do not claim it is deterministic.
    return "local_model"


_MODE_RANK = {"deterministic_offline": 0, "local_model": 1, "remote_model": 2}


def _combined_mode(*components: Any) -> ExecutionMode:
    modes = [_component_mode(component) for component in components]
    return max(modes, key=_MODE_RANK.__getitem__, default="deterministic_offline")


def _retrieval_components(engine: RAGEngine) -> tuple[Any, ...]:
    return (engine.embedding_provider, engine.query_planner, engine.reranker)


def _chunk_provenance(item: ScoredChunk) -> ProvenanceRef:
    chunk = item.chunk
    if chunk.source_span_sha256 is None or chunk.start_char is None or chunk.end_char is None:
        return ProvenanceRef(kind="chunk", id=chunk.id)
    return ProvenanceRef(
        kind="chunk",
        id=chunk.id,
        sha256=chunk.source_span_sha256,
        span=(chunk.start_char, chunk.end_char),
    )


def _result_row(item: ScoredChunk) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk.id,
        "document_id": item.chunk.document_id,
        "text": item.chunk.text,
        "text_sha256": sha256_text(item.chunk.text),
        "start_char": item.chunk.start_char,
        "end_char": item.chunk.end_char,
        "score": item.score,
        "lexical_score": item.lexical_score,
        "semantic_score": item.semantic_score,
        "rerank_score": item.rerank_score,
        "metadata": item.chunk.metadata,
    }


def _trace_dict(trace: Any) -> dict[str, Any]:
    return {
        "query": trace.query,
        "planned_queries": list(trace.planned_queries),
        "candidate_count": trace.candidate_count,
        "returned_count": trace.returned_count,
        "latency_ms": trace.latency_ms,
        "index_fingerprint": trace.index_fingerprint,
    }


# --------------------------------------------------------------------------- operations

OpResult = tuple[dict[str, Any], list[ProvenanceRef], ExecutionMode]


def _require_index(engine: RAGEngine) -> None:
    if engine.index_fingerprint() is None:
        raise _OperationFailure("INDEX_EMPTY", "No documents have been indexed.")


def _op_ingest(engine: RAGEngine, payload: IngestInput) -> OpResult:
    documents = [
        Document(id=item.id, text=item.text, metadata=dict(item.metadata))
        for item in payload.documents
    ]
    chunk_count = engine.index(documents)
    data = {
        "documents": len(documents),
        "document_ids": [document.id for document in documents],
        "indexed_chunks": chunk_count,
        "index_fingerprint": engine.index_fingerprint(),
    }
    provenance = [
        ProvenanceRef(
            kind="document",
            id=document.id,
            sha256=sha256_text(document.text),
            span=(0, len(document.text)),
        )
        for document in documents
    ]
    return data, provenance, _combined_mode(engine.embedding_provider)


def _op_retrieve(engine: RAGEngine, payload: QueryInput) -> OpResult:
    _require_index(engine)
    results, trace = engine.retrieve(payload.query, top_k=payload.top_k)
    data = {
        "results": [_result_row(item) for item in results],
        "trace": _trace_dict(trace),
    }
    provenance = [_chunk_provenance(item) for item in results]
    return data, provenance, _combined_mode(*_retrieval_components(engine))


def _op_answer(engine: RAGEngine, payload: QueryInput) -> OpResult:
    _require_index(engine)
    result = engine.answer(payload.query, top_k=payload.top_k)
    results: list[ScoredChunk] = result["results"]
    by_id = {item.chunk.id: item for item in results}
    citations: list[str] = result["citations"]
    retrieved_ids = [item.chunk.id for item in results]
    data = {
        "answer": result["answer"],
        "abstained": result["abstained"],
        "confidence": result["confidence"],
        "citations": citations,
        "answer_citations": result["answer_citations"],
        "unsupported_citations": result["unsupported_citations"],
        "citations_in_retrieved_set": set(citations).issubset(retrieved_ids),
        "context": result["context"],
        "retrieved_chunk_ids": retrieved_ids,
        "trace": _trace_dict(result["trace"]),
    }
    provenance = [_chunk_provenance(by_id[chunk_id]) for chunk_id in citations]
    components = _retrieval_components(engine)
    if not result["abstained"]:
        components = (*components, engine.generator)
    return data, provenance, _combined_mode(*components)


def _op_delete_document(engine: RAGEngine, payload: DeleteDocumentInput) -> OpResult:
    deleted = engine.delete(payload.document_id)
    data = {
        "document_id": payload.document_id,
        "deleted": deleted,
        "documents_remaining": len(engine.documents),
        "index_fingerprint": engine.index_fingerprint(),
    }
    return data, [], _combined_mode(engine.embedding_provider)


_DOCUMENT_SCHEMA = DocumentPayload.model_json_schema()


class _Operation:
    def __init__(
        self,
        handler: Callable[[RAGEngine, Any], OpResult],
        model: type[BaseModel],
        *,
        side_effects: bool,
        timeout_s: int,
    ) -> None:
        self.handler = handler
        self.model = model
        self.side_effects = side_effects
        self.timeout_s = timeout_s


OPERATIONS: dict[str, _Operation] = {
    "ingest": _Operation(_op_ingest, IngestInput, side_effects=True, timeout_s=120),
    "retrieve": _Operation(_op_retrieve, QueryInput, side_effects=False, timeout_s=30),
    "answer": _Operation(_op_answer, QueryInput, side_effects=False, timeout_s=60),
    "delete_document": _Operation(
        _op_delete_document, DeleteDocumentInput, side_effects=True, timeout_s=60
    ),
}


def _package_version() -> str:
    try:
        return package_version("agentic-rag-engine")
    except PackageNotFoundError:
        return "0+unknown"


def capabilities() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "system": SYSTEM,
        "version": _package_version(),
        "operations": [
            {
                "name": name,
                "side_effects": operation.side_effects,
                "requires_approval": False,
                "input_schema": operation.model.model_json_schema(),
                "timeout_s": operation.timeout_s,
            }
            for name, operation in OPERATIONS.items()
        ],
    }


_default_engine: RAGEngine | None = None


def default_engine() -> RAGEngine:
    """Process-wide engine used when callers do not inject one (built from RAG_* settings)."""
    global _default_engine
    if _default_engine is None:
        from rag_engine.settings import build_engine

        _default_engine = build_engine()
    return _default_engine


def _safe_operation_name(operation: object) -> str:
    # The envelope's operation field has a strict pattern; echo it only when it fits.
    if isinstance(operation, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", operation):
        return operation
    return "unknown"


def run_operation(
    operation: str,
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
    correlation_id: str | None = None,
    engine: RAGEngine | None = None,
) -> dict[str, Any]:
    """Execute one operation and return a ``makma.v1`` envelope as a plain dict."""
    started = perf_counter()
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    run = (run_id or uuid.uuid4().hex)[:128]
    correlation = (correlation_id or run)[:128]
    status: Status = "ok"
    data: Any = None
    error: ContractError | None = None
    provenance: list[ProvenanceRef] = []
    mode: ExecutionMode = "deterministic_offline"

    spec = OPERATIONS.get(operation) if isinstance(operation, str) else None
    if spec is None:
        status = "error"
        error = ContractError(
            code="UNKNOWN_OPERATION",
            message="Unknown operation.",
            details={"supported": sorted(OPERATIONS)},
        )
    else:
        active = engine if engine is not None else default_engine()
        try:
            if not isinstance(payload, dict):
                raise _OperationFailure("INVALID_INPUT", "Payload must be a JSON object.")
            try:
                parsed = spec.model.model_validate(payload)
            except ValidationError as exc:
                raise _OperationFailure(
                    "INVALID_INPUT",
                    "; ".join(
                        f"{'.'.join(str(p) for p in item['loc']) or 'payload'}: {item['msg']}"
                        for item in exc.errors()
                    )[:2000],
                ) from exc
            data, provenance, mode = spec.handler(active, parsed)
        except _OperationFailure as exc:
            status, data = "error", None
            error = ContractError(code=exc.code, message=exc.message, retryable=exc.retryable)
            provenance = []
        except ValueError as exc:
            status, data, provenance = "error", None, []
            error = ContractError(code="INVALID_INPUT", message=str(exc)[:2000])
        except OSError as exc:  # URLError/timeouts from configured model endpoints.
            status, data, provenance = "error", None, []
            error = ContractError(
                code="PROVIDER_UNAVAILABLE",
                message=f"Model provider call failed: {type(exc).__name__}.",
                retryable=True,
            )
        except Exception as exc:  # noqa: BLE001 - envelope must never leak a traceback.
            status, data, provenance = "error", None, []
            error = ContractError(
                code="INTERNAL_ERROR",
                message=f"Operation failed: {type(exc).__name__}.",
            )

    envelope = Envelope(
        operation=_safe_operation_name(operation),
        run_id=run,
        correlation_id=correlation,
        trace_id=secrets.token_hex(16),
        span_id=secrets.token_hex(8),
        status=status,
        execution_mode=mode,
        data=data,
        error=error,
        provenance=provenance,
        policy=None,
        started_at=started_at,
        duration_ms=max(0.0, (perf_counter() - started) * 1000),
    )
    return envelope.model_dump(mode="json")


__all__ = ["OPERATIONS", "capabilities", "default_engine", "run_operation"]
