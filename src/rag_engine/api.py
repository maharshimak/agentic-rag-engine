import os
import secrets
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from rag_engine.models import Document
from rag_engine.settings import build_engine


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=1_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentInput] = Field(min_length=1, max_length=500)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)


app = FastAPI(
    title="Agentic RAG Engine",
    version="1.1.0",
    description=(
        "Hybrid retrieval, query planning, reranking, citations, persistence and "
        "optional bearer-protected service access."
    ),
)
engine = build_engine()


async def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    token = os.environ.get("RAG_API_TOKEN")
    if token:
        scheme, _, supplied = (authorization or "").partition(" ")
        if (
            scheme.lower() != "bearer"
            or not supplied
            or not secrets.compare_digest(supplied, token)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Valid bearer token required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return

    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Remote access requires RAG_API_TOKEN.",
        )


protected = [Depends(require_auth)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/documents/index", dependencies=protected)
def index_documents(request: IndexRequest) -> dict[str, int]:
    documents = [
        Document(id=item.id, text=item.text, metadata=item.metadata) for item in request.documents
    ]
    chunk_count = engine.index(documents)
    return {"documents": len(documents), "chunks": chunk_count}


@app.post("/v1/retrieve", dependencies=protected)
def retrieve(request: QueryRequest) -> dict[str, object]:
    try:
        results, trace = engine.retrieve(request.query, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "results": [
            {
                "chunk_id": item.chunk.id,
                "document_id": item.chunk.document_id,
                "text": item.chunk.text,
                "score": item.score,
                "lexical_score": item.lexical_score,
                "semantic_score": item.semantic_score,
                "rerank_score": item.rerank_score,
                "metadata": item.chunk.metadata,
            }
            for item in results
        ],
        "trace": {
            "query": trace.query,
            "planned_queries": trace.planned_queries,
            "candidate_count": trace.candidate_count,
            "returned_count": trace.returned_count,
            "latency_ms": trace.latency_ms,
        },
    }


@app.post("/v1/answer", dependencies=protected)
def answer(request: QueryRequest) -> dict[str, object]:
    try:
        result = engine.answer(request.query, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    trace = result["trace"]
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "context": result["context"],
        "trace": {
            "query": trace.query,
            "planned_queries": trace.planned_queries,
            "candidate_count": trace.candidate_count,
            "returned_count": trace.returned_count,
            "latency_ms": trace.latency_ms,
        },
    }
