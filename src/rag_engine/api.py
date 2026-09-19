from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine.models import Document
from rag_engine.settings import build_engine


class DocumentInput(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[DocumentInput]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


app = FastAPI(
    title="Agentic RAG Engine",
    version="1.0.0",
    description="Hybrid retrieval, query planning, reranking, citations and evaluation.",
)
engine = build_engine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/documents/index")
def index_documents(request: IndexRequest) -> dict[str, int]:
    documents = [
        Document(id=item.id, text=item.text, metadata=item.metadata) for item in request.documents
    ]
    chunk_count = engine.index(documents)
    return {"documents": len(documents), "chunks": chunk_count}


@app.post("/v1/retrieve")
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


@app.post("/v1/answer")
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
