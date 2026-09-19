"""Backward-compatible helpers built on the production-style V2 modules."""

from rag_engine.context import ContextBuilder
from rag_engine.evaluation import recall_at_k
from rag_engine.models import Document
from rag_engine.service import RAGEngine


def hybrid_retrieve(
    query: str,
    docs: list[Document],
    top_k: int = 3,
) -> list[str]:
    engine = RAGEngine(chunk_size=180, overlap=30)
    engine.index(docs)
    results, _ = engine.retrieve(query, top_k=top_k)
    return [item.chunk.document_id for item in results]


def citation_context(ids: list[str], docs: list[Document]) -> str:
    lookup = {document.id: document.text for document in docs}
    return "\n\n".join(
        f"[{document_id}] {lookup[document_id]}" for document_id in ids if document_id in lookup
    )


__all__ = [
    "ContextBuilder",
    "Document",
    "citation_context",
    "hybrid_retrieve",
    "recall_at_k",
]
