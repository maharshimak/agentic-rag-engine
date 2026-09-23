from time import perf_counter

from rag_engine.agent import AgenticRetriever
from rag_engine.chunking import chunk_documents
from rag_engine.context import ContextBuilder
from rag_engine.embeddings import EmbeddingProvider
from rag_engine.generation import ExtractiveGenerator, Generator
from rag_engine.models import Document, RetrievalTrace, ScoredChunk
from rag_engine.retrieval import HybridRetriever
from rag_engine.storage import DocumentStore


class RAGEngine:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        generator: Generator | None = None,
        chunk_size: int = 180,
        overlap: int = 30,
        context_tokens: int = 700,
        document_store: DocumentStore | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.generator = generator or ExtractiveGenerator()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.context_builder = ContextBuilder(max_tokens=context_tokens)
        self.document_store = document_store
        stored = document_store.list_documents() if document_store is not None else []
        self.documents: dict[str, Document] = {document.id: document for document in stored}
        self._agent: AgenticRetriever | None = None
        if self.documents:
            self._rebuild_index()

    def _rebuild_index(self) -> int:
        if not self.documents:
            self._agent = None
            return 0
        chunks = chunk_documents(
            list(self.documents.values()),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        self._agent = AgenticRetriever(
            HybridRetriever(chunks, embedding_provider=self.embedding_provider)
        )
        return len(chunks)

    def index(self, documents: list[Document]) -> int:
        if not documents:
            return self._rebuild_index()
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Document IDs must be unique within an index batch.")
        for document in documents:
            if not document.id.strip() or not document.text.strip():
                raise ValueError("Documents require non-empty IDs and text.")
            self.documents[document.id] = document
        if self.document_store is not None:
            self.document_store.upsert_many(documents)
        return self._rebuild_index()

    def delete(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        if self.document_store is not None:
            self.document_store.delete(document_id)
        self._rebuild_index()
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[ScoredChunk], RetrievalTrace]:
        if self._agent is None:
            raise RuntimeError("No documents have been indexed")

        started = perf_counter()
        results, planned_queries, candidate_count = self._agent.retrieve(
            query,
            top_k=top_k,
        )
        latency_ms = (perf_counter() - started) * 1000
        trace = RetrievalTrace(
            query=query,
            planned_queries=planned_queries,
            candidate_count=candidate_count,
            returned_count=len(results),
            latency_ms=latency_ms,
        )
        return results, trace

    def answer(self, query: str, top_k: int = 5) -> dict[str, object]:
        results, trace = self.retrieve(query, top_k=top_k)
        context = self.context_builder.build(results)
        answer = self.generator.generate(query, context)

        return {
            "answer": answer,
            "citations": context.citations,
            "context": context.text,
            "trace": trace,
        }
