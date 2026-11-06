from dataclasses import asdict
from threading import Lock, RLock
from time import perf_counter

from rag_engine.agent import AgenticRetriever, QueryPlanner
from rag_engine.chunking import chunk_document
from rag_engine.confidence import assess_retrieval_confidence
from rag_engine.context import ContextBuilder
from rag_engine.embeddings import EmbeddingProvider, HashEmbeddingProvider
from rag_engine.generation import ExtractiveGenerator, Generator
from rag_engine.models import Chunk, Document, RetrievalTrace, ScoredChunk
from rag_engine.provenance import describe_embedding_provider, index_fingerprint, verify_citations
from rag_engine.rerank import TransparentReranker
from rag_engine.retrieval import HybridRetriever
from rag_engine.storage import DocumentStore

DEFAULT_MAX_INDEX_CHUNKS = 20_000


class RAGEngine:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        generator: Generator | None = None,
        query_planner: QueryPlanner | None = None,
        reranker: TransparentReranker | None = None,
        chunk_size: int = 180,
        overlap: int = 30,
        context_tokens: int = 700,
        document_store: DocumentStore | None = None,
        abstain_threshold: float = 0.45,
        max_index_chunks: int = DEFAULT_MAX_INDEX_CHUNKS,
    ) -> None:
        if not 0 <= abstain_threshold <= 1:
            raise ValueError("abstain_threshold must be between 0 and 1")
        if type(max_index_chunks) is not int or max_index_chunks <= 0:
            raise ValueError("max_index_chunks must be a positive integer.")
        self.max_index_chunks = max_index_chunks
        self.embedding_provider = embedding_provider
        self.generator = generator or ExtractiveGenerator()
        self.query_planner = query_planner
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.context_builder = ContextBuilder(max_tokens=context_tokens)
        self.document_store = document_store
        self.abstain_threshold = abstain_threshold
        self._lock = RLock()
        self._writer_lock = Lock()
        stored = document_store.list_documents() if document_store is not None else []
        self.documents: dict[str, Document] = {document.id: document for document in stored}
        self._agent, _ = self._build_index(self.documents)
        self._fingerprint: tuple[AgenticRetriever, str] | None = None

    def _build_index(
        self, documents: dict[str, Document]
    ) -> tuple[AgenticRetriever | None, int]:
        if not documents:
            return None, 0
        chunks: list[Chunk] = []
        for document in documents.values():
            chunks.extend(chunk_document(document, chunk_size=self.chunk_size, overlap=self.overlap))
            if len(chunks) > self.max_index_chunks:
                raise ValueError(f"Index exceeds the {self.max_index_chunks} chunk limit.")
        return (
            AgenticRetriever(
                HybridRetriever(chunks, embedding_provider=self.embedding_provider),
                planner=self.query_planner,
                reranker=self.reranker,
            ),
            len(chunks),
        )

    def _publish(self, documents: dict[str, Document], agent: AgenticRetriever | None) -> None:
        with self._lock:
            self.documents = documents
            self._agent = agent
            self._fingerprint = None

    def _index_transaction(
        self, documents: list[Document], *, with_receipt: bool
    ) -> tuple[int, str | None]:
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Document IDs must be unique within an index batch.")
        if any(not document.id.strip() or not document.text.strip() for document in documents):
            raise ValueError("Documents require non-empty IDs and text.")
        with self._writer_lock:
            next_documents = {**self.documents, **{doc.id: doc for doc in documents}}
            agent, chunk_count = self._build_index(next_documents)
            if documents and self.document_store is not None:
                self.document_store.upsert_many(documents)
            self._publish(next_documents, agent)
            # Keep the receipt inside the writer transaction. Otherwise a second
            # ingest can publish a newer index before the caller reads its hash.
            fingerprint = self.index_fingerprint() if with_receipt else None
            return chunk_count, fingerprint

    def index(self, documents: list[Document]) -> int:
        count, _ = self._index_transaction(documents, with_receipt=False)
        return count

    def index_with_receipt(self, documents: list[Document]) -> tuple[int, str | None]:
        """Return the count and hash from the same committed index snapshot."""

        return self._index_transaction(documents, with_receipt=True)

    def delete(self, document_id: str) -> bool:
        with self._writer_lock:
            if document_id not in self.documents:
                return False
            next_documents = self.documents.copy()
            del next_documents[document_id]
            agent, _ = self._build_index(next_documents)
            if self.document_store is not None:
                self.document_store.delete(document_id)
            self._publish(next_documents, agent)
            return True

    def _fingerprint_for(self, agent: AgenticRetriever) -> str:
        cached = self._fingerprint
        if cached is not None and cached[0] is agent:
            return cached[1]
        retriever = agent.retriever
        fingerprint = index_fingerprint(
            retriever.lexical.chunks,
            describe_embedding_provider(retriever.semantic.provider),
        )
        self._fingerprint = (agent, fingerprint)
        return fingerprint

    def index_fingerprint(self) -> str | None:
        """SHA-256 of the currently published index (``None`` when nothing is indexed)."""
        with self._lock:
            agent = self._agent
            return None if agent is None else self._fingerprint_for(agent)

    @property
    def uses_hash_embeddings(self) -> bool:
        provider = self.embedding_provider or HashEmbeddingProvider()
        return describe_embedding_provider(provider)["kind"] == "sha256_token_hash"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[ScoredChunk], RetrievalTrace]:
        with self._lock:
            agent = self._agent
            fingerprint = None if agent is None else self._fingerprint_for(agent)
        if agent is None:
            raise RuntimeError("No documents have been indexed")

        started = perf_counter()
        results, planned_queries, candidate_count = agent.retrieve(
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
            index_fingerprint=fingerprint,
        )
        return results, trace

    def answer(self, query: str, top_k: int = 5) -> dict[str, object]:
        results, trace = self.retrieve(query, top_k=top_k)
        confidence = assess_retrieval_confidence(results)
        context = self.context_builder.build(results)

        abstained = confidence.score < self.abstain_threshold
        if abstained:
            answer = (
                "I do not have enough independent evidence in the indexed documents "
                "to answer this reliably."
            )
        else:
            answer = self.generator.generate(query, context)
        citation_check = verify_citations(answer, context.citations)

        return {
            "answer": answer,
            "citations": context.citations,
            # Citation-shaped IDs that appear in the answer text and were in the context,
            # and any that were not (an ungrounded or hallucinated citation).
            "answer_citations": citation_check.cited_ids,
            "unsupported_citations": citation_check.unknown_ids,
            "results": results,
            "context": context.text,
            "trace": trace,
            "confidence": asdict(confidence),
            "abstained": abstained,
        }
