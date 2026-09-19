import re

from rag_engine.models import ScoredChunk
from rag_engine.rerank import TransparentReranker
from rag_engine.retrieval import HybridRetriever, reciprocal_rank_fusion


class QueryPlanner:
    """Small deterministic planner with a replaceable LLM-planner boundary."""

    def plan(self, query: str, max_queries: int = 3) -> list[str]:
        cleaned = " ".join(query.split())
        parts = [
            part.strip(" ,;")
            for part in re.split(
                r"\b(?:and|versus|vs\.?|then)\b|[?;]",
                cleaned,
                flags=re.IGNORECASE,
            )
            if part.strip(" ,;")
        ]
        planned = [cleaned]
        for part in parts:
            if part.lower() != cleaned.lower() and part not in planned:
                planned.append(part)
        return planned[:max_queries]


class AgenticRetriever:
    def __init__(
        self,
        retriever: HybridRetriever,
        planner: QueryPlanner | None = None,
        reranker: TransparentReranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.planner = planner or QueryPlanner()
        self.reranker = reranker or TransparentReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> tuple[list[ScoredChunk], list[str], int]:
        planned_queries = self.planner.plan(query)
        rankings = [
            self.retriever.retrieve(
                planned_query,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )
            for planned_query in planned_queries
        ]
        fused = reciprocal_rank_fusion(rankings)
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        return reranked, planned_queries, len(fused)
