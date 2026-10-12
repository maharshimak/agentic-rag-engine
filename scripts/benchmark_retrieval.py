"""Small, deterministic retrieval regression suite. Not a live model benchmark."""

from __future__ import annotations

import json
from statistics import mean

from rag_engine.evaluation import evaluate_retrieval
from rag_engine.models import Document
from rag_engine.service import RAGEngine

DOCS = [
    Document("routing", "Query routing chooses the appropriate retrieval source for each user question."),
    Document("storage", "SQLite snapshot transactions keep knowledge graph nodes and edges consistent."),
    Document("security", "SQL authorization enforces table allowlists and denies sensitive columns."),
    Document("evaluation", "LLM evaluation calculates citation coverage relevance latency and quality gates."),
    Document("deployment", "Canary deployment checks release regression thresholds before promoting production."),
    Document("rendering", "FFmpeg atomic rendering preserves the previous output when transcoding fails."),
]
CASES = [
    ("SQL authorization table allowlists sensitive columns", "security"),
    ("SQLite snapshot knowledge graph nodes and edges", "storage"),
    ("FFmpeg atomic rendering previous output", "rendering"),
    ("Canary deployment regression thresholds", "deployment"),
    ("LLM evaluation citation coverage relevance", "evaluation"),
    ("Query routing appropriate retrieval source", "routing"),
]


def benchmark() -> dict[str, object]:
    engine = RAGEngine(chunk_size=180, overlap=30)
    engine.index(DOCS)
    rows: list[dict[str, object]] = []
    for query, target in CASES:
        ranked, trace = engine.retrieve(query, top_k=3)
        # Ranking metrics apply to unique document IDs, not duplicated chunks.
        ids = list(dict.fromkeys(result.chunk.document_id for result in ranked))
        metrics = evaluate_retrieval({target}, ids, k=3)
        rows.append(
            {
                "query": query,
                "expected_doc": target,
                "retrieved": ids,
                "hit_at_3": metrics.hit_rate_at_k,
                "mrr": metrics.mrr,
                "latency_ms": round(trace.latency_ms, 3),
            }
        )
    return {
        "dataset": "six synthetic distinct-topic documents; deterministic offline mode",
        "cases": len(CASES),
        "hit_rate_at_3": mean(float(row["hit_at_3"]) for row in rows),
        "mrr": mean(float(row["mrr"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    result = benchmark()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["hit_rate_at_3"] < 1.0 or result["mrr"] < 0.75:
        raise SystemExit("Synthetic retrieval regression gate failed")


if __name__ == "__main__":
    main()
