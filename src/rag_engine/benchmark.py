"""Offline retrieval benchmark over a judged dataset file.

Usage::

    python -m rag_engine.benchmark benchmarks/retrieval_v1.json --k 5 [--output report.json]

Metrics are computed at document level (chunk rankings are collapsed to the first
occurrence of each document). The report is tied to the dataset by a SHA-256 of
its canonical JSON and to the index by ``RAGEngine.index_fingerprint()``. With the
default engine every number comes from SHA-256 token-hash vectors, BM25 and
heuristic reranking; it says nothing about learned embedding quality.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from rag_engine.embeddings import HashEmbeddingProvider
from rag_engine.evaluation import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from rag_engine.models import Document
from rag_engine.provenance import canonical_json_sha256, describe_embedding_provider
from rag_engine.service import RAGEngine

EngineFactory = Callable[[], RAGEngine]


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: dict[str, Any]) -> None:
    documents = dataset.get("documents")
    queries = dataset.get("queries")
    if not isinstance(documents, list) or not documents:
        raise ValueError("dataset.documents must be a non-empty list")
    if not isinstance(queries, list) or not queries:
        raise ValueError("dataset.queries must be a non-empty list")
    doc_ids = [doc.get("id") for doc in documents]
    if any(not isinstance(i, str) or not i.strip() for i in doc_ids):
        raise ValueError("every document needs a non-empty string id")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("document ids must be unique")
    if any(not isinstance(doc.get("text"), str) or not doc["text"].strip() for doc in documents):
        raise ValueError("every document needs non-empty text")
    deleted = dataset.get("delete_after_ingest", [])
    if not isinstance(deleted, list) or any(d not in doc_ids for d in deleted):
        raise ValueError("delete_after_ingest must list existing document ids")
    query_ids = [query.get("id") for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query ids must be unique")
    for query in queries:
        if not isinstance(query.get("query"), str) or not query["query"].strip():
            raise ValueError(f"query {query.get('id')!r} needs non-empty text")
        relevance = query.get("relevance")
        if not isinstance(relevance, dict):
            raise TypeError(f"query {query['id']!r} needs a relevance object")
        for doc_id, grade in relevance.items():
            if doc_id not in doc_ids:
                raise ValueError(f"query {query['id']!r} judges unknown document {doc_id!r}")
            if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
                raise ValueError(f"query {query['id']!r} has an invalid grade for {doc_id!r}")
            if grade > 0 and doc_id in deleted:
                raise ValueError(f"query {query['id']!r} marks deleted {doc_id!r} relevant")


def dataset_fingerprint(dataset: dict[str, Any]) -> str:
    """SHA-256 of the dataset's canonical JSON (formatting-independent)."""
    return canonical_json_sha256(dataset)


def _documents(dataset: dict[str, Any]) -> list[Document]:
    return [
        Document(doc["id"], doc["text"], dict(doc.get("metadata", {})))
        for doc in dataset["documents"]
    ]


def _build(dataset: dict[str, Any], factory: EngineFactory, *, reverse: bool) -> RAGEngine:
    engine = factory()
    documents = _documents(dataset)
    engine.index(list(reversed(documents)) if reverse else documents)
    for doc_id in dataset.get("delete_after_ingest", []):
        engine.delete(doc_id)
    return engine


def _unique_documents(chunk_document_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk_document_ids))


def run_benchmark(
    dataset: dict[str, Any],
    *,
    k: int = 5,
    engine_factory: EngineFactory = RAGEngine,
) -> dict[str, Any]:
    validate_dataset(dataset)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    engine = _build(dataset, engine_factory, reverse=False)
    # Same corpus, opposite insertion order: rankings and fingerprint must not change.
    mirror = _build(dataset, engine_factory, reverse=True)
    deleted = set(dataset.get("delete_after_ingest", []))

    rows: list[dict[str, Any]] = []
    unstable: list[str] = []
    deleted_leaks: list[str] = []
    citation_violations: list[str] = []
    unsupported_total = 0

    for case in dataset["queries"]:
        query = case["query"]
        relevance: dict[str, int] = case["relevance"]
        relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}

        results, _ = engine.retrieve(query, top_k=k)
        mirror_results, _ = mirror.retrieve(query, top_k=k)
        chunk_ids = [item.chunk.id for item in results]
        if chunk_ids != [item.chunk.id for item in mirror_results]:
            unstable.append(case["id"])
        ranked_docs = _unique_documents([item.chunk.document_id for item in results])
        leaked = sorted(deleted.intersection(ranked_docs))
        if leaked:
            deleted_leaks.append(case["id"])

        answer = engine.answer(query, top_k=k)
        citations: list[str] = answer["citations"]
        if not set(citations).issubset(chunk_ids):
            citation_violations.append(case["id"])
        unsupported_total += len(answer["unsupported_citations"])
        cited_docs = {
            item.chunk.document_id for item in answer["results"] if item.chunk.id in citations
        }

        row: dict[str, Any] = {
            "id": case["id"],
            "category": case.get("category", "uncategorized"),
            "query": query,
            "retrieved_documents": ranked_docs,
            "answerable": bool(relevant),
            "abstained": answer["abstained"],
            "confidence": round(answer["confidence"]["score"], 6),
            "deleted_leaks": leaked,
        }
        if relevant:
            row.update(
                {
                    "recall_at_k": recall_at_k(relevant, ranked_docs, k),
                    "precision_at_k": precision_at_k(relevant, ranked_docs, k),
                    "reciprocal_rank": reciprocal_rank(relevant, ranked_docs[:k]),
                    "ndcg_at_k": ndcg_at_k(relevance, ranked_docs, k),
                    "citation_coverage": len(relevant & cited_docs) / len(relevant),
                }
            )
        rows.append(row)

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    metric_names = ["recall_at_k", "precision_at_k", "reciprocal_rank", "ndcg_at_k"]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float | int]:
        summary: dict[str, float | int] = {"queries": len(selected)}
        for name in [*metric_names, "citation_coverage"]:
            summary[name] = round(mean(row[name] for row in selected), 6)
        return summary

    by_category: dict[str, dict[str, float | int]] = {}
    for category in sorted({row["category"] for row in answerable}):
        by_category[category] = summarize([r for r in answerable if r["category"] == category])

    fingerprint = engine.index_fingerprint()
    return {
        "dataset": dataset.get("name", "unnamed"),
        "dataset_version": dataset.get("version"),
        "dataset_sha256": dataset_fingerprint(dataset),
        "index_fingerprint": fingerprint,
        "k": k,
        "metric_level": "document",
        "engine": {
            "embedding": describe_embedding_provider(
                engine.embedding_provider or HashEmbeddingProvider()
            ),
            "chunk_size": engine.chunk_size,
            "overlap": engine.overlap,
            "abstain_threshold": engine.abstain_threshold,
        },
        "environment": {"python": platform.python_version()},
        "aggregate": {
            "answerable": summarize(answerable) if answerable else {"queries": 0},
            "by_category": by_category,
            "unanswerable": {
                "queries": len(unanswerable),
                "abstention_rate": (
                    round(mean(float(row["abstained"]) for row in unanswerable), 6)
                    if unanswerable
                    else None
                ),
            },
        },
        "invariants": {
            "rankings_stable_across_insertion_order": not unstable,
            "unstable_queries": unstable,
            "index_fingerprint_stable_across_insertion_order": (
                fingerprint == mirror.index_fingerprint()
            ),
            "deleted_document_leaks": deleted_leaks,
            "citations_outside_retrieved_set": citation_violations,
            "unsupported_answer_citations": unsupported_total,
        },
        "queries": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_benchmark(load_dataset(args.dataset), k=args.k)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    invariants = report["invariants"]
    ok = (
        invariants["rankings_stable_across_insertion_order"]
        and invariants["index_fingerprint_stable_across_insertion_order"]
        and not invariants["deleted_document_leaks"]
        and not invariants["citations_outside_retrieved_set"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
