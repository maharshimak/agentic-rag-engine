# Design and operating boundaries

Inspectable RAG pipeline with text chunking, BM25 retrieval, vector ranking, reciprocal-rank fusion, citations and evaluation.

## Scope

Offline embeddings use token hashing, not a trained semantic model. The reranker and query planner are deterministic heuristics. Indexes live in memory and are rebuilt during ingestion. Input is already-extracted text; PDF/OCR/HTML ingestion is not implemented. Citations identify supplied context but do not establish answer faithfulness. The small synthetic retrieval test is a regression fixture, not a general retrieval benchmark. Remote adapters require an independently hosted compatible service and are not validated against live providers by offline CI.

## Interfaces

Implementation lives in `src/rag_engine/`. Public examples in the README use its Python API. FastAPI exposes the same local capabilities; `/openapi.json` is the endpoint schema.

## Validation

Tests include synthetic regression fixtures. Package and container checks verify installation separately from source-tree imports. Tests do not certify general model quality, clinical correctness or multi-tenant isolation.

## Planned evolution

Persistent index adapter; labeled retrieval benchmark; citation-faithfulness checks; trained reranker; background ingestion.
