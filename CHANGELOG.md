# Changelog

All notable engineering changes to this repository are documented here.

The format follows the intent of [Keep a Changelog](https://keepachangelog.com/) and uses semantic versioning principles where releases are created.

## [Unreleased]

### Fixed
- Equal-score chunks are now ordered by chunk ID in BM25, vector, RRF and reranker rankings. Previously ties kept insertion order, so a restarted engine (SQLite reloads documents by ID) could rank the same corpus differently.
- `top_k` larger than the 20-candidate pool is now honoured; it was silently capped at 20 results.

### Added
- `makma.v1` contract: `rag_engine.contracts.capabilities()` / `run_operation()` with `ingest`, `retrieve`, `answer`, `delete_document`, vendored JSON Schema in `contracts/`, and authenticated `GET /v1/capabilities` and `POST /v1/ops/{operation}` routes.
- Chunk character offsets (`start_char`/`end_char`) into the source document, SHA-256 provenance for cited spans, and an index fingerprint reported on every retrieval trace.
- Answer citation verification: `answer_citations` and `unsupported_citations` (citation-shaped IDs not present in the supplied context).
- `ndcg_at_k` and an offline benchmark runner (`python -m rag_engine.benchmark`) over a judged synthetic dataset pinned by SHA-256.

### Engineering
- Added repository ownership and structured contribution workflows.
- Added Dependabot coverage for Python and GitHub Actions dependencies.
- Added issue and pull-request templates focused on reproducibility, validation, risk, and rollback.
- Added consistent editor configuration and engineering standards.

## Release policy

A release should describe user-visible behavior, compatibility notes, validation evidence, and any migration or rollback requirements. Repository history remains the source of truth for individual commits.
