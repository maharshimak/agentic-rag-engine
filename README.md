# 🧠 Agentic RAG Engine V2

**A MAK'MA Studio Product · MAK'MA Labs**

[Live Product Demo](https://maharshimak.github.io/makma-ai-os/projects/agentic-rag-engine/) · [MAK'MA Labs](https://maharshimak.github.io/makma-ai-os/projects/)

An inspectable Retrieval-Augmented Generation engine with **document chunking, BM25-style lexical search, semantic retrieval, reciprocal-rank fusion, multi-query planning, transparent reranking, context budgeting, citations, evaluation, FastAPI, Docker, and OpenAI-compatible model adapters**.


## Product contract — engineering upgrade

**Problem and audience:** A retrieval laboratory for engineers comparing evidence rankings against their own small corpus and judgments.

**Live tool:** https://maharshimak.github.io/makma-ai-os/projects/agentic-rag-engine/

**Implemented browser workflow:** Editable text/JSON corpus, strict Top K/Candidate K/BM25/RRF configuration, planned queries, per-stage scores, cited context, comparable baseline rankings and JSON export. No judgments displays N/A rather than a fake zero-quality score.

**Backend and parity contract:** Python and browser use SHA-256 token bytes, 256 buckets, digest sign bit, L2 normalization, BM25, RRF and transparent overlap reranking. Fixed fixtures execute the Python packages and compare JS outputs. Internal no-judgment convention: recall=1, precision=0, MRR=0, hit=0. Python additionally chunks documents, budgets context and supports external providers; the browser ranks supplied documents directly.

**Architecture:** `makma-ai-os/demo` is the shared web product source and Pages deployment. This repository owns its Python domain package. The central `tests/e2e` suite exercises all nine products; `tests/fixtures/python-parity.json` plus `scripts/generate_parity.py` guard shared mathematical contracts. Backend revisions used for regeneration are pinned in the central `backend-lock.json`.

**Safety and limitations:** Hash vectors are lexical hashing, not learned semantic embeddings. Browser limits: 200 documents, 100000 corpus characters, 2000 query characters. Baseline comparisons require identical corpus/query/judgments. Inputs are validated, rendered user values are escaped, and deterministic results are not presented as model inference.

**Verification:** Run `python -m ruff check .` and `python -m pytest -q`. `tests/test_engineering_upgrade.py` protects the new rejection/correctness paths. Central web checks: `npm ci`, `npm test`, `npm run build`, `npx playwright install --with-deps chromium`, `npm run test:e2e`. CI gates publishing on browser interactions and validates all public URLs after deployment.

**Highest-value next work:** Labeled domain benchmarks, persistent vector storage and calibrated trained reranking.

**Provenance:** Independent MAK’MA Studio engineering implementation; examples are synthetic and no employer code or data is included. Existing MIT license applies.


## Architecture

```mermaid
flowchart LR
Q[User Query] --> P[Query Planner]
P --> L[BM25 Retrieval]
P --> V[Semantic Retrieval]
L --> F[Reciprocal Rank Fusion]
V --> F
F --> R[Transparent Reranker]
R --> C[Context Budget]
C --> G[Grounded Generator]
G --> A[Answer + Citations]
```

## What is implemented

- typed `Document`, `Chunk`, scored-result and trace models
- deterministic overlapping document chunking
- real BM25-style lexical index
- deterministic offline embedding provider for reproducible CI
- OpenAI-compatible `/v1/embeddings` adapter for real local/cloud embedding servers
- cosine semantic retrieval
- reciprocal-rank fusion across lexical, semantic and planned-query rankings
- deterministic multi-query decomposition for compound questions
- interpretable reranker with a replaceable cross-encoder boundary
- token-budgeted citation context construction
- offline grounded extractive generation
- OpenAI-compatible `/v1/chat/completions` generator for real LLMs
- Recall@K, Precision@K, MRR and Hit Rate evaluation
- FastAPI indexing, retrieval and answer endpoints
- latency and retrieval tracing
- Docker support
- compatibility layer for the original V1 API
- automated tests and GitHub CI

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
uvicorn rag_engine.api:app --reload
```

Open `http://localhost:8000/docs`.

## API

- `GET /health`
- `POST /v1/documents/index`
- `POST /v1/retrieve`
- `POST /v1/answer`

## Use a real OpenAI-compatible model server

The default configuration is fully offline so tests and demos are reproducible. To connect Ollama/vLLM/LM Studio or another OpenAI-compatible gateway, configure the following environment variables:

```bash
export RAG_EMBEDDING_BASE_URL=http://localhost:11434
export RAG_EMBEDDING_MODEL=nomic-embed-text
export RAG_CHAT_BASE_URL=http://localhost:11434
export RAG_CHAT_MODEL=qwen3
export RAG_API_KEY=
```

The engine automatically switches from the deterministic offline providers to the configured model endpoints.

## Why this project matters

This repository is intentionally built beyond the common `embed → vector search → prompt` demo. The retrieval path separates planning, lexical search, semantic search, fusion, reranking, context construction, generation and evaluation so each stage can be tested, benchmarked and replaced independently.

## Next engineering milestones

- persistent Qdrant/pgvector adapter
- learned cross-encoder reranker
- LLM-driven query decomposition and iterative retrieval
- ingestion workers for PDF/HTML/Markdown
- groundedness and citation-faithfulness evaluation
- OpenTelemetry-compatible tracing
- benchmark datasets and regression gates

Built as an engineering portfolio project focused on production GenAI systems.

## Scope and limitations

Offline embeddings use token hashing, not a trained semantic model. The reranker and query planner are deterministic heuristics. Indexes live in memory and are rebuilt during ingestion. Input is already-extracted text; PDF/OCR/HTML ingestion is not implemented. Citations identify supplied context but do not establish answer faithfulness. The small synthetic retrieval test is a regression fixture, not a general retrieval benchmark. Remote adapters require an independently hosted compatible service and are not validated against live providers by offline CI.

## Installation and development

Requires Python 3.12 or newer. Run from this project directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
python -m pip wheel --no-deps . -w dist
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

## Library usage

```python
from rag_engine.models import Document
from rag_engine.service import RAGEngine

engine = RAGEngine()
engine.index([Document("memory", "SQLite stores persistent conversation memory.")])
print(engine.answer("What stores conversation memory?"))
```

## Configuration

See [.env.example](.env.example). Export variables into the process environment; the application does not automatically load that file. Keep real credentials out of Git.

## Service and API schema

```bash
python -m uvicorn rag_engine.api:app --host 127.0.0.1 --port 8000
```

Interactive endpoint schemas are at `http://127.0.0.1:8000/docs`; machine-readable schemas are at `/openapi.json`. These APIs have no built-in authentication. Use trusted local data and local access.

## Container

```bash
docker build -t agentic-rag-engine .
docker run --rm -p 127.0.0.1:8000:8000 agentic-rag-engine
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/rag_engine/` | Implementation |
| `tests/` | Offline unit and regression tests |
| `docs/DESIGN.md` | Architecture and trust boundaries |
| `.github/workflows/ci.yml` | Install, lint, tests, wheel and container build |
| `pyproject.toml` | Dependencies and package configuration |

## Next engineering work

Persistent index adapter; labeled retrieval benchmark; citation-faithfulness checks; trained reranker; background ingestion. These are planned work, not current capabilities.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). CI runs on every push and pull request through `.github/workflows/ci.yml`.

## License and provenance

[MIT](LICENSE), copyright 2026 Maharshi Patel. This public portfolio implementation is independent of employer systems and contains no confidential employer code or data. Examples and test fixtures are synthetic.
