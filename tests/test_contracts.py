import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

from rag_engine import api
from rag_engine.contracts import capabilities, run_operation
from rag_engine.embeddings import OpenAICompatibleEmbeddingProvider
from rag_engine.generation import OpenAICompatibleChatGenerator
from rag_engine.provenance import sha256_text
from rag_engine.service import RAGEngine

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "contracts" / "makma.v1.schema.json").read_text(
        encoding="utf-8"
    )
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())

DOCS = [
    {"id": "backup", "text": "Nightly  backups run at\n02:00 UTC and are kept 35 days.", "metadata": {}},
    {"id": "incident", "text": "SEV1 incidents page the on-call engineer within 5 minutes."},
]


def assert_valid(envelope: dict) -> dict:
    errors = sorted(VALIDATOR.iter_errors(envelope), key=lambda e: list(e.path))
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors]
    return envelope


@pytest.fixture
def engine() -> RAGEngine:
    engine = RAGEngine()
    assert_valid(run_operation("ingest", {"documents": DOCS}, engine=engine))
    return engine


def test_ingest_envelope_and_document_provenance() -> None:
    engine = RAGEngine()
    envelope = assert_valid(
        run_operation("ingest", {"documents": DOCS}, run_id="run-1", engine=engine)
    )
    assert envelope["status"] == "ok"
    assert envelope["run_id"] == envelope["correlation_id"] == "run-1"
    assert envelope["execution_mode"] == "deterministic_offline"
    assert envelope["data"]["indexed_chunks"] == 2
    assert envelope["data"]["index_fingerprint"] == engine.index_fingerprint()
    by_id = {ref["id"]: ref for ref in envelope["provenance"]}
    assert by_id["backup"]["sha256"] == sha256_text(DOCS[0]["text"])
    assert by_id["backup"]["span"] == [0, len(DOCS[0]["text"])]


def test_retrieve_provenance_spans_hash_the_source_text(engine) -> None:
    envelope = assert_valid(
        run_operation("retrieve", {"query": "backups 35 days", "top_k": 2}, engine=engine)
    )
    assert envelope["status"] == "ok"
    results = envelope["data"]["results"]
    assert results[0]["document_id"] == "backup"
    assert envelope["data"]["trace"]["index_fingerprint"] == engine.index_fingerprint()
    sources = {doc["id"]: doc["text"] for doc in DOCS}
    for row, ref in zip(results, envelope["provenance"], strict=True):
        assert ref["kind"] == "chunk" and ref["id"] == row["chunk_id"]
        start, end = ref["span"]
        span = sources[row["document_id"]][start:end]
        assert ref["sha256"] == sha256_text(span)
        assert " ".join(span.split()) == row["text"]


def test_answer_citations_are_retrieved_and_have_provenance(engine) -> None:
    envelope = assert_valid(
        run_operation("answer", {"query": "when do backups run", "top_k": 2}, engine=engine)
    )
    assert envelope["status"] == "ok"
    data = envelope["data"]
    assert data["citations_in_retrieved_set"] is True
    assert set(data["citations"]) <= set(data["retrieved_chunk_ids"])
    assert data["unsupported_citations"] == []
    assert [ref["id"] for ref in envelope["provenance"]] == data["citations"]
    assert envelope["execution_mode"] == "deterministic_offline"


def test_delete_document_removes_evidence(engine) -> None:
    envelope = assert_valid(
        run_operation("delete_document", {"document_id": "backup"}, engine=engine)
    )
    assert envelope["data"]["deleted"] is True
    assert envelope["data"]["documents_remaining"] == 1
    again = assert_valid(run_operation("delete_document", {"document_id": "backup"}, engine=engine))
    assert again["data"]["deleted"] is False

    retrieved = assert_valid(
        run_operation("retrieve", {"query": "backups 35 days", "top_k": 50}, engine=engine)
    )
    assert {row["document_id"] for row in retrieved["data"]["results"]} == {"incident"}


@pytest.mark.parametrize(
    "operation, payload",
    [
        ("ingest", {"documents": []}),
        ("ingest", {"documents": [{"id": "a", "text": "   "}]}),
        ("ingest", {"documents": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]}),
        ("retrieve", {"query": "x", "top_k": 0}),
        ("retrieve", {"query": "   "}),
        ("answer", {"query": "x", "unexpected": True}),
        ("delete_document", {}),
        ("retrieve", ["not", "an", "object"]),
    ],
)
def test_invalid_input_envelopes(engine, operation, payload) -> None:
    envelope = assert_valid(run_operation(operation, payload, engine=engine))
    assert envelope["status"] == "error"
    assert envelope["data"] is None
    assert envelope["error"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize("operation", ["summarize", "Bad-Name", ""])
def test_unknown_operation_envelope(operation) -> None:
    envelope = assert_valid(run_operation(operation, {}, engine=RAGEngine()))
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "UNKNOWN_OPERATION"


def test_retrieve_on_empty_index_is_a_typed_error() -> None:
    envelope = assert_valid(run_operation("retrieve", {"query": "x"}, engine=RAGEngine()))
    assert envelope["error"]["code"] == "INDEX_EMPTY"


class _Unavailable(OpenAICompatibleEmbeddingProvider):
    def embed(self, texts):
        raise OSError("connection refused")


def test_provider_failure_is_retryable_and_leaks_no_details() -> None:
    engine = RAGEngine(embedding_provider=_Unavailable("http://10.0.0.5:8080", "m", "k-secret"))
    envelope = assert_valid(run_operation("ingest", {"documents": DOCS}, engine=engine))
    assert envelope["error"]["code"] == "PROVIDER_UNAVAILABLE"
    assert envelope["error"]["retryable"] is True
    assert "k-secret" not in json.dumps(envelope)


class _FixedEmbeddings(OpenAICompatibleEmbeddingProvider):
    def embed(self, texts):
        return [[1.0, float(len(text) % 7)] for text in texts]


class _FixedChat(OpenAICompatibleChatGenerator):
    def generate(self, query, context):
        return f"[{context.citations[0]}]"


@pytest.mark.parametrize(
    "base_url, expected",
    [("http://localhost:11434", "local_model"), ("http://127.0.0.1:8000", "local_model"),
     ("https://api.example.test", "remote_model")],
)
def test_execution_mode_reflects_configured_model_endpoints(base_url, expected) -> None:
    engine = RAGEngine(
        embedding_provider=_FixedEmbeddings(base_url, "embed-model"),
        generator=_FixedChat(base_url, "chat-model"),
        abstain_threshold=0.0,
    )
    ingest = assert_valid(run_operation("ingest", {"documents": DOCS}, engine=engine))
    assert ingest["execution_mode"] == expected
    answer = assert_valid(run_operation("answer", {"query": "backups"}, engine=engine))
    assert answer["execution_mode"] == expected
    assert answer["data"]["unsupported_citations"] == []


def test_capabilities_declare_operations_with_schemas() -> None:
    caps = capabilities()
    assert caps["contract_version"] == "makma.v1"
    assert caps["system"] == "agentic-rag-engine"
    names = {op["name"]: op for op in caps["operations"]}
    assert set(names) == {"ingest", "retrieve", "answer", "delete_document"}
    assert names["delete_document"]["side_effects"] is True
    assert names["retrieve"]["side_effects"] is False
    for op in caps["operations"]:
        Draft202012Validator.check_schema(op["input_schema"])


def test_api_contract_routes_require_auth_and_return_envelopes(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_TOKEN", "test-token")
    monkeypatch.setattr(api, "engine", RAGEngine())
    client = TestClient(api.app)
    auth = {"Authorization": "Bearer test-token"}

    assert client.get("/v1/capabilities").status_code == 401
    assert client.post("/v1/ops/retrieve", json={"query": "x"}).status_code == 401

    caps = client.get("/v1/capabilities", headers=auth)
    assert caps.status_code == 200 and caps.json()["system"] == "agentic-rag-engine"

    empty = client.post("/v1/ops/retrieve", json={"query": "backups"}, headers=auth)
    assert empty.status_code == 409
    assert_valid(empty.json())

    ingest = client.post(
        "/v1/ops/ingest",
        json={"documents": DOCS},
        headers={**auth, "X-Correlation-ID": "corr-42"},
    )
    assert ingest.status_code == 200
    assert assert_valid(ingest.json())["correlation_id"] == "corr-42"

    ok = client.post("/v1/ops/retrieve", json={"query": "backups"}, headers=auth)
    assert ok.status_code == 200 and assert_valid(ok.json())["status"] == "ok"

    bad = client.post("/v1/ops/retrieve", json={"query": ""}, headers=auth)
    assert bad.status_code == 422 and assert_valid(bad.json())["error"]["code"] == "INVALID_INPUT"

    unknown = client.post("/v1/ops/nope", json={}, headers=auth)
    assert unknown.status_code == 404
    assert assert_valid(unknown.json())["error"]["code"] == "UNKNOWN_OPERATION"
