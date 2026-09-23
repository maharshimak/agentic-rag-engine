from fastapi.testclient import TestClient

from rag_engine.api import app


def test_remote_token_mode_requires_bearer(monkeypatch):
    monkeypatch.setenv("RAG_API_TOKEN", "test-secret")
    client = TestClient(app)

    denied = client.post("/v1/answer", json={"query": "hello", "top_k": 1})
    assert denied.status_code == 401

    allowed = client.post(
        "/v1/answer",
        json={"query": "hello", "top_k": 1},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert allowed.status_code in {200, 409}
