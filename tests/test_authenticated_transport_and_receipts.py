"""Regression tests for authenticated redirects and concurrent ingest receipts."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Thread
from urllib.error import HTTPError

import pytest

from rag_engine.context import ContextBlock
from rag_engine.contracts import run_operation
from rag_engine.embeddings import OpenAICompatibleEmbeddingProvider
from rag_engine.generation import OpenAICompatibleChatGenerator
from rag_engine.models import Document
from rag_engine.planning import OpenAICompatibleQueryPlanner
from rag_engine.service import RAGEngine


@contextmanager
def _serve(handler: type[BaseHTTPRequestHandler]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


@pytest.mark.parametrize("kind", ["embeddings", "planner", "generator"])
@pytest.mark.parametrize("redirect_status", [301, 302, 303, 307, 308])
def test_authenticated_adapter_does_not_follow_redirect_to_other_origin(
    kind, redirect_status
):
    forwarded: list[str | None] = []

    class Sink(BaseHTTPRequestHandler):
        def _capture(self):
            forwarded.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_GET(self):
            self._capture()

        def do_POST(self):
            self._capture()

        def log_message(self, *args):
            pass

    with _serve(Sink) as sink_url:

        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(redirect_status)
                self.send_header("Location", sink_url + "/redirect-target")
                self.end_headers()

            def log_message(self, *args):
                pass

        with _serve(Redirect) as origin_url:
            if kind == "embeddings":
                adapter = OpenAICompatibleEmbeddingProvider(
                    origin_url, "test-model", api_key="synthetic-test-only"
                )
                run = lambda: adapter.embed(["a"])
            elif kind == "planner":
                adapter = OpenAICompatibleQueryPlanner(
                    origin_url, "test-model", api_key="synthetic-test-only"
                )
                run = lambda: adapter.plan("question")
            else:
                adapter = OpenAICompatibleChatGenerator(
                    origin_url, "test-model", api_key="synthetic-test-only"
                )
                run = lambda: adapter.generate(
                    "question", ContextBlock("evidence", [], 1)
                )

            with pytest.raises(HTTPError) as error:
                run()
            assert error.value.code == redirect_status
            assert forwarded == []


def test_concurrent_ingests_return_hashes_from_their_own_snapshots():
    a = Document(id="a", text="alpha evidence")
    b = Document(id="b", text="beta evidence")

    def fingerprint(*documents):
        reference = RAGEngine(chunk_size=4, overlap=0)
        _, value = reference.index_with_receipt(list(documents))
        return value

    single_fingerprints = {fingerprint(a), fingerprint(b)}
    both_fingerprint = fingerprint(a, b)

    for _ in range(20):
        engine = RAGEngine(chunk_size=4, overlap=0)
        barrier = Barrier(2)
        receipts: list[tuple[int, str | None]] = []

        def ingest(document, current_engine=engine, ready=barrier, result=receipts):
            ready.wait()
            envelope = run_operation(
                "ingest",
                {"documents": [{"id": document.id, "text": document.text}]},
                engine=current_engine,
            )
            assert envelope["status"] == "ok"
            result.append(
                (
                    envelope["data"]["indexed_chunks"],
                    envelope["data"]["index_fingerprint"],
                )
            )

        workers = [Thread(target=ingest, args=(document,)) for document in (a, b)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=3)
            assert not worker.is_alive()

        assert len(receipts) == 2
        first = next(receipt for receipt in receipts if receipt[0] == 1)
        second = next(receipt for receipt in receipts if receipt[0] == 2)
        assert first[1] in single_fingerprints
        assert second[1] == both_fingerprint
