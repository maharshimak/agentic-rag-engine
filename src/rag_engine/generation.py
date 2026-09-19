import json
from typing import Protocol
from urllib import request

from rag_engine.context import ContextBlock


class Generator(Protocol):
    def generate(self, query: str, context: ContextBlock) -> str: ...


class ExtractiveGenerator:
    """Offline fallback that returns grounded evidence with explicit citations."""

    def generate(self, query: str, context: ContextBlock) -> str:
        del query
        if not context.text:
            return "I could not find grounded evidence for this question."
        first_section = context.text.split("\n\n", maxsplit=1)[0]
        return first_section


class OpenAICompatibleChatGenerator:
    """Adapter for OpenAI-compatible /v1/chat/completions endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(self, query: str, context: ContextBlock) -> str:
        system = (
            "Answer only from the supplied context. "
            "Cite supporting chunk ids in square brackets. "
            "If the context is insufficient, say so."
        )
        user = f"Question: {query}\n\nContext:\n{context.text}"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(self.endpoint, data=payload, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode())
        return data["choices"][0]["message"]["content"]
