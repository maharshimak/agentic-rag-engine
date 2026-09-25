from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request


@dataclass(slots=True)
class OpenAICompatibleQueryPlanner:
    """LLM-backed retrieval query decomposition with strict JSON validation."""

    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 30.0
    max_queries: int = 3

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("base_url and model are required.")
        if not 1 <= self.max_queries <= 8:
            raise ValueError("max_queries must be between 1 and 8.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

    def plan(self, query: str, max_queries: int | None = None) -> list[str]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty.")
        limit = self.max_queries if max_queries is None else max_queries
        if not 1 <= limit <= 8:
            raise ValueError("max_queries must be between 1 and 8.")

        system = (
            "Decompose the user's retrieval request into a small set of complementary "
            "search queries. Return only JSON with a 'queries' array. Preserve the original "
            "intent, avoid speculative facts, and do not generate more than "
            f"{limit} queries. The first query should be the original normalized request."
        )
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode())
        raw = str(body["choices"][0]["message"]["content"]).strip()
        return self.parse(query, raw, max_queries=limit)

    @staticmethod
    def parse(query: str, raw: str, *, max_queries: int = 3) -> list[str]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("Query planner returned invalid JSON.") from error
        queries = payload.get("queries")
        if not isinstance(queries, list):
            raise ValueError("Query planner JSON must contain a queries array.")
        cleaned: list[str] = []
        original = " ".join(query.split())
        for item in [original, *queries]:
            if not isinstance(item, str):
                raise ValueError("Every planned query must be text.")
            normalized = " ".join(item.split())
            if normalized and normalized.casefold() not in {x.casefold() for x in cleaned}:
                cleaned.append(normalized)
            if len(cleaned) >= max_queries:
                break
        if not cleaned:
            raise ValueError("Query planner produced no usable queries.")
        return cleaned
