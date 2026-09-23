import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol
from urllib import request


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic SHA-256 hash embeddings shared by backend and browser demos."""

    def __init__(self, dimensions: int = 256) -> None:
        if type(dimensions) is not int or not 1 <= dimensions <= 65536:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = -1.0 if digest[0] & 1 else 1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAICompatibleEmbeddingProvider:
    """Adapter for OpenAI-compatible /v1/embeddings endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/v1/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(self.endpoint, data=payload, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode())
        ordered = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


class SQLiteCachedEmbeddingProvider:
    """Persistent content-addressed cache around any embedding provider."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        path: str | Path,
        *,
        namespace: str = "default",
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace must be non-empty")
        self.provider = provider
        self.path = str(Path(path))
        self.namespace = namespace
        Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _key(self, text: str) -> str:
        payload = (self.namespace + "\0" + text).encode()
        return hashlib.sha256(payload).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        keys = [self._key(text) for text in texts]
        cached: dict[str, list[float]] = {}
        with closing(sqlite3.connect(self.path)) as connection:
            placeholders = ",".join("?" for _ in keys)
            rows = connection.execute(
                f"SELECT cache_key, vector_json FROM embedding_cache "
                f"WHERE cache_key IN ({placeholders})",
                keys,
            ).fetchall()
        for key, vector_json in rows:
            vector = json.loads(vector_json)
            if not isinstance(vector, list) or not vector:
                continue
            cached[str(key)] = [float(value) for value in vector]

        missing_positions = [index for index, key in enumerate(keys) if key not in cached]
        if missing_positions:
            missing_texts = [texts[index] for index in missing_positions]
            generated = self.provider.embed(missing_texts)
            if len(generated) != len(missing_texts):
                raise ValueError("Embedding provider returned the wrong number of vectors.")
            inserts: list[tuple[str, str]] = []
            for index, vector in zip(missing_positions, generated, strict=True):
                normalized = [float(value) for value in vector]
                if not normalized or not all(math.isfinite(value) for value in normalized):
                    raise ValueError("Embedding vectors must be non-empty and finite.")
                cached[keys[index]] = normalized
                inserts.append(
                    (
                        keys[index],
                        json.dumps(normalized, separators=(",", ":")),
                    )
                )
            with closing(sqlite3.connect(self.path)) as connection, connection:
                connection.executemany(
                    """
                    INSERT INTO embedding_cache(cache_key, vector_json)
                    VALUES (?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET vector_json = excluded.vector_json
                    """,
                    inserts,
                )

        return [cached[key] for key in keys]
