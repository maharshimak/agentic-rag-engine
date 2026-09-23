from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol

from rag_engine.models import Document


class DocumentStore(Protocol):
    def list_documents(self) -> list[Document]: ...

    def upsert_many(self, documents: list[Document]) -> None: ...

    def delete(self, document_id: str) -> bool: ...


class SQLiteDocumentStore:
    """Small local-first durable store for source documents and metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path))
        Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            connection.commit()

    def list_documents(self) -> list[Document]:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, text, metadata_json FROM documents ORDER BY id"
            ).fetchall()
        return [
            Document(
                id=row["id"],
                text=row["text"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def upsert_many(self, documents: list[Document]) -> None:
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Document IDs must be unique within an upsert batch.")
        payload = [
            (
                document.id,
                document.text,
                json.dumps(
                    document.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for document in documents
        ]
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO documents(id, text, metadata_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text = excluded.text,
                        metadata_json = excluded.metadata_json
                    """,
                    payload,
                )

    def delete(self, document_id: str) -> bool:
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM documents WHERE id = ?",
                    (document_id,),
                )
        return cursor.rowcount == 1
