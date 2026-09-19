import re

from rag_engine.models import Chunk, Document

_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN_RE.finditer(text)]


def chunk_document(
    document: Document,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    tokens = tokenize(document.text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(tokens), step)):
        end = min(start + chunk_size, len(tokens))
        text = " ".join(tokens[start:end])
        chunks.append(
            Chunk(
                id=f"{document.id}::c{index}",
                document_id=document.id,
                text=text,
                start_token=start,
                end_token=end,
                metadata=dict(document.metadata),
            )
        )
        if end >= len(tokens):
            break
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    ]
