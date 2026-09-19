from dataclasses import dataclass

from rag_engine.models import ScoredChunk


@dataclass(frozen=True)
class ContextBlock:
    text: str
    citations: list[str]
    used_tokens: int


class ContextBuilder:
    def __init__(self, max_tokens: int = 700) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    def build(self, items: list[ScoredChunk]) -> ContextBlock:
        sections: list[str] = []
        citations: list[str] = []
        used_tokens = 0

        for item in items:
            words = item.chunk.text.split()
            remaining = self.max_tokens - used_tokens
            if remaining <= 0:
                break
            selected = words[:remaining]
            if not selected:
                continue

            citation = item.chunk.id
            sections.append(f"[{citation}] {' '.join(selected)}")
            citations.append(citation)
            used_tokens += len(selected)

        return ContextBlock(
            text="\n\n".join(sections),
            citations=citations,
            used_tokens=used_tokens,
        )
