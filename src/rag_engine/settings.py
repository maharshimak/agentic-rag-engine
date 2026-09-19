from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_engine.embeddings import HashEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from rag_engine.generation import ExtractiveGenerator, OpenAICompatibleChatGenerator
from rag_engine.service import RAGEngine


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    embedding_base_url: str | None = None
    embedding_model: str | None = None
    chat_base_url: str | None = None
    chat_model: str | None = None
    api_key: str = ""
    chunk_size: int = 180
    chunk_overlap: int = 30
    context_tokens: int = 700


def build_engine(settings: Settings | None = None) -> RAGEngine:
    config = settings or Settings()

    if config.embedding_base_url and config.embedding_model:
        embedding_provider = OpenAICompatibleEmbeddingProvider(
            base_url=config.embedding_base_url,
            model=config.embedding_model,
            api_key=config.api_key,
        )
    else:
        embedding_provider = HashEmbeddingProvider()

    if config.chat_base_url and config.chat_model:
        generator = OpenAICompatibleChatGenerator(
            base_url=config.chat_base_url,
            model=config.chat_model,
            api_key=config.api_key,
        )
    else:
        generator = ExtractiveGenerator()

    return RAGEngine(
        embedding_provider=embedding_provider,
        generator=generator,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
        context_tokens=config.context_tokens,
    )
