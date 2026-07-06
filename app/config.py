from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_deployment: str
    azure_openai_embedding_deployment: str
    tavily_api_key: str
    redis_url: str
    redis_index_name: str
    memory_similarity_threshold: float



def get_settings() -> Settings:
    settings = Settings(
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        azure_openai_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        redis_index_name=os.getenv("REDIS_INDEX_NAME", "memory_idx"),
        memory_similarity_threshold=float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.7")),
    )

    missing = [
        key
        for key, value in {
            "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
            "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
            "AZURE_OPENAI_API_VERSION": settings.azure_openai_api_version,
            "AZURE_OPENAI_DEPLOYMENT": settings.azure_openai_deployment,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": settings.azure_openai_embedding_deployment,
            "TAVILY_API_KEY": settings.tavily_api_key,
        }.items()
        if not value
    ]

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {joined}")

    return settings
