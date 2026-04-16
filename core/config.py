"""Central configuration for the multi-agent AI system."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    GEMINI_MODEL: str
    SERPAPI_KEY: str
    REDIS_URL: str
    DB_PATH: str
    VECTOR_STORE_PATH: str
    DEFAULT_MODEL: str
    FAST_MODEL: str
    MAX_REFLECTION_ITERATIONS: int
    REFLECTION_SCORE_THRESHOLD: float
    CACHE_TTL_SECONDS: int

    @classmethod
    def from_env(cls) -> "Config":
        """Create a config instance by loading environment variables."""
        load_dotenv()
        gemini_key = (
            os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
            or os.getenv("GENAI_API_KEY", "")
        )
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        default_model = os.getenv("DEFAULT_MODEL", gemini_model if gemini_key else "gpt-4")
        fast_model = os.getenv("FAST_MODEL", "gemini-2.5-flash" if gemini_key else "gpt-3.5-turbo")
        return cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            GEMINI_API_KEY=gemini_key,
            GEMINI_MODEL=gemini_model,
            SERPAPI_KEY=os.getenv("SERPAPI_KEY", ""),
            REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379"),
            DB_PATH=os.getenv("DB_PATH", "memory.db"),
            VECTOR_STORE_PATH=os.getenv("VECTOR_STORE_PATH", "./faiss_index"),
            DEFAULT_MODEL=default_model,
            FAST_MODEL=fast_model,
            MAX_REFLECTION_ITERATIONS=int(os.getenv("MAX_REFLECTION_ITERATIONS", "3")),
            REFLECTION_SCORE_THRESHOLD=float(os.getenv("REFLECTION_SCORE_THRESHOLD", "0.8")),
            CACHE_TTL_SECONDS=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        )


config = Config.from_env()
