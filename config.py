"""
AI Assistant V3 — Core Configuration

Centralised, strongly-typed settings loaded from environment variables (.env).
All paths are explicitly anchored to /workspace/ai-assistant-v3/ for RunPod
deployment so the application is fully workspace-isolated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


WORKSPACE_ROOT: Final[str] = "/workspace/ai-assistant-v3"


class Settings(BaseSettings):
    """Application-wide settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -- Workspace-isolated paths ------------------------------------------------
    WORKSPACE_ROOT: str = WORKSPACE_ROOT
    CHROMA_PATH: str = f"{WORKSPACE_ROOT}/chroma_db"
    FRONTEND_PATH: str = f"{WORKSPACE_ROOT}/frontend.html"
    SOURCES_METADATA_PATH: str = f"{WORKSPACE_ROOT}/sources_metadata.json"
    UPLOAD_TMP_PATH: str = f"{WORKSPACE_ROOT}/uploads"

    # -- Authentication ----------------------------------------------------------
    API_KEY: str = Field(..., description="Gatekeeper key required in the X-API-KEY header.")

    # -- Cloud LLM credentials ---------------------------------------------------
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key.")
    GROK_API_KEY: str = Field(..., description="xAI Grok API key.")

    # -- Local model endpoints ---------------------------------------------------
    OLLAMA_HOST: str = Field(default="http://127.0.0.1:11434", description="Ollama HTTP endpoint.")
    OLLAMA_LLM_MODEL: str = Field(default="qwen2.5:14b", description="Local LLM model name.")
    OLLAMA_EMBED_MODEL: str = Field(default="nomic-embed-text", description="Local embedding model.")

    # -- Cloud LLM model identifiers --------------------------------------------
    GEMINI_LLM_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini chat model.")
    GEMINI_TRANSLATE_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Gemini translation model (AZ<->EN bridge).",
    )
    GROK_LLM_MODEL: str = Field(default="grok-2-latest", description="Grok chat model.")
    GROK_BASE_URL: str = Field(default="https://api.x.ai/v1", description="Grok API base URL.")

    # -- RAG tuning --------------------------------------------------------------
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    HYBRID_VECTOR_WEIGHT: float = 0.65
    HYBRID_BM25_WEIGHT: float = 0.35
    RAG_TOP_K: int = 10
    RAG_CANDIDATES_PER_INDEX: int = 30
    CHROMA_COLLECTION: str = "ai_assistant_v3_chunks"

    # -- Conversation memory -----------------------------------------------------
    MEMORY_TURNS: int = 5

    # -- Networking --------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"],
        description="Explicit CORS allow-list (NEVER use '*').",
    )

    def ensure_directories(self) -> None:
        """Create workspace directories on boot. Idempotent."""
        for path in (self.WORKSPACE_ROOT, self.CHROMA_PATH, self.UPLOAD_TMP_PATH):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
