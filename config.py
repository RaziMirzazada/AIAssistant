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
    RAG_TOP_K: int = 40                       # V4: deep retrieval, was 10
    RAG_CANDIDATES_PER_INDEX: int = 80        # V4: was 30
    CHROMA_COLLECTION: str = "ai_assistant_v3_chunks"

    # -- Depth & generation tuning (V4 adaptive depth) ---------------------------
    # MAX_OUTPUT_TOKENS is the ceiling; per-depth budgets below override it.
    MAX_OUTPUT_TOKENS: int = 16384
    TEMPERATURE: float = 0.6                  # a touch of warmth for analysis
    GEMINI_THINKING_BUDGET: int = 4096        # ceiling for Gemini 2.5 thinking
    ENABLE_TWO_PASS: bool = True              # plan -> expand pipeline
    OUTLINE_MIN_SECTIONS: int = 5
    OUTLINE_MAX_SECTIONS: int = 7
    OLLAMA_NUM_CTX: int = 16384               # large context window for local LLM

    # Adaptive depth — planner classifies question; user can override.
    # DEFAULT_DEPTH applies when the planner is uncertain / fails / disabled.
    DEFAULT_DEPTH: str = "moderate"           # simple | moderate | deep

    # -- HyDE (Hypothetical Document Embeddings) ---------------------------------
    # Short queries embed poorly; ask the LLM to write a plausible 100-200 word
    # answer first and embed THAT for the vector search. BM25 still uses raw.
    ENABLE_HYDE: bool = True
    HYDE_MIN_QUERY_TOKENS: int = 30           # only run on short queries
    HYDE_MAX_OUTPUT_TOKENS: int = 384         # ~ 200 words

    # Per-depth output budgets and snippet caps. The reranker still returns
    # RERANKER_TOP_N chunks; the cap below trims that pile further before
    # the LLM sees it, so simple answers don't get padded with weak context.
    DEPTH_SIMPLE_MAX_TOKENS: int = 1024       # ~400 words
    DEPTH_SIMPLE_TOP_N: int = 5
    DEPTH_MODERATE_MAX_TOKENS: int = 3072     # ~900 words
    DEPTH_MODERATE_TOP_N: int = 10
    DEPTH_DEEP_MAX_TOKENS: int = 16384        # ~4000+ words (current default)
    DEPTH_DEEP_TOP_N: int = 15

    # -- Reranker (V4 Stage 2 — cross-encoder, ACTIVE) ---------------------------
    # Set ENABLE_RERANKER=false in .env to revert to pure hybrid behaviour.
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str | None = None        # None = auto-detect (cuda > cpu)
    RERANKER_RETRIEVE_K: int = 50             # candidates fed into the reranker
    RERANKER_TOP_N: int = 15                  # results that survive to the LLM
    RERANKER_BATCH_SIZE: int = 32
    RERANKER_SCORE_MODE: str = "two_stage"    # two_stage | replace | blend
    # Eagerly download/load the model on boot so the first chat call doesn't
    # pay the ~3s warmup. Set to false on tiny boxes that may OOM at boot.
    RERANKER_WARMUP_ON_STARTUP: bool = True

    # -- Conversation memory -----------------------------------------------------
    MEMORY_TURNS: int = 10                    # V4: was 5

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
