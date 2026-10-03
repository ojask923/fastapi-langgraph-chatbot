import os
from typing import Literal
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(ENV_FILE_PATH, override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    APP_NAME: str = "Simple Local Chatbot"
    VERSION: str = "1.0.0"

    # Default LLM configurations
    DEFAULT_PROVIDER: Literal["groq", "ollama", "openai", "gemini", "anthropic", "openrouter"] = "groq"
    DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.7

    # API Keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Database settings (defaults to local SQLite, can be PostgreSQL)
    DATABASE_URL: str = "sqlite:///./chatbot.db"

    # Features
    ENABLE_TOOLS: bool = True

    # RAG Settings
    VECTOR_STORE_PATH: str = "./qdrant_data"
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # RAG Ingestion pipeline settings
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = ["pdf", "txt", "md"]
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 150

    # Hybrid Search Settings
    ENABLE_HYBRID_SEARCH: bool = True
    HYBRID_DENSE_WEIGHT: float = 0.5
    HYBRID_SPARSE_WEIGHT: float = 0.5
    ENABLE_RERANKING: bool = True

    # Reranking pipeline settings
    # RERANK_CANDIDATE_K: candidates fetched before reranking (20–50 recommended)
    RERANK_CANDIDATE_K: int = 20
    # RERANK_TOP_K: final chunks passed to the LLM after reranking
    RERANK_TOP_K: int = 5
    # RERANK_MODEL: FlashRank cross-encoder model name
    RERANK_MODEL: str = "ms-marco-TinyBERT-L-2-v2"

    # Minimum relevance thresholds — chunks below these scores are dropped before
    # being passed to the LLM, so the model never receives essentially irrelevant
    # context.  Set to 0.0 to disable the cutoff entirely.
    #
    # Applied in this priority order (only one branch runs per retrieval call):
    #   1. MIN_RERANK_SCORE  — when ENABLE_RERANKING=True (FlashRank cross-encoder, ~0.0-1.0)
    #   2. MIN_RRF_SCORE     — when ENABLE_HYBRID_SEARCH=True, reranking off (RRF score, ~0.01-0.03)
    #   3. MIN_DENSE_SCORE   — dense-only, no reranking (raw cosine similarity, ~0.0-1.0)
    MIN_RERANK_SCORE: float = 0.1
    MIN_RRF_SCORE: float = 0.005
    MIN_DENSE_SCORE: float = 0.3

    # RAG grounding/citation settings
    # When enabled, the ContextEngine injects citation rules into the system prompt
    # and the retrieve_documents tool returns structured 【Doc N】-marked context.
    RAG_GROUNDING_ENABLED: bool = True

    # ---- Memory / Context Architecture ----
    # Number of most-recent messages included in each LLM call.
    # Older messages are covered by the rolling summary instead.
    RECENT_MESSAGES_WINDOW: int = 20

    # Once the total stored message count exceeds this threshold the oldest
    # messages are compacted into a rolling summary and removed from state.
    SUMMARY_THRESHOLD: int = 30

    # Maximum number of Mem0 long-term memory results injected per query.
    MEM0_MAX_RESULTS: int = 5

    # Provider and model used by mem0 for memory extraction.
    # Intentionally decoupled from DEFAULT_PROVIDER so extraction always uses
    # a reliable, high-TPM model regardless of which chat LLM the user picks.
    # Groq options: "openai/gpt-oss-120b", "qwen/qwen3.8-27b"
    # OpenAI option (recommended): "gpt-4o-mini"
    MEM0_LLM_PROVIDER: str = "openai"
    MEM0_LLM_MODEL: str = "gpt-4o-mini"

    # Rough token budget for the full assembled context sent to the LLM.
    # Used by ContextEngine to guard against runaway context size.
    CONTEXT_TOKEN_BUDGET: int = 4000


settings = Settings()
