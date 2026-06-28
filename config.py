"""Centralized configuration for the Jama MCP Server.

All settings are read from environment variables (optionally a .env file)
so the same image runs in dev, test and production without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

# Project root = directory containing this config.py. All runtime artifacts
# (SQLite DB, HF model cache) default to a project-local ``user/`` folder so
# the server is self-contained and portable.
PROJECT_ROOT = Path(__file__).resolve().parent
USER_DIR = PROJECT_ROOT / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)

# Use the HuggingFace China mirror by default so model weights (e.g.
# Qwen3-Reranker-0.6B) can be downloaded from inside mainland China. This is
# read by huggingface_hub / transformers when fetching models. Override with
# HF_ENDPOINT in the environment if a different mirror is preferred.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Default the HF cache to a project-local folder (user/huggingface) so the
# 1.2GB reranker weights live inside the project, not in the user home dir.
# Only set if the caller hasn't already configured HF_HOME/HUGGINGFACE_HUB_CACHE.
os.environ.setdefault("HF_HOME", str(USER_DIR / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(USER_DIR / "huggingface" / "hub"))
# Once weights are cached we prefer offline mode so transient network errors
# never block reranker loading in production.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class JamaSettings:
    """Jama REST API connection."""
    url: str = _get("JAMA_URL", "https://your-tenant.jamacloud.com")
    client_id: str = _get("JAMA_CLIENT_ID", "")
    client_secret: str = _get("JAMA_CLIENT_SECRET", "")
    # Jama requires a versioned path segment; "latest" is the stable choice.
    api_prefix: str = _get("JAMA_API_PREFIX", "/rest/latest")
    # REST tuning
    page_size: int = _get_int("JAMA_PAGE_SIZE", 50)
    request_timeout: int = _get_int("JAMA_REQUEST_TIMEOUT", 30)
    max_retries: int = _get_int("JAMA_MAX_RETRIES", 6)
    # Polite delay between paged GETs to respect Jama rate limits.
    page_delay: float = _get_float("JAMA_PAGE_DELAY", 0.25)
    # Pre-flight bandwidth check (bytes/sec). If the Jama speed test falls
    # below this, downloads are aborted up-front with a network-error message.
    min_bytes_per_sec: int = _get_int("JAMA_MIN_BYTES_PER_SEC", 20_000)
    speed_test_timeout: int = _get_int("JAMA_SPEED_TEST_TIMEOUT", 15)
    # Per-page stall guard: if a single paged GET is slower than this
    # (bytes/sec over the response body) it's treated as a slow/stalled
    # network and retried instead of accepted. Kept low (500 B/s) because Jama
    # JSON pages are small and server-side processing latency can dominate the
    # effective rate on a healthy connection; this catches genuine stalls, not
    # transient server slowness.
    page_min_bytes_per_sec: int = _get_int("JAMA_PAGE_MIN_BYTES_PER_SEC", 500)
    # Max page-level retries on stall/timeout during a paginated fetch.
    page_max_retries: int = _get_int("JAMA_PAGE_MAX_RETRIES", 5)


@dataclass(frozen=True)
class EmbeddingSettings:
    """OpenAI-compatible text embedding endpoint (Azure gateway here)."""
    base_url: str = _get("EMBEDDING_BASE_URL", "https://your-embedding-endpoint.example.com")
    api_key: str = _get("EMBEDDING_API_KEY", _get("OPENAI_API_KEY", ""))
    model: str = _get("EMBEDDING_MODEL", "text-embedding-3-small")
    # text-embedding-3-small == 1536 dims.
    dimensions: int = _get_int("EMBEDDING_DIMENSIONS", 1536)
    batch_size: int = _get_int("EMBEDDING_BATCH_SIZE", 64)
    timeout: int = _get_int("EMBEDDING_TIMEOUT", 60)
    # Header used to carry the key (Azure uses "api-key", OpenAI uses "Authorization").
    key_header: str = _get("EMBEDDING_KEY_HEADER", "api-key")


@dataclass(frozen=True)
class LLMSettings:
    """Optional chat LLM for Multi-Query expansion.

    The provided Azure gateway only exposes embeddings, so multi-query LLM
    generation is opt-in: set these vars (e.g. to a local vLLM/Ollama or a
    full OpenAI deployment) and the RAG pipeline will expand queries;
    otherwise it falls back to deterministic lexical variants.
    """
    base_url: str = _get("LLM_BASE_URL", "")
    api_key: str = _get("LLM_API_KEY", "")
    model: str = _get("LLM_MODEL", "gpt-4o-mini")
    timeout: int = _get_int("LLM_TIMEOUT", 30)


@dataclass(frozen=True)
class RerankerSettings:
    """Local Qwen3-Reranker-0.6B (CPU)."""
    model_name: str = _get("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    # Max candidate pairs scored in one forward pass (memory bound).
    batch_size: int = _get_int("RERANKER_BATCH_SIZE", 16)
    max_length: int = _get_int("RERANKER_MAX_LENGTH", 512)
    # If model loading fails, degrade to RRF-only scoring instead of crashing.
    allow_fallback: bool = _get("RERANKER_ALLOW_FALLBACK", "1") == "1"
    device: str = _get("RERANKER_DEVICE", "cpu")
    # Pre-flight speed test before pulling weights from the HF mirror.
    hf_min_bytes_per_sec: int = _get_int("HF_MIN_BYTES_PER_SEC", 200_000)
    hf_speed_test_timeout: int = _get_int("HF_SPEED_TEST_TIMEOUT", 20)
    hf_download_retries: int = _get_int("HF_DOWNLOAD_RETRIES", 4)
    device: str = _get("RERANKER_DEVICE", "cpu")
    # Pre-flight speed test before pulling model weights from HuggingFace.
    # If the HF mirror is slower than this (bytes/sec), abort with a network
    # error instead of hanging for hours on a stalled download.
    hf_min_bytes_per_sec: int = _get_int("RERANKER_HF_MIN_BYTES_PER_SEC", 200_000)
    hf_speed_test_timeout: int = _get_int("RERANKER_HF_SPEED_TEST_TIMEOUT", 20)
    hf_max_retries: int = _get_int("RERANKER_HF_MAX_RETRIES", 5)


@dataclass(frozen=True)
class StorageSettings:
    # SQLite DB lives in the project-local user/ folder by default so all
    # runtime data is self-contained. Override with JAMA_MCP_DB_PATH.
    db_path: str = _get("JAMA_MCP_DB_PATH", str(USER_DIR / "jama_mcp.db"))
    # Busy timeout (ms) for SQLite write concurrency (init sync vs. reads).
    busy_timeout_ms: int = _get_int("SQLITE_BUSY_TIMEOUT_MS", 5000)


@dataclass(frozen=True)
class SyncSettings:
    """APScheduler incremental sync."""
    enabled: bool = _get("SYNC_ENABLED", "1") == "1"
    # Cron-style: every 2 hours by default.
    hours: int = _get_int("SYNC_INTERVAL_HOURS", 2)
    # Hard cap of items inspected per project per sync run (safety valve).
    max_items_per_run: int = _get_int("SYNC_MAX_ITEMS_PER_RUN", 5000)


@dataclass(frozen=True)
class ChunkSettings:
    """RecursiveCharacterTextSplitter tuning (data is 100% English, ~30% long)."""
    chunk_size: int = _get_int("CHUNK_SIZE", 512)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 80)
    separators: tuple = field(default=("\\n\\n", "\\n", ". ", "? ", "! ", " ", ""))


@dataclass(frozen=True)
class Settings:
    jama: JamaSettings = field(default_factory=JamaSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    sync: SyncSettings = field(default_factory=SyncSettings)
    chunk: ChunkSettings = field(default_factory=ChunkSettings)


settings = Settings()
