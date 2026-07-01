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
except Exception:  # pragma: no cover - dotenv is optional at runtime
    load_dotenv = None

# Project root = directory containing this config.py. All runtime artifacts
# (SQLite DB, HF model cache) default to a project-local ``user/`` folder so
# the server is self-contained and portable.
PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env from the project root (not the caller's cwd) so the server picks
# up its configuration no matter what directory an MCP client launches it
# from. ``load_dotenv()`` with no path searches the cwd, which is unreliable
# for a stdio server spawned by Claude Desktop / Copilot CLI / etc.
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")
USER_DIR = PROJECT_ROOT / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)

# Use the HuggingFace China mirror by default so model weights (the ~80MB
# cross-encoder reranker ONNX + ~130MB ONNX embedding model) can be downloaded
# from inside mainland China. Both models download via fastembed, which uses
# huggingface_hub and honours HF_ENDPOINT. Override with HF_ENDPOINT in the
# environment if a different mirror is preferred.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Default the HF cache to a project-local folder (user/huggingface) so the
# reranker + embedding weights live inside the project, not in the user home
# dir. Only set if the caller hasn't already configured HF_HOME/HUGGINGFACE_HUB_CACHE.
os.environ.setdefault("HF_HOME", str(USER_DIR / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(USER_DIR / "huggingface" / "hub"))
# Disable HF's Xet transfer protocol. fastembed pulls both ONNX models via
# huggingface_hub, which defaults to the Xet protocol; on some networks it
# fails mid-transfer. Forcing the plain HTTPS path avoids the failure.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Silence the tqdm "Fetching N files" / per-file progress bars that
# huggingface_hub (fastembed's ONNX downloads for both the embedding and the
# reranker) emit. In a non-interactive shell — especially the MCP stdio
# server, which runs headless — those bars spam thousands of carriage-return
# lines on stderr and look like a hang. Set here (not just in bootstrap.py) so
# server-driven syncs and selftest are silenced too; huggingface_hub reads
# HF_HUB_DISABLE_PROGRESS_BARS lazily on each download.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


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


def _project_path(path: str) -> str:
    """Resolve a path against PROJECT_ROOT when it is relative.

    Keeps path-like settings (e.g. the SQLite DB) inside the project
    regardless of the caller's cwd, so a stdio MCP server launched from an
    arbitrary directory still finds its files. Absolute paths pass through.
    """
    return path if Path(path).is_absolute() else str(PROJECT_ROOT / path)


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
    """Embedding configuration.

    Two providers are supported:
      * ``local`` (default) — runs bge-small-en-v1.5 on CPU via fastembed/ONNX.
        No API key, no network at query time; the model is downloaded once
        (China mirror first, then HuggingFace, then abort). CPU usage is capped
        at ``cpu_percent`` of the system's cores.
      * ``azure`` — calls an OpenAI-compatible embedding endpoint (Azure
        gateway). Requires base_url + api_key; every embed is a network round
        trip.
    """
    # "local" (CPU bge-small-en) or "azure" (OpenAI-compatible API endpoint).
    provider: str = _get("EMBEDDING_PROVIDER", "local")
    # --- local provider settings ---
    local_model: str = _get("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-en-v1.5")
    # Cap CPU threads used by the ONNX runtime at this % of system cores.
    # 60% leaves headroom for the MCP server + scheduler on a shared host.
    cpu_percent: int = _get_int("EMBEDDING_CPU_PERCENT", 60)
    # Minimum download throughput (bytes/s) for the model mirror speed test.
    # The bge-small-en-v1.5 ONNX model is ~130MB; at 200KB/s that's ~11min — acceptable.
    download_min_bps: int = _get_int("EMBEDDING_DOWNLOAD_MIN_BPS", 200_000)
    # --- azure provider settings ---
    base_url: str = _get("EMBEDDING_BASE_URL", "https://your-embedding-endpoint.example.com")
    api_key: str = _get("EMBEDDING_API_KEY", _get("OPENAI_API_KEY", ""))
    model: str = _get("EMBEDDING_MODEL", "text-embedding-3-small")
    # Header used to carry the key (Azure uses "api-key", OpenAI uses "Authorization").
    key_header: str = _get("EMBEDDING_KEY_HEADER", "api-key")
    timeout: int = _get_int("EMBEDDING_TIMEOUT", 60)
    # --- shared settings ---
    # Texts per embedding batch. For local CPU, 32 keeps latency low while
    # giving ONNX enough work for good thread utilisation. For azure, 32 was
    # the measured sweet spot (gateway per-request cost grows super-linearly).
    batch_size: int = _get_int("EMBEDDING_BATCH_SIZE", 32)
    # Concurrent embedding requests. Local is CPU-bound so concurrency just
    # thrashes — forced to 1. For azure, 2 avoids the gateway's server-side
    # queuing that makes high concurrency slower.
    concurrency: int = _get_int("EMBEDDING_CONCURRENCY", 2)

    @property
    def dimensions(self) -> int:
        """Embedding vector dimensionality, derived from the provider.

        local bge-small-en-v1.5 == 384; azure text-embedding-3-small == 1536
        (overridable via EMBEDDING_DIMENSIONS). Changing providers changes
        dimensions, which triggers a vec-index rebuild in db_setup.init_db.
        """
        if self.provider == "local":
            return 384  # bge-small-en-v1.5
        return _get_int("EMBEDDING_DIMENSIONS", 1536)


@dataclass(frozen=True)
class RerankerSettings:
    """Local cross-encoder reranker (CPU, ONNX via fastembed).

    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~80MB). This is
    the torch-style name; ``Reranker._resolve_model`` transparently remaps it
    to its ONNX port (``Xenova/ms-marco-MiniLM-L-6-v2``) so the reranker loads
    via fastembed/onnxruntime — the SAME runtime the bge embedding model uses —
    with no torch/transformers dependency (eliminating the Windows
    c10.dll/WinError 1114 load failure). A user's existing .env that names the
    torch model keeps working unchanged.

    RRF already fuses recall; the reranker only re-sorts the final 25
    candidates, so the small MiniLM cross-encoder gives comparable precision to
    the prior Qwen3-Reranker-0.6B at a fraction of the size. A cross-encoder
    scores (query, document) pairs directly via a sequence-classification head
    — no causal-LM prompt format, no large logits tensor.
    """
    model_name: str = _get("RERANKER_MODEL",
                           "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # If model loading fails, degrade to RRF-only scoring instead of crashing.
    allow_fallback: bool = _get("RERANKER_ALLOW_FALLBACK", "1") == "1"


@dataclass(frozen=True)
class StorageSettings:
    # SQLite DB lives in the project-local user/ folder by default so all
    # runtime data is self-contained. Override with JAMA_MCP_DB_PATH.
    db_path: str = _project_path(_get("JAMA_MCP_DB_PATH", str(USER_DIR / "jama_mcp.db")))
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
    # Concurrent page fetches during a project download. Jama REST caps pages
    # at 50 items and has no bulk-get, so a large project is thousands of
    # serial round-trips. Fetching pages wave-by-wave (each wave = up to
    # `download_concurrency` pages in parallel) cuts download time ~10x.
    # 16 is the measured sweet spot (borrowed from a prior production impl);
    # lower it if Jama returns 429s.
    download_concurrency: int = _get_int("SYNC_DOWNLOAD_CONCURRENCY", 16)


@dataclass(frozen=True)
class ChunkSettings:
    """LlamaIndex SentenceSplitter tuning (data is 100% English, ~30% long).

    ``make_splitter`` in rag_pipeline.py builds the actual SentenceSplitter from
    ``chunk_size`` / ``chunk_overlap`` (plus a paragraph separator and a
    secondary sentence regex hardcoded there, since SentenceSplitter takes those
    as constructor kwargs rather than a generic ``separators`` list).
    """
    chunk_size: int = _get_int("CHUNK_SIZE", 512)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 80)


# NOTE: ``Settings`` is intentionally NOT frozen so that ``reload_settings()``
# can swap in fresh inner dataclasses after the config wizard writes a new
# ``.env`` at runtime. Modules that did ``from config import settings`` hold a
# reference to this same instance, so replacing its attributes propagates to
# every caller without an import-time capture problem.
@dataclass
class Settings:
    jama: JamaSettings = field(default_factory=JamaSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    sync: SyncSettings = field(default_factory=SyncSettings)
    chunk: ChunkSettings = field(default_factory=ChunkSettings)


settings = Settings()


# --------------------------------------------------------------------------- #
# Configuration validation, persistence and live reload
# --------------------------------------------------------------------------- #
# (var, human label, which feature needs it). Required vars block every tool
# that talks to Jama or the embedding endpoint; optional ones only gate the
# features that use them. The embedding API vars are only required when
# EMBEDDING_PROVIDER=azure; local (CPU bge) needs no API credentials.
REQUIRED_VARS_JAMA = [
    ("JAMA_URL", "Jama tenant URL", "jama"),
    ("JAMA_CLIENT_ID", "Jama OAuth client id", "jama"),
    ("JAMA_CLIENT_SECRET", "Jama OAuth client secret", "jama"),
]
REQUIRED_VARS_AZURE_EMB = [
    ("EMBEDDING_BASE_URL", "Embedding endpoint URL", "embedding"),
    ("EMBEDDING_API_KEY", "Embedding API key", "embedding"),
]

# No optional chat-LLM vars remain: Multi-Query expansion is performed by the
# MCP LLM client and passed to the pipeline via ``search(sub_queries=...)``.
OPTIONAL_VARS: list[tuple[str, str, str]] = []


def _required_vars() -> list[tuple[str, str, str]]:
    """Build the required-var list for the active embedding provider."""
    rv = list(REQUIRED_VARS_JAMA)
    if os.environ.get("EMBEDDING_PROVIDER", "local") == "azure":
        rv.extend(REQUIRED_VARS_AZURE_EMB)
    return rv


def validate_config() -> list[dict]:
    """Return a list of issue dicts for missing/malformed config.

    Each issue is ``{"field","severity","message","feature"}`` where severity
    is ``"error"`` (blocks the feature) or ``"warn"`` (degraded mode). An empty
    list means the configuration is complete.
    """
    issues: list[dict] = []
    for name, label, feature in _required_vars():
        val = os.environ.get(name, "").strip()
        if not val or val.startswith("your-"):
            issues.append({
                "field": name, "severity": "error",
                "feature": feature,
                "message": f"{label} is not set. Configure it via the setup "
                           f"wizard (python setup_wizard.py) or the "
                           f"configure_jama tool.",
            })
    # URL shape sanity (cheap, no network). Also flag placeholder hosts
    # (your-tenant / example.com) so a never-configured .env is detected.
    url_checks = [("JAMA_URL", "your-tenant", "jama")]
    if os.environ.get("EMBEDDING_PROVIDER", "local") == "azure":
        url_checks.append(("EMBEDDING_BASE_URL", "your-embedding-endpoint",
                           "embedding"))
    for name, host, feature in url_checks:
        val = os.environ.get(name, "").strip()
        if val and not val.startswith(("http://", "https://")):
            issues.append({
                "field": name, "severity": "error", "feature": feature,
                "message": f"{name} must start with http:// or https://",
            })
        if val and (host in val or "example.com" in val):
            issues.append({
                "field": name, "severity": "error",
                "feature": feature,
                "message": f"{name} is still a placeholder ({val}). Set the "
                           f"real value via the setup wizard or configure_jama.",
            })
    # Orphan Azure config: EMBEDDING_BASE_URL / EMBEDDING_API_KEY are only
    # used when EMBEDDING_PROVIDER=azure. If they're set under the default
    # "local" provider they're silently ignored — a common misconfiguration
    # that wastes a configured endpoint. Warn (not error) so the rest of the
    # server still works; the user just needs to either set
    # EMBEDDING_PROVIDER=azure or drop the unused vars.
    if os.environ.get("EMBEDDING_PROVIDER", "local") != "azure":
        orphans = [v for v in ("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY",
                               "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS")
                   if os.environ.get(v, "").strip()
                   and not os.environ.get(v, "").strip().startswith("your-")]
        if orphans:
            issues.append({
                "field": "EMBEDDING_PROVIDER", "severity": "warn",
                "feature": "embedding",
                "message": f"Azure embedding vars ({', '.join(orphans)}) are "
                           f"set but EMBEDDING_PROVIDER=local — they are "
                           f"ignored. To use Azure, set EMBEDDING_PROVIDER="
                           f"azure (this rebuilds the vector index); otherwise "
                           f"remove the unused vars to avoid confusion.",
            })
    return issues


# All env keys the wizard/configure_jama can persist, in output order. This list
# MUST stay in sync with every ``_get*``/``os.environ`` read above — a key read
# by config but absent here is silently dropped by ``write_env_file`` (which only
# emits keys in this list). The two most consequential omissions historically
# were ``EMBEDDING_PROVIDER`` (a rewritten .env lost the azure->local switch and
# silently reverted to the default) and ``RERANKER_MODEL`` (emitted as a blank
# ``RERANKER_MODEL=`` line, which ``_get`` returns as ``""`` — overriding the
# code default and crashing reranker load with "Repo id must use alphanumeric...").
# Section headers are written as comments by ``write_env_file`` via ``_ENV_HEADERS``.
_ENV_KEYS = [
    # --- Jama REST API ---
    "JAMA_URL", "JAMA_CLIENT_ID", "JAMA_CLIENT_SECRET", "JAMA_API_PREFIX",
    "JAMA_PAGE_SIZE", "JAMA_PAGE_DELAY", "JAMA_REQUEST_TIMEOUT",
    "JAMA_MAX_RETRIES", "JAMA_MIN_BYTES_PER_SEC", "JAMA_SPEED_TEST_TIMEOUT",
    "JAMA_PAGE_MIN_BYTES_PER_SEC", "JAMA_PAGE_MAX_RETRIES",
    # --- Embedding provider ---
    "EMBEDDING_PROVIDER", "EMBEDDING_LOCAL_MODEL", "EMBEDDING_CPU_PERCENT",
    "EMBEDDING_DOWNLOAD_MIN_BPS", "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_CONCURRENCY", "EMBEDDING_TIMEOUT",
    # --- Azure embedding (only used when EMBEDDING_PROVIDER=azure) ---
    "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS", "EMBEDDING_KEY_HEADER",
    # --- Local cross-encoder reranker (CPU, ONNX via fastembed) ---
    "RERANKER_MODEL", "RERANKER_ALLOW_FALLBACK",
    # --- Storage ---
    "JAMA_MCP_DB_PATH", "SQLITE_BUSY_TIMEOUT_MS",
    # --- Incremental sync ---
    "SYNC_ENABLED", "SYNC_INTERVAL_HOURS", "SYNC_MAX_ITEMS_PER_RUN",
    "SYNC_DOWNLOAD_CONCURRENCY",
    # --- Chunking ---
    "CHUNK_SIZE", "CHUNK_OVERLAP",
]

# Comment headers inserted before each logical group in the written .env, so the
# file stays self-documenting. Keys not preceded by a header get no comment line.
_ENV_HEADERS: dict[str, str] = {
    "JAMA_URL": "Jama REST API",
    "EMBEDDING_PROVIDER": "Embedding provider",
    "EMBEDDING_BASE_URL": "Azure embedding (only used when EMBEDDING_PROVIDER=azure)",
    "RERANKER_MODEL": "Local cross-encoder reranker (CPU, ONNX via fastembed)",
    "JAMA_MCP_DB_PATH": "Storage",
    "SYNC_ENABLED": "Incremental sync",
    "CHUNK_SIZE": "Chunking",
}


def write_env_file(values: dict, path: str | None = None) -> str:
    """Write a ``.env`` file from a ``{var: value}`` mapping.

    Only the supplied keys are overridden; for every other key the *current
    environment* is consulted and the value is written through **only if it is
    set and non-empty**. A key that is unset in the environment (and not in
    ``values``, or given as ``None``/``""``) is OMITTED from the file rather than
    emitted as ``KEY=`` — a present-but-empty env var makes ``_get`` return ``""``,
    overriding the code default, so emitting blanks would silently break defaults
    like ``RERANKER_MODEL`` (empty -> reranker load fails) and
    ``EMBEDDING_LOCAL_MODEL``. Returns the absolute path written.
    """
    target = Path(path) if path else PROJECT_ROOT / ".env"
    # Effective value per key: caller override wins; otherwise the live env var
    # (only if set + non-empty, so we don't write blank lines that clobber code
    # defaults). An explicitly-empty override ("" / None) is treated as "unset"
    # rather than written as ``KEY=`` — none of these vars have a meaningful
    # empty value, and an empty ``RERANKER_MODEL=`` would break reranker load.
    eff: dict[str, str] = {}
    for k in _ENV_KEYS:
        v = values.get(k)
        if k in values and v not in (None, ""):
            eff[k] = str(v)
        elif k in os.environ and os.environ[k] != "":
            eff[k] = os.environ[k]
        # else: leave unset -> omitted from the file -> code default applies
    # Caller may also set a key NOT in _ENV_KEYS (escape hatch); write those too
    # so configure_jama never silently drops a value the user explicitly passed.
    for k, v in values.items():
        if k not in eff and v not in (None, ""):
            eff[k] = str(v)
    lines = [
        "# Jama MCP Server environment (managed by setup_wizard / configure_jama).",
        "# Copy to .env and fill in. All values are read by config.py.",
        "# Keys left out of this file fall back to the code defaults in config.py.",
        "",
    ]
    for k in _ENV_KEYS:
        if k in _ENV_HEADERS:
            lines.append(f"# --- {_ENV_HEADERS[k]} ---")
        if k in eff:
            lines.append(f"{k}={eff[k]}")
        # Unset + no default-override -> omitted (code default applies)
    # Any extra caller-supplied keys not in _ENV_KEYS, appended at the end.
    extra = [k for k in values if k not in _ENV_KEYS and values[k] is not None]
    if extra:
        lines.append("# --- Additional keys ---")
        for k in extra:
            lines.append(f"{k}={values[k]}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def reload_settings() -> None:
    """Re-read ``.env`` (overriding the live environment) and rebuild settings.

    Called after the config wizard / ``configure_jama`` writes a new ``.env``.
    Because ``Settings`` is mutable and every module shares the same
    ``settings`` instance, swapping the inner dataclasses here propagates to
    already-imported callers (e.g. JamaClient reads ``settings.jama.url`` at
    call time, not import time).
    """
    try:
        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except Exception:  # pragma: no cover
        pass
    settings.jama = JamaSettings()
    settings.embedding = EmbeddingSettings()
    settings.reranker = RerankerSettings()
    settings.storage = StorageSettings()
    settings.sync = SyncSettings()
    settings.chunk = ChunkSettings()
