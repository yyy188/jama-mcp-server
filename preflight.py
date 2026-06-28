"""Pre-flight dependency + configuration validation for the Jama MCP Server.

Every MCP tool gates on :func:`preflight` before doing any real work, so a
misconfigured server reports a clear, actionable error instead of failing
halfway through a Jama API call. Checks are split into three tiers:

1. **Python packages** — importable? (fast, cached after first call).
2. **Configuration** — required env vars present and well-formed?
3. **Storage** — SQLite DB + sqlite-vec extension openable?

Network/credential *validity* (does the Jama token actually work?) is NOT
checked here — that's a live, slow operation done only by the explicit
``validate_setup`` tool on demand. The per-call guard stays fast and offline.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

from config import settings, validate_config

log = logging.getLogger(__name__)

# (import name, what feature needs it, is it required for core operation?).
# Core deps block every tool; optional deps only block the feature that uses
# them and are reported as warnings so the rest of the server still works.
_CORE_DEPS = [
    ("requests", "HTTP client"),
    ("bs4", "HTML cleaning"),
    ("sqlite_vec", "sqlite-vec vector index"),
    ("apscheduler", "incremental sync scheduler"),
]
_OPTIONAL_DEPS = [
    ("mcp", "MCP framework (FastMCP)"),
    ("llama_index.core", "LlamaIndex chunking / Multi-Query"),
    ("llama_index.llms.openai", "LlamaIndex OpenAI LLM (optional Multi-Query)"),
    ("transformers", "local Qwen3 reranker"),
    ("torch", "local Qwen3 reranker"),
]

# Provider-specific embedding deps. Only required for the active provider.
_PROVIDER_DEPS = {
    "local": [("fastembed", "local CPU embedding (bge-small-en-v1.5)")],
    "azure": [],  # azure uses requests, already in _CORE_DEPS
}


def _try_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as exc:  # pragma: no cover - environment specific
        return False, str(exc)[:200]


def check_dependencies() -> dict[str, Any]:
    """Return a structured dependency report.

    Shape::
        {"ok": bool, "missing_core": [...], "missing_optional": [...],
         "checked": int}
    """
    missing_core: list[dict] = []
    missing_optional: list[dict] = []
    checked = 0
    for name, purpose in _CORE_DEPS:
        ok, err = _try_import(name)
        checked += 1
        if not ok:
            missing_core.append({"package": name, "purpose": purpose, "error": err})
    for name, purpose in _OPTIONAL_DEPS:
        ok, err = _try_import(name)
        checked += 1
        if not ok:
            missing_optional.append({"package": name, "purpose": purpose, "error": err})
    # The active embedding provider has its own required deps. fastembed is
    # CORE for local (no embedding works without it); azure only needs requests.
    provider = settings.embedding.provider
    for name, purpose in _PROVIDER_DEPS.get(provider, []):
        ok, err = _try_import(name)
        checked += 1
        if not ok:
            missing_core.append({"package": name, "purpose": purpose, "error": err})
    return {
        "ok": not missing_core,
        "missing_core": missing_core,
        "missing_optional": missing_optional,
        "checked": checked,
    }


def check_storage() -> dict[str, Any]:
    """Open the SQLite DB + load sqlite-vec to confirm storage is usable."""
    try:
        from db_setup import init_db
        conn = init_db()
        # Confirm the vec extension actually loaded (init_db loads it).
        conn.execute("SELECT 1 FROM chunks_vec LIMIT 0")
        conn.close()
        return {"ok": True, "db_path": settings.storage.db_path}
    except Exception as exc:
        return {"ok": False, "db_path": settings.storage.db_path,
                "error": str(exc)[:300]}


def preflight(*, require: set[str] | None = None) -> dict[str, Any]:
    """Full offline readiness check.

    Args:
        require: subset of ``{"jama","embedding","llm"}`` — which features the
                 caller needs. Missing required config for those features makes
                 the report blocking (``ok=False``). ``jama``+``embedding`` are
                 implied for most tools.

    Returns::
        {"ok": bool, "blocking": bool, "dependencies", "config", "storage",
         "issues": [human-readable strings], "hint": str}
    """
    require = require or set()
    deps = check_dependencies()
    storage = check_storage()
    config_issues = validate_config()

    issues: list[str] = []
    blocking = False

    if deps["missing_core"]:
        blocking = True
        for m in deps["missing_core"]:
            issues.append(f"Missing core dependency '{m['package']}' ({m['purpose']}). "
                          f"Run: pip install -r requirements.txt")

    if not storage["ok"]:
        blocking = True
        issues.append(f"Storage unavailable: {storage.get('error')}")
    else:
        issues.append(f"Storage OK ({storage['db_path']})")

    # Map config issues to the features this caller actually needs.
    needed = set(require)
    # Most tools need jama; embedding only for search/indexing.
    for issue in config_issues:
        feat = issue["feature"]
        if feat in needed and issue["severity"] == "error":
            blocking = True
            issues.append(f"[{feat}] {issue['message']}")
        elif issue["severity"] == "warn":
            issues.append(f"[{feat}] {issue['message']} (optional)")

    hint = ""
    if blocking:
        missing = [i["field"] for i in config_issues
                   if i["severity"] == "error" and i["feature"] in needed]
        if missing or deps["missing_core"] or not storage["ok"]:
            hint = ("Configuration or dependencies incomplete. Run the setup "
                    "wizard: `python setup_wizard.py`, or call the configure_jama "
                    "tool with the missing values, then call validate_setup.")

    return {
        "ok": not blocking,
        "blocking": blocking,
        "dependencies": deps,
        "config_issues": config_issues,
        "storage": storage,
        "issues": issues,
        "hint": hint,
    }
