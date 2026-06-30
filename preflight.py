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
    ("llama_index.core", "LlamaIndex chunking"),
    ("transformers", "local cross-encoder reranker"),
    ("torch", "local cross-encoder reranker"),
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


def _check_torch_env() -> list[str]:
    """Detect torch environment problems that silently break the reranker.

    The reranker loads ``torch`` + ``transformers``. Two common environment
    breakages that leave weights on disk but make loading fail (so search
    silently degrades to RRF):

    1. A CUDA build of torch installed (``+cuXXX``) — ~6GB, depends on a
       VC++ Runtime absent on many Windows machines (WinError 1114).
    2. A leftover ``torchaudio``/``torchvision`` from a prior CUDA install
       whose version doesn't match the installed (CPU) torch — its .pyd
       pollutes the torch load chain with DLL-not-found errors.

    Returns human-readable warning strings (empty list = healthy). Non-fatal:
    these are warnings, not blocking errors, because the rest of the server
    (embedding, browse, native query) still works without a usable torch.
    """
    warnings: list[str] = []
    try:
        import torch  # type: ignore
        ver = getattr(torch, "__version__", "")
        if "+cpu" not in ver and "cpu" not in ver.split("+")[-1:]:
            # Heuristic: official CPU builds carry the "+cpu" local-version
            # marker. Anything else (incl. plain "2.x.y" from a mirror, or
            # "+cuXXX") is suspect on a CPU-only reranker setup.
            warnings.append(
                f"torch {ver} is not the CPU build. The CUDA/default build "
                f"is ~6GB and may fail to load on Windows (WinError 1114). "
                f"Reinstall with: pip install torch==2.6.0+cpu --index-url "
                f"https://download.pytorch.org/whl/cpu")
    except Exception:
        # torch missing is reported separately by _OPTIONAL_DEPS; nothing to
        # add here.
        return warnings

    # Detect version-mismatched torchaudio/torchvision (CUDA leftovers). These
    # ship .pyd files that reference CUDA DLLs and break transformers' torch
    # import even when the torch package itself is the CPU build.
    try:
        from importlib.metadata import version, PackageNotFoundError
        for pkg in ("torchaudio", "torchvision"):
            try:
                pv = version(pkg)
            except PackageNotFoundError:
                continue
            # A CUDA local-version marker on a companion package means it was
            # built against CUDA torch; if the installed torch is CPU, the
            # companion's native libs will fail to load.
            if "+cu" in pv or "+cu" in ver:
                if ("+cpu" in ver) != ("+cpu" in pv) or \
                        (ver.split("+")[0] != pv.split("+")[0]):
                    warnings.append(
                        f"{pkg} {pv} does not match torch {ver}. A leftover "
                        f"CUDA companion package can break reranker loading "
                        f"via DLL errors. Uninstall it: pip uninstall {pkg}")
    except Exception:
        pass  # importlib.metadata quirks across versions; non-fatal.
    return warnings


def check_dependencies() -> dict[str, Any]:
    """Return a structured dependency report.

    Shape::
        {"ok": bool, "missing_core": [...], "missing_optional": [...],
         "env_warnings": [...], "checked": int}
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
    env_warnings = _check_torch_env()
    return {
        "ok": not missing_core,
        "missing_core": missing_core,
        "missing_optional": missing_optional,
        "env_warnings": env_warnings,
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

    # Non-blocking environment warnings (e.g. CUDA torch, mismatched
    # torchaudio) — these silently break the reranker so surface them, but
    # don't block tools that don't need torch.
    for w in deps.get("env_warnings", []):
        issues.append(f"[env] {w}")

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
