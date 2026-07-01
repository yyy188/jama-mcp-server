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
]

# Provider-specific embedding deps. fastembed serves BOTH the local bge
# embedding AND the cross-encoder reranker (both run on onnxruntime), so it is
# core for the local provider. azure only needs requests (already in _CORE_DEPS).
_PROVIDER_DEPS = {
    "local": [("fastembed", "local CPU embedding (bge-small-en-v1.5) + "
                            "cross-encoder reranker (MiniLM, ONNX)")],
    "azure": [],
}


def _try_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as exc:  # pragma: no cover - environment specific
        return False, str(exc)[:200]


# Download URL for the Microsoft VC++ Redistributable (x64). aka.ms is a
# Microsoft short-link that redirects to the current stable build on
# download.visualstudio.microsoft.com — verified reachable from mainland China
# (~420 KB/s, ~25MB). This is the ONLY system dependency onnxruntime 1.20.1
# needs on Windows (vcruntime140.dll); it ships with most Windows machines
# (any machine that has Chrome / Java / VS Code already has it), but a clean
# Windows install may lack it.
_VC_REDIST_URL = "https://aka.ms/vs/16/release/vc_redist.x64.exe"


def _check_vcruntime() -> tuple[bool, str]:
    """Check whether vcruntime140.dll is loadable on this Windows machine.

    onnxruntime 1.20.1 (the pinned version) is a C++ binary that dynamically
    links vcruntime140.dll. On a clean Windows install without the VC++
    Redistributable, importing onnxruntime fails with ``OSError: [WinError
    126]`` / ``WinError 1114``. This probe catches that BEFORE the user hits a
    confusing onnxruntime import error, and points them at the fix.

    Returns ``(ok, message)``. On non-Windows platforms always ``(True, "")``
    (onnxruntime ships the system libs in its Linux/macOS wheels).
    """
    import sys
    if sys.platform != "win32":
        return True, ""
    try:
        import ctypes
        ctypes.WinDLL("vcruntime140.dll")
        return True, ""
    except OSError:
        return False, (
            "vcruntime140.dll not found. onnxruntime (the ML runtime for the "
            "embedding + reranker models) needs the Microsoft VC++ Redistributable. "
            "Run `python -c \"from preflight import install_vcruntime; "
            "install_vcruntime()\"` to auto-install it, or download manually from "
            f"{_VC_REDIST_URL} (24MB, ~1min on a China connection).")


def install_vcruntime() -> str:
    """Download and silently install the VC++ Redistributable (Windows x64).

    Called when ``_check_vcruntime`` reports the DLL is missing. Downloads
    vc_redist.x64.exe from the Microsoft aka.ms short-link (verified reachable
    from mainland China), then runs it with ``/install /quiet /norestart``.
    Requires admin privileges (UAC prompt); on success vcruntime140.dll lands
    in C:\\Windows\\System32. Returns the path to the downloaded installer.
    """
    import os
    import subprocess
    import sys
    import tempfile
    import urllib.request

    if sys.platform != "win32":
        raise RuntimeError("VC++ Redistributable is Windows-only.")

    dest = os.path.join(tempfile.gettempdir(), "vc_redist.x64.exe")
    if not os.path.exists(dest):
        log.info("Downloading VC++ Redistributable from %s ...", _VC_REDIST_URL)
        urllib.request.urlretrieve(_VC_REDIST_URL, dest)
        log.info("Downloaded VC++ Redistributable (%d bytes).",
                 os.path.getsize(dest))
    else:
        log.info("VC++ Redistributable installer already cached at %s.", dest)

    log.info("Installing VC++ Redistributable (silent, may show a UAC prompt)...")
    # /install /quiet /norestart: install without UI, no reboot.
    result = subprocess.run(
        [dest, "/install", "/quiet", "/norestart"],
        capture_output=True, text=True)
    if result.returncode == 0:
        log.info("VC++ Redistributable installed successfully.")
    elif result.returncode == 1638:
        # 1638 = a newer version is already installed (treat as success).
        log.info("VC++ Redistributable already installed (newer version).")
    else:
        raise RuntimeError(
            f"VC++ Redistributable install failed (exit {result.returncode}): "
            f"{result.stderr[:300]}. Try running {dest} manually.")
    return dest


def check_dependencies() -> dict[str, Any]:
    """Return a structured dependency report.

    Shape::
        {"ok": bool, "missing_core": [...], "missing_optional": [...],
         "vcruntime": {"ok": bool, "message": str}, "checked": int}
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
    # CORE for local (no embedding OR reranker works without it); azure only
    # needs requests.
    provider = settings.embedding.provider
    for name, purpose in _PROVIDER_DEPS.get(provider, []):
        ok, err = _try_import(name)
        checked += 1
        if not ok:
            missing_core.append({"package": name, "purpose": purpose, "error": err})
    # Windows system DLL check: onnxruntime needs vcruntime140.dll. A clean
    # Windows without the VC++ Redistributable will fail to import onnxruntime
    # with a cryptic WinError — this surfaces the real cause + the fix.
    vc_ok, vc_msg = _check_vcruntime()
    return {
        "ok": not missing_core and vc_ok,
        "missing_core": missing_core,
        "missing_optional": missing_optional,
        "vcruntime": {"ok": vc_ok, "message": vc_msg},
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

    # Windows VC++ Runtime: onnxruntime needs vcruntime140.dll. Missing it is
    # blocking for the local embedding/reranker path (onnxruntime won't import).
    vc = deps.get("vcruntime", {})
    if not vc.get("ok", True):
        blocking = True
        issues.append(f"[system] {vc['message']}")

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
