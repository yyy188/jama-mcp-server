#!/usr/bin/env python3
"""Interactive setup wizard for the Jama MCP Server.

Prompts the user for every required configuration value (with sensible
defaults pulled from the existing environment / .env), writes a complete
``.env``, then runs the pre-flight dependency + config check and an optional
live connectivity self-test against Jama and the embedding endpoint.

Run with:  python setup_wizard.py [--non-interactive] [--self-test]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

# Ensure project root on sys.path so `import config` works when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (PROJECT_ROOT, OPTIONAL_VARS, _required_vars,  # noqa: E402
                    reload_settings, settings, write_env_file)


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def _ask(prompt: str, default: str = "", secret: bool = False,
         validator: Callable[[str], str | None] | None = None) -> str:
    """Prompt with a default; validate; re-ask on bad input. Never raises."""
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            val = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user.")
            sys.exit(1)
        if not val and default:
            val = default
        if secret:
            # Don't echo, but we already read via input(); mask only in logs.
            pass
        if validator:
            err = validator(val)
            if err:
                print(f"  ! {err}")
                continue
        return val


def _url_validator(name: str):
    def v(val: str):
        if val and not val.startswith(("http://", "https://")):
            return f"{name} must start with http:// or https://"
        return None
    return v


def _collect() -> dict:
    """Walk the user through required + optional config values."""
    print("=" * 70)
    print("  Jama MCP Server — Setup Wizard")
    print("=" * 70)
    print("Enter values (press Enter to keep the [default]). Secrets are not echoed")
    print("back; they are written to .env only.\n")

    values: dict = {}

    print("--- Jama REST API (required) ---")
    values["JAMA_URL"] = _ask("Jama tenant URL",
                              os.environ.get("JAMA_URL", "https://your-tenant.jamacloud.com"),
                              validator=_url_validator("JAMA_URL"))
    values["JAMA_CLIENT_ID"] = _ask("Jama OAuth client id",
                                    os.environ.get("JAMA_CLIENT_ID", ""))
    values["JAMA_CLIENT_SECRET"] = _ask("Jama OAuth client secret",
                                        os.environ.get("JAMA_CLIENT_SECRET", ""),
                                        secret=True)

    print("\n--- Embedding endpoint (required for semantic search) ---")
    values["EMBEDDING_BASE_URL"] = _ask("Embedding endpoint URL",
                                        os.environ.get("EMBEDDING_BASE_URL", ""),
                                        validator=_url_validator("EMBEDDING_BASE_URL"))
    values["EMBEDDING_API_KEY"] = _ask("Embedding API key",
                                       os.environ.get("EMBEDDING_API_KEY", ""),
                                       secret=True)

    print("\n--- Storage / sync (defaults are fine for most setups) ---")
    values["JAMA_MCP_DB_PATH"] = _ask("SQLite DB path",
                                      os.environ.get(
                                          "JAMA_MCP_DB_PATH",
                                          str(PROJECT_ROOT / "user" / "jama_mcp.db")))
    return values


# --------------------------------------------------------------------------- #
# Live connectivity self-test
# --------------------------------------------------------------------------- #
def _live_selftest() -> dict:
    """Probe Jama auth + embedding endpoint with the configured credentials."""
    from jama_client import JamaClient
    from rag_pipeline import EmbeddingClient
    report = {"jama": None, "embedding": None}

    # Jama: list projects (lightweight) to confirm OAuth + reachability.
    try:
        client = JamaClient()
        client.preflight_speed_check()
        projects = list(client.list_projects())
        report["jama"] = {"ok": True, "project_count": len(projects),
                          "sample": [{"id": p.get("id"), "name": (p.get("fields") or {}).get("name")}
                                     for p in projects[:3]]}
    except Exception as exc:
        report["jama"] = {"ok": False, "error": str(exc)[:300]}

    # Embedding: embed a probe string.
    try:
        emb = EmbeddingClient()
        vec = emb.embed_one("jama mcp self test")
        report["embedding"] = {"ok": True, "dimensions": len(vec)}
    except Exception as exc:
        report["embedding"] = {"ok": False, "error": str(exc)[:300]}
    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Jama MCP Server setup wizard.")
    ap.add_argument("--non-interactive", action="store_true",
                    help="Skip prompts; just write .env from current env and validate.")
    ap.add_argument("--self-test", action="store_true",
                    help="After writing .env, run a live connectivity probe.")
    args = ap.parse_args()

    if args.non_interactive:
        values = {k: os.environ.get(k, "") for k, *_ in _required_vars() + OPTIONAL_VARS}
    else:
        values = _collect()

    path = write_env_file(values)
    print(f"\nWrote {path}")
    # Re-read what we just wrote so the in-process settings reflect it.
    reload_settings()

    print("\n--- Pre-flight check ---")
    from preflight import preflight
    report = preflight(require={"jama", "embedding"})
    for line in report["issues"]:
        print(f"  • {line}")
    print(f"  blocking={report['blocking']}")

    if args.self_test:
        print("\n--- Live connectivity self-test ---")
        st = _live_selftest()
        print(f"  Jama:      {'OK' if st['jama'] and st['jama']['ok'] else 'FAIL'}"
              + ("" if st["jama"] and st["jama"]["ok"]
                 else f" — {st['jama']['error']}" if st["jama"] else ""))
        if st["jama"] and st["jama"]["ok"]:
            print(f"             {st['jama']['project_count']} project(s) visible; "
                  f"sample: {st['jama']['sample']}")
        print(f"  Embedding: {'OK' if st['embedding'] and st['embedding']['ok'] else 'FAIL'}"
              + ("" if st["embedding"] and st["embedding"]["ok"]
                 else f" — {st['embedding']['error']}" if st["embedding"] else ""))
        if st["embedding"] and st["embedding"]["ok"]:
            print(f"             embedding dims = {st['embedding']['dimensions']}")

    if report["blocking"]:
        print("\n✗ Setup incomplete — see issues above.")
        return 1
    print("\n✓ Setup complete. Start the server with: python server.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
