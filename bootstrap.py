#!/usr/bin/env python3
"""Pre-download the embedding + reranker models so the first sync isn't slow.

Run this once after installing the server (``pip install -r requirements.txt``)
and configuring it (``python setup_wizard.py``). It downloads BOTH models into
the project-local HF cache synchronously, with live progress, so the first
``init_jama_project`` doesn't pay the download cost (and a flaky network doesn't
fail mid-sync).

    python bootstrap.py

Re-running is a fast no-op: models already cached are skipped. This is the
synchronous, foreground counterpart of the async ``bootstrap_models`` MCP tool
(use the MCP tool if the server is already running; use this script before
first launch).
"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401  (loads .env + sets HF env vars)
from config import settings

# Silence the tqdm progress bars that fastembed / huggingface_hub emit during
# downloads. In a non-interactive shell (piped output, MCP-driven runs) those
# bars spam thousands of carriage-return lines that are unreadable and make it
# look like the process is stuck. We replace them with periodic one-line prints
# driven by the progress thread below.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:
    import tqdm  # type: ignore
    tqdm.tqdm.monitor_interval = 0  # disable the tqdm monitor thread
except Exception:
    pass


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1024:.1f} {unit}"
        n = n / 1024
    return f"{n:.1f} GB"


def _cache_size() -> int:
    """Total size of the HF cache dir (rough progress proxy)."""
    cache = os.environ.get("HF_HOME", str(config.USER_DIR / "huggingface"))
    total = 0
    for dirpath, _dirs, files in os.walk(cache):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


class _ProgressReporter:
    """Print periodic cache-size growth while a download runs.

    fastembed (used for both the embedding and the reranker ONNX downloads)
    gives no usable byte callback, so we poll the cache directory size from a
    daemon thread and emit one line every ~10s. This gives the user visible
    movement ("cache grew to 48 MB") instead of a silent multi-minute hang.
    Stopped via the ``stop()`` event.
    """

    def __init__(self, label: str):
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._start = _cache_size()
        self._t0 = time.monotonic()

        def _loop():
            while not self._stop.wait(10.0):
                grew = _cache_size() - self._start
                dt = time.monotonic() - self._t0
                print(f"    ... {self._label}: {_human_bytes(grew)} so far "
                      f"({dt:.0f}s)", flush=True)

        self._thread = threading.Thread(target=_loop, daemon=True,
                                        name="bootstrap-progress")
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


def _download_embedding() -> bool:
    """Download the embedding model (local provider only). Returns True on OK."""
    if settings.embedding.provider != "local":
        print(f"  embedding provider is {settings.embedding.provider} — "
              f"no model to download.")
        return True
    from rag_pipeline import LocalEmbeddingClient
    emb = LocalEmbeddingClient()
    if emb._model_present():
        print(f"  embedding model already cached ({settings.embedding.local_model}).")
        return True
    print(f"  downloading embedding model {settings.embedding.local_model} "
          f"(~130MB ONNX) ...")
    before = _cache_size()
    t0 = time.monotonic()
    with _ProgressReporter("embedding download"):
        emb._download_model()
    dt = time.monotonic() - t0
    grew = _cache_size() - before
    print(f"  embedding model downloaded ({_human_bytes(grew)} in {dt:.0f}s).")
    return True


def _download_reranker() -> bool:
    """Download the reranker model. Returns True on OK."""
    from rag_pipeline import Reranker
    rr = Reranker()
    if rr.weights_cached():
        print(f"  reranker model already cached ({settings.reranker.model_name}).")
        return True
    print(f"  downloading reranker model {settings.reranker.model_name} "
          f"(~80MB ONNX) ...")
    before = _cache_size()
    t0 = time.monotonic()
    with _ProgressReporter("reranker download"):
        rr.ensure_downloaded()
    dt = time.monotonic() - t0
    grew = _cache_size() - before
    print(f"  reranker model downloaded ({_human_bytes(grew)} in {dt:.0f}s).")
    return True


def main() -> int:
    print("=" * 60)
    print("  Jama MCP — model pre-download (bootstrap)")
    print("=" * 60)
    print("This downloads the embedding + reranker models into the local")
    print("cache so the first sync isn't slowed by a download. Already-cached")
    print("models are skipped.\n")

    ok = True
    print("[1/2] Embedding model")
    try:
        if not _download_embedding():
            ok = False
    except Exception as exc:
        print(f"  FAILED: {exc}")
        ok = False

    print("\n[2/2] Reranker model")
    try:
        if not _download_reranker():
            ok = False
    except Exception as exc:
        print(f"  FAILED: {exc}")
        ok = False

    print("\n" + "=" * 60)
    if ok:
        print("  Models ready. You can now run: python server.py")
        print("  (then call bootstrap_models via MCP to verify, or just init_jama_project)")
    else:
        print("  One or more models failed to download. Check network and retry.")
        print("  The server still works — models download on first sync as a fallback.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
