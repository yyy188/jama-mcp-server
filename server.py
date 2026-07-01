"""Jama MCP Server entry point.

Exposes MCP tools to LLM clients over three transports (selectable via the
JAMA_MCP_TRANSPORT env var):
  * stdio (default)     — local MCP client spawns this as a subprocess.
  * streamable-http     — server listens on a port; remote clients connect via
                          http://host:8000/mcp (the MCP new standard).
  * sse                 — server-sent events transport for older MCP clients
                          (http://host:8000/sse).
For HTTP/SSE modes set JAMA_MCP_HOST (0.0.0.0 for remote access) and
JAMA_MCP_PORT (default 8000).

Tools exposed:
  * bootstrap_models        - async pre-download of embedding + reranker models
  * init_jama_project        - async background init (returns job_id)
  * reinit_jama_project      - async full re-sync of an initialized project
  * get_sync_progress / get_bootstrap_progress - poll job progress
  * get_sync_status          - project monitor: in-flight + last run of each kind
  * search_jama_semantics    - high-precision RAG (client multi-query + hybrid + RRF + rerank)
  * query_jama_native_metadata - direct Jama REST filtering (exact metadata)
  * plus read-only Jama browse tools (items, relationships, releases, ...)

All sync operations (init / reinit / scheduled incremental / model bootstrap)
run as async background jobs in a thread pool and report progress into the
``sync_jobs`` table, pollable every ~2 min. An APScheduler job incrementally
syncs initialized projects every N hours: it walks items whose modifiedDate >
last_sync_time, cleans them, re-chunks and updates the FTS5 + sqlite-vec indexes.

Run with:  python server.py            (stdio, default)
           JAMA_MCP_TRANSPORT=streamable-http python server.py   (HTTP)
Configure an MCP client (Claude Desktop, etc.) to launch this as a stdio server,
or connect remotely in HTTP/SSE mode.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from apscheduler.schedulers.background import BackgroundScheduler
from mcp.server.fastmcp import FastMCP

from config import reload_settings, settings, write_env_file
from db_setup import (count_chunks, count_items, create_job,
                      get_active_job_for_project,
                      get_connection, get_job, get_latest_job_for_project,
                      get_project, indexed_item_ids, init_db,
                      list_initialized_projects,
                      reconcile_stale_jobs, update_job, upsert_item,
                      upsert_project, replace_chunks, write_txn)
from jama_client import JamaClient, utcnow_iso
from preflight import preflight
from rag_pipeline import RAGPipeline, chunk_item

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("jama_mcp")

# --------------------------------------------------------------------------- #
# Globals (lazy singletons)
# --------------------------------------------------------------------------- #
_db_conn = None
_jama: JamaClient | None = None
_rag: RAGPipeline | None = None
_init_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jama-job")

# Jama pre-flight speed-test retry. The test measures live throughput, so a
# momentary blip can drop it below the floor even on a healthy link; retry a
# couple of times with backoff before marking a project ERROR (a false ERROR
# needs a manual re-init to clear, so we err on the side of retrying).
_PREFLIGHT_RETRIES = 3
_PREFLIGHT_BACKOFF = 5  # seconds between attempts

# Per-project sync locks. ``_sync_project`` acquires the lock for its project
# so that a user-triggered init and a scheduler-triggered incremental sync can
# never run concurrently for the SAME project (which would race on upserts and
# could leave a split-brain terminal state). Different projects still run in
# parallel. The dict is guarded by ``_init_lock``; the per-project Lock is
# held only for the duration of one sync.
_project_locks: dict[int, threading.Lock] = {}


def _project_lock(project_id: int) -> threading.Lock:
    """Return (creating if needed) the sync lock for ``project_id``."""
    with _init_lock:
        lk = _project_locks.get(project_id)
        if lk is None:
            lk = threading.Lock()
            _project_locks[project_id] = lk
        return lk

# Server-level instructions sent to the LLM client alongside the tool list.
# This is the canonical MCP way to steer tool selection: it tells the model
# to DEFAULT to fusion retrieval and only fall back to native metadata for
# precise pointers, so fuzzy/keyword/natural-language questions always hit
# the hybrid (FTS5 BM25 + sqlite-vec + RRF + rerank) path.
JAMA_MCP_INSTRUCTIONS = """\
Jama MCP — tool selection guide (read before choosing a tool)

DEFAULT TO FUSION SEARCH. For any question that is not a precise lookup,
call `search_jama_semantics`. It already fuses three retrieval signals in
one call — keyword (FTS5/BM25), vector (sqlite-vec cosine) and RRF — then
reranks with a local cross-encoder model. So "like", "keyword" and "semantic"
queries are ALL best answered by this single tool. Use it for:
  • natural-language questions      ("how does volume sync work")
  • partial / fuzzy matches         ("something about login timeout")
  • topical / concept searches      ("test cases for payment flow")
  • "find items mentioning …"       (free-text containment)

USE NATIVE METADATA ONLY FOR PRECISE POINTERS. Reach for the structured
tools below when the user gives an exact, unambiguous key — not a topic:
  • `query_jama_native_metadata` — exact document_key (e.g. "SA-TC-7"),
    exact status ("BLOCKED"), or exact item_type filter on a project.
  • `get_jama_item`              — a specific numeric item id.
  • `get_jama_item_children` / `get_jama_item_relationships` /
    `get_jama_item_comments` / `get_jama_item_attachments` — drill into
    one known item id.
  • `list_jama_projects` / `list_jama_releases` / `list_jama_test_runs` /
    `list_jama_item_types` — enumerate, not search.

ROUTING RULE OF THUMB: if the user's intent can be expressed as a search
box query → `search_jama_semantics`. If it is "show me item X" or "list
all Y" → the native/browse tools. When unsure, prefer fusion search; it
is recall-oriented and surfaces the most relevant items even for
near-keyword phrasing, while native tools return empty on any
misspelling or mismatch.

QUERY EXPANSION: before calling `search_jama_semantics`, rewrite the
user's query into 3-5 diverse sub-queries (different semantic angles —
synonyms, broader/narrower scope, related concepts) and pass them via
the `sub_queries` parameter; keep the original query in `query` (it is
the rerank reference). This maximizes recall for RRF fusion and is
preferred over letting the server fall back to lexical variants.

MODEL BOOTSTRAP (first-run, before any init). The embedding model (~130MB
ONNX) and cross-encoder reranker (~80MB ONNX) are NOT bundled — they download
on first use. Both run on onnxruntime via fastembed (no torch dependency).
Call `bootstrap_models()` right after the server is configured: it downloads
BOTH models asynchronously and returns a job_id immediately, so the first
sync isn't slowed by a download and a download failure surfaces here instead
of mid-sync. Poll `get_bootstrap_progress(job_id)` roughly every 2 minutes,
reporting each sample (status, progress %, message) to the user, until status
is DONE or ERROR. Progress is phase-based (not live bytes): fastembed (used
for both models) lacks per-chunk byte callbacks, so `message` reports phase
transitions (e.g. "Downloading reranker model (...)" -> "Reranker model
ready") rather than byte counts. Already-cached models are skipped, so
re-running bootstrap after success is a fast no-op. OPTIONAL: if skipped,
init/sync still works (downloads on demand) but the first sync is slower and
a download failure surfaces mid-sync — prefer running bootstrap first.

SYNC MONITORING. `init_jama_project` (first init) and `reinit_jama_project`
(re-index an already-initialized project) run in the BACKGROUND and return a
job_id immediately — they are never blocking. Scheduled incremental syncs run
automatically (every ~2h) with no job_id returned to you. After starting an
init or reinit, poll `get_sync_progress(job_id)` roughly every 2 minutes,
reporting each sample (status, done/total, message) to the user, until status
is DONE or ERROR. For a single project-wide view — the in-flight job plus the
last init/reinit/sync run and live process metrics — call
`get_sync_status(project_id)` at the same ~2-minute cadence. Do NOT busy-poll
(every few seconds); syncs index many items and take minutes.

PREREQUISITE: `search_jama_semantics` needs the project initialized first
(`init_jama_project` → poll `get_sync_progress` until DONE, roughly every 2
minutes). The native and browse tools work immediately against the live Jama
API — no init required. If a project is not initialized, suggest
`init_jama_project` before falling back to native metadata.
"""

mcp = FastMCP("jama-mcp", instructions=JAMA_MCP_INSTRUCTIONS)


def db() -> "sqlite3.Connection":  # type: ignore[name-defined]
    global _db_conn
    if _db_conn is None:
        with _init_lock:
            if _db_conn is None:
                _db_conn = init_db()
    return _db_conn


def jama() -> JamaClient:
    global _jama
    if _jama is None:
        with _init_lock:
            if _jama is None:
                _jama = JamaClient()
    return _jama


def rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        with _init_lock:
            if _rag is None:
                _rag = RAGPipeline()
    return _rag


def reset_singletons() -> None:
    """Drop cached Jama/RAG/DB singletons so they rebuild with fresh config.

    Called after ``configure_jama`` rewrites ``.env`` and reloads settings:
    the old singletons were built from the previous config, so the next tool
    call must reconstruct them against the new values.
    """
    global _db_conn, _jama, _rag
    with _init_lock:
        _db_conn = None
        _jama = None
        _rag = None


# --------------------------------------------------------------------------- #
# Pre-flight guard: every tool gates on this before doing real work.
# --------------------------------------------------------------------------- #
def _ensure_ready(require: set[str]) -> dict | None:
    """Return an error dict if the server isn't configured for ``require``.

    ``require`` is a subset of ``{"jama","embedding","llm"}`` describing which
    backend features the calling tool needs. The check is offline and fast
    (packages + config + storage), so it adds negligible latency per call.
    Returns ``None`` when ready, letting the tool proceed.
    """
    report = preflight(require=require)
    if report["blocking"]:
        return {
            "error": "Server is not ready: configuration or dependencies are "
                     "incomplete.",
            "issues": report["issues"],
            "hint": report["hint"] or "Call validate_setup for a full report, "
                    "or configure_jama with the missing values.",
        }
    return None


# --------------------------------------------------------------------------- #
# Core sync logic (shared by init tool + scheduler)
# --------------------------------------------------------------------------- #
def _sync_project(project_id: int, *, job_id: str | None,
                  incremental: bool | str) -> None:
    """Download, clean, chunk, embed and index a project's items.

    Serializes per-project: a user init and a scheduler incremental sync for
    the SAME project can't run at once (they'd race on upserts). Different
    projects still sync in parallel.

    ``incremental``:
      * ``False``      — full init / reinit: fetch + embed every item.
      * ``True``       — incremental sync: only items modified since
                         ``last_sync_time``.
      * ``"resume"``   — resume an interrupted sync: fetch all items but SKIP
                         those that already have chunks (indexed before the
                         interruption). Saves re-embedding ~5000 items when
                         only a few hundred remain.
    """
    with _project_lock(project_id):
        _sync_project_locked(project_id, job_id=job_id, incremental=incremental)


def _sync_project_locked(project_id: int, *, job_id: str | None,
                         incremental: bool | str) -> None:
    """Download, clean, chunk, embed and index a project's items.

    ``incremental`` modes:
      * ``False``      — full init: fetch and index every item.
      * ``True``       — incremental: only items modified after last_sync_time.
      * ``"resume"``   — resume an interrupted sync: fetch all items but SKIP
                         those that already have chunks (indexed before the
                         interruption). Saves re-embedding ~5000 items when
                         only a few hundred remain.
    """
    conn = db()
    last_sync = None
    if incremental is True:
        proj = get_project(conn, project_id)
        last_sync = proj["last_sync_time"] if proj else None
    # Resume mode: build the skip-set of already-indexed item_ids (items that
    # have chunks). These are skipped during fetch — no re-download, no
    # re-embed. The `done` counter starts at this count so progress reflects
    # true completion, not just this run's work.
    skip_ids: set[int] = set()
    if incremental == "resume":
        skip_ids = indexed_item_ids(conn, project_id)
        if skip_ids:
            log.info("Resume sync for project %s: skipping %d already-indexed "
                     "items.", project_id, len(skip_ids))
            if job_id:
                update_job(conn, job_id, done=len(skip_ids),
                           message=f"Resuming: {len(skip_ids)} items already "
                                   f"indexed, fetching the rest")
    upsert_project(conn, project_id, status="INITIALIZING")

    # ---- Step 1: pre-download ML models BEFORE any indexing work. ----
    # Both the embedding model (~130MB ONNX) and the reranker (~80MB MiniLM)
    # are fetched first so that (a) a download failure surfaces immediately
    # with a clear message instead of mid-sync, and (b) the first search
    # after sync doesn't stall on a model download. Non-fatal: failures just
    # defer to lazy load (embedding) or RRF fallback (reranker) — sync still
    # proceeds.
    if job_id:
        update_job(conn, job_id, status="RUNNING", progress=0.0,
                   message="Downloading embedding + reranker models")
    pipeline = rag()
    try:
        pipeline.embedder.ensure_downloaded()
    except Exception as exc:
        log.warning("Embedding model pre-download skipped: %s", exc)
    try:
        pipeline.reranker.ensure_downloaded()
    except Exception as exc:
        log.warning("Reranker pre-download skipped: %s", exc)

    # ---- Step 2: Jama network pre-flight (with transient-failure retry). ----
    # The speed test measures live throughput; a momentary network blip can
    # drop it below the floor even though the link is fine a second later.
    # Retrying a couple of times with backoff avoids a flaky pre-flight marking
    # the whole project ERROR (which then needs a re-init to recover). Only a
    # SUSTAINED failure (all retries exhausted) is treated as a real error.
    client = jama()
    if job_id:
        update_job(conn, job_id, message="Pre-flight network speed test")
    preflight_err: Exception | None = None
    for attempt in range(1, _PREFLIGHT_RETRIES + 1):
        try:
            client.preflight_speed_check()
            preflight_err = None
            break
        except Exception as exc:
            preflight_err = exc
            if attempt < _PREFLIGHT_RETRIES:
                log.warning("Pre-flight attempt %d/%d failed (%s); retrying in %ds",
                            attempt, _PREFLIGHT_RETRIES, exc, _PREFLIGHT_BACKOFF)
                if job_id:
                    update_job(conn, job_id,
                               message=f"Pre-flight retry {attempt}/{_PREFLIGHT_RETRIES}")
                time.sleep(_PREFLIGHT_BACKOFF)
    if preflight_err is not None:
        msg = (f"Network pre-flight check failed after {_PREFLIGHT_RETRIES} "
               f"attempts: {preflight_err}")
        log.error(msg)
        if job_id:
            update_job(conn, job_id, status="ERROR", message=msg)
        upsert_project(conn, project_id, status="ERROR", error=msg)
        return

    # ---- Step 3: fetch + index items. ----
    if job_id:
        update_job(conn, job_id, message="Fetching items from Jama")

    project = client.get_project(project_id)
    proj_name = (project or {}).get("fields", {}).get("name") if project else None
    if project is None:
        msg = f"Project {project_id} not found in Jama"
        if job_id:
            update_job(conn, job_id, status="ERROR", message=msg)
        upsert_project(conn, project_id, status="ERROR", error=msg)
        return

    # Pipelined fetch + index: download and embed run in PARALLEL on two
    # threads, communicating through a bounded queue. The old interleaved
    # design fetched a wave, then STOPPED fetching while it embedded (8s/batch
    # of dead download time). Pipelining lets the downloader keep filling the
    # queue while the embedder consumes it, so overall throughput rises from
    # "serial alternation" (~1 item/s measured) to "max(download, embed)"
    # (~5 item/s, embedding-bound).
    #
    # CRASH SAFETY: `done`/`progress` and the per-item DB writes only ever
    # advance in the EMBED thread, AFTER the SQLite commit succeeds. So a crash
    # at any point leaves the DB consistent up to the last committed batch; the
    # project stays INITIALIZING and `_resume_interrupted_syncs` re-queues a
    # full resync (upsert is idempotent, so already-indexed items are just
    # overwritten — no duplicates, no lost data). Items sitting in the queue at
    # crash time were never written, so they're simply re-fetched on resume.
    import queue
    # max_items caps how many items a single run processes. Only incremental
    # sync (True) uses it as a safety valve; full init (False) and resume
    # ("resume") process every matching item with no cap.
    max_items = settings.sync.max_items_per_run if incremental is True else None
    dl_concurrency = settings.sync.download_concurrency
    batch_size = settings.embedding.batch_size
    total_holder: list[int] = [0]  # set by on_total callback
    # Bounded queue: caps memory at ~2 batches of items in flight. Small bound
    # so a crash never loses much unwritten work, and so the downloader
    # backpressures naturally when the embedder falls behind.
    item_queue: "queue.Queue[tuple[dict, list[dict]] | None]" = queue.Queue(maxsize=batch_size * 4)
    fetch_error: list[BaseException] = []

    def _on_total(n: int) -> None:
        total_holder[0] = n
        if job_id:
            update_job(conn, job_id, total=n, done=0,
                       message=f"Indexing {n} top-level items "
                               f"(Test Runs/Folders/Attachments excluded)")

    def _fetcher() -> None:
        """Download thread: pull items concurrently, chunk them, enqueue.

        In resume mode, items whose item_id is in ``skip_ids`` (already have
        chunks) are skipped entirely — no re-download, no re-embed, no enqueue.
        This is what makes resume fast: 4850/5079 indexed → only 229 fetched.
        """
        try:
            for item in client.iter_project_items(
                    project_id, modified_after=last_sync,
                    max_items=max_items, concurrency=dl_concurrency,
                    on_total=_on_total):
                # Resume: skip items that already have chunks (indexed before
                # the interruption). The item metadata is already in the items
                # table (upserted in the prior run); only items WITHOUT chunks
                # need re-processing.
                if skip_ids and item.get("item_id") in skip_ids:
                    continue
                try:
                    chunks = chunk_item(item)
                except Exception as exc:
                    log.warning("Failed to chunk item %s: %s",
                                item.get("item_id"), exc)
                    chunks = []
                # Items with no text still need metadata persisted + stale
                # chunks cleared; enqueue them so the embed thread handles
                # them uniformly (empty chunk list => upsert-only).
                item_queue.put((item, chunks))
        except BaseException as exc:
            fetch_error.append(exc)
        finally:
            item_queue.put(None)  # sentinel: no more items

    # In resume mode, `done` starts at the count of already-indexed items so
    # progress (done/total) reflects true completion, not just this run's work.
    done = len(skip_ids)
    chunk_total = 0

    def _embed_and_store(batch: list[tuple[dict, list[dict]]]) -> None:
        """Embed one batch of items and commit to SQLite (embed thread)."""
        nonlocal done, chunk_total
        # Split into text-bearing items (need embedding) and empty ones.
        to_embed: list[tuple[dict, list[dict]]] = []
        for item, chunks in batch:
            if chunks:
                to_embed.append((item, chunks))
            else:
                with write_txn(conn):
                    upsert_item(conn, item)
                replace_chunks(conn, item["item_id"], [], [])
                done += 1
        if not to_embed:
            return
        flat: list[dict] = []
        owners: list[tuple[dict, list[dict]]] = []
        for item, chunks in to_embed:
            owners.append((item, chunks))
            flat.extend(chunks)
        # embed_many packs flat into batch_size HTTP requests across items,
        # fired concurrently (EMBEDDING_CONCURRENCY).
        embeddings = pipeline.embed_many(flat)
        idx = 0
        for item, chunks in owners:
            n = len(chunks)
            item_embs = embeddings[idx:idx + n]
            idx += n
            try:
                with write_txn(conn):
                    upsert_item(conn, item)
                replace_chunks(conn, item["item_id"], chunks, item_embs)
                chunk_total += n
            except Exception as exc:
                log.warning("Failed to index item %s: %s",
                            item.get("item_id"), exc)
            done += 1

    # Start the download thread.
    fetch_thread = threading.Thread(target=_fetcher, daemon=True,
                                    name="jama-fetch")
    fetch_thread.start()

    # Embed thread (main thread): consume the queue, accumulate into
    # batch_size chunk pools, flush when full. This is the only place `done`
    # and the DB advance, preserving the crash-safety invariant.
    pending: list[tuple[dict, list[dict]]] = []
    pending_chunk_count = 0
    total = total_holder[0]
    while True:
        entry = item_queue.get()
        if entry is None:
            break  # sentinel from fetcher
        item, chunks = entry
        pending.append((item, chunks))
        pending_chunk_count += len(chunks)
        total = total_holder[0] or total
        if pending_chunk_count >= batch_size:
            _embed_and_store(pending)
            pending.clear()
            pending_chunk_count = 0
            if job_id:
                pct = round(done / total, 4) if total else 0.0
                update_job(conn, job_id, done=done, progress=pct,
                           message=f"Indexed {done}/{total or '?'} items")

    # Flush any trailing partial batch.
    if pending:
        _embed_and_store(pending)
        pending.clear()
    total = total_holder[0] or total
    fetch_thread.join(timeout=5)

    # Surface any error from the fetcher thread (e.g. Jama auth failure mid-sync).
    if fetch_error:
        exc = fetch_error[0]
        msg = f"Fetch failed: {exc}"
        log.error("Sync fetch thread failed for project %s: %s", project_id, exc)
        if job_id:
            update_job(conn, job_id, status="ERROR", message=msg)
        upsert_project(conn, project_id, name=proj_name, status="ERROR",
                       error=str(exc)[:500])
        raise exc

    if done == 0:
        if job_id:
            update_job(conn, job_id, status="DONE", progress=1.0,
                       done=0, message="No new/modified items")
        # Report the true persisted totals (an incremental run that found 0
        # changes must NOT clobber item_count with the per-run ``done``=0).
        upsert_project(conn, project_id, name=proj_name, status="READY",
                       last_sync_time=utcnow_iso(),
                       item_count=count_items(conn, project_id),
                       chunk_count=count_chunks(conn, project_id))
        return

    final_chunk_count = count_chunks(conn, project_id)
    # ``done`` is the count processed THIS run. For an incremental sync that's
    # only the modified items, not the project total — read the true total from
    # the items table so get_sync_status stays accurate after both run kinds.
    upsert_project(conn, project_id, name=proj_name, status="READY",
                   last_sync_time=utcnow_iso(),
                   item_count=count_items(conn, project_id),
                   chunk_count=final_chunk_count)
    if job_id:
        update_job(conn, job_id, status="DONE", progress=1.0, done=done,
                   message=f"Done: {done} items, {chunk_total} chunks "
                           f"indexed this run")
    log.info("Sync complete for project %s: %s items, %s chunks",
             project_id, done, chunk_total)


def _run_job(project_id: int, job_id: str, incremental: bool | str) -> None:
    """Background worker: runs sync then guarantees terminal job state.

    The error path is defensive: if marking the job/project ERROR itself fails
    (e.g. DB locked past busy_timeout), that secondary failure is logged rather
    than swallowed, so the failure is never silent and the job never lingers in
    a phantom RUNNING state without a breadcrumb.
    """
    conn = db()
    try:
        _sync_project(project_id, job_id=job_id, incremental=incremental)
    except Exception as exc:
        log.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        try:
            update_job(conn, job_id, status="ERROR", message=str(exc)[:500])
            upsert_project(conn, project_id, status="ERROR",
                           error=str(exc)[:500])
        except Exception as db_exc:
            # The terminal-state write itself failed. Log loudly so it isn't
            # lost; the job row stays RUNNING, but _resume_interrupted_syncs
            # will re-queue the (still-INITIALIZING) project on next startup.
            log.error("Job %s: could not persist ERROR state (%s). Project "
                      "%s may need manual recovery.", job_id, db_exc,
                      project_id)


def _run_bootstrap_job(job_id: str) -> None:
    """Background worker: pre-download embedding + reranker models.

    Reports live progress into the ``sync_jobs`` row (kind="bootstrap") so
    ``get_bootstrap_progress`` can be polled every ~2 min. The reranker
    (~80MB cross-encoder, ONNX via fastembed) and the embedding model
    (~130MB ONNX) both download through fastembed, which gives no per-chunk
    byte callback — so progress is reported as phase transitions
    (downloading -> ready), not live bytes. Either model already cached skips
    its download. Failures are non-fatal per-model (marked in the message) but
    the job still ends ERROR if any model is missing at the end.
    """
    conn = db()

    def _set(progress: float, message: str) -> None:
        try:
            update_job(conn, job_id, progress=progress, message=message)
        except Exception:
            pass

    errors: list[str] = []
    pipeline = rag()
    provider = settings.embedding.provider

    # --- Phase 1: embedding model (local only; azure needs no download) ------
    if provider == "local":
        _set(0.1, f"Checking/downloading embedding model "
                  f"({settings.embedding.local_model}, ~130MB)")
        emb = pipeline.embedder
        try:
            if emb._model_present():  # type: ignore[attr-defined]
                _set(0.2, "Embedding model already cached")
            else:
                emb._download_model()  # type: ignore[attr-defined]
                _set(0.45, "Embedding model downloaded")
        except Exception as exc:
            msg = f"Embedding model download failed: {exc}"
            log.error("Bootstrap %s: %s", job_id, msg)
            errors.append(msg)
    else:
        _set(0.45, f"Embedding provider is {provider} (no model to download)")

    # --- Phase 2: reranker model (ONNX via fastembed, no byte-level progress) --
    # fastembed (used by ensure_downloaded) gives no per-chunk callback, so we
    # can't show live bytes — only phase transitions. The progress band
    # 0.5..0.95 is held at 0.5 while downloading, then jumps to 0.95 on success.
    _set(0.5, f"Downloading reranker model ({settings.reranker.model_name}, ~80MB ONNX)")

    def _reranker_cb(received: int, expected: int | None) -> None:
        # Called once at completion by _ensure_weights_downloaded (snapshot_download
        # has no byte callback). Advance to 0.95 so the monitor sees movement.
        _set(0.95, "Reranker model downloaded")

    try:
        rr = pipeline.reranker
        if rr._model is not None or rr._load_error is not None:  # type: ignore[attr-defined]
            _set(0.95, "Reranker already available/failed-skip")
        else:
            rr.ensure_downloaded(progress_callback=_reranker_cb)
            _set(0.95, "Reranker model ready")
    except Exception as exc:
        msg = f"Reranker model download failed: {exc}"
        log.error("Bootstrap %s: %s", job_id, msg)
        errors.append(msg)

    # --- Terminal state ------------------------------------------------------
    if errors:
        update_job(conn, job_id, status="ERROR", progress=0.95,
                   message="Bootstrap failed: " + "; ".join(errors)[:400])
    else:
        update_job(conn, job_id, status="DONE", progress=1.0,
                   message="Models ready: embedding + reranker cached locally")
    log.info("Bootstrap job %s complete (errors=%d)", job_id, len(errors))


# --------------------------------------------------------------------------- #
# MCP Tools
# --------------------------------------------------------------------------- #
def sync_project_blocking(project_id: int, *, incremental: bool | str = False,
                          poll_interval: float = 60.0) -> dict:
    """Run a project sync synchronously (blocking) and return the final state.

    Unlike ``init_jama_project`` (which submits to ``_executor`` and returns
    immediately), this runs ``_run_job`` on the **calling thread** — so the
    process stays alive until the sync finishes. This is the correct entry
    point for standalone scripts (``python init_lyra.py``): a ``python -c``
    one-shot that calls ``init_jama_project`` exits immediately, triggering
    ``ThreadPoolExecutor`` shutdown which kills the background worker mid-sync.
    Here the main thread is blocked inside ``_run_job``, so no shutdown fires.

    A daemon progress-polling thread prints ``done/total (pct%)`` every
    ``poll_interval`` seconds so the user sees live progress.

    Args:
        project_id: Jama project id (int).
        incremental: False (full init), True (incremental), or "resume"
                     (skip already-indexed items — use after an interruption).
        poll_interval: seconds between progress prints (default 60).

    Returns:
        {"job_id", "status", "done", "total", "item_count", "chunk_count"}
    """
    not_ready = _ensure_ready({"jama", "embedding"})
    if not_ready:
        return not_ready
    conn = db()
    kind = "resume" if incremental == "resume" else "init"
    job_id = f"{kind}-{uuid.uuid4().hex[:12]}"
    with _init_lock:
        create_job(conn, job_id, project_id, kind)
        upsert_project(conn, project_id, status="INITIALIZING")

    # Progress poller: daemon thread printing job state every poll_interval.
    import threading as _t
    stop = _t.Event()

    def _poll():
        while not stop.wait(poll_interval):
            try:
                row = get_job(conn, job_id)
            except Exception:
                continue
            if row is None:
                continue
            tot = row["total"] or "?"
            pct = round(row["progress"] * 100, 1)
            log.info("[poll] %s  %s/%s  (%s%%)  %s", row["status"],
                     row["done"], tot, pct, row["message"] or "")

    poller = _t.Thread(target=_poll, daemon=True, name="sync-poll")
    poller.start()
    try:
        _run_job(project_id, job_id, incremental)
    finally:
        stop.set()
        poller.join(timeout=2)

    row = get_job(conn, job_id)
    proj = get_project(conn, project_id)
    return {
        "job_id": job_id,
        "status": row["status"] if row else "UNKNOWN",
        "done": row["done"] if row else 0,
        "total": row["total"] if row else 0,
        "item_count": proj["item_count"] if proj else 0,
        "chunk_count": proj["chunk_count"] if proj else 0,
    }


def _start_sync_job(pid: int, kind: str) -> dict:
    """Create a full (non-incremental) sync job for ``pid`` and submit it.

    Shared by ``init_jama_project`` (kind="init") and ``reinit_jama_project``
    (kind="reinit"). Both re-fetch and re-index every item from scratch
    (incremental=False). Enforces the reentrancy guard: if a job is already
    RUNNING/PENDING for a project that is still INITIALIZING, hand back its
    job_id instead of spawning a racing second worker (two concurrent syncs of
    the same project race on upserts and can leave a split-brain terminal state
    — one ERROR, one READY). A project already in a terminal state (READY/ERROR)
    has a stale "zombie" job row, so a new sync is always allowed.

    The check + create run under ``_init_lock``. (We can't wrap them in one
    write_txn because create_job/upsert_project each open their own transaction;
    SQLite forbids BEGIN within BEGIN. The lock is sufficient because there is a
    single shared db() connection and a single writer at a time.)

    Returns the immediate RUNNING response dict (caller returns it as-is), or an
    ``{"error": ...}`` dict if the job could not be submitted.
    """
    conn = db()
    with _init_lock:
        proj = get_project(conn, pid)
        proj_status = proj["status"] if proj else None
        active = get_active_job_for_project(conn, pid)
        if active is not None and proj_status == "INITIALIZING":
            return {"job_id": active["job_id"], "project_id": pid,
                    "status": "RUNNING",
                    "note": f"A sync is already in progress for project "
                            f"{pid}; reuse job_id {active['job_id']}"}
        job_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        create_job(conn, job_id, pid, kind)
        upsert_project(conn, pid, status="INITIALIZING")
    # Submit outside the txn (the worker opens its own conn). If submission
    # itself fails (executor shut down), roll back the job so the project
    # isn't left stuck INITIALIZING with a phantom RUNNING job.
    try:
        _executor.submit(_run_job, pid, job_id, incremental=False)
    except Exception as exc:
        log.error("Could not submit %s job %s: %s", kind, job_id, exc)
        try:
            update_job(conn, job_id, status="ERROR",
                       message=f"Submit failed: {exc}")
            upsert_project(conn, pid, status="ERROR",
                           error=f"Submit failed: {exc}")
        except Exception:
            pass
        return {"error": f"Could not start sync job: {exc}"}
    log.info("Started %s job %s for project %s", kind, job_id, pid)
    return {"job_id": job_id, "project_id": pid, "status": "RUNNING"}


@mcp.tool()
def init_jama_project(project_id: str) -> dict:
    """Initialize a Jama project: download, clean, vectorize and index its items.

    Runs as an async background task and returns a job_id immediately so the
    caller (LLM) is never blocked. Poll progress with get_sync_progress roughly
    every 2 minutes until status is DONE or ERROR, reporting each sample to the
    user. To re-index a project that is already initialized, prefer
    reinit_jama_project.

    Args:
        project_id: Jama project id (numeric string, e.g. "20571").

    Returns:
        {"job_id": "...", "project_id": ..., "status": "RUNNING"}
    """
    not_ready = _ensure_ready({"jama", "embedding"})
    if not_ready:
        return not_ready
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    return _start_sync_job(pid, "init")


@mcp.tool()
def reinit_jama_project(project_id: str) -> dict:
    """Re-initialize an already-initialized Jama project (full re-sync).

    Behaves like init_jama_project but is the explicit verb for re-fetching and
    re-indexing a project that has already reached READY/ERROR — e.g. after a
    config change, corrupted index, or to pull a fresh full copy. Runs as an
    async background task and returns a job_id immediately. Poll progress with
    get_sync_progress roughly every 2 minutes until status is DONE or ERROR,
    reporting each sample to the user.

    Args:
        project_id: Jama project id (numeric string, e.g. "20571").

    Returns:
        {"job_id": "...", "project_id": ..., "status": "RUNNING"}
    """
    not_ready = _ensure_ready({"jama", "embedding"})
    if not_ready:
        return not_ready
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    return _start_sync_job(pid, "reinit")


# Bootstrap job id guard — only one model pre-download runs at a time.
_bootstrap_lock = threading.Lock()
_bootstrap_job_id: str | None = None


@mcp.tool()
def bootstrap_models() -> dict:
    """Pre-download the embedding + reranker models so syncs never wait on them.

    Downloads the local embedding model (bge-small-en-v1.5, ~130MB ONNX) and the
    cross-encoder reranker (ms-marco-MiniLM-L-6-v2, ~80MB ONNX) into the
    project-local cache, ASYNCHRONOUSLY. Both run on onnxruntime via fastembed
    — no torch/transformers dependency. Returns a job_id immediately. This is
    the recommended first step after installing/configuring the server — call
    it BEFORE init_jama_project so the first sync isn't slowed by model
    downloads. Models already cached are skipped. Poll progress with
    get_bootstrap_progress roughly every 2 minutes, reporting each sample to
    the user, until status is DONE or ERROR.

    Returns:
        {"job_id": "...", "status": "RUNNING"} or, if a bootstrap is already
        running, {"job_id": "...", "status": "RUNNING", "note": "..."}.
    """
    not_ready = _ensure_ready(set())  # only needs storage + network, not Jama
    if not_ready:
        return not_ready
    global _bootstrap_job_id
    conn = db()
    with _bootstrap_lock:
        # Reentrancy: if a bootstrap job is still RUNNING, hand back its id
        # instead of spawning a second download (two concurrent downloads of
        # the same weights race on the cache files).
        if _bootstrap_job_id is not None:
            existing = get_job(conn, _bootstrap_job_id)
            if existing and existing["status"] in ("PENDING", "RUNNING"):
                return {"job_id": _bootstrap_job_id, "status": "RUNNING",
                        "note": "A model bootstrap is already in progress; "
                                f"reuse job_id {_bootstrap_job_id}"}
        job_id = f"bootstrap-{uuid.uuid4().hex[:12]}"
        create_job(conn, job_id, 0, "bootstrap")  # project_id=0 (no project)
        _bootstrap_job_id = job_id
    try:
        _executor.submit(_run_bootstrap_job, job_id)
    except Exception as exc:
        log.error("Could not submit bootstrap job %s: %s", job_id, exc)
        try:
            update_job(conn, job_id, status="ERROR",
                       message=f"Submit failed: {exc}")
        except Exception:
            pass
        with _bootstrap_lock:
            _bootstrap_job_id = None
        return {"error": f"Could not start bootstrap job: {exc}"}
    log.info("Started bootstrap job %s", job_id)
    return {"job_id": job_id, "status": "RUNNING"}


@mcp.tool()
def get_bootstrap_progress(job_id: str) -> dict:
    """Poll the progress of a bootstrap_models job.

    After calling bootstrap_models, poll this roughly every 2 minutes, reporting
    each sample (status, progress %, message) to the user, until status is DONE
    or ERROR. Progress is phase-based, not live bytes: both the embedding
    (~130MB ONNX, via fastembed) and the reranker (~80MB ONNX, via fastembed)
    lack per-chunk byte callbacks, so `message` reports phase transitions (e.g.
    "Downloading reranker model (...)" -> "Reranker model ready") rather than
    byte counts.

    Returns:
        {"job_id","project_id","kind","status","progress","total","done",
         "message","started_at","finished_at"} (project_id is 0 for a
        bootstrap job — it has no project). status is one of
        PENDING | RUNNING | DONE | ERROR.
    """
    not_ready = _ensure_ready(set())
    if not_ready:
        return not_ready
    conn = db()
    row = get_job(conn, job_id)
    if row is None:
        return {"error": f"Unknown job_id: {job_id}"}
    return _job_summary(row)


@mcp.tool()
def get_sync_progress(job_id: str) -> dict:
    """Poll the progress of an init, reinit or sync job.

    After calling init_jama_project or reinit_jama_project, poll this roughly
    every 2 minutes until status is DONE or ERROR, reporting each sample to the
    user. For a project-wide view of all operations and their last runs, use
    get_sync_status instead.

    Returns:
        {"job_id","project_id","kind","status","progress","total","done",
         "message","started_at","finished_at"}
        status is one of PENDING | RUNNING | DONE | ERROR.
    """
    not_ready = _ensure_ready(set())
    if not_ready:
        return not_ready
    conn = db()
    row = get_job(conn, job_id)
    if row is None:
        return {"error": f"Unknown job_id: {job_id}"}
    return _job_summary(row)


def _process_metrics(pid: int, conn) -> dict | None:
    """Lightweight live process + index metrics for the monitor.

    Mirrors the sampling in ``monitor_lyra_init.py`` but reuses the shared
    ``db()`` connection and scopes the chunk count to the requested project.
    Returns ``None`` if psutil is unavailable or sampling fails, so the monitor
    degrades gracefully instead of erroring the whole tool. cpu_pct is omitted
    (it needs a baseline interval that would block the tool call).
    """
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        return {
            "rss_mb": round(mem.rss / 1024 / 1024, 1),
            "threads": proc.num_threads(),
            "db_mb": round(os.path.getsize(settings.storage.db_path)
                           / 1024 / 1024, 1),
            "chunks": count_chunks(conn, pid),
        }
    except Exception:
        return None


def _job_summary(row) -> dict | None:
    """Compact job dict for the monitor (None when there's no row).

    Shared by get_sync_progress / get_bootstrap_progress / get_sync_status so
    the job field shape stays consistent across all monitor tools.
    """
    if row is None:
        return None
    return {
        "job_id": row["job_id"],
        "project_id": row["project_id"],
        "kind": row["kind"],
        "status": row["status"],
        "progress": round(row["progress"] * 100, 1),  # percent
        "total": row["total"],
        "done": row["done"],
        "message": row["message"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


@mcp.tool()
def get_sync_status(project_id: str) -> dict:
    """Monitor a project's sync operations: current run + last run of each kind.

    Use this to check on init_jama_project / reinit_jama_project / scheduled
    sync for one project. It returns the in-flight job (if any), the most recent
    init / reinit / sync job (terminal or running), the project's current state,
    and lightweight process metrics. After starting an init or reinit, you may
    poll this roughly every 2 minutes (reporting each sample to the user) until
    active_job is null and the relevant recent.* entry is DONE/ERROR.

    Args:
        project_id: Jama project id (numeric string, e.g. "20571").

    Returns:
        {"project_id","project_status","last_sync_time","item_count",
         "chunk_count","active_job": {...}|null,
         "recent": {"init": {...}|null, "reinit": {...}|null,
                    "sync": {...}|null},
         "process": {"rss_mb","threads","db_mb","chunks"}|null}
        Returns {"error": ...} if the project_id is not numeric or the server
        is not ready.
    """
    not_ready = _ensure_ready(set())  # always allowed, like get_sync_progress
    if not_ready:
        return not_ready
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    conn = db()
    proj = get_project(conn, pid)
    proj_status = proj["status"] if proj else "NEW"
    return {
        "project_id": pid,
        "project_status": proj_status,
        "last_sync_time": proj["last_sync_time"] if proj else None,
        "item_count": proj["item_count"] if proj else 0,
        "chunk_count": proj["chunk_count"] if proj else 0,
        "active_job": _job_summary(get_active_job_for_project(conn, pid)),
        "recent": {
            "init": _job_summary(get_latest_job_for_project(conn, pid, "init")),
            "reinit": _job_summary(
                get_latest_job_for_project(conn, pid, "reinit")),
            "sync": _job_summary(get_latest_job_for_project(conn, pid, "sync")),
        },
        "process": _process_metrics(pid, conn),
    }


@mcp.tool()
def search_jama_semantics(project_id: str, query: str,
                          sub_queries: list[str] = None,
                          item_type: str = None, top_k: int = 5,
                          candidate_k: int = 100,
                          modified_after: str = None,
                          modified_before: str = None) -> dict:
    """Semantic search over an initialized Jama project using high-precision RAG.

    This is the DEFAULT tool for any non-precise question. It fuses keyword
    (FTS5/BM25), vector (sqlite-vec cosine) and RRF in one call, then reranks
    with a local cross-encoder model (ONNX via fastembed/onnxruntime) — so "like",
    "keyword" and "semantic" queries are all best answered here. Prefer it over
    native metadata unless the user gives an exact document key / status / item id.

    Pipeline: client Multi-Query expansion -> hybrid recall (sqlite-vec +
    FTS5) -> RRF fusion -> local cross-encoder reranker -> top_k results.

    Args:
        project_id: numeric string Jama project id (must be initialized first).
        query: the ORIGINAL natural-language search query, verbatim. It is
               always the rerank reference, so even when `sub_queries` is
               supplied you MUST pass the original user query here too.
        sub_queries: RECOMMENDED. Rewrite `query` into 3-5 diverse search
                     sub-queries capturing different semantic angles
                     (synonyms, broader/narrower scope, related concepts) to
                     maximize recall for RRF fusion. Pass as a JSON array of
                     strings. The server normalizes them (forces `query` to the
                     front, de-duplicates, caps at 5). If omitted, the server
                     falls back to deterministic lexical variants.
                     Example for query "how does login timeout work":
                       ["login session expiration",
                        "authentication timeout policy",
                        "user inactivity logout"]
        item_type: optional Jama item-type id to filter (e.g. "89011" for Test
                   Cases, "89009" for Requirements). Pass None for all.
        top_k: final results to return (default 5).
        candidate_k: candidate pool size before reranking (default 100). A
                     larger pool improves recall (vector+FTS recall is capped
                     by this): measured vecR@25=7%, @50=13%, @100=21%, @200=34%.
                     The MiniLM reranker scores 100 candidates in ~1.5s on CPU.
                     Range 1-500; must be >= top_k.
        modified_after: optional ISO-8601 lower bound on item modified date
                        (inclusive). Naive timestamps are assumed UTC.
                        e.g. "2024-01-01" or "2024-06-01T00:00:00Z".
        modified_before: optional ISO-8601 upper bound on item modified date
                         (inclusive). Naive timestamps are assumed UTC.

    Returns:
        {"project_id","query","sub_queries_used","results":
        [{document_key,name,item_type_name,section,modified_date,text,
        score,strategy}, ...]}
    """
    not_ready = _ensure_ready({"jama", "embedding"})
    if not_ready:
        return not_ready
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    if not query or not query.strip():
        return {"error": "query is required"}

    # Normalize caller-supplied sub-queries defensively. A non-list value is a
    # client bug (the schema declares array of string); report it clearly. We
    # filter out non-string / blank entries and drop to None when nothing
    # usable remains, so the pipeline falls back to lexical expansion.
    if sub_queries is not None:
        if not isinstance(sub_queries, list):
            return {"error": "sub_queries must be an array of strings"}
        cleaned = [s.strip() for s in sub_queries
                   if isinstance(s, str) and s.strip()]
        sub_queries = cleaned or None

    conn = db()
    proj = get_project(conn, pid)
    if proj is None or proj["status"] not in ("READY", "INITIALIZING"):
        return {"error": f"Project {pid} is not initialized. "
                         f"Call init_jama_project first."}
    if count_chunks(conn, pid) == 0:
        return {"error": f"Project {pid} has no indexed chunks yet. "
                         f"Wait for init to finish or re-run it."}

    it = None
    if item_type is not None and str(item_type).strip():
        try:
            it = int(item_type)
        except (TypeError, ValueError):
            return {"error": "item_type must be a numeric string or null"}

    # Clamp result/pool sizes to sane bounds. candidate_k=0 would yield an
    # empty pool (silently zero results); top_k > candidate_k returns fewer
    # than requested; huge values cause excessive vector/FTS work.
    try:
        top_k = int(top_k)
        candidate_k = int(candidate_k)
    except (TypeError, ValueError):
        return {"error": "top_k and candidate_k must be integers"}
    if top_k < 1 or top_k > 50:
        return {"error": "top_k must be between 1 and 50"}
    if candidate_k < 1 or candidate_k > 500:
        return {"error": "candidate_k must be between 1 and 500"}
    if top_k > candidate_k:
        return {"error": f"top_k ({top_k}) cannot exceed candidate_k "
                         f"({candidate_k})"}

    # Validate (and UTC-normalize) the time bounds early so a bad format is
    # reported as a clean error rather than a 500 mid-search.
    try:
        from db_setup import _normalize_iso_utc
        if modified_after and modified_after.strip():
            _normalize_iso_utc(modified_after)
        else:
            modified_after = None
        if modified_before and modified_before.strip():
            _normalize_iso_utc(modified_before)
        else:
            modified_before = None
    except ValueError as exc:
        return {"error": f"Invalid timestamp: {exc}"}

    try:
        pipeline = rag()
        results = pipeline.search(pid, query.strip(),
                                  sub_queries=sub_queries,
                                  item_type=it,
                                  top_k=top_k, candidate_k=candidate_k,
                                  modified_after=modified_after,
                                  modified_before=modified_before)
        warnings = list(pipeline.last_warnings)
    except Exception as exc:
        log.error("search failed: %s\n%s", exc, traceback.format_exc())
        return {"error": f"Search failed: {exc}"}
    # Echo the query variants actually used by the pipeline so the caller can
    # verify expansion happened. Read from the pipeline (which records both
    # caller-supplied and lexical-fallback variants) so the lexical-fallback
    # path reports the real variants used, not just [query]. Computed
    # defensively so a reflection failure never discards search results.
    used = list(getattr(pipeline, "last_sub_queries", []) or [])
    if not used:
        used = [query.strip()] if query.strip() else []
    resp = {"project_id": pid, "query": query,
            "sub_queries_used": used,
            "count": len(results),
            "modified_after": modified_after,
            "modified_before": modified_before,
            "results": results}
    if warnings:
        # Surface silent degradations (e.g. reranker load failure -> RRF
        # fallback) so the LLM can tell the user precision may be lower.
        resp["warnings"] = warnings
    return resp


@mcp.tool()
def query_jama_native_metadata(project_id: str, document_key: str = None,
                               item_type: str = None, status: str = None,
                               keyword: str = None) -> dict:
    """Query Jama's native REST API directly for exact metadata filtering.

    Use ONLY for precise lookups (exact document key, exact status, exact
    item_type) — it returns empty on any misspelling. For topical, fuzzy or
    natural-language questions, prefer `search_jama_semantics` instead.

    Bypasses the vector store to answer precise questions (exact document key,
    specific status, specific item type). Handles pagination internally and
    returns up to 20 core metadata records.

    Args:
        project_id: numeric string Jama project id.
        document_key: exact Jama document key (e.g. "SA-TC-7").
        item_type: Jama item-type id as a numeric string (e.g. "89011" for Test
                   Case). Pass None for all types. Kept as a string (not int)
                   to match `search_jama_semantics` and the other MCP tools,
                   which all take ids as numeric strings.
        status: exact status string (e.g. "BLOCKED", "APPROVED").
        keyword: full-text 'contains' filter delegated to Jama.

    Returns:
        {"project_id","count","results":[{document_key,name,item_type_name,
        status,modified_date,description}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    # Normalize item_type to int (or None), matching search_jama_semantics.
    # The schema declares it as a string so the tool surface stays consistent
    # across all id-bearing parameters (project_id / item_type / item_id are
    # all numeric strings); convert here before handing to the REST client.
    it = None
    if item_type is not None and str(item_type).strip():
        try:
            it = int(item_type)
        except (TypeError, ValueError):
            return {"error": "item_type must be a numeric string or null"}
    try:
        rows = jama().query_items_native(
            pid, document_key=document_key, item_type=it,
            status=status, keyword=keyword, limit=20)
    except Exception as exc:
        log.error("native query failed: %s\n%s", exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"project_id": pid, "count": len(rows), "results": rows}


# --------------------------------------------------------------------------- #
# Read-only Jama browse tools (extend the native query surface)
# --------------------------------------------------------------------------- #
# Each gate on {"jama"} only (no embedding/index needed). They are thin
# wrappers over JamaClient methods that reuse the OAuth + pagination machinery.

@mcp.tool()
def list_jama_projects() -> dict:
    """List all Jama projects visible to the OAuth client.

    Returns:
        {"count","results":[{id,project_key,name,status,description}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        rows = jama().list_projects()
    except Exception as exc:
        log.error("list_jama_projects failed: %s\n%s", exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"count": len(rows), "results": rows}


@mcp.tool()
def get_jama_item(item_id: str) -> dict:
    """Fetch a single Jama item by id (full metadata + cleaned text).

    Args:
        item_id: numeric string Jama item id.

    Returns:
        {"item":{item_id,document_key,item_type_name,name,status,
        description,test_steps,modified_date,...}}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        iid = int(item_id)
    except (TypeError, ValueError):
        return {"error": "item_id must be a numeric string"}
    try:
        item = jama().get_item(iid)
    except Exception as exc:
        log.error("get_jama_item failed: %s\n%s", exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    if item is None:
        return {"error": f"Item {iid} not found"}
    return {"item": item}


@mcp.tool()
def get_jama_item_relationships(item_id: str, limit: int = 50) -> dict:
    """List relationships (source/target) for an item.

    Args:
        item_id: numeric string Jama item id.
        limit: max relationships to return (default 50).

    Returns:
        {"item_id","count","results":[{id,relationship_type,source_item,
        target_item,name,modified_date}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        iid = int(item_id)
    except (TypeError, ValueError):
        return {"error": "item_id must be a numeric string"}
    try:
        rows = jama().get_item_relationships(iid, limit=limit)
    except Exception as exc:
        log.error("get_jama_item_relationships failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"item_id": iid, "count": len(rows), "results": rows}


@mcp.tool()
def get_jama_item_children(item_id: str, limit: int = 50) -> dict:
    """List decomposition children of an item.

    Args:
        item_id: numeric string Jama item id.
        limit: max children to return (default 50).

    Returns:
        {"item_id","count","results":[{item_id,document_key,item_type_name,
        name,status,modified_date}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        iid = int(item_id)
    except (TypeError, ValueError):
        return {"error": "item_id must be a numeric string"}
    try:
        rows = jama().get_item_children(iid, limit=limit)
    except Exception as exc:
        log.error("get_jama_item_children failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"item_id": iid, "count": len(rows), "results": rows}


@mcp.tool()
def list_jama_project_relationships(project_id: str, item_id: str = None,
                                    limit: int = 50) -> dict:
    """List relationships for a project (cursor-paginated Jama endpoint).

    Jama's ``/relationships`` endpoint requires a ``project`` filter and uses
    ``lastId`` cursor pagination. Optionally filter to relationships involving
    a specific item (client-side on fromItem/toItem).

    Args:
        project_id: numeric string Jama project id.
        item_id: optional numeric string item id to filter on.
        limit: max relationships to return (default 50).

    Returns:
        {"project_id","count","results":[{id,relationship_type,source_item,
        target_item,suspect,name,modified_date}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    iid = None
    if item_id is not None and str(item_id).strip():
        try:
            iid = int(item_id)
        except (TypeError, ValueError):
            return {"error": "item_id must be a numeric string"}
    try:
        rows = jama().list_project_relationships(
            pid, limit=limit, item_id=iid)
    except Exception as exc:
        log.error("list_jama_project_relationships failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"project_id": pid, "count": len(rows), "results": rows}


@mcp.tool()
def get_jama_item_comments(item_id: str, limit: int = 50) -> dict:
    """List comments threaded on an item.

    Args:
        item_id: numeric string Jama item id.
        limit: max comments to return (default 50).

    Returns:
        {"item_id","count","results":[{id,body,created_by,created_date,
        modified_date}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        iid = int(item_id)
    except (TypeError, ValueError):
        return {"error": "item_id must be a numeric string"}
    try:
        rows = jama().get_item_comments(iid, limit=limit)
    except Exception as exc:
        log.error("get_jama_item_comments failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"item_id": iid, "count": len(rows), "results": rows}


@mcp.tool()
def get_jama_item_attachments(item_id: str, limit: int = 50) -> dict:
    """List attachment metadata for an item (no binary download).

    Args:
        item_id: numeric string Jama item id.
        limit: max attachments to return (default 50).

    Returns:
        {"item_id","count","results":[{id,name,file_type,file_size,
        mime_type,created_date,modified_date}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        iid = int(item_id)
    except (TypeError, ValueError):
        return {"error": "item_id must be a numeric string"}
    try:
        rows = jama().get_item_attachments(iid, limit=limit)
    except Exception as exc:
        log.error("get_jama_item_attachments failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"item_id": iid, "count": len(rows), "results": rows}


@mcp.tool()
def list_jama_releases(project_id: str, limit: int = 50) -> dict:
    """List releases / versions for a project.

    Args:
        project_id: numeric string Jama project id.
        limit: max releases to return (default 50).

    Returns:
        {"project_id","count","results":[{id,name,release_date,status,
        description,modified_date}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    try:
        rows = jama().list_releases(pid, limit=limit)
    except Exception as exc:
        log.error("list_jama_releases failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"project_id": pid, "count": len(rows), "results": rows}


@mcp.tool()
def list_jama_test_runs(project_id: str = None,
                        test_cycle_id: str = None,
                        limit: int = 50) -> dict:
    """List test runs for a project and/or test cycle.

    At least one of ``project_id`` / ``test_cycle_id`` must be provided.

    Args:
        project_id: optional numeric string Jama project id.
        test_cycle_id: optional numeric string Jama test cycle id.
        limit: max test runs to return (default 50).

    Returns:
        {"count","results":[{id,name,status,test_cycle,item,assigned_to,
        modified_date}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    pid = None
    tcid = None
    try:
        if project_id is not None and str(project_id).strip():
            pid = int(project_id)
        if test_cycle_id is not None and str(test_cycle_id).strip():
            tcid = int(test_cycle_id)
    except (TypeError, ValueError):
        return {"error": "project_id/test_cycle_id must be numeric strings"}
    if pid is None and tcid is None:
        return {"error": "Provide project_id and/or test_cycle_id"}
    try:
        rows = jama().list_test_runs(project_id=pid, test_cycle_id=tcid,
                                     limit=limit)
    except Exception as exc:
        log.error("list_jama_test_runs failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"count": len(rows), "results": rows}


@mcp.tool()
def list_jama_item_types() -> dict:
    """List all Jama item types (id -> display name) for the tenant.

    Returns:
        {"count","results":[{id,name}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    try:
        rows = jama().list_item_types()
    except Exception as exc:
        log.error("list_jama_item_types failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"count": len(rows), "results": rows}


@mcp.tool()
def find_jama_project_by_name(name: str, exact: bool = False,
                              limit: int = 20) -> dict:
    """Find Jama projects by name (case-insensitive) and return their info.

    Useful when you only know a project's name (or a fragment of it) and need
    its numeric id to feed into other tools (init_jama_project,
    list_jama_releases, list_jama_test_runs, …). Matching is substring by
    default; pass exact=True for full case-insensitive equality.

    Args:
        name: project name or fragment (e.g. "acre" matches "Acrelec").
        exact: if True, require full case-insensitive name equality.
        limit: max matches to return (default 20).

    Returns:
        {"count","results":[{id,project_key,name,status,description}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    if not name or not str(name).strip():
        return {"error": "name is required"}
    try:
        rows = jama().find_projects(str(name), exact=exact, limit=limit)
    except Exception as exc:
        log.error("find_jama_project_by_name failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"count": len(rows), "results": rows}


@mcp.tool()
def find_jama_item_type_by_name(name: str, exact: bool = False,
                                limit: int = 20) -> dict:
    """Find Jama item types by display name (case-insensitive) and return info.

    Returns the type id (needed by search_jama_semantics / query_jama_native_metadata
    item_type filters) plus category, display plural and description. Matching is
    substring by default; pass exact=True for full case-insensitive equality.

    Args:
        name: type name or fragment (e.g. "test" matches "Test Case", "Test Plan").
        exact: if True, require full case-insensitive name equality.
        limit: max matches to return (default 20).

    Returns:
        {"count","results":[{id,display,display_plural,category,category_name,
        description}, ...]}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    if not name or not str(name).strip():
        return {"error": "name is required"}
    try:
        rows = jama().find_item_types(str(name), exact=exact, limit=limit)
    except Exception as exc:
        log.error("find_jama_item_type_by_name failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"count": len(rows), "results": rows}


@mcp.tool()
def query_jama_endpoint(path: str, params: str = None,
                        all_pages: bool = False) -> dict:
    """Power-user escape hatch: GET any Jama REST endpoint (read-only).

    ``path`` is appended to ``{JAMA_URL}{API_PREFIX}`` (e.g. ``"/projects"``).
    Only GET is ever issued; the client is read-only by design.

    Args:
        path: REST path beginning with '/', e.g. "/items/12345".
        params: optional 'k1=v1&k2=v2' query string.
        all_pages: if True, walk all pages and return a flat list of ``data``;
                   if False (default), return only the first page.

    Returns:
        {"path","data": <first-page data or flat list>}
    """
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
    if not path or not path.startswith("/"):
        return {"error": "path must start with '/' (e.g. '/projects')"}
    parsed: dict | None = None
    if params:
        from urllib.parse import parse_qs
        parsed = {k: v[0] if len(v) == 1 else v
                  for k, v in parse_qs(params).items()}
    try:
        # Cap all-pages walks at 50 pages (≤2500 rows at page_size=50) so a
        # broad endpoint can't trigger an unbounded full-scan that exhausts
        # memory/network. Callers needing more should page explicitly.
        data = jama().get_raw(path, params=parsed,
                              max_pages=50 if all_pages else 1)
    except ValueError as exc:
        # Path sanitization rejection from get_raw.
        return {"error": str(exc)}
    except Exception as exc:
        log.error("query_jama_endpoint failed: %s\n%s",
                  exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"path": path, "data": data}


# --------------------------------------------------------------------------- #
# Configuration tools: validate_setup + configure_jama
# --------------------------------------------------------------------------- #
@mcp.tool()
def validate_setup(live: bool = False) -> dict:
    """Validate all dependencies, configuration and storage.

    Runs the offline pre-flight (packages + env vars + SQLite). When
    ``live=True`` it also probes the Jama OAuth token and the embedding
    endpoint with a real request, so credentials can be verified without
    running a full project init.

    Args:
        live: if True, perform live connectivity probes against Jama and the
              embedding endpoint (slower; uses one Jama + one embedding call).

    Returns:
        {"blocking","issues":[...],"dependencies","config_issues","storage",
         "live": {"jama","embedding"} | null, "hint"}
    """
    report = preflight(require={"jama", "embedding"})
    out: dict = {
        "blocking": report["blocking"],
        "issues": report["issues"],
        "dependencies": report["dependencies"],
        "config_issues": report["config_issues"],
        "storage": report["storage"],
        "live": None,
        "hint": report["hint"],
    }
    if live and not report["blocking"]:
        out["live"] = _live_probe()
    return out


def _live_probe() -> dict:
    """Live (network) checks for Jama auth + the active embedding backend."""
    result: dict = {"jama": None, "embedding": None}
    # Jama: OAuth + connectivity. The pre-flight speed test probes a single
    # /projects page (~20KB) — too small for a stable bandwidth measurement
    # (server-side processing latency dominates the body-transfer clock), so on
    # a healthy link it routinely dips below the 20KB/s floor that's meant for
    # the multi-MB model download. Treat a "too slow" verdict here as a
    # non-fatal warning instead of a failure: ``list_projects()`` is the real
    # proof that OAuth + connectivity work. The bandwidth gate still applies in
    # the sync path, where a real download is about to start.
    from net_guard import NetworkTooSlowError
    client = jama()
    speed_warning: str | None = None
    try:
        client.preflight_speed_check()
    except NetworkTooSlowError as exc:
        speed_warning = str(exc)[:200]
    except Exception as exc:
        result["jama"] = {"ok": False, "error": str(exc)[:300]}
    if result["jama"] is None:
        try:
            projects = list(client.list_projects())
            info: dict = {"ok": True, "project_count": len(projects)}
            if speed_warning:
                info["speed_warning"] = speed_warning
            result["jama"] = info
        except Exception as exc:
            result["jama"] = {"ok": False, "error": str(exc)[:300]}
    # Embedding: probe the ACTIVE provider's embedder (local CPU bge or azure),
    # not a hard-coded Azure client. The old code always instantiated
    # ``EmbeddingClient()`` against the placeholder Azure URL, so under the
    # default ``local`` provider ``validate_setup(live=True)`` always reported
    # embedding failure even though local embedding worked fine.
    try:
        vec = rag().embedder.embed_one("jama mcp validate")
        result["embedding"] = {"ok": True, "dimensions": len(vec)}
    except Exception as exc:
        result["embedding"] = {"ok": False, "error": str(exc)[:300]}
    return result


@mcp.tool()
def configure_jama(values: dict) -> dict:
    """Apply configuration values at runtime and persist them to .env.

    Accepts a mapping of env-var names to values (e.g.
    ``{"JAMA_URL":"...","JAMA_CLIENT_SECRET":"..."}`). Writes a complete
    ``.env`` (merging with existing values), reloads settings in-process, and
    resets the Jama/RAG/DB singletons so subsequent calls use the new config.
    Secrets are written to ``.env`` on disk only; they are never echoed back.

    Args:
        values: dict of {ENV_VAR: value}. Recognized keys: JAMA_URL,
                JAMA_CLIENT_ID, JAMA_CLIENT_SECRET, EMBEDDING_BASE_URL,
                EMBEDDING_API_KEY, JAMA_MCP_DB_PATH,
                and any other key in the .env template.

    Returns:
        {"ok": true, "written": <abs .env path>, "applied_keys": [...]}
        or {"error": ...}
    """
    if not isinstance(values, dict) or not values:
        return {"error": "values must be a non-empty dict {ENV_VAR: value}"}
    try:
        path = write_env_file(values)
    except Exception as exc:
        log.error("configure_jama write failed: %s", exc)
        return {"error": f"Failed to write .env: {exc}"}
    reload_settings()
    reset_singletons()
    log.info("Configuration applied; .env rewritten with keys: %s",
             list(values.keys()))
    return {"ok": True, "written": path, "applied_keys": list(values.keys())}


# --------------------------------------------------------------------------- #
# APScheduler incremental sync
# --------------------------------------------------------------------------- #
def _incremental_sync_all() -> None:
    """Sync every initialized project for newly-modified items (async).

    Each project's sync is submitted to the shared ``_executor`` thread pool
    (like init/reinit) so this scheduler tick returns immediately instead of
    blocking the scheduler thread for the whole batch. The per-project
    ``_project_lock`` still serializes a scheduled sync against any concurrent
    user-initiated sync for the same project.

    A project that already has an in-flight job is skipped: submitting another
    would just queue behind the per-project lock and occupy a worker slot, and
    the next scheduled tick will pick up any newer modifications. The check +
    create run under ``_init_lock`` to avoid a TOCTOU with a concurrent
    init/reinit (same pattern as ``_start_sync_job``).
    """
    conn = db()
    projects = list_initialized_projects(conn)
    if not projects:
        return
    log.info("Incremental sync: %s project(s)", len(projects))
    for p in projects:
        pid = p["project_id"]
        with _init_lock:
            if get_active_job_for_project(conn, pid) is not None:
                log.info("Skip scheduled sync for project %s: a job is "
                         "already in flight", pid)
                continue
            job_id = f"sync-{uuid.uuid4().hex[:12]}"
            try:
                create_job(conn, job_id, pid, "sync")
            except Exception as exc:
                log.error("Could not create sync job for %s: %s", pid, exc)
                continue
        try:
            _executor.submit(_run_job, pid, job_id, incremental=True)
        except Exception as exc:
            log.error("Could not submit sync job for %s: %s", pid, exc)
            try:
                update_job(conn, job_id, status="ERROR",
                           message=f"Submit failed: {exc}")
            except Exception:
                pass


def _start_scheduler() -> BackgroundScheduler | None:
    if not settings.sync.enabled:
        log.info("Incremental sync disabled (SYNC_ENABLED=0)")
        return None
    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        _incremental_sync_all, "interval",
        hours=settings.sync.hours, next_run_time=None,
        id="jama-incremental-sync", max_instances=1,
        coalesce=True,
    )
    sched.start()
    log.info("APScheduler started: incremental sync every %s hour(s)",
             settings.sync.hours)
    return sched


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _resume_interrupted_syncs() -> None:
    """Crash recovery: re-queue any project left in INITIALIZING.

    A sync that was interrupted by a crash/process kill leaves the project in
    INITIALIZING with a stale (or absent) last_sync_time. Already-flushed items
    are safely persisted (each flush commits), but the project was never marked
    READY, so it must be re-synced to guarantee completeness.

    Uses ``incremental="resume"`` mode: items that already have chunks are
    SKIPPED (no re-download, no re-embed), so only the items that were in-flight
    or not-yet-fetched when the crash happened are re-processed. This turns a
    ~15min full re-sync of a 5000-item project into a seconds-fast resume when
    most items were already indexed. upsert is idempotent regardless.
    """
    conn = db()
    stuck = conn.execute(
        "SELECT project_id FROM projects WHERE status = 'INITIALIZING'"
    ).fetchall()
    if not stuck:
        return
    for row in stuck:
        pid = row["project_id"]
        job_id = f"resume-{uuid.uuid4().hex[:12]}"
        log.warning("Recovery: project %s was left INITIALIZING (crash during "
                    "sync); re-queuing resume sync as job %s", pid, job_id)
        create_job(conn, job_id, pid, "init")
        _executor.submit(_run_job, pid, job_id, incremental="resume")


def _warn_if_models_missing() -> None:
    """Log a clear hint if the embedding/reranker models aren't cached yet.

    Uses the lightweight on-disk presence checks (no model load) so startup
    stays fast. Doesn't block or auto-download — just nudges the user/LLM to
    call `bootstrap_models` so the first sync isn't slowed by a download.
    """
    missing = []
    try:
        pipeline = rag()
        # Embedding: only the local provider downloads a model; azure hits an
        # API endpoint, so there's nothing to cache.
        if settings.embedding.provider == "local":
            if not pipeline.embedder._model_present():  # type: ignore[attr-defined]
                missing.append("embedding (bge-small-en-v1.5)")
        # Reranker: lightweight on-disk check (no load, no network).
        if not pipeline.reranker.weights_cached():
            missing.append(f"reranker ({settings.reranker.model_name})")
    except Exception:
        return  # don't let a probe failure noise up startup
    if missing:
        log.warning("Models not yet cached: %s. Call the `bootstrap_models` MCP "
                    "tool to pre-download them (poll get_bootstrap_progress "
                    "every ~2 min) BEFORE the first init_jama_project.",
                    ", ".join(missing))


def main() -> None:
    # Eagerly initialize the DB so schema/extension errors surface at startup.
    # Reuse the shared singleton connection — calling init_db() standalone here
    # would open (and leak) a SECOND connection alongside db()'s lazy one.
    db()
    # Clear crash-leftover RUNNING jobs before any worker runs, so the monitor
    # never reports a phantom in-flight job. Must precede _resume_interrupted_syncs
    # so resumed INITIALIZING projects get a fresh job row rather than reusing a
    # now-stale one.
    reconciled = reconcile_stale_jobs(db())
    if reconciled:
        log.info("Reconciled %s stale job(s) left RUNNING by a prior crash",
                 reconciled)
    # Recover any project interrupted mid-sync by a prior crash before serving.
    _resume_interrupted_syncs()
    _start_scheduler()
    _warn_if_models_missing()
    # Transport selection. stdio (default) is for local MCP clients that spawn
    # the server as a subprocess. streamable-http / sse expose the server over
    # HTTP so remote clients (or a Docker container) can connect by URL.
    # FASTMCP_HOST/FASTMCP_PORT env vars do NOT work (FastMCP's constructor
    # kwargs override them), so we set mcp.settings.host/port directly before
    # run() — the Settings pydantic model is mutable.
    transport = os.environ.get("JAMA_MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        mcp.settings.host = os.environ.get("JAMA_MCP_HOST", "127.0.0.1")
        try:
            mcp.settings.port = int(os.environ.get("JAMA_MCP_PORT", "8000"))
        except ValueError:
            mcp.settings.port = 8000
    elif transport not in ("stdio",):
        log.warning("Unknown JAMA_MCP_TRANSPORT=%r; falling back to stdio",
                    transport)
        transport = "stdio"
    log.info("Jama MCP Server starting (%s)...", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
