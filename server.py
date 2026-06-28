"""Jama MCP Server entry point.

Exposes four tools to LLM clients via the Model Context Protocol:
  * init_jama_project        - async background init (returns job_id)
  * get_sync_progress        - poll job progress
  * search_jama_semantics    - high-precision RAG (multi-query + hybrid + RRF + rerank)
  * query_jama_native_metadata - direct Jama REST filtering (exact metadata)

Also runs an APScheduler job that incrementally syncs initialized projects
every N hours: it walks items whose modifiedDate > last_sync_time, cleans
them, re-chunks and updates the FTS5 + sqlite-vec indexes.

Run with:  python server.py
Configure an MCP client (Claude Desktop, etc.) to launch this as a stdio server.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from apscheduler.schedulers.background import BackgroundScheduler
from mcp.server.fastmcp import FastMCP

from config import settings
from db_setup import (count_chunks, create_job, get_connection, get_job,
                      get_project, init_db, list_initialized_projects,
                      update_job, upsert_item, upsert_project,
                      replace_chunks, write_txn)
from jama_client import JamaClient, utcnow_iso
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

mcp = FastMCP("jama-mcp")


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


# --------------------------------------------------------------------------- #
# Core sync logic (shared by init tool + scheduler)
# --------------------------------------------------------------------------- #
def _sync_project(project_id: int, *, job_id: str | None,
                  incremental: bool) -> None:
    """Download, clean, chunk, embed and index a project's items."""
    conn = db()
    last_sync = None
    if incremental:
        proj = get_project(conn, project_id)
        last_sync = proj["last_sync_time"] if proj else None
    upsert_project(conn, project_id, status="INITIALIZING")

    if job_id:
        update_job(conn, job_id, status="RUNNING", progress=0.0,
                   message="Pre-flight network speed test")

    client = jama()
    # Pre-flight: abort early with a clear network error if the Jama host is
    # too slow, instead of timing out partway through pagination.
    try:
        client.preflight_speed_check()
    except Exception as exc:
        msg = f"Network pre-flight check failed: {exc}"
        log.error(msg)
        if job_id:
            update_job(conn, job_id, status="ERROR", message=msg)
        upsert_project(conn, project_id, status="ERROR", error=msg)
        return

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

    # First pass: collect items to process (for progress + batching).
    items = list(client.iter_project_items(
        project_id, modified_after=last_sync,
        max_items=settings.sync.max_items_per_run if incremental else None))
    total = len(items)
    if job_id:
        update_job(conn, job_id, total=total, done=0,
                   message=f"Indexing {total} items")

    if total == 0:
        if job_id:
            update_job(conn, job_id, status="DONE", progress=1.0,
                       done=0, message="No new/modified items")
        upsert_project(conn, project_id, name=proj_name, status="READY",
                       last_sync_time=utcnow_iso())
        return

    pipeline = rag()
    done = 0
    chunk_total = 0
    for item in items:
        try:
            # Persist item metadata.
            with write_txn(conn):
                upsert_item(conn, item)
            # Chunk + embed + index.
            chunks = chunk_item(item)
            if chunks:
                embeddings = pipeline.embed_chunks(chunks)
                replace_chunks(conn, item["item_id"], chunks, embeddings)
                chunk_total += len(chunks)
            else:
                # Remove stale chunks for items that no longer have text.
                replace_chunks(conn, item["item_id"], [], [])
        except Exception as exc:
            log.warning("Failed to index item %s: %s", item.get("item_id"), exc)
        done += 1
        if job_id and total:
            update_job(conn, job_id, done=done,
                       progress=round(done / total, 4),
                       message=f"Indexed {done}/{total} items")

    final_chunk_count = count_chunks(conn, project_id)
    upsert_project(conn, project_id, name=proj_name, status="READY",
                   last_sync_time=utcnow_iso(),
                   item_count=total, chunk_count=final_chunk_count)
    if job_id:
        update_job(conn, job_id, status="DONE", progress=1.0, done=done,
                   message=f"Done: {total} items, {chunk_total} chunks "
                           f"indexed this run")
    log.info("Sync complete for project %s: %s items, %s chunks",
             project_id, total, chunk_total)


def _run_job(project_id: int, job_id: str, incremental: bool) -> None:
    """Background worker: runs sync then guarantees terminal job state."""
    conn = db()
    try:
        _sync_project(project_id, job_id=job_id, incremental=incremental)
    except Exception as exc:
        log.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        update_job(conn, job_id, status="ERROR", message=str(exc)[:500])
        upsert_project(conn, project_id, status="ERROR", error=str(exc)[:500])


# --------------------------------------------------------------------------- #
# MCP Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def init_jama_project(project_id: str) -> dict:
    """Initialize a Jama project: download, clean, vectorize and index its items.

    Runs as an async background task and returns a job_id immediately so the
    caller (LLM) is never blocked. Poll progress with get_sync_progress.

    Args:
        project_id: Jama project id (numeric string, e.g. "20571").

    Returns:
        {"job_id": "...", "project_id": ..., "status": "RUNNING"}
    """
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    job_id = f"init-{uuid.uuid4().hex[:12]}"
    conn = db()
    create_job(conn, job_id, pid, "init")
    upsert_project(conn, pid, status="INITIALIZING")
    _executor.submit(_run_job, pid, job_id, incremental=False)
    log.info("Started init job %s for project %s", job_id, pid)
    return {"job_id": job_id, "project_id": pid, "status": "RUNNING"}


@mcp.tool()
def get_sync_progress(job_id: str) -> dict:
    """Poll the progress of an init or sync job.

    Returns:
        {"job_id","project_id","kind","status","progress","total","done",
         "message","started_at","finished_at"}
        status is one of PENDING | RUNNING | DONE | ERROR.
    """
    conn = db()
    row = get_job(conn, job_id)
    if row is None:
        return {"error": f"Unknown job_id: {job_id}"}
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
def search_jama_semantics(project_id: str, query: str,
                          item_type: str = None, top_k: int = 5,
                          candidate_k: int = 50,
                          modified_after: str = None,
                          modified_before: str = None) -> dict:
    """Semantic search over an initialized Jama project using high-precision RAG.

    Pipeline: Multi-Query expansion -> hybrid recall (sqlite-vec + FTS5) ->
    RRF fusion -> local Qwen3-Reranker-0.6B -> top_k results.

    Args:
        project_id: numeric string Jama project id (must be initialized first).
        query: natural-language search query.
        item_type: optional Jama item-type id to filter (e.g. "89011" for Test
                   Cases, "89009" for Requirements). Pass None for all.
        top_k: final results to return (default 5).
        candidate_k: candidate pool size before reranking (default 50).
        modified_after: optional ISO-8601 lower bound on item modified date
                        (inclusive). Naive timestamps are assumed UTC.
                        e.g. "2024-01-01" or "2024-06-01T00:00:00Z".
        modified_before: optional ISO-8601 upper bound on item modified date
                         (inclusive). Naive timestamps are assumed UTC.

    Returns:
        {"project_id","query","results":[{document_key,name,item_type_name,
        section,modified_date,text,score,strategy}, ...]}
    """
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    if not query or not query.strip():
        return {"error": "query is required"}

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
        results = rag().search(pid, query.strip(), item_type=it,
                               top_k=top_k, candidate_k=candidate_k,
                               modified_after=modified_after,
                               modified_before=modified_before)
    except Exception as exc:
        log.error("search failed: %s\n%s", exc, traceback.format_exc())
        return {"error": f"Search failed: {exc}"}
    return {"project_id": pid, "query": query, "count": len(results),
            "modified_after": modified_after,
            "modified_before": modified_before,
            "results": results}


@mcp.tool()
def query_jama_native_metadata(project_id: str, document_key: str = None,
                               item_type: int = None, status: str = None,
                               keyword: str = None) -> dict:
    """Query Jama's native REST API directly for exact metadata filtering.

    Bypasses the vector store to answer precise questions (exact document key,
    specific status, specific item type). Handles pagination internally and
    returns up to 20 core metadata records.

    Args:
        project_id: numeric string Jama project id.
        document_key: exact Jama document key (e.g. "SA-TC-7").
        item_type: Jama item-type id (e.g. 89011 = Test Case).
        status: exact status string (e.g. "BLOCKED", "APPROVED").
        keyword: full-text 'contains' filter delegated to Jama.

    Returns:
        {"project_id","count","results":[{document_key,name,item_type_name,
        status,modified_date,description}, ...]}
    """
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"error": "project_id must be a numeric string"}
    try:
        rows = jama().query_items_native(
            pid, document_key=document_key, item_type=item_type,
            status=status, keyword=keyword, limit=20)
    except Exception as exc:
        log.error("native query failed: %s\n%s", exc, traceback.format_exc())
        return {"error": f"Jama API query failed: {exc}"}
    return {"project_id": pid, "count": len(rows), "results": rows}


# --------------------------------------------------------------------------- #
# APScheduler incremental sync
# --------------------------------------------------------------------------- #
def _incremental_sync_all() -> None:
    """Sync every initialized project for newly-modified items."""
    conn = db()
    projects = list_initialized_projects(conn)
    if not projects:
        return
    log.info("Incremental sync: %s project(s)", len(projects))
    for p in projects:
        pid = p["project_id"]
        job_id = f"sync-{uuid.uuid4().hex[:12]}"
        try:
            create_job(conn, job_id, pid, "sync")
            _run_job(pid, job_id, incremental=True)
        except Exception as exc:
            log.error("Incremental sync failed for %s: %s", pid, exc)


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
def main() -> None:
    # Eagerly initialize the DB so schema/extension errors surface at startup.
    init_db()
    _start_scheduler()
    log.info("Jama MCP Server starting (stdio)...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
