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

from config import reload_settings, settings, write_env_file
from db_setup import (count_chunks, create_job, get_connection, get_job,
                      get_project, init_db, list_initialized_projects,
                      update_job, upsert_item, upsert_project,
                      replace_chunks, write_txn)
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
reranks with a local Qwen3 model. So "like", "keyword" and "semantic"
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
is召回-oriented and surfaces the most relevant items even for near-keyword
phrasing, while native tools return empty on any misspelling or mismatch.

PREREQUISITE: `search_jama_semantics` needs the project initialized first
(`init_jama_project` → poll `get_sync_progress` until DONE). The native
and browse tools work immediately against the live Jama API — no init
required. If a project is not initialized, suggest `init_jama_project`
before falling back to native metadata.
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
    not_ready = _ensure_ready({"jama", "embedding"})
    if not_ready:
        return not_ready
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
    not_ready = _ensure_ready(set())
    if not_ready:
        return not_ready
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

    This is the DEFAULT tool for any non-precise question. It fuses keyword
    (FTS5/BM25), vector (sqlite-vec cosine) and RRF in one call, then reranks
    with a local Qwen3 model — so "like", "keyword" and "semantic" queries are
    all best answered here. Prefer it over native metadata unless the user
    gives an exact document key / status / item id.

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
    not_ready = _ensure_ready({"jama", "embedding"})
    if not_ready:
        return not_ready
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

    Use ONLY for precise lookups (exact document key, exact status, exact
    item_type) — it returns empty on any misspelling. For topical, fuzzy or
    natural-language questions, prefer `search_jama_semantics` instead.

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
    not_ready = _ensure_ready({"jama"})
    if not_ready:
        return not_ready
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
        data = jama().get_raw(path, params=parsed,
                              max_pages=None if all_pages else 1)
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
    """Live (network) checks for Jama auth + embedding endpoint."""
    result: dict = {"jama": None, "embedding": None}
    try:
        client = jama()
        client.preflight_speed_check()
        projects = list(client.list_projects())
        result["jama"] = {"ok": True, "project_count": len(projects)}
    except Exception as exc:
        result["jama"] = {"ok": False, "error": str(exc)[:300]}
    try:
        from rag_pipeline import EmbeddingClient
        vec = EmbeddingClient().embed_one("jama mcp validate")
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
                EMBEDDING_API_KEY, LLM_BASE_URL, LLM_API_KEY, JAMA_MCP_DB_PATH,
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
