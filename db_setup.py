"""SQLite schema + extension loading for the Jama MCP Server.

Storage layout
--------------
* ``projects``           - per-project sync state (last_sync_time, status).
* ``items``              - one row per Jama item (de-normalized metadata).
* ``chunks``             - one row per text chunk (the unit of retrieval).
* ``chunks_fts`` (FTS5)  - full-text index over chunk text (keyword recall).
* ``chunks_vec`` (vec0)  - sqlite-vec vector index over embeddings (semantic recall).
* ``sync_jobs``          - progress tracking for async init/sync jobs.

Concurrency
-----------
All connections are opened with ``check_same_thread=False`` and a busy timeout
so that the MCP server's reader threads and the APScheduler writer thread can
coexist. Writers serialize through a process-level ``RLock`` plus SQLite's own
database-level locking; the busy timeout lets readers wait briefly instead of
failing with SQLITE_BUSY.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable

log = logging.getLogger(__name__)

import sqlite_vec

from config import settings

# Process-wide write lock. SQLite handles file locking, but this avoids
# pointless "database is locked" contention between the scheduler and the
# init job within a single process.
_write_lock = threading.RLock()


# --------------------------------------------------------------------------- #
# Connection helpers
# --------------------------------------------------------------------------- #
def _load_extensions(conn: sqlite3.Connection) -> None:
    """Enable FTS5 (built into CPython's sqlite) and sqlite-vec."""
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    except Exception as exc:  # pragma: no cover - hard environment failure
        raise RuntimeError(
            "Failed to load sqlite-vec extension. "
            "Install it via `pip install sqlite-vec`."
        ) from exc
    conn.enable_load_extension(False)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a configured SQLite connection (extensions + pragmas)."""
    path = db_path or settings.storage.db_path
    conn = sqlite3.connect(
        path,
        check_same_thread=False,
        isolation_level=None,  # autocommit; we manage txns explicitly
        timeout=settings.storage.busy_timeout_ms / 1000.0,
    )
    _load_extensions(conn)
    conn.row_factory = sqlite3.Row
    # WAL gives concurrent readers + a single writer without blocking reads.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={settings.storage.busy_timeout_ms};")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def write_txn(conn: sqlite3.Connection):
    """Serialize writes and wrap them in a single transaction."""
    with _write_lock:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id   INTEGER PRIMARY KEY,
    name         TEXT,
    status       TEXT DEFAULT 'NEW',          -- NEW | INITIALIZING | READY | ERROR
    last_sync_time TEXT,                       -- ISO-8601 UTC
    item_count   INTEGER DEFAULT 0,
    chunk_count  INTEGER DEFAULT 0,
    error        TEXT,
    updated_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS items (
    item_id      INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL,
    document_key TEXT,
    global_id    TEXT,
    item_type    INTEGER,                      -- Jama itemType id
    item_type_name TEXT,                       -- e.g. "Requirement", "Test Case"
    name         TEXT,
    status       TEXT,                         -- fields.status / testCaseStatus
    description  TEXT,                         -- cleaned plain text
    test_steps   TEXT,                         -- cleaned plain text (Test Cases)
    modified_date TEXT,                         -- ISO-8601 from Jama
    created_date  TEXT,
    raw_json     TEXT,                          -- full payload for debugging
    updated_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);
CREATE INDEX IF NOT EXISTS idx_items_project ON items(project_id);
CREATE INDEX IF NOT EXISTS idx_items_type ON items(project_id, item_type);
CREATE INDEX IF NOT EXISTS idx_items_modified ON items(project_id, modified_date);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,             -- "{item_id}#{n}"
    item_id      INTEGER NOT NULL,
    project_id   INTEGER NOT NULL,
    item_type    INTEGER,
    item_type_name TEXT,
    document_key TEXT,
    name         TEXT,                          -- item name (for result context)
    status       TEXT,
    section      TEXT,                          -- "description" | "test_steps"
    chunk_index  INTEGER,
    text         TEXT,
    modified_date TEXT,                         -- ISO-8601 UTC from parent item (range-filter metadata)
    updated_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (item_id) REFERENCES items(item_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_item ON chunks(item_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_modified ON chunks(project_id, modified_date);

CREATE TABLE IF NOT EXISTS sync_jobs (
    job_id       TEXT PRIMARY KEY,
    project_id   INTEGER,
    kind         TEXT,                          -- "init" | "reinit" | "sync"
    status       TEXT DEFAULT 'PENDING',        -- PENDING|RUNNING|DONE|ERROR
    progress     REAL DEFAULT 0.0,              -- 0.0 .. 1.0
    total        INTEGER DEFAULT 0,
    done         INTEGER DEFAULT 0,
    message      TEXT,
    started_at   TEXT,
    finished_at  TEXT
);

-- FTS5 keyword index (porter + unicode for English text). External content
-- table: we mirror text manually to keep the vec/fts/chunks in one place.
-- modified_date is an UNINDEXED metadata column so BM25 ranking ignores it
-- but we can filter by it (range query) at recall time.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    project_id UNINDEXED,
    item_type UNINDEXED,
    modified_date UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);

-- sqlite-vec vector index. Embedding dim comes from config (1536 by default).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[{DIM}]
);
"""


def _existing_vec_dim(conn: sqlite3.Connection) -> int | None:
    """Return the embedding dimension of an existing chunks_vec index, or None.

    Reads the column declaration of the ``embedding`` column from the vec0
    table's schema. Returns None when the table doesn't exist yet (fresh DB).
    """
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        if not row or not row["sql"]:
            return None
        # The CREATE statement looks like: ... embedding FLOAT[1536] ...
        import re
        m = re.search(r"embedding\s+FLOAT\[(\d+)\]", row["sql"], re.IGNORECASE)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Create all tables/extensions if missing and return a ready connection.

    If the ``chunks_vec`` index already exists but its embedding dimension
    differs from the configured one (e.g. after switching EMBEDDING_PROVIDER
    from azure/1536 to local/384), the vec index, FTS index and chunk rows are
    dropped and rebuilt so the new dimension takes effect. Item metadata rows
    are preserved (re-sync re-embeds them).
    """
    conn = get_connection(db_path)
    want_dim = settings.embedding.dimensions

    # Detect a dimension mismatch on an existing vec index and rebuild.
    existing_dim = _existing_vec_dim(conn)
    if existing_dim is not None and existing_dim != want_dim:
        log.warning("Embedding dimension changed %d -> %d; rebuilding vector "
                    "index (chunks will be re-embedded on next sync).",
                    existing_dim, want_dim)
        with write_txn(conn):
            conn.execute("DROP TABLE IF EXISTS chunks_vec")
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM chunks")
        # Mark all projects as needing re-sync (not READY).
        conn.execute("UPDATE projects SET status='NEW', chunk_count=0, "
                     "last_sync_time=NULL, error=NULL WHERE status='READY'")

    schema = SCHEMA.replace("{DIM}", str(want_dim))
    conn.executescript(schema)
    return conn


# --------------------------------------------------------------------------- #
# Low-level data access used by jama_client / rag_pipeline / server
# --------------------------------------------------------------------------- #
def upsert_project(conn: sqlite3.Connection, project_id: int, name: str = None,
                   status: str = None, last_sync_time: str = None,
                   item_count: int = None, chunk_count: int = None,
                   error: str = None) -> None:
    fields = {
        "name": name, "status": status, "last_sync_time": last_sync_time,
        "item_count": item_count, "chunk_count": chunk_count, "error": error,
    }
    cols = ["project_id"] + [k for k, v in fields.items() if v is not None]
    ph = ",".join("?" * len(cols))
    vals = [project_id] + [fields[k] for k in cols[1:]]
    updates = ", ".join(f"{k}=excluded.{k}" for k in cols[1:] if k != "project_id")
    updates += ", updated_at=datetime('now')"
    sql = (f"INSERT INTO projects ({','.join(cols)}) VALUES ({ph}) "
           f"ON CONFLICT(project_id) DO UPDATE SET {updates}")
    with write_txn(conn):
        conn.execute(sql, vals)


def get_project(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM projects WHERE project_id=?", (project_id,)
    ).fetchone()


def list_initialized_projects(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Projects eligible for scheduled incremental sync (status=READY only).

    INITIALIZING is deliberately EXCLUDED: a project in that state was either
    just kicked off by init/reinit (its own job row drives it) or left stuck by
    a crash (handled by ``_resume_interrupted_syncs`` on startup). Letting the
    scheduler also submit a sync for it would race the in-progress/recovery
    worker on the per-project lock — the loser queues behind the lock and burns
    a worker slot, and on a stuck project the scheduled sync would never
    advance past INITIALIZING anyway. Only READY projects have a meaningful
    ``last_sync_time`` to increment from.
    """
    return conn.execute(
        "SELECT * FROM projects WHERE status = 'READY' ORDER BY project_id"
    ).fetchall()


def upsert_item(conn: sqlite3.Connection, item: dict) -> None:
    """Insert/update one item row (caller holds write_txn if batching)."""
    conn.execute(
        """
        INSERT INTO items (item_id, project_id, document_key, global_id, item_type,
            item_type_name, name, status, description, test_steps,
            modified_date, created_date, raw_json, updated_at)
        VALUES (:item_id,:project_id,:document_key,:global_id,:item_type,
            :item_type_name,:name,:status,:description,:test_steps,
            :modified_date,:created_date,:raw_json, datetime('now'))
        ON CONFLICT(item_id) DO UPDATE SET
            project_id=excluded.project_id, document_key=excluded.document_key,
            global_id=excluded.global_id, item_type=excluded.item_type,
            item_type_name=excluded.item_type_name, name=excluded.name,
            status=excluded.status, description=excluded.description,
            test_steps=excluded.test_steps, modified_date=excluded.modified_date,
            created_date=excluded.created_date, raw_json=excluded.raw_json,
            updated_at=datetime('now')
        """,
        item,
    )


def replace_chunks(conn: sqlite3.Connection, item_id: int,
                   chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Atomically replace all chunks (text + FTS + vec) for one item."""
    with write_txn(conn):
        # Remove old chunks + indexes for this item.
        old_ids = [r["chunk_id"] for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE item_id=?", (item_id,))]
        conn.execute("DELETE FROM chunks WHERE item_id=?", (item_id,))
        if old_ids:
            conn.executemany("DELETE FROM chunks_fts WHERE chunk_id=?",
                             [(c,) for c in old_ids])
            conn.executemany("DELETE FROM chunks_vec WHERE chunk_id=?",
                             [(c,) for c in old_ids])
        # Insert new.
        rows, fts_rows, vec_rows = [], [], []
        for ch, emb in zip(chunks, embeddings):
            rows.append(ch)
            fts_rows.append((ch["chunk_id"], ch["project_id"], ch["item_type"],
                             ch.get("modified_date"), ch["text"]))
            vec_rows.append((ch["chunk_id"], _vec_blob(emb)))
        if rows:
            # INSERT OR REPLACE so a re-sync (or a chunk_id collision from a
            # partially-cleaned prior state) upserts instead of raising
            # UNIQUE — replace_chunks is meant to be idempotent per item.
            conn.executemany(
                """INSERT OR REPLACE INTO chunks (chunk_id, item_id, project_id, item_type,
                   item_type_name, document_key, name, status, section,
                   chunk_index, text, modified_date, updated_at)
                   VALUES (:chunk_id,:item_id,:project_id,:item_type,:item_type_name,
                   :document_key,:name,:status,:section,:chunk_index,:text,
                   :modified_date, datetime('now'))""", rows)
            conn.executemany(
                "INSERT OR REPLACE INTO chunks_fts (chunk_id, project_id, item_type, "
                "modified_date, text) VALUES (?,?,?,?,?)", fts_rows)
            conn.executemany(
                "INSERT OR REPLACE INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                vec_rows)


def _vec_blob(vec: list[float]) -> bytes:
    """sqlite-vec accepts float32 little-endian blobs."""
    import struct
    return struct.pack(f"{len(vec)}f", *vec)


def _normalize_iso_utc(ts: str | None) -> str | None:
    """Normalize an ISO-8601 timestamp to ``YYYY-MM-DDTHH:MM:SS.ffffff+0000``.

    Accepts bare dates (``2024-01-01`` -> start of day), naive datetimes
    (assumed UTC), and offset datetimes; all converted to UTC. Returns None
    for None/empty input so callers can treat it as "no bound". Comparison
    against Jama's modified_date strings is then a lexicographic string
    comparison (valid because the format is fixed-width and same timezone).
    """
    if not ts:
        return None
    from datetime import datetime, timezone
    s = ts.strip()
    # Try a few formats; the trailing Z is shorthand for +00:00.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    else:
        # Fall back to fromisoformat (Python 3.11+ handles most variants).
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            raise ValueError(f"Unparseable timestamp: {ts!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f+0000")


def vector_search(conn: sqlite3.Connection, query_vec: list[float],
                  project_id: int, item_type: int | None,
                  limit: int, modified_after: str | None = None,
                  modified_before: str | None = None) -> list[sqlite3.Row]:
    """Semantic recall via sqlite-vec, joined back to chunk metadata.

    Uses vec0's native KNN: ``WHERE embedding MATCH ? AND k=?`` returns the
    k nearest vectors with a ``distance`` column. We over-fetch (k = limit*4)
    then filter by project_id / item_type / date range in the outer query, so
    enough candidates survive the filter to fill ``limit``.

    Optional ``modified_after``/``modified_before`` (ISO-8601, normalized to
    UTC) restrict results to items modified within a [after, before] range
    (inclusive on both ends).
    """
    # Build the query blob once.
    qblob = _vec_blob(query_vec)
    # Over-fetch: KNN returns the k nearest across ALL projects; we then
    # filter to this project. A large project (5000+ items) needs a generous
    # k so enough same-project candidates survive. k = limit * 8 handles
    # projects up to ~40k items (8x over-fetch).
    k = max(limit * 8, 200)

    q = """
        SELECT c.chunk_id, c.item_id, c.project_id, c.item_type,
               c.item_type_name, c.document_key, c.name, c.status,
               c.section, c.chunk_index, c.text, c.modified_date,
               knn.distance AS score
        FROM (
            SELECT chunk_id, distance
            FROM chunks_vec
            WHERE embedding MATCH ? AND k = ?
        ) knn
        JOIN chunks c ON c.chunk_id = knn.chunk_id
        WHERE c.project_id = ?
    """
    params: list[Any] = [qblob, k, project_id]
    if item_type is not None:
        q += " AND c.item_type = ?"
        params.append(item_type)
    after = _normalize_iso_utc(modified_after)
    if after is not None:
        q += " AND c.modified_date >= ?"
        params.append(after)
    before = _normalize_iso_utc(modified_before)
    if before is not None:
        q += " AND c.modified_date <= ?"
        params.append(before)
    q += " ORDER BY knn.distance ASC LIMIT ?"
    params.append(limit)
    return conn.execute(q, params).fetchall()


def fts_search(conn: sqlite3.Connection, query: str, project_id: int,
               item_type: int | None, limit: int,
               modified_after: str | None = None,
               modified_before: str | None = None) -> list[sqlite3.Row]:
    """Keyword recall via FTS5 (BM25 ranking) with optional time range."""
    q = """
        SELECT c.chunk_id, c.item_id, c.project_id, c.item_type,
               c.item_type_name, c.document_key, c.name, c.status,
               c.section, c.chunk_index, c.text, c.modified_date,
               bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ? AND chunks_fts.project_id = ?
    """
    params: list[Any] = [query, project_id]
    if item_type is not None:
        q += " AND chunks_fts.item_type = ?"
        params.append(item_type)
    after = _normalize_iso_utc(modified_after)
    if after is not None:
        q += " AND c.modified_date >= ?"
        params.append(after)
    before = _normalize_iso_utc(modified_before)
    if before is not None:
        q += " AND c.modified_date <= ?"
        params.append(before)
    q += " ORDER BY bm25(chunks_fts) LIMIT ?"
    params.append(limit)
    return conn.execute(q, params).fetchall()


def fetch_chunks_by_ids(conn: sqlite3.Connection,
                        chunk_ids: Iterable[str]) -> list[sqlite3.Row]:
    ids = list(chunk_ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    return conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
    ).fetchall()


def count_chunks(conn: sqlite3.Connection, project_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE project_id=?", (project_id,)
    ).fetchone()[0]


def count_items(conn: sqlite3.Connection, project_id: int) -> int:
    """Total indexed items for a project (the ``items`` table row count).

    Used to report a project's true ``item_count`` after a sync. A full init
    indexes every item, but an incremental sync only re-processes MODIFIED
    items — so the per-run ``done`` counter is NOT the project total. Reading
    the persisted item count keeps ``get_sync_status`` accurate after both
    kinds of run.
    """
    return conn.execute(
        "SELECT COUNT(*) FROM items WHERE project_id=?", (project_id,)
    ).fetchone()[0]


def indexed_item_ids(conn: sqlite3.Connection, project_id: int) -> set[int]:
    """Return the set of item_ids that already have VECTORS indexed.

    Used by resume-mode sync to skip items that were fully indexed (text +
    FTS + vec) before an interruption — only items WITHOUT vectors are
    re-fetched and re-embedded.

    Reads from ``chunks_vec`` (the vector index), NOT ``chunks`` (the text
    table). This is the critical distinction: ``replace_chunks`` writes
    chunks + FTS + vec in one transaction, but if a prior sync was
    interrupted after the chunks table write but before vec insertion (or
    if vec insertion failed silently), the chunks table would have rows
    while chunks_vec would not. Checking chunks_vec ensures we only skip
    items whose vectors are truly present — otherwise resume would skip
    items that lack vectors, leaving the vector index permanently incomplete
    (vector search returns empty → zero recall).
    """
    return {
        row[0] for row in conn.execute(
            "SELECT DISTINCT c.item_id FROM chunks_vec v "
            "JOIN chunks c ON c.chunk_id = v.chunk_id "
            "WHERE c.project_id=?",
            (project_id,)
        )
    }


def indexed_item_ids(conn: sqlite3.Connection, project_id: int) -> set[int]:
    """Return the set of item_ids that already have chunks indexed.

    Used by the resume mode of ``_sync_project_locked`` to skip items whose
    text + FTS + vector data is already written — so an interrupted sync
    doesn't re-fetch and re-embed the whole project. An item is "indexed"
    when it has at least one row in ``chunks`` (the unit of retrieval); items
    with metadata-only rows in ``items`` but no chunks are NOT skipped, so a
    partial write (item upserted but chunks not yet written) is retried.

    ``idx_chunks_item`` (on ``chunks.item_id``) makes this a cheap indexed scan.
    """
    return {row[0] for row in conn.execute(
        "SELECT DISTINCT item_id FROM chunks WHERE project_id=?", (project_id,)
    )}


def create_job(conn: sqlite3.Connection, job_id: str, project_id: int,
               kind: str) -> None:
    with write_txn(conn):
        conn.execute(
            "INSERT INTO sync_jobs (job_id, project_id, kind, status, "
            "started_at) VALUES (?,?,?,?, datetime('now'))",
            (job_id, project_id, kind, "RUNNING"))


def update_job(conn: sqlite3.Connection, job_id: str, *,
               status: str | None = None, progress: float | None = None,
               total: int | None = None, done: int | None = None,
               message: str | None = None) -> None:
    sets, vals = [], []
    if status is not None:
        sets.append("status=?"); vals.append(status)
    if progress is not None:
        sets.append("progress=?"); vals.append(progress)
    if total is not None:
        sets.append("total=?"); vals.append(total)
    if done is not None:
        sets.append("done=?"); vals.append(done)
    if message is not None:
        sets.append("message=?"); vals.append(message)
    if status in ("DONE", "ERROR"):
        sets.append("finished_at=datetime('now')")
    if not sets:
        return
    vals.append(job_id)
    with write_txn(conn):
        conn.execute(f"UPDATE sync_jobs SET {','.join(sets)} WHERE job_id=?", vals)


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM sync_jobs WHERE job_id=?", (job_id,)
    ).fetchone()


def get_active_job_for_project(conn: sqlite3.Connection,
                               project_id: int) -> sqlite3.Row | None:
    """Return the most recent non-terminal job for a project, or None.

    A job is "active" while in PENDING or RUNNING status. Used by
    ``init_jama_project`` to refuse a duplicate concurrent sync for a project
    that already has one in flight (prevents racing upserts and split-brain
    terminal states). Call within a write_txn for a consistent read.
    """
    return conn.execute(
        "SELECT * FROM sync_jobs WHERE project_id=? AND status IN "
        "('PENDING','RUNNING') ORDER BY started_at DESC LIMIT 1",
        (project_id,)
    ).fetchone()


def get_latest_job_for_project(conn: sqlite3.Connection, project_id: int,
                               kind: str | None = None) -> sqlite3.Row | None:
    """Most recent job of a given kind for a project (terminal or in-flight).

    Unlike :func:`get_active_job_for_project` (which only finds non-terminal
    jobs), this returns the latest job row regardless of status — used by the
    ``get_sync_status`` monitor to report each operation's last run. Pass
    ``kind`` to scope to ``"init"`` / ``"reinit"`` / ``"sync"``; ``None`` returns
    the latest job of any kind.
    """
    if kind:
        return conn.execute(
            "SELECT * FROM sync_jobs WHERE project_id=? AND kind=? "
            "ORDER BY started_at DESC LIMIT 1",
            (project_id, kind)
        ).fetchone()
    return conn.execute(
        "SELECT * FROM sync_jobs WHERE project_id=? "
        "ORDER BY started_at DESC LIMIT 1",
        (project_id,)
    ).fetchone()


def reconcile_stale_jobs(conn: sqlite3.Connection) -> int:
    """Mark every PENDING/RUNNING job as ERROR (interrupted by restart).

    Called once at startup, before any worker thread is running, so any
    non-terminal job row is a crash leftover from a prior process — its worker
    is gone and will never advance it. Without this, ``get_active_job_for_project``
    would forever report a phantom RUNNING job and the monitor would look stuck.

    Returns the number of rows reconciled. Project-level recovery (re-queuing
    INITIALIZING projects) is handled separately by ``_resume_interrupted_syncs``
    in server.py; a reconciled ``sync`` job belongs to a READY project that the
    next scheduled tick will catch up.
    """
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE sync_jobs SET status='ERROR', "
            "message=COALESCE(message,'')||' [interrupted by restart]', "
            "finished_at=datetime('now') WHERE status IN ('PENDING','RUNNING')"
        )
    return cur.rowcount
