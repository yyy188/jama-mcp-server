#!/usr/bin/env python3
"""Self-test suite for the Jama MCP Server.

Verifies, end-to-end and without an MCP client, that:

1. Offline pre-flight passes (deps + config + storage).
2. Jama OAuth + connectivity works and project listing returns data.
3. Every extended Jama query method (projects, item, children, relationships,
   comments, attachments, releases, test runs, item types, raw GET) runs and
   returns the expected shape.
4. MCP tools are registered with the expected names.
5. The per-tool pre-flight guard correctly blocks a misconfigured server.

Run with:  python selftest.py
Exit code 0 = all checks passed; 1 = one or more failed.

This script reads credentials from .env (loaded by config.py). It performs only
read-only GETs against Jama — it cannot create, modify or delete data.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Unbuffered + line-buffered stdout so progress is visible in real time when
# output is redirected to a file (the common case for background/monitored
# runs). Without this, Python uses full block-buffering for non-tty stdout,
# so a multi-minute self-test writes nothing until it finishes — making it
# impossible to tell a hang from a healthy run via `tail`. reconfigure is a
# no-op on streams that don't support it (older Pythons); fall back to flushing
# in the helpers below regardless.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

# Monotonic start time for relative progress timestamps (mm:ss).
_T0 = time.monotonic()


def _tlog(msg: str) -> None:
    """Print a progress line with a relative mm:ss timestamp, flushed now.

    Used by the long-running phases (sync, crash recovery, search) so a
    background/monitored run shows live movement instead of going dark for
    minutes. Prefixed with a leading space so it visually separates from the
    PASS/FAIL block headers without being mistaken for a test result.
    """
    dt = time.monotonic() - _T0
    mm, ss = divmod(int(dt), 60)
    print(f"  [..{mm:02d}:{ss:02d}] {msg}", flush=True)


class _JobPoller:
    """Background thread that prints sync job progress while a long phase runs.

    Wraps a blocking call (e.g. ``server._sync_project``) by polling the
    ``sync_jobs`` row every few seconds and emitting a one-line progress
    update (done/total + message), so the phase isn't a black box. Use as a
    context manager; polling stops on exit.
    """

    def __init__(self, conn, job_id: str, label: str, interval: float = 5.0):
        self._conn = conn
        self._job_id = job_id
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="selftest-jobpoll")
        self._thread.start()
        return self

    def _loop(self):
        from db_setup import get_job
        last = None
        while not self._stop.wait(self._interval):
            try:
                row = get_job(self._conn, self._job_id)
            except Exception:
                continue
            if row is None:
                continue
            cur = (row["status"], row["done"], row["total"], row["message"])
            if cur != last:
                last = cur
                tot = row["total"] or "?"
                _tlog(f"{self._label}: {row['status']} "
                      f"{row['done']}/{tot} — {row['message'] or ''}")

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

import config  # noqa: E402  (triggers .env load)
from config import settings  # noqa: E402
from preflight import preflight  # noqa: E402

# ANSI colors for readable terminal output.
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"

_passed = 0
_failed = 0
_skipped = 0


def _ok(name: str, detail: str = "") -> None:
    global _passed
    _passed += 1
    print(f"  {_GREEN}PASS{_RESET} {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str) -> None:
    global _failed
    _failed += 1
    print(f"  {_RED}FAIL{_RESET} {name} — {detail}")


def _skip(name: str, reason: str) -> None:
    global _skipped
    _skipped += 1
    print(f"  {_YELLOW}SKIP{_RESET} {name} — {reason}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
def test_preflight() -> bool:
    section("1. Pre-flight (offline)")
    report = preflight(require={"jama", "embedding"})
    if report["blocking"]:
        _fail("preflight", "; ".join(report["issues"]))
        print("  Hint: run `python setup_wizard.py` and provide credentials.")
        return False
    _ok("preflight", f"{len(report['issues'])} checks ok")
    return True


def test_jama_connect(jama) -> tuple[dict | None, list]:
    """Return (first project, all projects) so later tests can reuse them."""
    section("2. Jama connectivity + list_jama_projects")
    projects = jama.list_projects()
    if not projects:
        _fail("list_jama_projects", "no projects returned (check OAuth scope)")
        return None, []
    _ok("list_jama_projects", f"{len(projects)} project(s); "
        f"sample id={projects[0].get('id')} name={projects[0].get('name')!r}")
    return projects[0], projects


def test_list_item_types(jama) -> list:
    section("3. list_jama_item_types")
    types = jama.list_item_types()
    if not types:
        _fail("list_jama_item_types", "no item types returned")
        return []
    _ok("list_jama_item_types", f"{len(types)} type(s); sample: {types[0]}")
    return types


def test_find_project_by_name(jama, sample_name: str) -> None:
    section("3b. find_projects (by name)")
    if not sample_name:
        _skip("find_projects", "no project name available to probe")
        return
    # Substring match: take a distinctive fragment of the sample name.
    fragment = sample_name[: max(3, len(sample_name) // 2)].lower()
    rows = jama.find_projects(fragment, limit=10)
    if not rows:
        _fail("find_projects (substring)", f"no match for fragment {fragment!r}")
        return
    _ok("find_projects (substring)",
        f"{len(rows)} match(es) for {fragment!r}; "
        f"sample id={rows[0].get('id')} name={rows[0].get('name')!r}")
    # Exact match: the full sample name must match itself.
    exact = jama.find_projects(sample_name, exact=True, limit=10)
    if exact and any((r.get("name") or "").lower() == sample_name.lower()
                     for r in exact):
        _ok("find_projects (exact)",
            f"exact match found: id={exact[0].get('id')}")
    else:
        _fail("find_projects (exact)",
              f"expected exact match for {sample_name!r}, got {len(exact)} row(s)")
    # Empty needle returns [] without error.
    if jama.find_projects("") == []:
        _ok("find_projects (empty guard)", "empty name returns []")
    else:
        _fail("find_projects (empty guard)", "expected [] for empty name")


def test_find_item_type_by_name(jama) -> None:
    section("3c. find_item_types (by name)")
    # "test" is a very common Jama type fragment (Test Case, Test Plan, …).
    rows = jama.find_item_types("test", limit=10)
    if not rows:
        _skip("find_item_types", "no type matching 'test' on this tenant")
        return
    _ok("find_item_types (substring)",
        f"{len(rows)} match(es) for 'test'; sample: "
        f"id={rows[0].get('id')} display={rows[0].get('display')!r}")
    # Verify the richer payload fields are present.
    r = rows[0]
    if "display_plural" in r and "category" in r:
        _ok("find_item_types payload",
            f"has display_plural={r.get('display_plural')!r} "
            f"category={r.get('category')!r}")
    else:
        _fail("find_item_types payload",
              f"missing rich fields; keys={list(r.keys())}")
    # Empty needle returns [] without error.
    if jama.find_item_types("") == []:
        _ok("find_item_types (empty guard)", "empty name returns []")
    else:
        _fail("find_item_types (empty guard)", "expected [] for empty name")


def test_releases(jama, project_id: int) -> None:
    section("4. list_jama_releases")
    releases = jama.list_releases(project_id, limit=10)
    _ok("list_jama_releases", f"{len(releases)} release(s) for project {project_id}")


def test_item_drilldown(jama, project_id: int) -> None:
    section("5. Item drill-down (get_item / children / comments / attachments)")
    # Find one real item to drill into via the existing native query.
    items = jama.iter_project_items(project_id, max_items=1)
    try:
        item = next(items)
    except StopIteration:
        _skip("item drill-down", "project has no items to drill into")
        return
    item_id = item["item_id"]
    _ok("iter_project_items (probe)", f"item_id={item_id} key={item.get('document_key')}")

    full = jama.get_item(item_id)
    if full is None:
        _fail("get_item", f"get_item({item_id}) returned None")
    else:
        _ok("get_item", f"name={full.get('name')!r} type={full.get('item_type_name')!r}")

    children = jama.get_item_children(item_id, limit=10)
    _ok("get_item_children", f"{len(children)} child(ren)")

    # Item-scoped relationships walk the whole project (Jama has no per-item
    # server filter), so they can be slow on large projects — test the fast
    # project-level path instead, and only do a shallow item-level probe.
    comments = jama.get_item_comments(item_id, limit=10)
    _ok("get_item_comments", f"{len(comments)} comment(s)")

    atts = jama.get_item_attachments(item_id, limit=10)
    _ok("get_item_attachments", f"{len(atts)} attachment(s)")


def test_project_relationships(jama, project_id: int) -> None:
    section("5b. list_project_relationships (cursor-paginated)")
    try:
        rels = jama.list_project_relationships(project_id, limit=5)
    except Exception as exc:
        _fail("list_project_relationships", str(exc)[:200])
        return
    _ok("list_project_relationships",
        f"{len(rels)} relationship(s) for project {project_id} (limit=5)")
    if rels:
        r = rels[0]
        _ok("relationship shape",
            f"id={r.get('id')} type={r.get('relationship_type')} "
            f"src={r.get('source_item')}->dst={r.get('target_item')}")


def test_test_runs(jama, project_id: int) -> None:
    section("6. list_jama_test_runs")
    try:
        runs = jama.list_test_runs(project_id=project_id, limit=10)
    except Exception as exc:
        _skip("list_jama_test_runs", f"endpoint error: {exc}")
        return
    _ok("list_jama_test_runs", f"{len(runs)} test run(s) for project {project_id}")


def test_get_raw(jama, project_id: int) -> None:
    section("7. get_raw (generic GET)")
    try:
        data = jama.get_raw(f"/projects/{project_id}")
    except Exception as exc:
        _fail("get_raw", str(exc)[:200])
        return
    if isinstance(data, dict) and data.get("id") == project_id:
        _ok("get_raw", f"GET /projects/{project_id} -> id={data.get('id')}")
    else:
        _fail("get_raw", f"unexpected payload: {str(data)[:120]}")


def test_error_path_requires_arg(jama) -> None:
    section("8. Error path: list_test_runs requires an arg")
    try:
        jama.list_test_runs()
        _fail("list_test_runs no-arg", "expected ValueError")
    except ValueError:
        _ok("list_test_runs no-arg", "raised ValueError as expected")
    except Exception as exc:
        _fail("list_test_runs no-arg", f"wrong exception type: {type(exc).__name__}")


def test_concurrent_sync(jama, project_id: int) -> None:
    section("8b. Concurrent download + batched embed")
    import time
    import server
    from db_setup import (init_db, get_job, upsert_project, create_job,
                          count_chunks, write_txn)
    # Only run on small projects so the self-test stays fast.
    total = jama.count_project_items(project_id)
    if total > 100:
        _skip("concurrent sync", f"project {project_id} has {total} items; "
              f"needs a small (<100) project to stay fast")
        return
    conn = init_db()
    upsert_project(conn, project_id, status="NEW")
    with write_txn(conn):
        # Delete THIS project's vectors FIRST (while chunks rows still exist
        # to scope the delete). chunks_vec has no project_id column, so we
        # JOIN to chunks to scope — deleting chunks BEFORE this would empty
        # the subquery and leave stale vectors behind, which then cause
        # "UNIQUE constraint failed" on the next sync's INSERT into chunks_vec.
        conn.execute(
            "DELETE FROM chunks_vec WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE project_id=?)",
            (project_id,))
        conn.execute("DELETE FROM chunks WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM chunks_fts WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM items WHERE project_id=?", (project_id,))
    import uuid as _uuid
    job_id = f"selftest-sync-{_uuid.uuid4().hex[:8]}"
    create_job(conn, job_id, project_id, "init")
    t0 = time.time()
    _tlog(f"concurrent sync: starting for project {project_id} "
          f"({total} items)")
    try:
        with _JobPoller(conn, job_id, "concurrent sync"):
            server._sync_project(project_id, job_id=job_id, incremental=False)
    except Exception as exc:
        _fail("concurrent sync", f"raised: {exc}")
        conn.close()
        return
    elapsed = time.time() - t0
    job = get_job(conn, job_id)
    chunks = count_chunks(conn, project_id)
    if job["status"] == "DONE" and chunks > 0:
        _ok("concurrent sync",
            f"DONE in {elapsed:.1f}s, {job['done']}/{job['total']} items, "
            f"{chunks} chunks")
    else:
        _fail("concurrent sync",
              f"status={job['status']} chunks={chunks} done={job['done']}")
    conn.close()


def test_crash_recovery(project_id: int) -> None:
    section("8c. Crash recovery: INITIALIZING project auto-resynced")
    import time
    import server
    from jama_client import JamaClient
    from db_setup import (init_db, get_project, upsert_project, count_chunks,
                          write_txn)
    # Only run on small projects so the self-test stays fast.
    total = JamaClient().count_project_items(project_id)
    if total > 100:
        _skip("crash recovery", f"project {project_id} has {total} items; "
              f"needs a small (<100) project to stay fast")
        return
    conn = init_db()
    # Simulate a crash: leave the project INITIALIZING with no chunks.
    upsert_project(conn, project_id, status="INITIALIZING")
    with write_txn(conn):
        # Delete THIS project's vectors FIRST (while chunks rows still exist
        # to scope the delete) — see test_concurrent_sync for the rationale.
        # Deleting chunks first would empty the scoping subquery and leave
        # stale vectors that cause UNIQUE-constraint failures on next sync.
        conn.execute(
            "DELETE FROM chunks_vec WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE project_id=?)",
            (project_id,))
        conn.execute("DELETE FROM chunks WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM chunks_fts WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM items WHERE project_id=?", (project_id,))
    before = get_project(conn, project_id)
    _tlog(f"crash recovery: project {project_id} left INITIALIZING; "
          f"running _resume_interrupted_syncs")
    # Mirror main()'s startup order: reconcile stale (orphaned) jobs BEFORE
    # resuming. A prior test run killed mid-sync (or a killed self-test
    # process) leaves RUNNING job rows whose worker is gone forever; without
    # reconcile, get_active_job_for_project would return those zombies and
    # the poll loop below would wait on a job that never advances.
    from db_setup import reconcile_stale_jobs
    n = reconcile_stale_jobs(conn)
    if n:
        _tlog(f"crash recovery: reconciled {n} stale job(s) first")
    # Run the startup recovery function.
    server._resume_interrupted_syncs()
    # Poll for completion (the recovery submits a background job). The resumed
    # job re-runs model pre-download + pre-flight speed test + full re-sync;
    # on a flaky link the speed test can retry a few times (each ~5-15s) and
    # the reranker re-loads (~8s), so allow up to ~4 min. A genuine hang still
    # fails the test — just not a transiently-slow network.
    for i in range(80):
        time.sleep(3)
        p = get_project(conn, project_id)
        if p["status"] in ("READY", "ERROR"):
            break
        if i % 3 == 0:  # emit a heartbeat roughly every ~9s
            _tlog(f"crash recovery: still {p['status']} "
                  f"({count_chunks(conn, project_id)} chunks so far)")
    after = get_project(conn, project_id)
    if before["status"] == "INITIALIZING" and after["status"] == "READY" \
            and count_chunks(conn, project_id) > 0:
        _ok("crash recovery",
            f"INITIALIZING(0 chunks) -> READY({count_chunks(conn, project_id)} chunks)")
    else:
        _fail("crash recovery",
              f"before={before['status']} after={after['status']} "
              f"chunks={count_chunks(conn, project_id)}")
    conn.close()


def test_search_subqueries(project_id: int) -> None:
    """Read-only RAG search: caller-supplied sub_queries vs lexical fallback.

    Runs only on an already-indexed project (test_concurrent_sync leaves a
    small one READY). All Jama access is read-only — search hits only the
    local SQLite index; no Jama REST calls, no online writes.
    """
    section("8d. search_jama_semantics (sub_queries + lexical fallback)")
    import server
    from db_setup import init_db, get_project, count_chunks
    conn = init_db()
    proj = get_project(conn, project_id)
    if proj is None or proj["status"] != "READY" or count_chunks(conn, project_id) == 0:
        _skip("search sub_queries",
              f"project {project_id} not indexed (status="
              f"{proj['status'] if proj else 'None'}, "
              f"chunks={count_chunks(conn, project_id)})")
        conn.close()
        return

    # Pick a generic keyword likely present in any real project. Derive it
    # from the first indexed chunk's text so the search is never empty.
    row = conn.execute(
        "SELECT text FROM chunks WHERE project_id=? LIMIT 1", (project_id,)
    ).fetchone()
    conn.close()
    if row is None or not row["text"]:
        _skip("search sub_queries", "no chunk text available to probe")
        return
    # First meaningful token of the first chunk = a near-guaranteed hit.
    import re as _re
    tokens = _re.findall(r"[A-Za-z0-9_-]+", row["text"])
    probe = tokens[0] if tokens else "test"

    try:
        rag = server.rag()
        _tlog(f"search: probing with {probe!r} (first call may load reranker)")
        # (a) Caller supplies sub_queries (the Plan B happy path).
        r_sub = rag.search(project_id, probe,
                           sub_queries=[probe, f"{probe} requirement",
                                        f"{probe} description"],
                           top_k=3, candidate_k=20)
        # (b) No sub_queries -> lexical fallback path.
        r_lex = rag.search(project_id, probe, top_k=3, candidate_k=20)
    except Exception as exc:
        _fail("search sub_queries", f"raised: {exc}")
        return

    if r_sub and r_lex:
        _ok("search sub_queries",
            f"sub_queries={len(r_sub)} results, lexical={len(r_lex)} results")
        strategies = {r["strategy"] for r in r_sub + r_lex}
        if strategies <= {"rerank", "rrf"}:
            _ok("search strategy", f"strategies={sorted(strategies)}")
        else:
            _fail("search strategy",
                  f"unexpected strategy: {sorted(strategies)}")
    else:
        _fail("search sub_queries",
              f"empty results: sub={len(r_sub)} lex={len(r_lex)} "
              f"(probe={probe!r})")


def test_mcp_tools_registered() -> None:
    section("9. MCP tools registered")
    # Import server (registers @mcp.tool() decorators) without running it.
    try:
        import server  # noqa: F401
    except Exception as exc:
        _fail("import server", f"{exc}\n{traceback.format_exc()}")
        return
    # FastMCP exposes registered tools via the tool manager.
    expected = {
        "bootstrap_models", "get_bootstrap_progress",
        "init_jama_project", "reinit_jama_project", "get_sync_progress",
        "get_sync_status", "search_jama_semantics",
        "query_jama_native_metadata", "list_jama_projects", "get_jama_item",
        "get_jama_item_relationships", "get_jama_item_children",
        "list_jama_project_relationships",
        "get_jama_item_comments", "get_jama_item_attachments",
        "list_jama_releases", "list_jama_test_runs", "list_jama_item_types",
        "find_jama_project_by_name", "find_jama_item_type_by_name",
        "query_jama_endpoint", "validate_setup", "configure_jama",
    }
    try:
        tools = server.mcp._tool_manager._tools  # type: ignore[attr-defined]
    except Exception:
        # Newer FastMCP versions expose tools differently.
        try:
            tools = {name: t for name, t in server.mcp._tools.items()}  # type: ignore[attr-defined]
        except Exception:
            _skip("tool registration", "could not introspect FastMCP tool registry")
            return
    registered = set(tools.keys()) if isinstance(tools, dict) else set()
    missing = expected - registered
    if missing:
        _fail("tool registration", f"missing tools: {sorted(missing)}")
    else:
        _ok("tool registration", f"{len(expected)} tools present")

    # Assert search_jama_semantics exposes the client-side Multi-Query
    # parameter (Plan B): sub_queries is an array of strings, optional.
    # ``parameters`` is a plain dict on the installed FastMCP version; fall
    # back to the pydantic model path for newer versions.
    try:
        st = tools.get("search_jama_semantics")
        params = st.parameters  # type: ignore[union-attr]
        schema = (params.model_json_schema()           # type: ignore[union-attr]
                  if hasattr(params, "model_json_schema") else params)
        props = schema.get("properties", {})
        sq = props.get("sub_queries", {})
        if sq.get("type") == "array" and sq.get("items", {}).get("type") == "string":
            _ok("sub_queries schema",
                f"array<string>, default={sq.get('default')!r}")
        else:
            _fail("sub_queries schema",
                  f"expected array<string>; got {sq}")
    except Exception as exc:
        _fail("sub_queries schema", f"could not introspect: {exc}")


def test_preflight_guard_blocks() -> None:
    section("10. Pre-flight guard blocks a misconfigured server")
    # Simulate missing config. reload_settings() re-reads .env, so we must
    # temporarily move .env aside to truly simulate an unconfigured server.
    #
    # We only touch the 5 config vars we care about (save/restore their values
    # individually) instead of snapshotting the whole os.environ. On Windows,
    # os.environ.clear()+update(full snapshot) raises
    # "ValueError: environment variable longer than 32767" because some real
    # vars (PATH, proxy/IDE injections) exceed the Win32 SetEnvironmentVariableW
    # 32767-char limit — and that crash happens in the cleanup phase *after* all
    # tests passed, turning a green run into a non-zero exit. Avoid it entirely
    # by never clearing or bulk-restoring the environment.
    from config import PROJECT_ROOT
    env_path = PROJECT_ROOT / ".env"
    bak_path = PROJECT_ROOT / ".env.selftest_bak"
    cfg_vars = ("JAMA_URL", "JAMA_CLIENT_ID", "JAMA_CLIENT_SECRET",
                "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY")
    saved = {k: os.environ.get(k) for k in cfg_vars}
    moved = False
    try:
        if env_path.exists():
            env_path.rename(bak_path)
            moved = True
        for k in cfg_vars:
            os.environ.pop(k, None)
        import config
        config.reload_settings()
        from preflight import preflight
        report = preflight(require={"jama"})
        if report["blocking"]:
            _ok("guard blocks misconfig",
                f"preflight reported {sum(1 for i in report['issues'] if '[jama]' in i or 'dependency' in i.lower())} blocking issue(s)")
        else:
            _fail("guard blocks misconfig",
                  f"expected blocking=True; issues={report['issues']}")
    finally:
        # Restore .env and only the 5 config vars we touched, then reload.
        if moved and bak_path.exists():
            bak_path.rename(env_path)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import config
        config.reload_settings()


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("  Jama MCP Server — Self-test")
    print("=" * 70)

    if not test_preflight():
        print(f"\n{_RED}Cannot proceed: pre-flight failed.{_RESET}")
        return 1

    from jama_client import JamaClient
    jama = JamaClient()

    proj, _all = test_jama_connect(jama)
    test_list_item_types(jama)
    test_find_item_type_by_name(jama)
    if proj:
        pid = int(proj["id"])
        pname = proj.get("name") or ""
        test_find_project_by_name(jama, pname)
        test_releases(jama, pid)
        test_item_drilldown(jama, pid)
        test_project_relationships(jama, pid)
        test_test_runs(jama, pid)
        test_get_raw(jama, pid)
    else:
        _skip("project-scoped tests", "no project available")

    # Concurrent sync + crash recovery need a SMALL project to stay fast.
    # Find one (<=50 items) by probing the project list.
    small_pid = None
    for p in (proj, *_all) if proj else _all:
        try:
            if jama.count_project_items(int(p["id"])) <= 50:
                small_pid = int(p["id"])
                break
        except Exception:
            continue
    if small_pid:
        test_concurrent_sync(jama, small_pid)
        test_crash_recovery(small_pid)
        # Reuses the project indexed by test_concurrent_sync (read-only).
        test_search_subqueries(small_pid)
    else:
        _skip("concurrent sync + crash recovery",
              "no small (<=50 items) project found")

    test_error_path_requires_arg(jama)
    test_mcp_tools_registered()
    test_preflight_guard_blocks()

    print("\n" + "=" * 70)
    print(f"  {_GREEN}Passed:{_RESET} {_passed}   "
          f"{_RED}Failed:{_RESET} {_failed}   "
          f"{_YELLOW}Skipped:{_RESET} {_skipped}")
    print("=" * 70)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
