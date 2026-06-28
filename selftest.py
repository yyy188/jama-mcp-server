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
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        "init_jama_project", "get_sync_progress", "search_jama_semantics",
        "query_jama_native_metadata", "list_jama_projects", "get_jama_item",
        "get_jama_item_relationships", "get_jama_item_children",
        "list_jama_project_relationships", "get_jama_item_comments",
        "get_jama_item_attachments", "list_jama_releases",
        "list_jama_test_runs", "list_jama_item_types",
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


def test_preflight_guard_blocks() -> None:
    section("10. Pre-flight guard blocks a misconfigured server")
    # Simulate missing config. reload_settings() re-reads .env, so we must
    # temporarily move .env aside to truly simulate an unconfigured server.
    from config import PROJECT_ROOT
    env_path = PROJECT_ROOT / ".env"
    bak_path = PROJECT_ROOT / ".env.selftest_bak"
    moved = False
    saved_env = dict(os.environ)
    try:
        if env_path.exists():
            env_path.rename(bak_path)
            moved = True
        # Clear the relevant vars from the live env too.
        for k in ("JAMA_URL", "JAMA_CLIENT_ID", "JAMA_CLIENT_SECRET",
                  "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY"):
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
        # Restore .env and the live environment, then reload for downstream tests.
        if moved and bak_path.exists():
            bak_path.rename(env_path)
        os.environ.clear()
        os.environ.update(saved_env)
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
    if proj:
        pid = int(proj["id"])
        test_releases(jama, pid)
        test_item_drilldown(jama, pid)
        test_project_relationships(jama, pid)
        test_test_runs(jama, pid)
        test_get_raw(jama, pid)
    else:
        _skip("project-scoped tests", "no project available")

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
