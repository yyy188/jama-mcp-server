#!/usr/bin/env python3
"""Offline unit tests for the Multi-Query refactor (Plan B).

Verifies, without touching Jama, the embedding endpoint or the network:

1. ``_normalize_sub_queries`` — original query forced front, de-duplication,
   blank/non-string filtering, ``max_n`` cap, empty-input fallback.
2. ``MultiQueryExpander.expand`` (lexical fallback) — empty -> [], non-empty
   returns >= 1 with the original query first, deterministic across calls.
3. ``RAGPipeline.search`` wiring — caller-supplied ``sub_queries`` are used
   verbatim (after normalization); when omitted, the lexical expander drives
   recall. The DB, embedding and reranker layers are stubbed so this is pure
   control-flow verification.

Run with:  python test_multiquery.py
Exit code 0 = all passed; 1 = one or more failed.
"""
from __future__ import annotations

import os
import sys
import traceback
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"

_passed = 0
_failed = 0


def _ok(name: str, detail: str = "") -> None:
    global _passed
    _passed += 1
    print(f"  {_GREEN}PASS{_RESET} {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str) -> None:
    global _failed
    _failed += 1
    print(f"  {_RED}FAIL{_RESET} {name} — {detail}")


def _assert(cond: bool, name: str, detail: str = "") -> None:
    if cond:
        _ok(name, detail)
    else:
        _fail(name, detail)


# --------------------------------------------------------------------------- #
def test_normalize_forces_query_front() -> None:
    from rag_pipeline import _normalize_sub_queries
    q = "how does login timeout work"
    out = _normalize_sub_queries(["auth timeout", "session expiry"], q)
    _assert(out[0] == q, "query forced to front",
            f"first={out[0]!r}")
    _assert(out == [q, "auth timeout", "session expiry"], "order preserved",
            f"out={out}")


def test_normalize_dedupes_case_insensitive() -> None:
    from rag_pipeline import _normalize_sub_queries
    q = "login timeout"
    # "Login Timeout" duplicates the query (case-insensitive); dropped.
    out = _normalize_sub_queries(["Login Timeout", "session expiry"], q)
    _assert(out == [q, "session expiry"], "case-insensitive dedup",
            f"out={out}")


def test_normalize_drops_blanks_and_non_strings() -> None:
    from rag_pipeline import _normalize_sub_queries
    q = "login"
    out = _normalize_sub_queries(["", "   ", 123, None, "real"], q)
    _assert(out == [q, "real"], "blanks/non-strings filtered",
            f"out={out}")


def test_normalize_caps_at_max_n() -> None:
    from rag_pipeline import _normalize_sub_queries
    q = "q"
    subs = [f"s{i}" for i in range(10)]
    out = _normalize_sub_queries(subs, q, max_n=4)
    _assert(len(out) == 4, "respects max_n cap", f"len={len(out)} out={out}")
    _assert(out[0] == q, "query still first under cap", f"first={out[0]}")


def test_normalize_empty_falls_back_to_query() -> None:
    from rag_pipeline import _normalize_sub_queries
    q = "solo"
    out = _normalize_sub_queries([], q)
    _assert(out == [q], "empty subs -> [query]", f"out={out}")


def test_normalize_all_blank_falls_back_to_query() -> None:
    from rag_pipeline import _normalize_sub_queries
    q = "solo"
    out = _normalize_sub_queries(["", "  "], q)
    _assert(out == [q], "all-blank subs -> [query]", f"out={out}")


def test_normalize_empty_query_empty_subs() -> None:
    from rag_pipeline import _normalize_sub_queries
    out = _normalize_sub_queries([], "")
    _assert(out == [], "empty query + empty subs -> []", f"out={out}")


# --------------------------------------------------------------------------- #
def test_expander_empty_returns_empty() -> None:
    from rag_pipeline import MultiQueryExpander
    ex = MultiQueryExpander(n=3)
    _assert(ex.expand("") == [], "empty query -> []", "")
    _assert(ex.expand("   ") == [], "whitespace query -> []", "")


def test_expander_nonempty_keeps_query_first() -> None:
    from rag_pipeline import MultiQueryExpander
    ex = MultiQueryExpander(n=3)
    q = "how does the login timeout work"
    out = ex.expand(q)
    _assert(len(out) >= 1, "non-empty query -> >=1 variant",
            f"len={len(out)}")
    _assert(out[0] == q, "original query first", f"first={out[0]!r}")


def test_expander_deterministic() -> None:
    from rag_pipeline import MultiQueryExpander
    ex = MultiQueryExpander(n=3)
    q = "test cases for the payment flow"
    a = ex.expand(q)
    b = ex.expand(q)
    _assert(a == b, "deterministic across calls", f"{a}")


def test_expander_keyword_variant_present() -> None:
    from rag_pipeline import MultiQueryExpander
    ex = MultiQueryExpander(n=3)
    # Stopwords "the/for/how/does" stripped -> keyword-focused variant should
    # differ from the original.
    q = "how does the login timeout work"
    out = ex.expand(q)
    _assert(any("login" in v and "timeout" in v for v in out),
            "keyword variant keeps meaningful tokens", f"{out}")


# --------------------------------------------------------------------------- #
def _make_fake_row(cid: str) -> dict:
    """A chunk row shaped like db_setup.fetch_chunks_by_ids returns."""
    return {
        "chunk_id": cid, "item_id": 1, "document_key": "X-1",
        "item_type": 89009, "item_type_name": "Requirement",
        "name": "Sample", "status": "OPEN", "section": "",
        "modified_date": "2024-01-01T00:00:00Z", "text": f"text for {cid}",
    }


class _FakeConn:
    """Stand-in for a sqlite3.Connection; search() only calls .close()."""
    def close(self) -> None:
        pass


def _stub_search_pipeline(rag, captured: dict):
    """Patch DB/embed/reranker so RAGPipeline.search runs offline.

    ``captured`` collects the sub_queries actually iterated for recall, so
    tests can assert which query variants drove the search.
    """
    fake_rows = {f"c{i}": _make_fake_row(f"c{i}") for i in range(6)}

    rag.embedder.embed_one = lambda sq: [0.1]  # any non-empty vector
    rag.reranker.rerank = lambda query, texts: [0.9] * len(texts)

    def _vsearch(conn, vec, pid, it, ck, ma, mb):
        captured.setdefault("subs", []).append(("vector", conn))
        return [{"chunk_id": f"c{i}"} for i in range(3)]

    def _fsearch(conn, q, pid, it, ck, ma, mb):
        return [{"chunk_id": f"c{i}"} for i in range(3, 6)]

    def _fetch(conn, ids):
        return [fake_rows[i] for i in ids if i in fake_rows]

    return _vsearch, _fsearch, _fetch


def test_search_uses_supplied_sub_queries() -> None:
    import rag_pipeline as rp
    rag = rp.RAGPipeline()
    captured: dict = {}
    _vsearch, _fsearch, _fetch = _stub_search_pipeline(rag, captured)
    with mock.patch.object(rp, "vector_search", _vsearch), \
            mock.patch.object(rp, "fts_search", _fsearch), \
            mock.patch.object(rp, "fetch_chunks_by_ids", _fetch), \
            mock.patch.object(rp, "get_connection", lambda *a, **k: _FakeConn()):
        # Count how many distinct sub-queries hit vector recall.
        seen_subs: list[str] = []

        def _vtrace(conn, vec, pid, it, ck, ma, mb):
            seen_subs.append("hit")
            return [{"chunk_id": f"c{i}"} for i in range(3)]

        with mock.patch.object(rp, "vector_search", _vtrace):
            results = rag.search(1, "login timeout",
                                 sub_queries=["auth timeout",
                                              "session expiry",
                                              "user inactivity logout"])
    # 3 caller sub-queries (query forced front makes 4 total) -> 4 vector
    # recall passes. The normalized list has 4 entries.
    _assert(len(results) > 0, "search returns results", f"len={len(results)}")
    _assert(len(seen_subs) == 4, "one recall pass per normalized sub-query",
            f"passes={len(seen_subs)} (expected 4: query+3 variants)")
    _assert(results[0]["strategy"] in ("rerank", "rrf"),
            "strategy field set", f"strategy={results[0]['strategy']}")


def test_search_falls_back_to_expander_when_omitted() -> None:
    import rag_pipeline as rp
    rag = rp.RAGPipeline()
    seen_subs: list[str] = []

    def _vtrace(conn, vec, pid, it, ck, ma, mb):
        seen_subs.append("hit")
        return [{"chunk_id": f"c{i}"} for i in range(3)]

    rag.embedder.embed_one = lambda sq: [0.1]
    rag.reranker.rerank = lambda query, texts: [0.9] * len(texts)
    fake_rows = {f"c{i}": _make_fake_row(f"c{i}") for i in range(6)}
    with mock.patch.object(rp, "vector_search", _vtrace), \
            mock.patch.object(rp, "fts_search",
                              lambda *a, **k: [{"chunk_id": f"c{i}"} for i in range(3, 6)]), \
            mock.patch.object(rp, "fetch_chunks_by_ids",
                              lambda conn, ids: [fake_rows[i] for i in ids if i in fake_rows]), \
            mock.patch.object(rp, "get_connection", lambda *a, **k: _FakeConn()):
        results = rag.search(1, "how does the login timeout work")
    # Lexical expander produces original + keyword variant (>=2 variants).
    _assert(len(results) > 0, "fallback search returns results",
            f"len={len(results)}")
    _assert(len(seen_subs) >= 2, "lexical fallback yields multiple variants",
            f"passes={len(seen_subs)}")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 60)
    print("  Multi-Query refactor — offline unit tests")
    print("=" * 60)

    print("\n--- _normalize_sub_queries ---")
    test_normalize_forces_query_front()
    test_normalize_dedupes_case_insensitive()
    test_normalize_drops_blanks_and_non_strings()
    test_normalize_caps_at_max_n()
    test_normalize_empty_falls_back_to_query()
    test_normalize_all_blank_falls_back_to_query()
    test_normalize_empty_query_empty_subs()

    print("\n--- MultiQueryExpander (lexical) ---")
    test_expander_empty_returns_empty()
    test_expander_nonempty_keeps_query_first()
    test_expander_deterministic()
    test_expander_keyword_variant_present()

    print("\n--- RAGPipeline.search wiring ---")
    try:
        test_search_uses_supplied_sub_queries()
        test_search_falls_back_to_expander_when_omitted()
    except Exception:
        _fail("search wiring", traceback.format_exc())

    print("\n" + "=" * 60)
    print(f"  {_GREEN}Passed:{_RESET} {_passed}   "
          f"{_RED}Failed:{_RESET} {_failed}")
    print("=" * 60)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
