#!/usr/bin/env python3
"""Recall + Precision test suite for Lyra project (id=20675).

Methodology:
  - Ground truth: FTS5 full-text search with topic keywords returns ALL
    items whose text contains the keywords — this is the "should be found"
    universe for that topic.
  - Search recall: run RAGPipeline.search() with a natural-language query;
    the returned document_keys are the "actually found" set.
  - Recall@k = |search_results ∩ ground_truth| / |ground_truth|
  - Precision@k = |search_results ∩ ground_truth| / k
  - Also measures pure vector recall and pure FTS recall separately to
    diagnose which recall path is weak.

Run: uv run python recall_test.py
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
from db_setup import get_connection, vector_search, fts_search
from rag_pipeline import RAGPipeline, _to_fts_query

PROJECT_ID = 20675

# Test cases: (natural_language_query, [fts_keywords_for_ground_truth], topic)
# The FTS keywords define the ground-truth universe as a UNION — every item
# whose text matches ANY of the keyword variants is "relevant". Including
# synonyms ("three way" + "3 way" + "3-way") avoids false-precision-zero
# caused by different spellings of the same concept.
TEST_CASES = [
    ("how does volume sync work between devices",
     ["volume sync", "volume synchronization", "synchronization of volume"], "volume sync"),
    ("call transfer functionality",
     ["call transfer", "transfer call"], "call transfer"),
    ("music control and media focus",
     ["music control", "music focus", "media focus"], "music control"),
    ("three way call handling",
     ["three way call", "3 way call", "3-way"], "three/3-way call"),
    ("answering incoming call from device",
     ["incoming call", "answer call"], "incoming call"),
    ("mute and unmute from device",
     ["mute unmute", "mute unmut"], "mute"),
    ("PC reboot behavior",
     ["reboot", "restart", "power cycle"], "reboot"),
    ("LE Audio device connection",
     ["LEA", "LE Audio", "LE Audio"], "LE Audio"),
    ("USB device enumeration",
     ["USB", "enumeration"], "USB"),
    ("MS Teams certification",
     ["teams", "MS Teams", "Microsoft Teams"], "MS Teams"),
]


def get_ground_truth(conn, fts_keywords: list[str]) -> set[str]:
    """Get the set of document_keys matching ANY of the keyword variants (union)."""
    gt: set[str] = set()
    for kw in fts_keywords:
        fts_q = _to_fts_query(kw)
        if not fts_q:
            continue
        rows = fts_search(conn, fts_q, PROJECT_ID, None, 5000)
        gt |= {r["document_key"] for r in rows}
    return gt


def run_search(rag, query: str, top_k: int = 5, candidate_k: int = 25) -> list[str]:
    """Run the full RAG search and return document_keys."""
    results = rag.search(PROJECT_ID, query, top_k=top_k, candidate_k=candidate_k)
    return [r["document_key"] for r in results]


def run_vector_only(conn, rag, query: str, limit: int = 25) -> set[str]:
    """Pure vector search recall (no FTS, no rerank)."""
    qvec = rag.embedder.embed_one(query)
    rows = vector_search(conn, qvec, PROJECT_ID, None, limit)
    return {r["document_key"] for r in rows}


def run_fts_only(conn, query: str, limit: int = 25) -> set[str]:
    """Pure FTS search recall (no vector, no rerank)."""
    fts_q = _to_fts_query(query)
    if not fts_q:
        return set()
    rows = fts_search(conn, fts_q, PROJECT_ID, None, limit)
    return {r["document_key"] for r in rows}


def main():
    conn = get_connection()
    rag = RAGPipeline()

    # Verify vec index completeness
    r = conn.execute(
        "SELECT COUNT(*) FROM chunks_vec v JOIN chunks c ON c.chunk_id=v.chunk_id "
        "WHERE c.project_id=?", (PROJECT_ID,)).fetchone()
    vec_count = r[0]
    r = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE project_id=?", (PROJECT_ID,)).fetchone()
    chunk_count = r[0]
    print(f"Lyra: chunks={chunk_count}, vec={vec_count} "
          f"(complete={'YES' if vec_count == chunk_count else 'NO'})")
    if vec_count < chunk_count:
        print(f"WARNING: vector index incomplete ({vec_count}/{chunk_count}), "
              f"recall will be degraded.\n")

    top_k = 5
    candidate_k = 25

    print(f"=== Recall + Precision Test (top_k={top_k}, candidate_k={candidate_k}) ===\n")

    total_recall = 0
    total_precision = 0
    total_vec_recall = 0
    total_fts_recall = 0
    n = len(TEST_CASES)

    for nl_query, fts_keywords_list, topic in TEST_CASES:
        # Ground truth: all items matching ANY keyword variant (union)
        gt = get_ground_truth(conn, fts_keywords_list)
        if not gt:
            print(f"[{topic}] ground truth empty, skipping")
            n -= 1
            continue

        # Full search (vector + FTS + RRF + rerank)
        search_keys = set(run_search(rag, nl_query, top_k, candidate_k))

        # Pure vector recall (candidate_k limit)
        vec_keys = run_vector_only(conn, rag, nl_query, candidate_k)

        # Pure FTS recall (candidate_k limit)
        fts_keys = run_fts_only(conn, nl_query, candidate_k)

        # Metrics
        search_hits = search_keys & gt
        vec_hits = vec_keys & gt
        fts_hits = fts_keys & gt

        recall = len(search_hits) / len(gt)
        precision = len(search_hits) / top_k
        vec_recall = len(vec_hits) / len(gt)
        fts_recall = len(fts_hits) / len(gt)

        total_recall += recall
        total_precision += precision
        total_vec_recall += vec_recall
        total_fts_recall += fts_recall

        print(f"[{topic}]")
        print(f"  query: {nl_query!r}")
        print(f"  ground truth: {len(gt)} items")
        print(f"  search Recall@{top_k}: {recall:.1%} ({len(search_hits)}/{len(gt)})")
        print(f"  search Precision@{top_k}: {precision:.1%} ({len(search_hits)}/{top_k})")
        print(f"  vector Recall@{candidate_k}: {vec_recall:.1%} ({len(vec_hits)}/{len(gt)})")
        print(f"  fts    Recall@{candidate_k}: {fts_recall:.1%} ({len(fts_hits)}/{len(gt)})")
        if search_keys - gt:
            print(f"  irrelevant results: {search_keys - gt}")
        print()

    if n > 0:
        print(f"=== SUMMARY (avg over {n} queries) ===")
        print(f"  Search  Recall@{top_k}:    {total_recall/n:.1%}")
        print(f"  Search  Precision@{top_k}: {total_precision/n:.1%}")
        print(f"  Vector  Recall@{candidate_k}: {total_vec_recall/n:.1%}")
        print(f"  FTS     Recall@{candidate_k}: {total_fts_recall/n:.1%}")

    conn.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
