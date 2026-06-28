#!/usr/bin/env python3
"""Benchmark search latency on the Lyra project (5822 indexed chunks).

Measures, via the real RAGPipeline.search path:
1. End-to-end latency vs candidate_k (recall pool size) — isolates the
   reranker cost (the reranker scores candidate_k chunks).
2. End-to-end latency vs top_k (final result count).
3. Embedding-only vs rerank breakdown by toggling allow_fallback.
4. Embedding-one latency (the per-sub-query vector recall input).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server
from rag_pipeline import RAGPipeline

PID = 20675
QUERY = "how does the login authentication timeout work"
SUBS = ["login session expiration", "authentication timeout policy",
        "user inactivity logout"]


def t_ms(fn, *a, **k):
    t0 = time.monotonic()
    r = fn(*a, **k)
    return (time.monotonic() - t0) * 1000, r


def main() -> int:
    rag = server.rag()
    print(f"=== Search benchmark: Lyra (project_id={PID}) ===")
    print(f"query: {QUERY!r}")
    print(f"sub_queries: {SUBS}\n")

    # Warm up the reranker + embedder (first call loads models).
    print("warming up models...")
    t, r = t_ms(rag.search, PID, QUERY, sub_queries=SUBS, top_k=5,
                candidate_k=50)
    print(f"  warmup: {t:.0f}ms, {len(r)} results, strategy={r[0].get('strategy') if r else None}\n")

    print(f"{'candidate_k':>11} {'top_k':>5} {'latency_ms':>11} {'n':>3} {'strategy':>9}")
    print("-" * 48)
    # 1. Vary candidate_k (reranker cost grows with this)
    for ck in (10, 25, 50, 100, 200):
        t, r = t_ms(rag.search, PID, QUERY, sub_queries=SUBS, top_k=5,
                    candidate_k=ck)
        strat = r[0].get("strategy") if r else "-"
        print(f"{ck:>11} {5:>5} {t:>11.0f} {len(r):>3} {strat:>9}")

    print()
    # 2. Vary top_k (reranker scores candidate_k regardless; top_k just trims)
    for tk in (1, 5, 10, 20):
        t, r = t_ms(rag.search, PID, QUERY, sub_queries=SUBS, top_k=tk,
                    candidate_k=50)
        strat = r[0].get("strategy") if r else "-"
        print(f"{50:>11} {tk:>5} {t:>11.0f} {len(r):>3} {strat:>9}")

    print()
    # 3. Component breakdown: embed-one vs full search
    print("--- component breakdown ---")
    t_emb, _ = t_ms(rag.embedder.embed_one, QUERY)
    print(f"  embed_one(1 query):     {t_emb:.0f}ms")
    # avg over 3 calls
    ts = [t_ms(rag.embedder.embed_one, QUERY)[0] for _ in range(3)]
    print(f"  embed_one (avg of 3):   {sum(ts)/len(ts):.0f}ms")
    # full search avg of 3
    ts = [t_ms(rag.search, PID, QUERY, sub_queries=SUBS, top_k=5,
               candidate_k=50)[0] for _ in range(3)]
    print(f"  full search (avg of 3): {sum(ts)/len(ts):.0f}ms  (candidate_k=50)")

    # 4. Reranker-only timing: score 50 candidate texts directly
    print()
    print("--- reranker direct (50 texts) ---")
    import db_setup
    conn = server.db()
    rows = db_setup.fts_search(conn, "login timeout", PID, None, 50, None, None)
    texts = [r.get("text", "")[:512] for r in rows[:50]]
    if texts:
        t, _ = t_ms(rag.reranker.rerank, QUERY, texts)
        print(f"  rerank(50 texts):       {t:.0f}ms")
        t, _ = t_ms(rag.reranker.rerank, QUERY, texts[:10])
        print(f"  rerank(10 texts):       {t:.0f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
