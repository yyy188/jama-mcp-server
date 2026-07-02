#!/usr/bin/env python3
"""Pure-recall test: RRF fusion only, NO reranker, top_k=100, candidate_k=100.

Measures the candidate-pool ceiling: of the known-relevant qrel items, how
many make it into the top-100 after vector+FTS recall + RRF fusion + item
dedup, WITHOUT any reranker re-sorting. This isolates the recall stage from
the rerank stage.

Temporary script — safe to delete (git restore recall_test.py unaffected).
Run: python recall_no_rerank.py
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_setup import get_connection, vector_search, fts_search, fetch_chunks_by_ids
from rag_pipeline import RAGPipeline, _to_fts_query, rrf_fuse, _normalize_sub_queries

PROJECT_ID = 20675
K = 50  # top_k = candidate_k = 50

# Same curated qrels as recall_test.py
TEST_CASES = [
    ("how does volume sync work between devices", "volume sync", {"Ly-FEAT-395","Ly-FEAT-420"}),
    ("call transfer between devices", "call transfer", {"Ly-FEAT-308","Ly-FEAT-392"}),
    ("music streaming and playback control", "music control", {"Ly-FEAT-310","Ly-FEAT-397"}),
    ("three way conference call", "3-way call", {"Ly-FEAT-308","Ly-FEAT-392"}),
    ("auto reject incoming call", "auto reject", {"Ly-FEAT-321"}),
    ("mute reminder tones on off", "mute", {"Ly-FEAT-311","Ly-FEAT-323"}),
    ("factory reset the device", "factory reset", {"Ly-FEAT-309","Ly-FEAT-453"}),
    ("LE Audio unicast and Auracast", "LE Audio", {"Ly-FEAT-387","Ly-FEAT-389","Ly-FEAT-390"}),
    ("USB Type-C interface", "USB Type-C", {"Ly-FEAT-307","Ly-FEAT-407"}),
    ("MS Teams certification", "MS Teams", {"Ly-FEAT-376","Ly-FEAT-428"}),
    ("battery status and performance", "battery", {"Ly-FEAT-322","Ly-FEAT-347","Ly-FEAT-357"}),
    ("active noise cancelling ANC", "ANC", {"Ly-FEAT-330","Ly-FEAT-336"}),
    ("firmware update over the air", "firmware OTA", {"Ly-FEAT-382","Ly-FEAT-393"}),
    ("Bluetooth pairing and multi use", "BT pairing", {"Ly-FEAT-383","Ly-FEAT-385","Ly-FEAT-386"}),
    ("hearthrough ambient sound", "hearthrough", {"Ly-FEAT-315","Ly-FEAT-327"}),
]


def pure_recall(rag, conn, query, candidate_k=100):
    """Vector + FTS recall + RRF fusion + item dedup, NO rerank. Returns doc keys."""
    subs = rag.expander.expand(query) or [query]
    ranked_lists = []
    for sq in subs:
        qvec = rag.embedder.embed_one(sq)
        v_rows = vector_search(conn, qvec, PROJECT_ID, None, candidate_k)
        fts_q = _to_fts_query(sq)
        f_rows = fts_search(conn, fts_q, PROJECT_ID, None, candidate_k) if fts_q else []
        ranked_lists.append([r["chunk_id"] for r in v_rows])
        ranked_lists.append([r["chunk_id"] for r in f_rows])
    fused = rrf_fuse(ranked_lists)
    if not fused:
        return []
    # Item-level dedup (same as the search method fix D)
    need_ids = [cid for cid, _ in fused]
    meta = {r["chunk_id"]: r for r in fetch_chunks_by_ids(conn, need_ids)}
    seen_items, deduped = set(), []
    for cid, sc in fused:
        if cid not in meta:
            continue
        item_id = meta[cid]["item_id"]
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        deduped.append((cid, sc))
        if len(deduped) >= candidate_k:
            break
    # Return doc keys in RRF order (no rerank)
    return [meta[cid]["document_key"] for cid, _ in deduped if cid in meta]


def main():
    conn = get_connection()
    rag = RAGPipeline()
    print(f"=== Pure Recall (RRF only, NO rerank, top_k={K}, candidate_k={K}) ===\n")

    total_recall = 0.0
    n = 0
    for query, topic, qrels in TEST_CASES:
        retrieved = pure_recall(rag, conn, query, candidate_k=K)
        retrieved_set = set(retrieved[:K])
        hits = retrieved_set & qrels
        recall = len(hits) / len(qrels) if qrels else 0
        total_recall += recall
        n += 1
        # Find rank of each hit
        hit_ranks = {k: retrieved.index(k)+1 for k in hits if k in retrieved}
        miss = qrels - retrieved_set
        status = "✅" if recall == 1.0 else ("⚠️" if recall > 0 else "❌")
        print(f"[{topic}] {status} Recall@{K}: {recall:.0%} ({len(hits)}/{len(qrels)})")
        print(f"  query: {query!r}")
        if hit_ranks:
            for k, rank in sorted(hit_ranks.items(), key=lambda x: x[1]):
                print(f"  hit:   {k} (RRF rank {rank})")
        if miss:
            print(f"  miss:  {sorted(miss)}")
        print()

    print(f"=== SUMMARY (avg over {n} queries) ===")
    print(f"  Pure Recall@{K} (RRF, no rerank): {total_recall/n:.1%}")
    conn.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
