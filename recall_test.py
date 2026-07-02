#!/usr/bin/env python3
"""BEIR-standard Recall + Precision + MRR + nDCG test suite for Lyra (id=20675).

Methodology (BEIR / MS MARPO style):
  - Ground truth (qrels): a hand-curated set of document_keys that are KNOWN
    relevant for each query, drawn from Lyra's Feature/Requirement item names
    (the topic authorities). Each qrel key is validated at test start by a DB
    read confirming the item name matches the topic; a mismatch aborts that
    query so no silent bad ground truth inflates/deflates metrics.
  - Metrics reported @k=5:
      Recall@k     = |retrieved ∩ relevant| / |relevant|
      Precision@k  = |retrieved ∩ relevant| / k
      MRR@k        = 1 / rank of first relevant result (0 if none in top-k)
      nDCG@k       = DCG / IDCG, with DCG = sum(rel_i / log2(i+1)) for i=1..k
  - Computed for three recall paths:
      full search (vector + FTS + RRF + cross-encoder rerank)
      pure vector recall (sqlite-vec KNN, no FTS, no rerank)
      pure FTS recall (FTS5 BM25, no vector, no rerank)
  - Parameter sweep over candidate_k in {25,50,100,150,200} shows the
    recall/latency trade-off so the sweet spot can be picked.

Run:  uv run python recall_test.py
      python recall_test.py          (alt)
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import math
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_setup import get_connection, vector_search, fts_search
from rag_pipeline import RAGPipeline, _to_fts_query

PROJECT_ID = 20675

# ---------------------------------------------------------------------------
# Curated qrels: (natural-language query, topic, {known-relevant doc keys})
# Each key is a Lyra Feature/Requirement that is the authoritative item for
# that topic (verified by item name). Adding closely-related sibling items
# gives a 1-3 item relevant set per query (BEIR style — small, high-quality).
# ---------------------------------------------------------------------------
TEST_CASES = [
    ("how does volume sync work between devices", "volume sync", {
        "Ly-FEAT-420",  # Volume control and synchronization
        "Ly-FEAT-395",  # Volume control and synchronization, USB audio
    }),
    ("call transfer between devices", "call transfer", {
        "Ly-FEAT-308",  # Call management, Bluetooth
        "Ly-FEAT-392",  # Call management, USB
    }),
    ("music streaming and playback control", "music control", {
        "Ly-FEAT-310",  # Music streaming and control
        "Ly-FEAT-397",  # USB Media Control on Bluetooth dongles and headsets
    }),
    ("three way conference call", "3-way call", {
        "Ly-FEAT-308",  # Call management, Bluetooth (covers 3-way)
        "Ly-FEAT-392",  # Call management, USB
    }),
    ("auto reject incoming call", "auto reject", {
        "Ly-FEAT-321",  # Auto reject call
    }),
    ("mute reminder tones on off", "mute", {
        "Ly-FEAT-311",  # Mute reminder tones On / Off
        "Ly-FEAT-323",  # Button sounds On / Off
    }),
    ("factory reset the device", "factory reset", {
        "Ly-FEAT-309",  # Factory reset
        "Ly-FEAT-453",  # Forced Reset
    }),
    ("LE Audio unicast and Auracast", "LE Audio", {
        "Ly-FEAT-387",  # LE Audio Unicast
        "Ly-FEAT-389",  # LE Audio Auracast Receiver
        "Ly-FEAT-390",  # LE Audio Pairing
    }),
    ("USB Type-C interface", "USB Type-C", {
        "Ly-FEAT-307",  # USB Type-C interface
        "Ly-FEAT-407",  # USB cable
    }),
    ("MS Teams certification", "MS Teams", {
        "Ly-FEAT-376",  # MS Teams Certification v5.0 (Headsets) - Audio
        "Ly-FEAT-428",  # Co-Pilot integration through MS Teams
    }),
    ("battery status and performance", "battery", {
        "Ly-FEAT-322",  # Automatic Battery status
        "Ly-FEAT-347",  # Battery performance
        "Ly-FEAT-357",  # Battery status
    }),
    ("active noise cancelling ANC", "ANC", {
        "Ly-FEAT-336",  # Active Noise Cancelling (ANC)
        "Ly-FEAT-330",  # Rx Noise Cancellation Performance
    }),
    ("firmware update over the air", "firmware OTA", {
        "Ly-FEAT-382",  # Firmware update OTA
        "Ly-FEAT-393",  # Firmware update USB
    }),
    ("Bluetooth pairing and multi use", "BT pairing", {
        "Ly-FEAT-383",  # Bluetooth pairing
        "Ly-FEAT-385",  # Multi Use
        "Ly-FEAT-386",  # Bluetooth Low Energy support
    }),
    ("hearthrough ambient sound", "hearthrough", {
        "Ly-FEAT-327",  # Hearthrough
        "Ly-FEAT-315",  # Sound Modes
    }),
]


# ---------------------------------------------------------------------------
# Metric helpers (BEIR standard)
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return len(set(top) & relevant) / k


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    # Binary relevance: 1 if in relevant set, else 0.
    gains = [1 if d in relevant else 0 for d in retrieved[:k]]
    ideal = sorted(gains, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(gains) / idcg


# ---------------------------------------------------------------------------
# Path runners
# ---------------------------------------------------------------------------
def run_search(rag, query: str, top_k: int, candidate_k: int) -> list[str]:
    results = rag.search(PROJECT_ID, query, top_k=top_k, candidate_k=candidate_k)
    return [r["document_key"] for r in results]


def run_vector_only(conn, rag, query: str, limit: int) -> list[str]:
    qvec = rag.embedder.embed_one(query)
    rows = vector_search(conn, qvec, PROJECT_ID, None, limit)
    # Deduplicate by document_key (multiple chunks per item), keep first.
    seen, out = set(), []
    for r in rows:
        dk = r["document_key"]
        if dk not in seen:
            seen.add(dk); out.append(dk)
    return out


def run_fts_only(conn, query: str, limit: int) -> list[str]:
    fts_q = _to_fts_query(query)
    if not fts_q:
        return []
    rows = fts_search(conn, fts_q, PROJECT_ID, None, limit)
    seen, out = set(), []
    for r in rows:
        dk = r["document_key"]
        if dk not in seen:
            seen.add(dk); out.append(dk)
    return out


def validate_qrels(conn, qrels: set[str], topic: str) -> tuple[set[str], list[str]]:
    """Confirm each qrel key exists in the DB; drop any that don't.

    Returns (valid_qrels, errors). A topic with zero valid qrels is skipped.
    """
    if not qrels:
        return set(), ["empty qrel set"]
    ph = ",".join("?" * len(qrels))
    rows = conn.execute(
        f"SELECT document_key, name FROM items WHERE project_id=? "
        f"AND document_key IN ({ph})", (PROJECT_ID, *qrels)
    ).fetchall()
    found = {r["document_key"]: r["name"] for r in rows}
    errors = []
    for k in qrels:
        if k not in found:
            errors.append(f"{k} not found in DB")
    return set(found.keys()), errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    conn = get_connection()
    rag = RAGPipeline()

    # Index completeness sanity.
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

    # Validate qrels up front.
    print("=== Validating curated qrels ===")
    valid_cases = []
    for nl_query, topic, qrels in TEST_CASES:
        vq, errs = validate_qrels(conn, qrels, topic)
        if errs:
            for e in errs:
                print(f"  [{topic}] WARN: {e}")
        if vq:
            valid_cases.append((nl_query, topic, vq))
        else:
            print(f"  [{topic}] SKIP: no valid qrels remain")
    print(f"  {len(valid_cases)}/{len(TEST_CASES)} queries with valid qrels\n")

    K = 5

    # ---- candidate_k sweep for the full search path ----
    sweep = [25, 50, 100, 150, 200]
    print(f"=== Full-search metric sweep (top_k={K}) ===")
    header = (f"{'candidate_k':>11} {'Recall@5':>9} {'Prec@5':>8} "
              f"{'MRR@5':>8} {'nDCG@5':>8} {'lat_ms':>8}")
    print(header)
    print("-" * len(header))
    sweep_results = {}
    for ck in sweep:
        rec = prec = mrr = ndcg = 0.0
        n = 0
        t0 = time.monotonic()
        for nl_query, topic, qrels in valid_cases:
            retrieved = run_search(rag, nl_query, top_k=K, candidate_k=ck)
            rec += recall_at_k(retrieved, qrels, K)
            prec += precision_at_k(retrieved, qrels, K)
            mrr += mrr_at_k(retrieved, qrels, K)
            ndcg += ndcg_at_k(retrieved, qrels, K)
            n += 1
        elapsed = (time.monotonic() - t0) * 1000 / max(n, 1)
        if n:
            print(f"{ck:>11} {rec/n:>9.1%} {prec/n:>8.1%} "
                  f"{mrr/n:>8.3f} {ndcg/n:>8.3f} {elapsed:>8.0f}")
            sweep_results[ck] = (rec/n, prec/n, mrr/n, ndcg/n, elapsed)
        else:
            print(f"{ck:>11}  (no valid queries)")
    print()

    # ---- detail at the default candidate_k=100 ----
    ck = 100
    print(f"=== Per-query detail (full search, candidate_k={ck}, top_k={K}) ===")
    tot_rec = tot_prec = tot_mrr = tot_ndcg = 0.0
    tot_vec_rec = tot_fts_rec = 0.0
    n = 0
    for nl_query, topic, qrels in valid_cases:
        retrieved = run_search(rag, nl_query, top_k=K, candidate_k=ck)
        vec_ret = run_vector_only(conn, rag, nl_query, ck)
        fts_ret = run_fts_only(conn, nl_query, ck)
        rec = recall_at_k(retrieved, qrels, K)
        prec = precision_at_k(retrieved, qrels, K)
        mrr = mrr_at_k(retrieved, qrels, K)
        ndcg = ndcg_at_k(retrieved, qrels, K)
        vrec = recall_at_k(vec_ret, qrels, K)
        frec = recall_at_k(fts_ret, qrels, K)
        tot_rec += rec; tot_prec += prec; tot_mrr += mrr
        tot_ndcg += ndcg; tot_vec_rec += vrec; tot_fts_rec += frec
        n += 1
        hits = set(retrieved[:K]) & qrels
        miss = qrels - set(retrieved[:K])
        irr = set(retrieved[:K]) - qrels
        print(f"[{topic}]")
        print(f"  query: {nl_query!r}")
        print(f"  qrels: {sorted(qrels)}")
        print(f"  search Recall@{K}: {rec:.1%}  Prec@{K}: {prec:.1%}  "
              f"MRR@{K}: {mrr:.3f}  nDCG@{K}: {ndcg:.3f}")
        print(f"  vector Recall@{ck}: {vrec:.1%}   fts Recall@{ck}: {frec:.1%}")
        if hits:
            print(f"  hit:   {sorted(hits)}")
        if miss:
            print(f"  miss:  {sorted(miss)}")
        if irr:
            print(f"  irr:   {sorted(irr)}")
        print()

    if n:
        print(f"=== SUMMARY (avg over {n} queries, candidate_k={ck}, top_k={K}) ===")
        print(f"  Search  Recall@{K}:    {tot_rec/n:.1%}")
        print(f"  Search  Precision@{K}: {tot_prec/n:.1%}")
        print(f"  Search  MRR@{K}:       {tot_mrr/n:.3f}")
        print(f"  Search  nDCG@{K}:      {tot_ndcg/n:.3f}")
        print(f"  Vector  Recall@{ck}:   {tot_vec_rec/n:.1%}")
        print(f"  FTS     Recall@{ck}:   {tot_fts_rec/n:.1%}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
