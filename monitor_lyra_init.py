#!/usr/bin/env python3
"""Drive the MCP server tools in-process to init + monitor the Lyra project.

Exercises the SAME code paths an MCP client would hit (init_jama_project,
get_sync_progress) while sampling process + DB metrics on a timer, so we can
identify the sync bottleneck (download vs embed vs write).
"""
from __future__ import annotations

import os
import sys
import time
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psutil
import server
from config import settings

PID = 20675
DB_PATH = settings.storage.db_path

proc = psutil.Process()


def _db_size_mb() -> float:
    try:
        return os.path.getsize(DB_PATH) / 1024 / 1024
    except OSError:
        return 0.0


def _chunk_count() -> int:
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return -1


def _sample() -> dict:
    mem = proc.memory_info()
    return {
        "rss_mb": round(mem.rss / 1024 / 1024, 1),
        "cpu_pct": proc.cpu_percent(interval=None),
        "threads": proc.num_threads(),
        "db_mb": round(_db_size_mb(), 1),
        "chunks": _chunk_count(),
    }


def main() -> int:
    print(f"=== Init + monitor Lyra (project_id={PID}) ===")
    # Trigger init via the MCP tool function. This submits a background
    # _run_job thread; init_jama_project returns immediately with a job_id.
    resp = server.init_jama_project(str(PID))
    print("init_jama_project ->", resp)
    if "error" in resp:
        return 1
    # get_sync_progress takes a JOB_ID (not project_id). init returns it.
    job_id = resp.get("job_id")
    if not job_id:
        print("No job_id returned; aborting monitor.")
        return 1

    samples = []
    t0 = time.monotonic()
    last_done = -1
    last_status = None
    print(f"\n{'t(s)':>6} {'status':<14} {'done':>6} {'total':>6} {'pct':>6} "
          f"{'rss_mb':>8} {'cpu%':>6} {'thr':>4} {'db_mb':>7} {'chunks':>7}")
    while True:
        # get_sync_progress takes the JOB_ID (init returns it), not the
        # project_id — passing the project id would hit "Unknown job_id".
        prog = server.get_sync_progress(job_id)
        status = prog.get("status", "?")
        done = prog.get("done", 0)
        total = prog.get("total", 0)
        pct = prog.get("progress", 0)
        s = _sample()
        s["t"] = round(time.monotonic() - t0, 1)
        s["done"] = done
        s["status"] = status
        samples.append(s)
        print(f"{s['t']:>6.1f} {status:<14} {done:>6} {total:>6} {pct:>6.1f} "
              f"{s['rss_mb']:>8} {s['cpu_pct']:>6} {s['threads']:>4} "
              f"{s['db_mb']:>7} {s['chunks']:>7}")
        if status in ("DONE", "READY", "ERROR") and done == last_done and status == last_status:
            # terminal and stable: take one more sample then stop
            time.sleep(3)
            s2 = _sample()
            s2["t"] = round(time.monotonic() - t0, 1)
            s2["done"] = done
            s2["status"] = status
            samples.append(s2)
            break
        last_done = done
        last_status = status
        time.sleep(5)

    print("\n=== Sync phase summary ===")
    total_t = samples[-1]["t"] if samples else 0
    final = samples[-1] if samples else {}
    print(f"Total wall time : {total_t:.1f}s")
    print(f"Items indexed   : {final.get('done', 0)}")
    print(f"Chunks written  : {final.get('chunks', 0)}")
    print(f"Peak RSS        : {max(s['rss_mb'] for s in samples):.0f} MB")
    print(f"Final DB size   : {final.get('db_mb', 0):.0f} MB")
    print(f"Throughput      : "
          f"{final.get('done', 0)/max(total_t,1):.1f} items/s, "
          f"{final.get('chunks',0)/max(total_t,1):.1f} chunks/s")
    print(f"Final status    : {final.get('status')}")

    # Save samples for the post-sync analyzer.
    import json
    with open("lyra_sync_metrics.json", "w") as f:
        json.dump({"pid": PID, "job_id": job_id, "total_t": total_t,
                   "samples": samples, "final": final}, f, indent=2)
    print("\nMetrics saved to lyra_sync_metrics.json")
    return 0 if final.get("status") in ("DONE", "READY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
