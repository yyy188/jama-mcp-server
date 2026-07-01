#!/usr/bin/env python3
"""Initialize the Lyra project (id=20675) with blocking sync + resume support.

Run:
  uv run python init_lyra.py              # full init (or resume if interrupted)
  uv run python init_lyra.py --full       # force full re-init (no skip)
  uv run python init_lyra.py --resume     # resume only (skip indexed items)
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server

PROJECT_ID = 20675

def main():
    # Default: resume if the project is INITIALIZING/ERROR (interrupted),
    # otherwise full init.
    proj = None
    try:
        from db_setup import get_project
        conn = server.db()
        proj = get_project(conn, int(PROJECT_ID))
    except Exception:
        pass

    if "--full" in sys.argv:
        mode = False  # full init
    elif "--resume" in sys.argv:
        mode = "resume"
    elif proj and proj["status"] in ("INITIALIZING", "ERROR"):
        # Auto-resume: project was interrupted, skip already-indexed items.
        mode = "resume"
        print(f"Project is {proj['status']} — resuming (skipping indexed items)")
    else:
        mode = False  # full init

    print(f"Starting sync: project={PROJECT_ID}, mode={mode!r}", flush=True)
    r = server.sync_project_blocking(PROJECT_ID, incremental=mode, poll_interval=60)
    print(f"\nResult: {r}", flush=True)
    return 0 if r.get("status") == "DONE" else 1

if __name__ == "__main__":
    raise SystemExit(main())
