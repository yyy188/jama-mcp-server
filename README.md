# Jama MCP Server

A production-grade Model Context Protocol (MCP) server for the Jama requirements
management system, combining **high-precision RAG retrieval** with **native REST
API filtering**. An LLM client (Claude Desktop, etc.) can autonomously choose
between semantic search and structured metadata queries.

## Architecture

```
            ┌──────────────────────── MCP (stdio) ────────────────────────┐
            │                                                              │
  LLM ──────┤  init_jama_project      get_sync_progress                   │
  Client    │  search_jama_semantics  query_jama_native_metadata           │
            │                                                              │
            │   server.py  (FastMCP + APScheduler + thread pool)          │
            │      │                                                       │
            │      ├── rag_pipeline.py  (Multi-Query + Hybrid + RRF +     │
            │      │                       cross-encoder reranker)         │
            │      ├── jama_client.py   (OAuth2 + pagination + HTML clean)│
            │      └── db_setup.py      (SQLite + FTS5 + sqlite-vec)      │
            │                                                              │
            └──────────────────────────────────────────────────────────────┘
                         │                        │
                  Jama REST API            Local CPU embeddings (default:
                  (read-only GET)          bge-small-en-v1.5) + Azure OpenAI
                                           (optional, text-embedding-3-small)
```

## Retrieval pipeline (`search_jama_semantics`)

1. **Multi-Query** — the query is expanded into 3-5 sub-queries. The MCP LLM
   client performs the expansion and passes the variants via the
   `sub_queries` parameter; when none are supplied, the server falls back to
   deterministic lexical variants (stopword-stripped + truncated) so RRF
   fusion still benefits from multiple recall angles. No server-side chat
   LLM is configured or called.
2. **Hybrid recall** — for each sub-query: vector recall (sqlite-vec, cosine)
   + keyword recall (FTS5, BM25), each capped at `candidate_k`.
3. **RRF fusion** — Reciprocal Rank Fusion merges all ranked lists into one
   candidate pool of ≤ `candidate_k` unique chunks.
4. **Rerank** — a local **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
   ~80MB, CPU, via `transformers`) scores `(query, chunk)` pairs via a sequence-
   classification head; top `top_k` returned. If the model is unavailable, the
   pipeline gracefully falls back to RRF scores. Model weights are fetched from
   the HuggingFace China mirror (`HF_ENDPOINT=https://hf-mirror.com`) on first
   use, then served from cache.

## Reliability & crash recovery

The server is designed to survive crashes without losing data and to come back
up consistent on restart:

- **Atomic per-item indexing** — each item's chunks (text + FTS5 + sqlite-vec)
  are replaced in a single `write_txn` (`BEGIN IMMEDIATE`), so a crash mid-sync
  never leaves a half-written item. `done`/`progress` only advance *after* the
  commit, so the DB is consistent up to the last flushed batch.
- **Idempotent re-sync** — upserts overwrite (never duplicate), so
  re-processing already-indexed items on resume is harmless.
- **Startup recovery** — `_resume_interrupted_syncs` re-queues any project left
  `INITIALIZING` by a prior crash, so the server self-heals without manual
  action.
- **Concurrency guard** — `init_jama_project` refuses a duplicate concurrent
  sync for a project that already has a job in flight, returning the existing
  `job_id` instead of spawning a racing second worker.
- **Bounded HTTP retries** — 429 rate-limit handling is a bounded loop (not
  recursion), so a persistent rate-limit fails cleanly instead of overflowing
  the stack; `Retry-After` parsing tolerates non-numeric values; a 401 mid-sync
  refreshes the token and retries the page; malformed JSON bodies are retried.
- **WAL mode + write lock** — SQLite runs in WAL with a process-wide write
  lock, so the scheduler's writer and MCP reader threads coexist without
  `SQLITE_BUSY` failures.

## Chunking (LlamaIndex)

Jama rich-text (Description / Test Case Steps) is cleaned to plain text with
BeautifulSoup **before** being wrapped in LlamaIndex `Document` objects. The
documents are split into `TextNode` chunks by LlamaIndex's `SentenceSplitter`
(recursive, sentence-aware; `chunk_size=512`, `chunk_overlap=80` to preserve
context for the ~30% long-form items). The item name is prepended to each chunk
so the title is always retrievable.

## Native API (`query_jama_native_metadata`)

Bypasses the vector store for exact-match questions (specific document key,
status, item type). Uses `/abstractitems` which honours `itemType`,
`contains` and `documentKey` server-side; `status` is refined client-side.
Handles pagination internally, returns up to 20 core metadata records.

## Incremental sync

On startup, APScheduler registers a job (every 2h by default) that reads the
`projects` table for projects in `READY` status (`INITIALIZING` is deliberately
excluded — those are handled by crash recovery) along with their
`last_sync_time`, then walks Jama items whose `modifiedDate > last_sync_time`,
re-cleans/re-chunks them and updates the FTS5 + sqlite-vec indexes. New items
are added; modified items have their old chunks replaced atomically. A project
that already has an in-flight job is skipped so a scheduled sync never races a
user-initiated one.

## Setup

```bash
# 1a. Install CPU torch FIRST, on its own, forcing the PyTorch CPU index.
#     This is the reliable way to get the ~200MB CPU build — installing torch
#     alongside the rest of the deps can let a mirror hand back the 6GB CUDA
#     wheel instead (it loads fine but breaks the reranker on Windows via
#     WinError 1114 and wastes ~6GB of disk).
pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 1b. Then install the rest of the dependencies (Aliyun mirror for speed).
pip install -r requirements.txt

# 2. Configure + pre-download models (the wizard writes .env, validates deps +
#    config, probes connectivity with --self-test, AND downloads the embedding +
#    reranker models so the first sync isn't slow). Use --skip-models to defer.
python setup_wizard.py --self-test

# 3. Run (stdio transport for an MCP client)
python server.py
```

> **Upgrading from a prior CUDA torch install?** Uninstall the CUDA packages
> together, or a leftover `torchaudio`/`torchvision` will keep its CUDA `.pyd`
> and silently break reranker loading:
> ```bash
> pip uninstall torch torchaudio torchvision -y
> ```
> Then run step 1a above. The setup wizard's pre-flight (`validate_setup`)
> now warns about a CUDA torch build or a version-mismatched companion package.

To (re)download just the models later without re-running the wizard:

```bash
python bootstrap.py
```

The models live in `user/huggingface/` (project-local, ~210MB total: a ~130MB
ONNX embedding + ~80MB cross-encoder reranker). They're plain data files —
portable across machines, so you can also copy that folder from another machine
to skip the download entirely.

### Why pinned torch / onnxruntime versions

`requirements.txt` pins **`torch==2.6.0+cpu`** (from the PyTorch CPU index) and
**`onnxruntime==1.20.1`** (or `1.21.1` on Python 3.13+) on purpose:

- The default CUDA `torch` from the Aliyun mirror is ~6 GB and depends on a new
  VC++ Runtime (`vcruntime140_1.dll`) absent on many Windows machines, causing
  `WinError 1114` DLL load failures. The CPU build (~200 MB) has no such
  dependency and is all an ~80MB MiniLM CPU reranker needs.
- `onnxruntime` 1.27.0 (what `fastembed` pulls on Python 3.13+) triggers the
  same VC++ DLL issue on Windows. 1.20.1 satisfies `fastembed`'s constraint on
  Python 3.10–3.12 (`>=1.17.0, !=1.20.0`) and loads cleanly; Python 3.13+ needs
  `>1.21`, so it's pinned to 1.21.1 there.

If you upgrade these, re-test on a clean Windows machine **without** the latest
VC++ Redistributable installed.

After the server starts, the LLM client should call `bootstrap_models` (and poll
`get_bootstrap_progress` every ~2 min) to pre-download the embedding + reranker
models BEFORE the first `init_jama_project` — see [Model bootstrap](#model-bootstrap).
On startup the server logs a hint if the models aren't cached yet.

### First-run configuration guard

Every MCP tool runs an offline **pre-flight check** before doing any work:
Python dependencies, required env vars (JAMA_URL / JAMA_CLIENT_ID /
JAMA_CLIENT_SECRET — plus EMBEDDING_BASE_URL / EMBEDDING_API_KEY only when
EMBEDDING_PROVIDER=azure; the default `local` CPU provider needs no embedding
credentials) and the SQLite store. If anything is missing the tool returns a
clear error dict with a `hint` instead of failing midway through a Jama API
call. Configure via the wizard, or call the `configure_jama` / `validate_setup`
tools at runtime.

### MCP client config (Claude Desktop example)

```json
{
  "mcpServers": {
    "jama-mcp": {
      "command": "python",
      "args": ["/absolute/path/to/jama-mcp-server/server.py"],
      "env": { "JAMA_MCP_DB_PATH": "/absolute/path/to/jama-mcp-server/jama_mcp.db" }
    }
  }
}
```

## Usage flow (for the LLM)

0. `bootstrap_models()` → pre-download embedding + reranker models (first run
   only). Returns `job_id` immediately; poll `get_bootstrap_progress(job_id)`
   every ~2 min until `DONE`. Skip if models are already cached (re-running is
   a fast no-op).
1. `init_jama_project("20571")` → returns `job_id` immediately (non-blocking).
2. `get_sync_progress(job_id)` → poll until `status == "DONE"`, roughly every
   2 minutes (syncs index many items and take minutes — don't busy-poll).
3. `search_jama_semantics("20571", "how does volume sync work", top_k=5)` → RAG.
4. `query_jama_native_metadata("20314", document_key="SA-TC-7")` → exact match.

To re-index a project that is already initialized, use
`reinit_jama_project("20571")` (full re-sync) and poll the same way. Scheduled
incremental syncs run automatically (~every 2h); check any project's in-flight
job plus its last init/reinit/sync run at any time with
`get_sync_status("20571")`.

## Model bootstrap

The embedding model (~130MB ONNX, bge-small-en-v1.5) and the cross-encoder
reranker (~80MB) are **not bundled** — they download on first use. To keep the
first sync from stalling on a model download, call `bootstrap_models` right
after the server is configured. It downloads BOTH models **asynchronously** (a
`kind="bootstrap"` job in `sync_jobs`, run on the same thread pool as syncs) and
returns a `job_id` immediately.

- `bootstrap_models()` — start the async pre-download (no-op per model if
  already cached). Reentrancy-guarded: a second call while one is RUNNING
  returns the existing `job_id`.
- `get_bootstrap_progress(job_id)` — poll every ~2 min. Progress is
  **phase-based, not live bytes**: the reranker downloads via
  `snapshot_download` and the embedding via fastembed, neither of which gives a
  per-chunk byte callback, so `message` reports phase transitions (e.g.
  "Downloading reranker model (...)" → "Reranker model ready") rather than byte
  counts. `status` → `DONE` (both cached) or `ERROR`.

On startup, if either model isn't cached, the server logs a hint to call
`bootstrap_models`. The sync-time `ensure_downloaded` calls remain as a fallback
so a skipped bootstrap still works (the first sync downloads the models inline).

## Monitoring

`get_sync_status(project_id)` is the one-call monitor for a project's sync
operations. All three operations — `init_jama_project`, `reinit_jama_project`
and the scheduled incremental sync — run **asynchronously** as background jobs
(recorded in the `sync_jobs` table with `kind` = `init` / `reinit` / `sync`),
so each is pollable. The tool returns:

- `active_job` — the in-flight job for this project (or `null` if idle);
- `recent.{init,reinit,sync}` — the most recent job of each kind, terminal or
  running, so you can see the last result even when nothing is running now;
- `project_status` / `last_sync_time` / `item_count` / `chunk_count` — current
  project state;
- `process` — lightweight live metrics (RSS, threads, DB size, chunk count) for
  the server process; `null` if `psutil` is unavailable.

After starting an init or reinit, poll `get_sync_progress(job_id)` (or
`get_sync_status(project_id)`) roughly every 2 minutes, reporting each sample
to the user, until the job reaches `DONE`/`ERROR`. On startup, any job left
`RUNNING` by a prior crash is reconciled to `ERROR (interrupted by restart)`
so the monitor never shows a phantom in-flight job.

## Resilience

- **Jama API**: OAuth token auto-refresh on expiry + 401 retry; urllib3 `Retry`
  with exponential backoff on 429/5xx; explicit `Retry-After` handling; SSL
  connection-reset tolerated (transient on this network).
- **Embeddings**: same retry/backoff session on the embedding endpoint.
- **SQLite concurrency**: WAL mode + busy timeout + a process-level write lock
  so the APScheduler writer and MCP reader threads coexist without
  `SQLITE_BUSY` errors; chunk replacement is atomic per item.
- **Reranker**: lazy-loaded singleton; failure degrades to RRF-only scoring
  instead of crashing the search.
- **Read-only**: `JamaClient` only issues GET requests — it cannot create,
  modify or delete data on the Jama instance.

## Files

| File | Purpose |
|------|---------|
| `requirements.txt` | deps + Aliyun mirror config |
| `config.py` | env-driven settings (dataclasses) + validation/persistence/reload |
| `db_setup.py` | SQLite schema, FTS5 + sqlite-vec loading, CRUD |
| `jama_client.py` | OAuth, paginated fetch, HTML cleaning, native query, browse API |
| `rag_pipeline.py` | chunking, embeddings, Multi-Query, hybrid recall, RRF, rerank |
| `server.py` | MCP tools, async jobs, APScheduler incremental sync, pre-flight guards |
| `preflight.py` | offline dependency + config + storage validation |
| `net_guard.py` | pre-download bandwidth speed test (`NetworkTooSlowError`) |
| `bootstrap.py` | foreground model pre-download CLI (`python bootstrap.py`) |
| `setup_wizard.py` | interactive configuration wizard (`python setup_wizard.py`) |
| `selftest.py` | end-to-end self-test suite (`python selftest.py`) |
| `.env.example` | template for environment configuration |

## Tools

**Configuration & validation**
- `validate_setup(live=False)` — offline pre-flight (+ optional live Jama/embedding probe).
- `configure_jama(values)` — apply config at runtime, persist to `.env`, reload.

**Jama browse (read-only, gated by pre-flight)**
- `list_jama_projects()` — all visible projects.
- `find_jama_project_by_name(name, exact?)` — find projects by name → get id + info.
- `get_jama_item(item_id)` — full single item (cleaned text).
- `get_jama_item_children(item_id)` — decomposition children.
- `get_jama_item_relationships(item_id)` / `list_jama_project_relationships(project_id, item_id?)` — relationships (cursor-paginated `/relationships`).
- `get_jama_item_comments(item_id)` — item comments (cleaned body).
- `get_jama_item_attachments(item_id)` — attachment metadata (no binary).
- `list_jama_releases(project_id)` — project releases/versions.
- `list_jama_test_runs(project_id?, test_cycle_id?)` — test runs.
- `list_jama_item_types()` — tenant item types (id → name).
- `find_jama_item_type_by_name(name, exact?)` — find item types by display name → get the id needed by item_type filters.
- `query_jama_endpoint(path, params?, all_pages?)` — generic read-only GET escape hatch.

**RAG / retrieval / sync monitoring**
- `bootstrap_models()` — async pre-download of embedding + reranker models (returns `job_id`).
- `get_bootstrap_progress(job_id)` — poll a bootstrap job (every ~2 min) until DONE/ERROR.
- `init_jama_project(project_id)` — async background init (returns `job_id`).
- `reinit_jama_project(project_id)` — async full re-sync of an already-initialized project.
- `get_sync_progress(job_id)` — poll one init/reinit/sync job's progress.
- `get_sync_status(project_id)` — project monitor: in-flight job + last init/reinit/sync run + process metrics.
- `search_jama_semantics(project_id, query, ...)` — Multi-Query + hybrid + RRF + cross-encoder rerank.
- `query_jama_native_metadata(project_id, ...)` — exact-match metadata via `/abstractitems`.

## Verified

All components self-tested against the live Jama instance and the local CPU
embedding backend: OAuth + paginated fetch, HTML→text cleaning, Test Case step
rendering, item-type mapping, DB schema (FTS5 + vec0), full RAG search, async
init with progress polling, incremental sync (0 new items), concurrent
download + batched embed, crash recovery (INITIALIZING → auto-resynced READY),
native metadata filters (item_type / status / keyword / document_key),
APScheduler startup, MCP stdio handshake, and error paths (bad project id,
unknown job, nonexistent project, missing args).

The **cross-encoder reranker** (ms-marco-MiniLM-L-6-v2) was downloaded from the
HuggingFace China mirror (`hf-mirror.com`) and loaded on CPU; verified it produces non-zero relevance
scores with correct ordering (related=0.55 > unrelated=0.0002) and that the
end-to-end RAG search returns `strategy=rerank` results. **LlamaIndex** is the
primary RAG framework: `SentenceSplitter` + `Document`/`TextNode` for chunking.
Multi-Query expansion is performed by the MCP LLM client and passed to the
pipeline via `search(sub_queries=...)`; when omitted, deterministic lexical
variants are used.
