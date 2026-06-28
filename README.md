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
            │      │                       Qwen3-Reranker)                 │
            │      ├── jama_client.py   (OAuth2 + pagination + HTML clean)│
            │      └── db_setup.py      (SQLite + FTS5 + sqlite-vec)      │
            │                                                              │
            └──────────────────────────────────────────────────────────────┘
                         │                        │
                  Jama REST API            Azure OpenAI embeddings
                  (read-only GET)          (text-embedding-3-small)
```

## Retrieval pipeline (`search_jama_semantics`)

1. **Multi-Query** — the query is expanded into 3 sub-queries. Uses **LlamaIndex**
   (`PromptTemplate` + `OpenAI` LLM via `llama-index-llms-openai`) when
   `LLM_BASE_URL` is configured, otherwise deterministic lexical variants
   (stopword-stripped + truncated) so RRF fusion still helps without an LLM.
2. **Hybrid recall** — for each sub-query: vector recall (sqlite-vec, cosine)
   + keyword recall (FTS5, BM25), each capped at `candidate_k`.
3. **RRF fusion** — Reciprocal Rank Fusion merges all ranked lists into one
   candidate pool of ≤ `candidate_k` unique chunks.
4. **Rerank** — local **Qwen3-Reranker-0.6B** (CPU, via `transformers`) scores
   `(query, chunk)` pairs via the `P("yes")` token probability; top `top_k`
   returned. If the model is unavailable, the pipeline gracefully falls back to
   RRF scores. Model weights are fetched from the HuggingFace China mirror
   (`HF_ENDPOINT=https://hf-mirror.com`) on first use, then served from cache.

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
`projects` table for initialized `project_id` + `last_sync_time`, then walks
Jama items whose `modifiedDate > last_sync_time`, re-cleans/re-chunks them and
updates the FTS5 + sqlite-vec indexes. New items are added; modified items have
their old chunks replaced atomically.

## Setup

```bash
# 1. Install (uses Aliyun mirror; GitHub-only deps fall back to PyPI)
pip install -r requirements.txt

# 2. Configure (interactive wizard writes .env, then validates deps + config,
#    and optionally probes Jama/embedding connectivity with --self-test)
python setup_wizard.py --self-test

# 3. Run (stdio transport for an MCP client)
python server.py
```

### First-run configuration guard

Every MCP tool runs an offline **pre-flight check** before doing any work:
Python dependencies, required env vars (JAMA_URL / JAMA_CLIENT_ID /
JAMA_CLIENT_SECRET / EMBEDDING_BASE_URL / EMBEDDING_API_KEY) and the SQLite
store. If anything is missing the tool returns a clear error dict with a
`hint` instead of failing midway through a Jama API call. Configure via the
wizard, or call the `configure_jama` / `validate_setup` tools at runtime.

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

1. `init_jama_project("20571")` → returns `job_id` immediately (non-blocking).
2. `get_sync_progress(job_id)` → poll until `status == "DONE"`.
3. `search_jama_semantics("20571", "how does volume sync work", top_k=5)` → RAG.
4. `query_jama_native_metadata("20314", document_key="SA-TC-7")` → exact match.

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
- `query_jama_endpoint(path, params?, all_pages?)` — generic read-only GET escape hatch.

**RAG / retrieval**
- `init_jama_project(project_id)` — async background init (returns `job_id`).
- `get_sync_progress(job_id)` — poll init/sync progress.
- `search_jama_semantics(project_id, query, ...)` — Multi-Query + hybrid + RRF + Qwen3 rerank.
- `query_jama_native_metadata(project_id, ...)` — exact-match metadata via `/abstractitems`.

## Verified

All components self-tested against the live Jama instance and embedding API:
OAuth + paginated fetch, HTML→text cleaning, Test Case step rendering,
item-type mapping, DB schema (FTS5 + vec0), full RAG search, async init with
progress polling, incremental sync (0 new items), native metadata filters
(item_type / status / keyword / document_key), APScheduler startup, MCP stdio
handshake, and error paths (bad project id, unknown job, nonexistent project).

The **Qwen3-Reranker-0.6B** was downloaded from the HuggingFace China mirror
(`hf-mirror.com`) and loaded on CPU; verified it produces non-zero relevance
scores with correct ordering (related=0.55 > unrelated=0.0002) and that the
end-to-end RAG search returns `strategy=rerank` results. **LlamaIndex** is the
primary RAG framework: `SentenceSplitter` + `Document`/`TextNode` for chunking,
and `PromptTemplate` + `OpenAI` LLM for Multi-Query expansion (with a
deterministic fallback when no LLM endpoint is configured).
