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

# 2. Configure
cp .env.example .env
# edit .env with your Jama + embedding credentials

# 3. Run (stdio transport for an MCP client)
python server.py
```

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
| `config.py` | env-driven settings (dataclasses) |
| `db_setup.py` | SQLite schema, FTS5 + sqlite-vec loading, CRUD |
| `jama_client.py` | OAuth, paginated fetch, HTML cleaning, native query |
| `rag_pipeline.py` | chunking, embeddings, Multi-Query, hybrid recall, RRF, rerank |
| `server.py` | MCP tools, async jobs, APScheduler incremental sync |
| `.env.example` | template for environment configuration |

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
