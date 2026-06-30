# Jama MCP Server — 整体设计说明

> 一份面向工程读者的高层 + 细节设计文档。基于对项目全部源码（约 6266 行 Python，14 个源文件）的逐文件阅读整理。

---

## 1. 项目概述

Jama MCP Server 是一个**生产级 Model Context Protocol (MCP) 服务器**，把 Jama 需求管理系统接入大语言模型客户端（如 Claude Desktop）。它把两种检索能力统一暴露给 LLM：

- **语义 RAG 检索**（`search_jama_semantics`）：Multi-Query 扩展 + 混合召回（向量 + 关键词）+ RRF 融合 + 交叉编码器重排；
- **原生 REST 元数据查询**（`query_jama_native_metadata` 及一组只读 browse 工具）：精确字段匹配。

LLM 客户端可自主在两种模式间选择。服务器通过 **stdio** 传输与 MCP 客户端通信，对 Jama 实例**只读**（仅发 GET，外加 OAuth token 的 POST）。

### 设计目标

| 目标 | 实现手段 |
|------|----------|
| 高精度检索 | Multi-Query + 混合召回 + RRF + MiniLM 交叉编码器重排 |
| 崩溃不丢数据 | 单条目原子写入（`BEGIN IMMEDIATE` 单事务替换 chunks+FTS+vec）；进度只在提交后推进 |
| 自愈 | 启动时 `reconcile_stale_jobs` + `_resume_interrupted_syncs` |
| 非阻塞 | init/reinit/sync/bootstrap 全部异步后台任务，返回 `job_id` 供轮询 |
| 可移植自包含 | DB 与模型缓存默认落在项目内 `user/` 目录；纯数据文件可跨机拷贝 |
| 国内可用 | Aliyun pip 镜像 + HuggingFace 中国镜像（hf-mirror.com）+ CPU 版 torch |
| 配置零侵入 | 全部设置来自环境变量/.env，同一份代码跑 dev/test/prod |

---

## 2. 系统架构

```
            ┌──────────────────────── MCP (stdio) ────────────────────────┐
            │                                                              │
  LLM ──────┤  init_jama_project      get_sync_progress                   │
  Client    │  search_jama_semantics  query_jama_native_metadata           │
            │  bootstrap_models       get_sync_status   (+ 17 browse tools)│
            │                                                              │
            │   server.py   (FastMCP + APScheduler + ThreadPoolExecutor)   │
            │      │                                                       │
            │      ├── rag_pipeline.py  (Multi-Query + Hybrid + RRF +      │
            │      │                       MiniLM 交叉编码器重排)           │
            │      ├── jama_client.py   (OAuth2 + 并发分页 + HTML 清洗)     │
            │      ├── db_setup.py      (SQLite + FTS5 + sqlite-vec)       │
            │      ├── preflight.py     (离线依赖/配置/存储校验)            │
            │      ├── net_guard.py     (带宽预检 + 断点续传下载)           │
            │      └── config.py        (env→dataclass + 校验/持久化/reload)│
            │                                                              │
            └──────────────────────────────────────────────────────────────┘
                         │                        │
                  Jama REST API            本地 CPU 嵌入 (bge-small-en-v1.5)
                  (只读 GET)               + Azure OpenAI (可选)
```

### 分层

1. **MCP 服务层**（`server.py`，1708 行）：FastMCP 工具暴露、异步任务系统、APScheduler 增量同步、预检守卫、监控工具。
2. **RAG 流水线**（`rag_pipeline.py`，970 行）：分块、嵌入提供者抽象、Multi-Query、混合召回、RRF、重排。
3. **Jama 客户端**（`jama_client.py`，915 行）：OAuth、并发分页拉取、HTML 清洗、browse API、原生元数据查询。
4. **存储层**（`db_setup.py`，550 行）：SQLite schema、FTS5 + sqlite-vec 加载、并发模型、CRUD。
5. **横切关注点**：`config.py`（348 行）、`preflight.py`（167 行）、`net_guard.py`（206 行）。
6. **运维脚本**：`setup_wizard.py`、`bootstrap.py`、`selftest.py`、`bench_lyra_search.py`、`monitor_lyra_init.py`、`test_multiquery.py`。

---

## 3. 模块总览

| 文件 | 行数 | 职责 |
|------|------|------|
| `server.py` | 1708 | MCP 入口：23 个工具、异步任务池、增量同步调度、预检守卫、崩溃恢复 |
| `rag_pipeline.py` | 970 | 分块(LlamaIndex)、嵌入(local/azure)、Multi-Query、混合召回、RRF、MiniLM 重排 |
| `jama_client.py` | 915 | OAuth2、并发分页、HTML→文本、browse API、原生查询、只读保证 |
| `config.py` | 348 | env→dataclass、校验、`.env` 持久化、热重载 |
| `db_setup.py` | 550 | SQLite schema、FTS5+vec0、WAL+写锁、原子替换、CRUD、维度变更重建 |
| `selftest.py` | 555 | 端到端自测套件 |
| `setup_wizard.py` | 216 | 交互式配置向导（写 .env + 预检 + 自检 + 预下载模型） |
| `net_guard.py` | 206 | 带宽预检 `speed_test`、断点续传 `download_with_retry` |
| `preflight.py` | 167 | 离线依赖+配置+存储三段式校验 |
| `bootstrap.py` | 127 | 前台同步预下载嵌入+重排模型 |
| `monitor_lyra_init.py` | 129 | 进程内 init + 监控采样（性能调优用） |
| `bench_lyra_search.py` | 93 | 搜索延迟基准（candidate_k / top_k / 重排开关） |
| `test_multiquery.py` | 282 | Multi-Query 重构的离线单元测试 |

---

## 4. 配置系统（`config.py`）

所有设置来自环境变量（可由 `.env` 加载，`python-dotenv` 可选）。用 `@dataclass(frozen=True)` 分组：

- `JamaSettings` — REST 连接：`url`/`client_id`/`client_secret`/`api_prefix`、分页调优（`page_size=50`、`page_delay=0.25`）、重试（`max_retries=6`）、**带宽/卡顿阈值**（`min_bytes_per_sec`、`page_min_bytes_per_sec=500`、`page_max_retries=5`）。
- `EmbeddingSettings` — 双提供者：
  - `local`（默认）：`bge-small-en-v1.5` 跑在 CPU（fastembed/ONNX），无需 API key；`cpu_percent=60` 限制线程；`download_min_bps=200000`。
  - `azure`：OpenAI 兼容端点，`base_url`+`api_key`+`key_header`（Azure 用 `api-key`）。
  - `dimensions` 属性按提供者推导：local=384，azure=1536（可覆盖）。**维度变化会触发 `db_setup.init_db` 重建向量索引**。
- `RerankerSettings` — 默认 `cross-encoder/ms-marco-MiniLM-L-6-v2`（~80MB），`batch_size=16`、`max_length=256`、`allow_fallback=True`（加载失败降级为 RRF）。
- `StorageSettings` — `db_path` 默认 `user/jama_mcp.db`；`busy_timeout_ms=5000`。
- `SyncSettings` — `enabled`、`hours=2`、`max_items_per_run=5000`、`download_concurrency=16`（并发分页拉取）。
- `ChunkSettings` — `chunk_size=512`、`chunk_overlap=80`。

### 关键设计点

- **HuggingFace 环境预设**（模块加载时 `os.environ.setdefault`）：`HF_ENDPOINT=https://hf-mirror.com`（中国镜像）、`HF_HOME`/`HUGGINGFACE_HUB_CACHE` 指向项目内 `user/huggingface`、`HF_HUB_DISABLE_XET=1`（避免 Xet 协议在某些网络中断）。
- **`Settings` 故意非 frozen**：`reload_settings()` 替换内部 dataclass，因为所有模块共享同一个 `settings` 实例（`from config import settings`），运行时替换属性即可传播，无 import 时捕获问题。
- **校验** `validate_config()` 返回 issue 列表（`{field,severity,message,feature}`）。URL 形状与占位符（`your-tenant`/`example.com`）会被标为 error。所需变量按提供者动态拼接（azure 才要求 embedding 端点变量）。
- **持久化** `write_env_file(values)` 合并现有环境写入完整 `.env`，部分配置不会清空其余项。

---

## 5. 数据存储层（`db_setup.py`）

### 5.1 Schema

```sql
-- 项目同步状态
projects(project_id PK, name, status, last_sync_time, item_count,
         chunk_count, error, updated_at)
-- status 域: NEW | INITIALIZING | READY | ERROR

-- 单条 Jama 项的元数据（反范式化）
items(item_id PK, project_id FK->projects, document_key, global_id,
      item_type, item_type_name, name, status, description, test_steps,
      modified_date, created_date, raw_json, updated_at)
-- 索引: idx_items_project, idx_items_type(project_id,item_type),
--       idx_items_modified(project_id,modified_date)

-- 检索单元：一个文本块一行
chunks(chunk_id PK "{item_id}#{section}#{i}", item_id FK->items,
       project_id, item_type, item_type_name, document_key, name, status,
       section("description"|"test_steps"), chunk_index, text,
       modified_date, updated_at)
-- 索引: idx_chunks_item, idx_chunks_project, idx_chunks_modified

-- 异步任务进度
sync_jobs(job_id PK, project_id, kind("init"|"reinit"|"sync"|"bootstrap"),
          status("PENDING"|"RUNNING"|"DONE"|"ERROR"), progress REAL,
          total, done, message, started_at, finished_at)

-- FTS5 全文索引（porter+unicode61 分词，仅 text 可搜索）
chunks_fts USING fts5(chunk_id UNINDEXED, project_id UNINDEXED,
                      item_type UNINDEXED, modified_date UNINDEXED, text,
                      tokenize='porter unicode61')

-- sqlite-vec 向量索引（{DIM} 由 embedding 维度替换）
chunks_vec USING vec0(chunk_id TEXT PK, embedding FLOAT[{DIM}])
```

`init_db()` 用 `CREATE ... IF NOT EXISTS` 幂等建表；`{DIM}` 在执行前替换为 `settings.embedding.dimensions`。

### 5.2 FTS5 与 sqlite-vec 加载

`_load_extensions(conn)`：开启扩展加载→`sqlite_vec.load(conn)`→关闭扩展加载。FTS5 内置于 CPython 自带 SQLite，无需显式加载。失败抛 `RuntimeError`。`preflight.check_storage()` 还会 `SELECT 1 FROM chunks_vec LIMIT 0` 验证 vec0 真正可用。

### 5.3 并发模型

```python
_write_lock = threading.RLock()        # 进程级写锁（RLock 允许同线程嵌套）

@contextmanager
def write_txn(conn):
    with _write_lock:                  # 先串行化进程内写者
        conn.execute("BEGIN IMMEDIATE;")  # 立即取保留锁，避免升级死锁
        try:
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
```

连接配置（`get_connection`）：`check_same_thread=False`、`isolation_level=None`（autocommit，事务显式管理）、`timeout=busy_timeout_ms`。PRAGMA：`journal_mode=WAL`（读不阻塞写）、`synchronous=NORMAL`、`busy_timeout`、`foreign_keys=ON`。

WAL + 进程级 RLock + busy_timeout 三者配合：APScheduler 写线程与 MCP 读线程共存，不出现 `SQLITE_BUSY`。

### 5.4 关键 CRUD

- `upsert_project(...)` — 动态列 `ON CONFLICT(project_id) DO UPDATE`，包在 `write_txn`。
- `upsert_item(item)` — `ON CONFLICT(item_id) DO UPDATE`；**调用方持 `write_txn`**（批处理时）。
- `replace_chunks(conn, item_id, chunks, embeddings)` — **单条目原子替换**：在单个 `write_txn` 内删旧→`INSERT OR REPLACE` 写 `chunks`/`chunks_fts`/`chunks_vec` 三表。`INSERT OR REPLACE` 使其**幂等**（重同步/碰撞安全）。向量经 `_vec_blob` 打包为 float32 LE blob（`struct.pack(f"{n}f", *vec)`）。
- `vector_search(...)` — `chunks_vec v JOIN chunks c ... WHERE c.project_id=? ORDER BY v.distance ASC LIMIT ?`，`distance` 作为 `score`（越小越好）。
- `fts_search(...)` — `chunks_fts MATCH ? ... ORDER BY bm25(chunks_fts) LIMIT ?`，`bm25` 作为 `score`（越小越好）。
- 两者都支持 `item_type` 与 `modified_after/before`（经 `_normalize_iso_utc` 归一为定宽 UTC 字符串做字典序范围比较）。
- sync_jobs：`create_job`（直接以 `RUNNING`+`started_at` 插入）、`update_job`（动态 SET，DONE/ERROR 时填 `finished_at`）、`get_active_job_for_project`（最近非终止任务，用于拒绝重复并发同步）、`get_latest_job_for_project`（按 kind 取最近任务，供监控）、`reconcile_stale_jobs`（见 5.6）。

### 5.5 维度变更重建

`init_db` 用 `_existing_vec_dim`（正则解析 `sqlite_master` 的 `FLOAT[N]`）比对当前维度。若不同：在单个 `write_txn` 内 `DROP chunks_vec` + 清空 `chunks_fts`/`chunks`，**保留 `items` 元数据**（重同步时只重嵌不重取），把所有 `READY` 项目置回 `NEW`（`chunk_count=0`、`last_sync_time=NULL`），再用新维度重建 `chunks_vec`。

### 5.6 崩溃恢复与幂等

- **幂等**：schema 全 `IF NOT EXISTS`；`replace_chunks`/`upsert_*` 全 `INSERT OR REPLACE`/`ON CONFLICT`。
- **`reconcile_stale_jobs(conn)`**：启动时把所有 `PENDING`/`RUNNING` 任务标为 `ERROR` 并追加 ` [interrupted by restart]`，否则 `get_active_job_for_project` 会永远报告幽灵任务。
- **WAL + 每次提交持久化**：中断的同步已提交的条目已落盘；仅项目状态未到 `READY`，由 `server._resume_interrupted_syncs` 补救。

---

## 6. Jama 客户端（`jama_client.py`）

### 6.1 OAuth2 与会话

- `_ensure_token()`：`threading.Lock` 保护，过期前 60s 提前刷新；POST `/rest/oauth/token`（`grant_type=client_credentials`）。
- 401 中途失效：`_get`/`_get_page_with_stall_retry`/`list_project_relationships` 内单次重试（`refreshed` 标志保证每调用只刷一次）。
- `_build_session()`：`urllib3 Retry(total=max_retries, backoff_factor=1.5, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)`；连接池 `pool_maxsize=max(download_concurrency,10)` 避免 16 并发时池过小导致 TLS 重握手。

### 6.2 分页拉取（串行 + 并发波）

- `_get(path, params)`：单页，`_MAX_429_RETRIES=5` 的**有界循环**处理 429（`_parse_retry_after` 仅认整数秒，非法回退 5s，夹到 ≤30s），**避免旧的无限递归**。
- `_get_page_with_stall_retry(...)`：每页卡顿守卫——计算吞吐 `bps=len(body)/elapsed`，若 `bps < page_min_bytes_per_sec` 且 `len(body)>1024` 则抛 `NetworkTooSlowError`；200 但 JSON 损坏（`ValueError`）按瞬时错误重试；指数退避 `min(2**attempt,10)`。
- `_paginate(...)`：串行 `startAt`/`maxResults` 偏移分页，读 `meta.pageInfo.totalResults`，页间 `page_delay`。
- `iter_pages_concurrent(...)`：先取第 1 页获 `totalResults`，再用 `ThreadPoolExecutor(max_workers=concurrency)` **波次并发**拉后续页（每波 `concurrency` 页），保序合并。`concurrency` 来自 `settings.sync.download_concurrency`（默认 16）。大项目下载提速约 10×。
- `iter_project_items(...)`：高层流式迭代，`_emit_wave` 做条目归一化 + 客户端 `modified_after` 过滤（Jama 无服务端修改时间过滤）+ `max_items` 上限。

### 6.3 HTML 清洗

- `clean_html(raw)`：BeautifulSoup+lxml，`<br>`→`\n`，块元素加换行，`_normalize_ws` 统一空白。
- `render_test_steps(steps)`：把 `testCaseSteps` 渲染为 `Step i: action | Expected: ... | Notes: ...`。
- 在 `_normalize_item` 中清洗 `description` 与 `test_steps`，并拍平 `item_id/project_id/document_key/item_type/item_type_name/name/status/modified_date/created_date/raw_json`。

### 6.4 原生元数据查询

`query_items_native(project_id, *, document_key, item_type, status, keyword, limit=20)` 用 `/abstractitems`（它支持 `itemType`/`contains`/`documentKey` 服务端过滤，而 `/items` 忽略 `itemType`）：

- **服务端**：`project`、`itemType`、`contains`(keyword)、`documentKey`。
- **客户端**：`status`（无服务端过滤，按 `fields.status`/`testCaseStatus`/`testRunStatus` 大小写不敏感比较）；`document_key` 客户端再校验一道。
- **安全帽**：`max_pages=50`、累计 `seen>=2000` 行即停。

### 6.5 Browse API（全部只读）

`list_projects`、`find_projects(name, exact?)`、`get_project`、`get_item`、`get_item_children`、`get_item_relationships`、`list_project_relationships`（**游标分页** `lastId`，非 `startAt`）、`get_item_comments`、`get_item_attachments`（仅元数据无二进制）、`list_releases`、`list_test_runs`、`load_item_types`/`item_type_name`/`list_item_types`/`find_item_types`、`count_project_items`、`get_raw(path, params, max_pages)`（通用 GET 逃生舱，**SSRF 守卫**：必须以 `/` 开头、无 `://`、无 `?`/`#`）、`preflight_speed_check()`（取一整页 50 项目做带宽探测）。

`load_item_types` 采用**仅成功才缓存**不变式：异常时返回部分结果但不缓存，避免永久冻结残缺映射。

### 6.6 只读保证

所有数据访问方法仅用 GET（`requests.get`/`_get`/`_paginate`/`iter_pages_concurrent`）。无任何 PUT/POST/PATCH/DELETE 打到 `/rest/...` 数据端点。`Retry.allowed_methods` 含 `POST` 仅为 token 获取可重试。模块与类 docstring 明确声明只读。

---

## 7. RAG 流水线（`rag_pipeline.py`）

### 7.1 嵌入提供者抽象

`EmbeddingClient`（azure）与 `LocalEmbeddingClient`（local）实现同一接口（`embed_texts`/`embed_one`/`embed_many_concurrent`），`RAGPipeline` 多态选用。

- **azure**：POST `/openai/v1/embeddings`，按 `batch_size` 切片串行/并发（`embed_many_concurrent` 用 `ThreadPoolExecutor`，按记录的 `(start_index,batch)` 位置重装）；返回 `data` 按 `index` 排序（API 可能乱序）。
- **local（默认）**：fastembed/ONNX `bge-small-en-v1.5`，单例（`__new__`+`_lock`+`_initialized`）。
  - 查询前缀 `Represent this sentence for searching relevant passages: `（文档不加前缀）。
  - CPU 线程上限：`max(1, ceil(cores * cpu_percent/100))`。
  - **镜像下载**：fastembed 配置名是 `BAAI/bge-small-en-v1.5`，实际从 `qdrant/bge-small-en-v1.5-onnx-q` 拉 ONNX；缓存目录按真实 repo 命名。下载按 `[hf-mirror.com, huggingface.co]` 顺序，每个镜像先 `net_guard.speed_test` 探 `config.json`，达标才 `TextEmbedding(...)`+预热。
  - 嵌入在 `_embed_lock` 下串行（ONNX session 非重入安全）；`embed_many_concurrent` 的 concurrency 被**故意忽略**（CPU 受限，并行只会过载）。
  - 失败设粘性 `_load_error`，后续调用直接返回同一错误而非反复重试。

### 7.2 重排器（交叉编码器）

`Reranker` 单例，默认 `cross-encoder/ms-marco-MiniLM-L-6-v2`（~80MB）：

- `_load()`：`transformers` `AutoModelForSequenceClassification`/`AutoTokenizer`，**fp32** on CPU（MiniLM 极小，fp32 任意 CPU 可跑，无需 AVX512_BF16/AMX）。`.eval()`。失败设粘性错误返回 `False`。
- `_ensure_weights_downloaded()`：`huggingface_hub.snapshot_download` 用 `allow_patterns` 只拉 transformers 需要的文件（`config.json`+safetensors+tokenizer，无 safetensors 时回退 `pytorch_model.bin`），`_weights_lock` 双检。
- `_score`：`tokenizer([(query,t)...], padding=True, truncation=True, max_length)` → `model(**tok)` → `torch.sigmoid(logits)`。
- **降级**：`rerank` 任何失败返回全零；`search` 用 `used_rerank = any(s!=0.0)` 检测，False 时改用 RRF 分数并标 `strategy="rrf"`。

> 设计说明：MiniLM 交叉编码器用 sequence-classification head 直接打分 (query,doc) 对，**无因果 LM 的 `[N,seq,152k-vocab]` logits 张量**，因此 `batch_size=16` 即可（旧 Qwen3-Reranker-0.6B 需 batch=4，体积 1.2GB）。RRF 已融合召回，重排只重排最终 25 条，小模型精度足够。

### 7.3 分块（LlamaIndex）

`make_splitter()` 用 `SentenceSplitter`（递归、句感知，`chunk_size=512`/`overlap=80`，主分隔符 `\n\n`，次级正则 `[^,.;。]+[,.;。]?`）。

`chunk_item(item)`：

- 构造 `sections`：`description` +（Test Case 的）`test_steps`；两者皆空但有 `name` 时回退 `("description", name)`，保证至少可按标题检索。
- 每段包成 LlamaIndex `Document`（带 `item_id/section/document_key/modified_date` 元数据）→ `get_nodes_from_documents` 得 `TextNode`。
- **每个 chunk 文本前置 item name**（`f"{name}\n{part}"`），保证标题始终可召回。
- 产出存储行，`chunk_id = f"{item_id}#{section}#{i}"`。

HTML 已在 `jama_client` 用 BeautifulSoup 清洗，`chunk_item` 收到的是纯文本。

### 7.4 检索流程 `RAGPipeline.search`

```
search(project_id, query, *, sub_queries=None, item_type=None,
       top_k=5, candidate_k=25, modified_after=None, modified_before=None)
```

1. **Multi-Query 扩展**：调用方（MCP LLM 客户端）传 `sub_queries`（3–5 条），`_normalize_sub_queries` 强制原 query 在首位、去重、去空、封顶 5；若为 `None`，回退 `MultiQueryExpander.expand(query)`（确定性词法变体：原 query + 去停用词聚焦 + 截断 4 token 拓宽），再不行用 `[query]`。**服务端不配置/不调用 chat LLM**。
2. **混合召回（每个子查询）**：
   - 向量召回：`embed_one(sq)` → `vector_search(..., candidate_k)`（sqlite-vec 余弦距离 ASC）。
   - 关键词召回：`_to_fts_query(sq)`（构造安全 FTS5 MATCH：引号前缀 AND）→ `fts_search(..., candidate_k)`（BM25 ASC）。
   - 每子查询产出两个 `chunk_id` 排序列表。
3. **RRF 融合**：`rrf_fuse(ranked_lists, k=60)[:candidate_k]` 合并为 ≤ `candidate_k` 唯一 chunk。
4. **取详情 + 重排**：`fetch_chunks_by_ids` 取行，按融合序取 `texts`，`reranker.rerank(query, texts)`。`used_rerank = any(s!=0.0)`；False 则用每 cid 的 RRF 分数替代。
5. **top_k + 策略标签**：按分数降序取 `top_k`，`_row_to_result` 整形，`strategy` 为 `"rerank"` 或 `"rrf"`。

`modified_after/before` 在召回层应用，RRF/重排只见范围内候选。连接在 `finally` 关闭。

### 7.5 索引流程（编排于 `server._sync_project_locked`）

生产者/消费者两线程 + 有界队列：

- **取数线程** `_fetcher`：`iter_project_items(..., concurrency=dl_concurrency, on_total=_on_total)` → 每条 `chunk_item` → 入 `item_queue`（maxsize=`batch_size*4`，背压）→ 末尾 `None` 哨兵。
- **嵌入线程**（主）：累积到 `pending_chunk_count >= batch_size` 时 `_embed_and_store(batch)`：把有文本的项扁平化 → `pipeline.embed_many(flat)`（跨项打包，按 `EMBEDDING_CONCURRENCY` 并发）→ **逐项** `with write_txn: upsert_item + replace_chunks`；无文本项 `replace_chunks(...,[],[])` 清旧块。
- **崩溃安全不变式**：`done`/`progress` 与 DB 写入只在嵌入线程**提交成功后**推进；崩溃时 DB 一致到最后一批，项目留 `INITIALIZING` 由 `_resume_interrupted_syncs` 全量重同步（upsert 幂等）。

### 7.6 `ensure_downloaded` 桥接

- `LocalEmbeddingClient.ensure_downloaded()`：已加载/已出错/已存在则 no-op，否则 `_download_model()`，失败仅告警（实际报错延后到首次 embed）。
- `Reranker.ensure_downloaded(progress_callback)`：no-op 或 `_ensure_weights_downloaded`，失败保持未加载（search 降级 RRF）。`progress_callback` 因 `snapshot_download` 无字节回调，仅在完成时触发。
- 三入口共用同一逻辑：MCP 工具 `bootstrap_models`（异步）、`bootstrap.py`（前台 CLI）、同步启动时 `ensure_downloaded` 兜底。

---

## 8. MCP 服务层（`server.py`）

### 8.1 启动序列（`main()`，严格顺序）

1. `init_db()` — 建表/加载扩展，启动即暴露 schema 错误。
2. `reconcile_stale_jobs(db())` — **必须在 resume 之前**，让恢复项目拿到新 job 行而非复用陈旧 RUNNING 行。
3. `_resume_interrupted_syncs()` — 重排队 `INITIALIZING` 项目（kind=`init`，full resync）。
4. `_start_scheduler()` — 启动 APScheduler（返回值被丢弃，无全局单例变量）。
5. `_warn_if_models_missing()` — 磁盘存在性检查（不加载不联网），缺失则告警提示 `bootstrap_models`。
6. `mcp.run(transport="stdio")` — 阻塞服务。

### 8.2 预检守卫

`_ensure_ready(require) -> dict | None`：`require ⊆ {"jama","embedding","llm"}`，调 `preflight.preflight(require=...)`（离线：包+配置+存储）。若 `report["blocking"]` 真则返回错误字典，否则 `None` 放行。

```python
{"error": "Server is not ready: ...",
 "issues": [...],
 "hint": "...configure_jama...validate_setup..."}
```

每个受守卫工具开头统一 `not_ready = _ensure_ready({...}); if not_ready: return not_ready`。守卫矩阵：

- `{"jama","embedding"}`：`init_jama_project`、`reinit_jama_project`、`search_jama_semantics`。
- `{"jama"}`：所有 browse/native 工具。
- `set()`：`bootstrap_models`、`get_bootstrap_progress`、`get_sync_progress`、`get_sync_status`。
- 不受守卫：`configure_jama`（写配置在 backend 存在前）、`validate_setup`（它本身是预检）。

### 8.3 异步任务系统

- `_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jama-job")` — init/reinit/bootstrap/调度同步共用，`max_workers=2` 允许不同项目并行（同项目由 `_project_lock` 串行）。
- `_project_lock(pid)` — `_init_lock` 下创建的 per-project `Lock`，串行化同项目的用户 init 与调度增量同步。
- **并发守卫**（`_start_sync_job`，`_init_lock` 下）：`get_active_job_for_project` 非空且项目 `INITIALIZING` → 返回已有 `job_id` + note，**拒绝重复并发同步**（避免 upsert 竞争与 split-brain 终态）。已终止项目（READY/ERROR）的"僵尸" job 行允许新同步。
- **job_id**：`f"{kind}-{uuid.uuid4().hex[:12]}"`（kinds: init/reinit/bootstrap/sync/resume）。
- **提交回滚**：job 行创建+项目置 `INITIALIZING` 后，**锁外**提交 `_run_job`；提交失败则回滚 job→ERROR、project→ERROR，避免留幽灵 RUNNING。
- **`_run_job(pid, job_id, incremental)`**：调 `_sync_project`；异常时防御性把 job+project 都置 ERROR，若终态写入本身失败则大声告警（留 RUNNING 由下次启动恢复）。
- **`_sync_project`**：取 `_project_lock` 后转 `_sync_project_locked`。
- **`_sync_project_locked`** 核心步骤：置 `INITIALIZING` → **模型预下载**（embedder+reranker 各自 try/except，失败非致命）→ **Jama 带宽预检**（最多 3 次重试，间隔 5s）→ 取项目名 → 生产者/消费者取数+嵌入+原子写 → 终态（`done==0` 也算 DONE；成功置 `READY`+`last_sync_time`）。
- **`_run_bootstrap_job`**：Phase1 嵌入（仅 local，progress 0.1→0.45）、Phase2 重排（0.5→0.95，相位推进无字节级）；终态 ERROR 或 DONE。

### 8.4 增量同步（APScheduler）

`_start_scheduler()`：`BackgroundScheduler(timezone="UTC")`，`add_job(_incremental_sync_all, "interval", hours=sync.hours, max_instances=1, coalesce=True, next_run_time=None)`（不在启动即触发，仅首个间隔后）。`max_instances=1`+`coalesce=True` 防重叠/堆积。

`_incremental_sync_all()`：`list_initialized_projects` → 每个项目在 `_init_lock` 下查 active job，有则跳过，无则 `create_job(kind="sync")` 提交 `_run_job(..., incremental=True)`。增量逻辑在 `_sync_project_locked`：读 `last_sync_time` 作 `modified_after` 传给 `iter_project_items`，重分块/重嵌入/`replace_chunks` 原子替换，成功后推进 `last_sync_time=utcnow_iso()`，`max_items` 封顶。

### 8.5 监控工具

- `get_sync_progress(job_id)` / `get_bootstrap_progress(job_id)`：查 `get_job`，返回 `_job_summary`。
- `_job_summary(row)`：`{job_id,project_id,kind,status, progress(0..1→×100 取一位), total,done,message,started_at,finished_at}`。
- `get_sync_status(project_id)`：`{project_id, project_status, last_sync_time, item_count, chunk_count, active_job|null, recent:{init,reinit,sync}各取最新|null, process:{rss_mb,threads,db_mb,chunks}|null}`（`_process_metrics` 用 psutil，不可用则 null；故意不含 cpu_pct，因需阻塞基线间隔）。

### 8.6 搜索工具接线

`search_jama_semantics`：守卫 `{"jama","embedding"}` → 校验 `project_id` 数字、`query` 非空、项目 `READY`/`INITIALIZING`（允许同步中搜索）、`count_chunks>0`、`item_type` int、夹 `top_k(1..50)`/`candidate_k(1..500)` 且 `top_k<=candidate_k`、提前用 `_normalize_iso_utc` 校验时间格式 → 调 `rag().search(...)` → 异常返回 `{"error":...}`；**回显实际使用的子查询**（防御性调 `_normalize_sub_queries`，反射失败不丢已算结果）。返回 `{project_id, query, sub_queries_used, count, modified_after, modified_before, results:[{document_key,name,item_type_name,section,modified_date,text,score,strategy}]}`。

`query_jama_native_metadata`：守卫 `{"jama"}`，转 `jama().query_items_native(...,limit=20)`，返回 `{project_id,count,results}`。

### 8.7 配置/校验工具

- `configure_jama(values)`：校验非空 dict → `write_env_file` → `reload_settings` → `reset_singletons`（清 DB/Jama/RAG 单例下次重建）；密钥只写盘不回显；返回 `{ok, written, applied_keys}`。
- `validate_setup(live=False)`：直接调 `preflight(require={"jama","embedding"})`；`live=True` 且非阻塞时 `_live_probe()`（Jama `preflight_speed_check`+`list_projects`、嵌入 `embed_one` 探测）。

---

## 9. 模型引导（Model Bootstrap）

模型（嵌入 ~67MB + 重排 ~80MB）**不打包**，首次使用下载。三入口：

1. **MCP 工具 `bootstrap_models`**：异步后台任务（`kind="bootstrap"` job），`get_bootstrap_progress` 每 ~2min 轮询。重入守卫（`_bootstrap_lock`+`_bootstrap_job_id`）。同步内 `ensure_downloaded` 作兜底。
2. **`bootstrap.py`**：前台同步 CLI，`_download_embedding`（仅 local）+ `_download_reranker`，重跑是快速 no-op。
3. **启动提示** `_warn_if_models_missing`：磁盘存在性检查，缺失告警。

单例 + 锁（`_download_lock`/`_weights_lock`）保证并发 bootstrap 与 sync 不损坏共享 HF 缓存。

---

## 10. 网络守卫（`net_guard.py`）

- `NetworkTooSlowError` — 携带实测速度，供调用方给出清晰消息。
- `speed_test(url, min_bytes_per_sec, timeout)` — 下载有界探针（≤1MB 或 `timeout` 秒），**吞吐时钟从首字节到达开始**（排除 TLS 握手/服务端处理延迟），低于地板抛 `NetworkTooSlowError`。
- `download_with_retry(url, dest, min_bytes_per_sec, max_retries=4, chunk_timeout=30, progress_callback)` — **断点续传**：每 attempt 从已有字节 `Range:` 续传；5s 滚动窗口测速低于地板则中断重试；`progress_callback(received, expected)` 每块后触发（供 bootstrap 监控）；`_content_length` HEAD 取期望大小。

---

## 11. 预检（`preflight.py`）

三段式离线校验（快、首次后缓存）：

1. **Python 包**：核心（requests/bs4/sqlite_vec/apscheduler + 按提供者的 fastembed）缺失即 blocking；可选（mcp/llama_index/transformers/torch）缺失仅 warn。
2. **配置**：`validate_config()`，按 `require` 把对应 feature 的 error issue 标 blocking。
3. **存储**：`init_db()` + `SELECT 1 FROM chunks_vec LIMIT 0` 验证可用。

返回 `{ok, blocking, dependencies, config_issues, storage, issues[], hint}`。不校验网络/凭据有效性（那是 `validate_setup` 的 live 慢操作）。

---

## 12. 关键设计决策与权衡

| 决策 | 理由 |
|------|------|
| MiniLM 交叉编码器替代 Qwen3-Reranker-0.6B | 80MB vs 1.2GB；无 152k-vocab logits 张量；RRF 已融合召回，只重排 25 条，小模型足够 |
| 默认 local 嵌入（bge-small-en CPU） | 查询时无网络、无 API key；中国镜像可下；CPU 线程 60% 留余量 |
| Multi-Query 由 MCP 客户端扩展 | 服务端不配 chat LLM；客户端传 `sub_queries`；缺省确定性词法回退保 RRF 多角度收益 |
| 单条目原子写入（`write_txn`） | 崩溃不丢已提交数据；进度只在提交后推进；项目留 INITIALIZING 由 resume 补 |
| `INSERT OR REPLACE` 幂等 | 重同步/碰撞安全，`replace_chunks` 文档明示"per-item idempotent" |
| 有界 429 循环（非递归） | 持续限流干净失败而非栈溢出 |
| WAL + 进程级 RLock + busy_timeout | 写者串行化、读不阻塞写、读线程不遇 SQLITE_BUSY |
| 维度变更重建保留 items | 切提供者只重嵌不重取 Jama |
| torch pin `2.6.0+cpu` / onnxruntime pin | 默认 CUDA torch ~6GB 依赖新 VC++ Runtime（WinError 1114）；CPU 版无此依赖 |
| `check_same_thread=False` | MCP 读线程与调度写线程共享连接 |

---

## 13. 数据流总览

### 13.1 首次 init（`init_jama_project`）

```
LLM → init_jama_project(pid)
  → _ensure_ready({jama,embedding}) → _start_sync_job(pid,"init")
     → 并发守卫 → create_job(RUNNING) → upsert_project(INITIALIZING)
     → _executor.submit(_run_job)
  ← 立即返回 {job_id, status:RUNNING}

_run_job → _sync_project(_project_lock) → _sync_project_locked:
  1. ensure_downloaded(embedder) ; ensure_downloaded(reranker)  [非致命]
  2. client.preflight_speed_check()  [≤3 重试]
  3. _fetcher 线程: iter_project_items(concurrency=16)
       → chunk_item → item_queue(背压)
  4. 主线程: 攒批 → embed_many → 每项 write_txn{upsert_item+replace_chunks}
       → update_job(done/total/progress)
  5. 终态: READY + last_sync_time ; job DONE

LLM 轮询 get_sync_progress(job_id) ~每2min → DONE
```

### 13.2 搜索（`search_jama_semantics`）

```
LLM → search_jama_semantics(pid, query, sub_queries=[...], top_k=5)
  → _ensure_ready → 校验 → rag().search():
     Multi-Query 规范化 → 每子查询 {vector_search + fts_search}
     → rrf_fuse[:candidate_k] → fetch_chunks_by_ids
     → Reranker.rerank(query, texts) → [全零? 用 RRF 分数]
     → 取 top_k, 标 strategy
  ← {results:[{document_key,name,...,text,score,strategy}], ...}
```

### 13.3 增量同步（APScheduler，每 2h）

```
定时 → _incremental_sync_all
  → list_initialized_projects → 每项目 [有 active job 跳过]
     → create_job(kind="sync") → submit(_run_job, incremental=True)
        → _sync_project_locked: modified_after=last_sync_time
           → 只取 modifiedDate>last_sync 的项 → 重分块/重嵌入/replace_chunks
           → 推进 last_sync_time ; job DONE
```

### 13.4 崩溃恢复（启动）

```
main():
  init_db
  reconcile_stale_jobs   # PENDING/RUNNING → ERROR [interrupted by restart]
  _resume_interrupted_syncs   # status=INITIALIZING → create_job(init) → 全量重同步
  _start_scheduler
```

---

## 14. 部署与运行

### 安装

```bash
pip install -r requirements.txt        # Aliyun 镜像 + PyTorch CPU index
python setup_wizard.py --self-test     # 写 .env + 预检 + 自检 + 预下载模型
python server.py                        # stdio 传输
```

`requirements.txt` 关键 pin：`torch==2.6.0+cpu`、`onnxruntime==1.20.1`(py<3.13)/`1.21.1`(py≥3.13)、`setuptools<81`（APScheduler 3.x 依赖 pkg_resources）。

### MCP 客户端配置（Claude Desktop 示例）

```json
{
  "mcpServers": {
    "jama-mcp": {
      "command": "python",
      "args": ["/abs/path/server.py"],
      "env": { "JAMA_MCP_DB_PATH": "/abs/path/jama_mcp.db" }
    }
  }
}
```

### 推荐使用流（给 LLM）

0. `bootstrap_models()` → 轮询 `get_bootstrap_progress` 至 DONE（已缓存则跳过）。
1. `init_jama_project("20571")` → 返回 `job_id`（非阻塞）。
2. `get_sync_progress(job_id)` → ~每 2min 轮询至 `DONE`。
3. `search_jama_semantics("20571","...",top_k=5)` → RAG。
4. `query_jama_native_metadata("20314",document_key="SA-TC-7")` → 精确匹配。
5. 重索引：`reinit_jama_project`；查状态：`get_sync_status`。

### 自测

`python selftest.py` 端到端验证：预检、OAuth、所有 browse 方法形状、MCP 工具注册（23 个）、预检守卫拦截误配、并发同步+批量嵌入、崩溃恢复（INITIALIZING→READY 自动重同步）、`search` 子查询路径。全部只读 GET。

---

## 15. 辅助脚本

| 脚本 | 用途 |
|------|------|
| `setup_wizard.py` | 交互式配置（写 `.env` + 预检 + 可选 live 自检 + 预下载模型，`--skip-models` 可跳过） |
| `bootstrap.py` | 前台同步预下载嵌入+重排模型；重跑 no-op |
| `selftest.py` | 端到端自测套件（只读） |
| `bench_lyra_search.py` | 搜索延迟基准：`candidate_k`/`top_k`/重排开关/单嵌入延迟 |
| `monitor_lyra_init.py` | 进程内 init + 定时采样进程/DB 指标，定位同步瓶颈 |
| `test_multiquery.py` | Multi-Query 重构离线单测（`_normalize_sub_queries`/`MultiQueryExpander`/`search` 接线，DB/嵌入/重排全 stub） |

---

## 16. MCP 工具清单（共 23 个）

**配置/校验**
- `validate_setup(live=False)` — 离线预检（+可选 live Jama/嵌入探测）
- `configure_jama(values)` — 运行时应用配置、写 `.env`、reload、重置单例
- `bootstrap_models()` — 异步预下载嵌入+重排模型，返回 `job_id`
- `get_bootstrap_progress(job_id)` — 轮询 bootstrap 任务

**RAG / 同步监控**
- `init_jama_project(project_id)` — 异步全量初始化，返回 `job_id`
- `reinit_jama_project(project_id)` — 已初始化项目的全量重同步
- `get_sync_progress(job_id)` — 轮询单个 init/reinit/sync 任务
- `get_sync_status(project_id)` — 项目监控：在飞任务 + 最近 init/reinit/sync + 进程指标
- `search_jama_semantics(project_id, query, sub_queries?, item_type?, top_k=5, candidate_k=25, modified_after?, modified_before?)` — Multi-Query+混合+RRF+交叉编码器重排
- `query_jama_native_metadata(project_id, document_key?, item_type?, status?, keyword?)` — `/abstractitems` 精确元数据（≤20 条）

**Jama browse（只读，预检守卫）**
- `list_jama_projects()` / `find_jama_project_by_name(name, exact?, limit=20)`
- `get_jama_item(item_id)` / `get_jama_item_children(item_id, limit=50)`
- `get_jama_item_relationships(item_id, limit=50)` / `list_jama_project_relationships(project_id, item_id?, limit=50)`
- `get_jama_item_comments(item_id, limit=50)` / `get_jama_item_attachments(item_id, limit=50)`
- `list_jama_releases(project_id, limit=50)` / `list_jama_test_runs(project_id?, test_cycle_id?, limit=50)`
- `list_jama_item_types()` / `find_jama_item_type_by_name(name, exact?, limit=20)`
- `query_jama_endpoint(path, params?, all_pages?)` — 通用只读 GET 逃生舱（all_pages 封顶 50 页）

---

## 17. 可靠性总结

- **Jama API**：OAuth 自动刷新+401 重试；urllib3 Retry 指数退避 429/5xx；显式 `Retry-After`；SSL 重置容忍；有界 429 循环；卡顿页重试；损坏 JSON 重试。
- **嵌入**：同一 retry/backoff session（azure）；local 单例+粘性错误+镜像顺序下载+带宽预检。
- **SQLite 并发**：WAL + busy_timeout + 进程级写锁；单条目原子替换。
- **重排器**：懒加载单例；失败降级 RRF 而非崩溃搜索。
- **崩溃恢复**：原子写入 + 进度后置推进 + 启动 `reconcile_stale_jobs` + `_resume_interrupted_syncs` + 幂等 upsert。
- **只读**：JamaClient 仅发 GET，无法创建/修改/删除 Jama 数据。

---

## 附录 A：项目目录布局

```
jama/
├── server.py            # MCP 入口
├── rag_pipeline.py      # RAG 流水线
├── jama_client.py       # Jama REST 客户端
├── db_setup.py          # SQLite 存储层
├── config.py            # 配置
├── preflight.py         # 预检
├── net_guard.py         # 网络守卫
├── setup_wizard.py      # 配置向导
├── bootstrap.py         # 模型预下载 CLI
├── selftest.py          # 自测
├── bench_lyra_search.py # 基准
├── monitor_lyra_init.py # 监控采样
├── test_multiquery.py   # Multi-Query 单测
├── requirements.txt     # 依赖 + 镜像
├── .env.example         # 配置模板
└── user/                # 运行时数据（gitignore）
    ├── jama_mcp.db          # SQLite + WAL
    └── huggingface/         # 模型缓存（~150MB）
```

## 附录 B：环境变量速查

| 变量 | 默认 | 说明 |
|------|------|------|
| `JAMA_URL` / `JAMA_CLIENT_ID` / `JAMA_CLIENT_SECRET` | — | Jama OAuth（必填） |
| `JAMA_API_PREFIX` | `/rest/latest` | REST 版本路径 |
| `JAMA_PAGE_SIZE` / `JAMA_PAGE_DELAY` | 50 / 0.25 | 分页调优 |
| `EMBEDDING_PROVIDER` | `local` | `local`(bge CPU) / `azure` |
| `EMBEDDING_LOCAL_MODEL` | `BAAI/bge-small-en-v1.5` | local 模型 |
| `EMBEDDING_CPU_PERCENT` | 60 | ONNX 线程占比 |
| `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY`/`EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` | — | azure 提供者 |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 重排器 |
| `RERANKER_BATCH_SIZE`/`RERANKER_MAX_LENGTH` | 16 / 256 | 重排调优 |
| `RERANKER_ALLOW_FALLBACK` | 1 | 加载失败降级 RRF |
| `JAMA_MCP_DB_PATH` | `user/jama_mcp.db` | SQLite 路径 |
| `SQLITE_BUSY_TIMEOUT_MS` | 5000 | 写并发等待 |
| `SYNC_ENABLED` / `SYNC_INTERVAL_HOURS` | 1 / 2 | 增量同步 |
| `SYNC_DOWNLOAD_CONCURRENCY` | 16 | 并发分页 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 80 | 分块 |
