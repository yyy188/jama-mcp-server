"""RAG pipeline: chunking, embeddings, Multi-Query, hybrid recall, RRF, rerank.

Retrieval chain (``search`` method):
1. Multi-Query   - sub-queries supplied by the caller (the MCP LLM client is
                   expected to expand the user query into 3-5 variants); when
                   none are supplied, deterministic lexical variants are used
                   so RRF fusion still benefits from multiple query angles.
2. Hybrid recall - for each sub-query: vector recall (sqlite-vec) + keyword
                   recall (FTS5), each limited to ``candidate_k``.
3. RRF fusion    - Reciprocal Rank Fusion merges all candidate lists into one
                   ranked list of <= ``candidate_k`` unique chunks.
4. Rerank        - local Qwen3-Reranker-0.6B scores (query, chunk) pairs; the
                   top ``top_k`` are returned. If the model is unavailable and
                   ``allow_fallback`` is set, RRF scores are used directly.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

import requests

from config import settings
from db_setup import (fts_search, get_connection, vector_search,
                      fetch_chunks_by_ids)

log = logging.getLogger(__name__)

# LlamaIndex is the primary RAG framework: it provides the recursive splitter
# (SentenceSplitter) and the Document/TextNode document model used for
# chunking. Multi-Query expansion is performed by the MCP LLM client and
# passed in via ``search(sub_queries=...)``; no server-side chat LLM is used.
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode


# --------------------------------------------------------------------------- #
# Chunking (LlamaIndex SentenceSplitter - recursive, sentence-aware)
# --------------------------------------------------------------------------- #
def make_splitter() -> SentenceSplitter:
    """LlamaIndex's recursive sentence-aware splitter.

    SentenceSplitter recursively splits on paragraph/sentence boundaries and
    only falls back to word-level splitting when a single sentence exceeds
    ``chunk_size``. ``chunk_overlap`` preserves context between chunks, which
    matters because ~30% of Jama items are long-form.
    """
    return SentenceSplitter(
        chunk_size=settings.chunk.chunk_size,
        chunk_overlap=settings.chunk.chunk_overlap,
        # Primary separator is paragraph breaks; SentenceSplitter handles the
        # full recursive fallback chain (sentence -> word) internally.
        separator="\n\n",
        paragraph_separator="\n\n\n",
        chunking_tokenizer_fn=None,
        secondary_chunking_regex="[^,.;。]+[,.;。]?",
        include_metadata=True,
        include_prev_next_rel=False,
    )


def chunk_item(item: dict) -> list[dict]:
    """Build chunk rows for one normalized item using LlamaIndex Documents.

    The item's description and (for Test Cases) rendered steps are wrapped in
    LlamaIndex ``Document`` objects, split into ``TextNode`` chunks via the
    ``SentenceSplitter`` (cleaned plain text is already present - HTML was
    stripped with BeautifulSoup in jama_client before reaching here), and then
    flattened back into storage rows. The item name is prepended to every chunk
    so semantic/keyword recall can always match on the title.
    """
    splitter = make_splitter()
    chunks: list[dict] = []
    item_id = item["item_id"]
    name = item.get("name") or ""

    sections: list[tuple[str, str]] = []
    if item.get("description"):
        sections.append(("description", item["description"]))
    if item.get("test_steps"):
        sections.append(("test_steps", item["test_steps"]))
    if not sections and name:  # keep at least the name searchable
        sections.append(("description", name))

    for section, text in sections:
        # Build a LlamaIndex Document and split into TextNodes. Metadata is
        # carried through so downstream LlamaIndex consumers can use it.
        doc = Document(
            text=text,
            metadata={
                "item_id": item_id,
                "section": section,
                "document_key": item.get("document_key"),
                "modified_date": item.get("modified_date"),
            },
        )
        nodes: list[TextNode] = splitter.get_nodes_from_documents([doc])
        for i, node in enumerate(nodes):
            part = (node.text or "").strip()
            if not part:
                continue
            # Prefix with the item name for better retrieval signal.
            body = f"{name}\n{part}".strip() if name else part
            chunks.append({
                "chunk_id": f"{item_id}#{section}#{i}",
                "item_id": item_id,
                "project_id": item["project_id"],
                "item_type": item.get("item_type"),
                "item_type_name": item.get("item_type_name"),
                "document_key": item.get("document_key"),
                "name": name,
                "status": item.get("status"),
                "section": section,
                "chunk_index": i,
                "text": body,
                "modified_date": item.get("modified_date"),
            })
    return chunks


# --------------------------------------------------------------------------- #
# Embedding client (OpenAI-compatible, Azure gateway friendly)
# --------------------------------------------------------------------------- #
class EmbeddingClient:
    """Batched embedding calls against an OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        s = settings.embedding
        self._url = f"{s.base_url.rstrip('/')}/openai/v1/embeddings"
        self._model = s.model
        self._batch = s.batch_size
        self._timeout = s.timeout
        self._headers = {"Content-Type": "application/json"}
        if s.key_header.lower() == "authorization":
            self._headers["Authorization"] = f"Bearer {s.api_key}"
        else:
            self._headers[s.key_header] = s.api_key
        self._sess = requests.Session()
        # Resilience: retry transient network resets and 429s with backoff.
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=5, backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        self._sess.mount("https://", HTTPAdapter(max_retries=retry))
        self._sess.mount("http://", HTTPAdapter(max_retries=retry))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i:i + self._batch]
            r = self._sess.post(self._url, headers=self._headers,
                                json={"model": self._model, "input": batch},
                                timeout=self._timeout)
            if r.status_code != 200:
                raise RuntimeError(
                    f"Embedding API {r.status_code}: {r.text[:300]}")
            data = r.json()["data"]
            # API may return out of order; sort by index to be safe.
            data.sort(key=lambda d: d.get("index", 0))
            out.extend(d["embedding"] for d in data)
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_many_concurrent(self, texts: list[str],
                              concurrency: int = 4) -> list[list[float]]:
        """Embed a large text list with concurrent batch requests.

        Splits ``texts`` into ``batch_size`` slices and fires up to
        ``concurrency`` of them in parallel (the embedding endpoint is
        stateless, so parallel HTTP requests are safe). Results are reassembled
        in input order. Falls back to the serial ``embed_texts`` when
        ``concurrency <= 1`` or there's only one batch.

        This is the indexing path used by ``_sync_project``: a 5000-item
        project with ~7000 chunks / batch_size 64 = ~110 batches; at
        concurrency=4 the wall-time is ~4x shorter than serial.
        """
        if not texts:
            return []
        if concurrency <= 1 or len(texts) <= self._batch:
            return self.embed_texts(texts)
        from concurrent.futures import ThreadPoolExecutor

        # Slice into batches; record (start_index, batch) so we can place
        # each result back at the right position.
        slices: list[tuple[int, list[str]]] = []
        for i in range(0, len(texts), self._batch):
            slices.append((i, texts[i:i + self._batch]))

        out: list[list[float] | None] = [None] * len(texts)

        def _embed_slice(start_batch: list[str], batch_texts: list[str]) -> None:
            return start_batch, batch_texts

        def _do(batch_texts: list[str]) -> list[list[float]]:
            r = self._sess.post(self._url, headers=self._headers,
                                json={"model": self._model, "input": batch_texts},
                                timeout=self._timeout)
            if r.status_code != 200:
                raise RuntimeError(
                    f"Embedding API {r.status_code}: {r.text[:300]}")
            data = r.json()["data"]
            data.sort(key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]

        with ThreadPoolExecutor(max_workers=concurrency,
                                thread_name_prefix="embed") as ex:
            futures = {ex.submit(_do, bt): (start, bt) for start, bt in slices}
            for fut in futures:
                start, bt = futures[fut]
                embs = fut.result()
                for j, e in enumerate(embs):
                    out[start + j] = e
        # All positions filled (batches are contiguous); cast away None.
        return [v for v in out if v is not None]


# --------------------------------------------------------------------------- #
# Local embedding client (bge-small-en-v1.5 on CPU via fastembed / ONNX)
# --------------------------------------------------------------------------- #
# bge-small-en-v1.5 query prefix. The v1.5 model was trained so the prefix is
# "not so necessary", but adding it for the query side still measurably helps
# retrieval recall against documents indexed without the prefix. Documents are
# embedded raw; only queries get the prefix.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddingClient:
    """CPU-only local embedding via fastembed (bge-small-en-v1.5, ONNX).

    Production-grade characteristics:
      * **Lazy singleton load** — the ~130MB ONNX model loads on first embed,
        never at import, so server startup is fast and a missing model never
        blocks MCP serving.
      * **Model presence check + mirrored download** — if the model isn't
        cached, download it. China mirror (hf-mirror.com) is tried first with
        a pre-flight speed test; if too slow or unreachable, fall back to the
        global HuggingFace endpoint; if that's also too slow, abort with a
        clear error instead of hanging for hours.
      * **CPU cap** — ONNX runtime threads are capped at ``cpu_percent`` of the
        system's cores (default 60%) so the MCP server and scheduler keep
        headroom on a shared host.
      * **Thread-safe** — a lock serialises embed calls because the underlying
        ONNX session is not re-entrant-safe for concurrent forward passes on
        the same instance.
      * **Graceful failure** — a load error is sticky; subsequent calls return
        the same error rather than retrying on every request.
    """

    _instance: "LocalEmbeddingClient | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._cfg = settings.embedding
        self._model = None  # fastembed.TextEmbedding, lazy
        self._load_error: str | None = None
        self._embed_lock = threading.Lock()

    # ----- model loading + download ------------------------------------- #
    def _thread_count(self) -> int:
        """ONNX threads = ceil(cpu_count * cpu_percent / 100), min 1."""
        import math
        cores = os.cpu_count() or 4
        return max(1, math.ceil(cores * self._cfg.cpu_percent / 100))

    def _cache_dir(self) -> str:
        """fastembed cache root — project-local user/huggingface for portability."""
        from config import USER_DIR
        return os.environ.get("HF_HOME", str(USER_DIR / "huggingface"))

    def _model_present(self) -> bool:
        """True if the bge-small-en-v1.5 ONNX model is already cached locally."""
        # fastembed stores models under <cache>/models--<org>--<name>. We check
        # for the snapshot dir with the ONNX file present.
        model = self._cfg.local_model
        repo_dir = "models--" + model.replace("/", "--")
        base = os.path.join(self._cache_dir(), "models", model)
        # fastembed uses a slightly different layout: <cache>/models/<org>/<name>
        # and also <cache>/models--<org>--<name>. Probe both.
        candidates = [
            os.path.join(self._cache_dir(), "models", model),
            os.path.join(self._cache_dir(), repo_dir),
        ]
        for c in candidates:
            if os.path.isdir(c):
                for dirpath, _dirs, files in os.walk(c):
                    if any(f.endswith(".onnx") for f in files):
                        return True
        return False

    def _download_model(self) -> None:
        """Download the model via fastembed, China mirror first then global.

        Sets ``HF_ENDPOINT`` to the mirror before invoking fastembed (which uses
        huggingface_hub under the hood and honours that env var). Runs a
        pre-flight speed test against the mirror; if it's below the configured
        floor or unreachable, switches to the global endpoint and retries. If
        both fail, raises ``RuntimeError`` with an actionable message.
        """
        from net_guard import speed_test, NetworkTooSlowError
        from fastembed import TextEmbedding

        # fastembed downloads the ONNX model from a Qdrant-published repo, not
        # the original BAAI repo. Resolve the real HF source so the speed-test
        # probe hits a file that actually exists.
        model_meta = next(
            (m for m in TextEmbedding.list_supported_models()
             if m["model"] == self._cfg.local_model), None)
        hf_repo = (model_meta or {}).get("sources", {}).get("hf", self._cfg.local_model)
        model_file = (model_meta or {}).get("model_file", "model_optimized.onnx")

        mirrors = [
            ("China mirror", "https://hf-mirror.com"),
            ("HuggingFace (global)", "https://huggingface.co"),
        ]
        min_bps = self._cfg.download_min_bps
        # Probe URL for the speed test: a small file from the ONNX model repo.
        probe_path = f"{hf_repo}/resolve/main/config.json"

        last_err: str | None = None
        for label, endpoint in mirrors:
            os.environ["HF_ENDPOINT"] = endpoint
            probe_url = f"{endpoint}/{probe_path}"
            try:
                bps = speed_test(probe_url, min_bytes_per_sec=min_bps,
                                 timeout=20, label=f"emb-{label}")
                log.info("Embedding model mirror %s OK: %.0f bytes/s", label, bps)
            except NetworkTooSlowError as exc:
                last_err = f"{label} ({endpoint}): {exc}"
                log.warning("Embedding mirror %s too slow/unreachable: %s",
                            label, exc)
                continue
            # Mirror is fast enough — trigger the actual download by
            # constructing TextEmbedding (it downloads on first use).
            try:
                log.info("Downloading %s (ONNX from %s) via %s ...",
                         self._cfg.local_model, hf_repo, label)
                te = TextEmbedding(
                    model_name=self._cfg.local_model,
                    cache_dir=self._cache_dir(),
                    threads=self._thread_count(),
                    cuda=False,
                )
                # Force the actual model load (download + ONNX init) now so a
                # download failure surfaces here, not on the first embed.
                _ = list(te.embed(["warmup"], batch_size=1))
                self._model = te
                log.info("Local embedding model ready (%s, %d CPU threads).",
                         self._cfg.local_model, self._thread_count())
                return
            except Exception as exc:
                last_err = f"{label} download failed: {exc}"
                log.warning("Embedding download from %s failed: %s", label, exc)
                continue

        # Both mirrors failed.
        self._load_error = (
            f"Could not download embedding model {self._cfg.local_model} "
            f"(ONNX source {hf_repo}/{model_file}) from any mirror. "
            f"Last error: {last_err}. Check network connectivity or "
            f"pre-download the model manually into {self._cache_dir()}.")
        raise RuntimeError(self._load_error)

    def _load(self) -> bool:
        """Load the model (cached or freshly downloaded). True on success."""
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            if self._model_present():
                log.info("Loading cached local embedding model %s ...",
                         self._cfg.local_model)
                from fastembed import TextEmbedding
                self._model = TextEmbedding(
                    model_name=self._cfg.local_model,
                    cache_dir=self._cache_dir(),
                    threads=self._thread_count(),
                    cuda=False,
                )
                _ = list(self._model.embed(["warmup"], batch_size=1))
                log.info("Local embedding model ready (%d CPU threads).",
                         self._thread_count())
                return True
            self._download_model()
            return self._model is not None
        except Exception as exc:
            self._load_error = str(exc)
            log.error("Local embedding model unavailable: %s", exc)
            return False

    def ensure_downloaded(self) -> None:
        """Pre-download the embedding model without loading it.

        Called at the very start of project sync so the (64 MB) download
        happens before any indexing work, alongside the reranker weights. If
        the model is already cached this is a fast no-op. A failure here is
        non-fatal: the actual load (and error) is deferred to first embed.
        """
        if self._model is not None or self._load_error is not None:
            return
        if self._model_present():
            return
        try:
            self._download_model()
        except Exception as exc:
            log.warning("Embedding model pre-download failed (%s); "
                        "will retry on first embed.", exc)

    # ----- embed API (mirrors EmbeddingClient) ------------------------- #
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts (no query prefix)."""
        if not texts:
            return []
        if not self._load():
            raise RuntimeError(f"Local embedding unavailable: {self._load_error}")
        out: list[list[float]] = []
        with self._embed_lock:
            for i in range(0, len(texts), self._cfg.batch_size):
                batch = texts[i:i + self._cfg.batch_size]
                for vec in self._model.embed(batch,
                                             batch_size=self._cfg.batch_size):
                    out.append([float(x) for x in vec])
        return out

    def embed_one(self, text: str) -> list[float]:
        """Embed a single query (with the bge query prefix for better recall)."""
        return self.embed_texts([_BGE_QUERY_PREFIX + text])[0]

    def embed_many_concurrent(self, texts: list[str],
                              concurrency: int = 1) -> list[list[float]]:
        """Embed a large document list. Local is CPU-bound so concurrency is
        ignored — a single ONNX session already saturates the capped threads,
        and parallel sessions would oversubscribe the CPU. Serial batched embed
        is the correct and fastest path here."""
        return self.embed_texts(texts)


# --------------------------------------------------------------------------- #
# Multi-Query expansion
# --------------------------------------------------------------------------- #
def _normalize_sub_queries(subs: list[str], query: str,
                           max_n: int = 5) -> list[str]:
    """Normalize caller-supplied sub-queries for RRF fusion.

    The original ``query`` is always forced to the front (rerank scores
    ``(query, chunk)`` pairs against it, so it must be one of the recall
    angles). Empty/non-string entries are dropped and the list is
    de-duplicated case-insensitively (preserving first-seen order) and capped
    at ``max_n`` — each sub-query triggers its own vector + FTS5 recall pass,
    so an unbounded list would multiply recall cost. Falls back to ``[query]``
    when nothing usable remains.
    """
    out: list[str] = []
    seen: set[str] = set()
    # Original query first, so rerank always has its native phrasing.
    for s in [query, *subs]:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    if not out:
        out = [query] if query else []
    return out[: max(max_n, 1)]


class MultiQueryExpander:
    """Deterministic lexical Multi-Query fallback.

    Query expansion is normally done by the MCP LLM client and passed to
    :meth:`RAGPipeline.search` via ``sub_queries``. This expander is the
    zero-configuration fallback used when no sub-queries are supplied: it
    derives deterministic lexical variants (stopword-stripped + truncated) so
    RRF fusion still benefits from multiple recall angles. It makes no network
    calls and needs no configuration.
    """

    def __init__(self, n: int = 3) -> None:
        self.n = max(n, 1)

    def expand(self, query: str) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []
        return self._lexical_expand(query)

    def _lexical_expand(self, query: str) -> list[str]:
        """Deterministic variants: original + keyword-focused + broadened."""
        subs = [query]
        # Keep only alphanumeric tokens (drop common stopwords).
        stop = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
                "is", "are", "be", "with", "that", "this", "it", "as", "by",
                "how", "does", "do", "what", "which", "why", "when", "can",
                "will", "should", "would", "could", "i", "we", "you"}
        tokens = [t for t in re.findall(r"[A-Za-z0-9_-]+", query)
                  if t.lower() not in stop]
        if tokens:
            # keyword-focused (non-stopword tokens only)
            kw = " ".join(tokens)
            if kw.lower() != query.lower():
                subs.append(kw)
            # broadened / truncated to first few meaningful tokens
            if len(tokens) > 4:
                subs.append(" ".join(tokens[:4]))
        # Deduplicate while preserving order; always keep the original first.
        seen, out = set(), []
        for s in subs:
            if s and s.lower() not in seen:
                seen.add(s.lower()); out.append(s)
        # Pad with the original if we don't have enough diverse variants.
        while len(out) < self.n and query:
            out.append(query)
        return out[: max(self.n, 1)]


# --------------------------------------------------------------------------- #
# RRF fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Returns [(chunk_id, score)] sorted desc."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- #
# Qwen3 Reranker (lazy-loaded singleton)
# --------------------------------------------------------------------------- #
class QwenReranker:
    """Local Qwen3-Reranker-0.6B scorer (CPU).

    Loads lazily on first use so the MCP server starts fast and so a missing
    model never blocks startup. With ``allow_fallback`` the pipeline degrades
    to RRF-only scoring if loading fails.
    """

    _instance: "QwenReranker | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._cfg = settings.reranker
        self._tokenizer = None
        self._model = None
        self._load_error: str | None = None

    def ensure_downloaded(self) -> None:
        """Pre-download reranker weights without loading the model.

        Called during project sync so the (1.2 GB) download happens up front,
        alongside the embedding model — not deferred to the first search
        (where a download failure would surprise the user mid-query). If the
        weights are already cached this is a no-op. A failure here is
        non-fatal: the reranker stays unloaded and search degrades to RRF.
        """
        if self._model is not None or self._load_error is not None:
            return
        try:
            self._ensure_weights_downloaded()
            log.info("Qwen3-Reranker weights ready (pre-downloaded during sync).")
        except Exception as exc:
            log.warning("Qwen3-Reranker pre-download failed (%s); "
                        "will retry on first search or fall back to RRF.", exc)

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            log.info("Loading Qwen3-Reranker %s on %s ...",
                     self._cfg.model_name, self._cfg.device)
            # Ensure model weights are present locally. On first use this runs
            # a pre-flight speed test against the HF mirror and downloads the
            # safetensors with resume+stall retry; aborts with a clear network
            # error if the mirror is too slow instead of hanging. Returns the
            # local path to load from (cached snapshot or the manual download
            # dir), or None to load by model name from the HF cache.
            load_from = self._ensure_weights_downloaded()
            src = load_from or self._cfg.model_name
            self._tokenizer = AutoTokenizer.from_pretrained(
                src, trust_remote_code=True)
            # Load in bf16 (half the memory of fp32; modern CPUs support it via
            # AVX512_BF16/AMX) with low_cpu_mem_usage to avoid a transient 2x
            # peak while the weights are materialized. fp32 was ~2.4 GB; bf16
            # is ~1.2 GB, leaving headroom for the forward-pass activations.
            self._model = AutoModelForCausalLM.from_pretrained(
                src, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True).to(self._cfg.device).eval()
            log.info("Qwen3-Reranker loaded.")
            return True
        except Exception as exc:
            self._load_error = str(exc)
            log.warning("Qwen3 reranker unavailable (%s); "
                        "RRF fallback will be used.", exc)
            return False

    def _ensure_weights_downloaded(self) -> str | None:
        """Make sure Qwen3-Reranker weights are cached locally.

        Returns the local directory to load from, or None if the model is
        already in the HF cache (load by model name). If the weights aren't
        cached, run a pre-flight speed test against the HF mirror and download
        the safetensors with resume + stall retry into a local dir (with the
        tokenizer/config files copied alongside) so transformers can load
        offline. Raises ``NetworkTooSlowError`` on a slow/failed mirror.
        """
        import os
        from huggingface_hub import hf_hub_download
        from net_guard import (NetworkTooSlowError, download_with_retry,
                               speed_test)

        cfg = self._cfg
        # Fast path: weights already in the HF cache -> load by model name.
        try:
            path = hf_hub_download(repo_id=cfg.model_name,
                                   filename="model.safetensors")
            if os.path.getsize(path) > 0:
                return None  # use model_name; transformers resolves the cache
        except Exception:
            pass  # not cached yet — fall through to guarded download

        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        url = (f"{endpoint.rstrip('/')}/{cfg.model_name}/resolve/main/"
               f"model.safetensors")
        # Fast path: weights already downloaded into the project-local cache.
        # We MUST check this BEFORE the network speed test, otherwise a
        # transient network blip makes speed_test fail and degrades to RRF even
        # though the model is fully present on disk. Re-verifying the small
        # tokenizer/config files too lets us return a fully-offline load path.
        from config import USER_DIR
        cache_dir = os.environ.get("HF_HOME", str(USER_DIR / "huggingface"))
        dest_dir = os.path.join(cache_dir, "hub",
                                "models--" + cfg.model_name.replace("/", "--"),
                                "snapshots", "manual")
        dest = os.path.join(dest_dir, "model.safetensors")
        small_files = ("config.json", "tokenizer.json",
                       "tokenizer_config.json", "vocab.json", "merges.txt",
                       "chat_template.jinja")
        if (os.path.exists(dest) and os.path.getsize(dest) > 0
                and all(os.path.exists(os.path.join(dest_dir, f))
                        and os.path.getsize(os.path.join(dest_dir, f)) > 0
                        for f in small_files)):
            log.info("Qwen3-Reranker weights cached locally at %s; "
                     "skipping network check.", dest_dir)
            return dest_dir
        # Pre-flight speed test against the mirror (1 MB probe). Aborts with a
        # clear network error if the mirror is too slow.
        speed_test(url, min_bytes_per_sec=cfg.hf_min_bytes_per_sec,
                   timeout=cfg.hf_speed_test_timeout, label="hf-mirror")
        # Download weights (resumable, stall-retried) into a local snapshot dir
        # under the project-local HF cache (user/huggingface/hub/...).
        os.makedirs(dest_dir, exist_ok=True)
        download_with_retry(url, dest,
                            min_bytes_per_sec=cfg.hf_min_bytes_per_sec,
                            max_retries=cfg.hf_download_retries,
                            label="qwen3-reranker")
        # Fetch the small tokenizer/config files into the same snapshot dir so
        # AutoTokenizer/AutoModel can load fully offline. These are tiny (<15MB
        # total), so we download them directly by URL with light retry rather
        # than relying on hf_hub_download (which may HEAD-fail on this network).
        import shutil
        small_files = ("config.json", "tokenizer.json",
                       "tokenizer_config.json", "vocab.json", "merges.txt",
                       "chat_template.jinja")
        for fname in small_files:
            link = os.path.join(dest_dir, fname)
            if os.path.exists(link) and os.path.getsize(link) > 0:
                continue
            furl = (f"{endpoint.rstrip('/')}/{cfg.model_name}/resolve/main/"
                    f"{fname}")
            try:
                download_with_retry(furl, link,
                                    min_bytes_per_sec=1_000,  # tiny files: low floor
                                    max_retries=cfg.hf_download_retries,
                                    chunk_timeout=20,
                                    label=f"qwen3-{fname}")
            except Exception as exc:
                log.warning("Could not fetch %s: %s", fname, exc)
        return dest_dir

    def rerank(self, query: str, texts: list[str],
               batch_size: int | None = None) -> list[float]:
        """Return relevance scores (higher = more relevant)."""
        if not texts:
            return []
        if not self._load():
            return [0.0] * len(texts)
        try:
            return self._score(query, texts, batch_size or self._cfg.batch_size)
        except Exception as exc:
            log.warning("Reranker scoring failed (%s); using zeros.", exc)
            return [0.0] * len(texts)

    def _score(self, query: str, texts: list[str], batch_size: int) -> list[float]:
        import torch
        # Qwen3-Reranker prompt format (official): prefix tokens mark
        # query/document boundaries, and relevance = P("yes") from the
        # token-scored last position.
        prefix = ("<|im_start|>system\nJudge whether the Document meets the "
                  "requirements based on the Query and only answer with yes "
                  "or no.<|im_end|>\n<|im_start|>user\n")
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        no_id = self._tokenizer.convert_tokens_to_ids("no")
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                prompts = [
                    f"{prefix}\nQuery: {query}\nDocument: {t}\n{suffix}"
                    for t in batch
                ]
                tok = self._tokenizer(prompts, padding=True, truncation=True,
                                      max_length=self._cfg.max_length,
                                      return_tensors="pt").to(self._cfg.device)
                # logits_to_keep=1: only compute logits for the LAST token
                # position. Without this, the model materializes a full
                # [batch, seq_len, vocab(~152k)] tensor (~5 GB at batch=16,
                # seq=512, fp32) — the root cause of the prior OOM. We only
                # need P("yes") at the final position, so 1 is sufficient and
                # cuts that tensor by ~seq_len× (hundreds-fold).
                out = self._model(**tok, logits_to_keep=1)
                logits = out.logits[:, -1, :]
                yes = torch.softmax(logits, dim=-1)[:, yes_id]
                scores.extend(yes.tolist())
        return scores


# --------------------------------------------------------------------------- #
# Top-level pipeline
# --------------------------------------------------------------------------- #
class RAGPipeline:
    def __init__(self) -> None:
        # Pick the embedding backend by provider: local (CPU bge) or azure
        # (OpenAI-compatible API). Both expose the same embed_texts /
        # embed_one / embed_many_concurrent surface.
        if settings.embedding.provider == "local":
            self.embedder = LocalEmbeddingClient()
        else:
            self.embedder = EmbeddingClient()
        self.expander = MultiQueryExpander(n=3)
        self.reranker = QwenReranker()

    # ----- indexing ------------------------------------------------------ #
    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        return self.embedder.embed_texts([c["text"] for c in chunks])

    def embed_many(self, chunks: list[dict]) -> list[list[float]]:
        """Embed a flat list of chunks, packing them into full API batches.

        Unlike calling ``embed_chunks`` once per item (which underfills each
        HTTP request when items have only 1-3 chunks), this batches across
        items: it walks the whole chunk list in ``batch_size`` slices and
        issues embedding requests for each slice. For a project with ~1.4
        chunks/item and ``batch_size=64`` this cuts the number of HTTP round
        trips ~45x, turning minutes of serial network wait into seconds.

        When ``EMBEDDING_CONCURRENCY > 1`` (default 4) the batches are fired
        concurrently, cutting indexing wall-time a further ~3-4x.

        Embeddings are returned position-aligned with the input ``chunks``
        so callers can ``zip(chunks, embeddings)`` directly.
        """
        if not chunks:
            return []
        texts = [c["text"] for c in chunks]
        return self.embedder.embed_many_concurrent(
            texts, concurrency=settings.embedding.concurrency)

    # ----- retrieval ----------------------------------------------------- #
    def search(self, project_id: int, query: str, *,
               sub_queries: list[str] | None = None,
               item_type: int | None = None,
               top_k: int = 5, candidate_k: int = 25,
               modified_after: str | None = None,
               modified_before: str | None = None) -> list[dict]:
        """Full RAG chain -> list of result dicts (best first).

        ``sub_queries`` (optional) are caller-supplied query expansions — the
        MCP LLM client is expected to rewrite ``query`` into 3-5 diverse
        variants. When supplied they are normalized (original query forced to
        the front, de-duplicated, capped); when omitted, deterministic lexical
        variants are used. ``query`` itself is always the rerank reference.

        ``modified_after``/``modified_before`` (ISO-8601, UTC-normalized)
        restrict recall to items modified within the inclusive range; applied
        at the recall layer so RRF fusion and reranking only see in-range
        candidates.
        """
        # Caller-supplied sub-queries win; otherwise fall back to the
        # deterministic lexical expander. Either way ``query`` itself is
        # guaranteed present (it's the rerank reference) and the list is
        # non-empty for a non-empty query.
        if sub_queries is not None:
            sub_queries = _normalize_sub_queries(sub_queries, query)
        else:
            sub_queries = self.expander.expand(query) or [query]

        conn = get_connection()
        try:
            ranked_lists: list[list[str]] = []
            for sq in sub_queries:
                # Vector recall.
                qvec = self.embedder.embed_one(sq)
                v_rows = vector_search(conn, qvec, project_id, item_type,
                                       candidate_k, modified_after,
                                       modified_before)
                # Keyword recall (FTS5). Quote/escape for MATCH safety.
                fts_q = _to_fts_query(sq)
                f_rows = []
                if fts_q:
                    f_rows = fts_search(conn, fts_q, project_id, item_type,
                                        candidate_k, modified_after,
                                        modified_before)
                ranked_lists.append([r["chunk_id"] for r in v_rows])
                ranked_lists.append([r["chunk_id"] for r in f_rows])

            # RRF fusion -> candidate_k unique chunk_ids.
            fused = rrf_fuse(ranked_lists)[:candidate_k]
            if not fused:
                return []
            cand_ids = [cid for cid, _ in fused]
            rrf_scores = {cid: sc for cid, sc in fused}
            rows = {r["chunk_id"]: r for r in fetch_chunks_by_ids(conn, cand_ids)}

            # Rerank.
            ordered = [cid for cid in cand_ids if cid in rows]
            texts = [rows[cid]["text"] for cid in ordered]
            rr_scores = self.reranker.rerank(query, texts)
            used_rerank = any(s != 0.0 for s in rr_scores)
            if not used_rerank:
                # Fallback to RRF ordering/scores.
                rr_scores = [rrf_scores[cid] for cid in ordered]

            scored = sorted(zip(ordered, rr_scores),
                            key=lambda x: x[1], reverse=True)[:top_k]
            return [_row_to_result(rows[cid], score,
                                   "rerank" if used_rerank else "rrf")
                    for cid, score in scored if cid in rows]
        finally:
            conn.close()


def _to_fts_query(query: str) -> str:
    """Make a safe FTS5 MATCH expression (prefix + AND of terms)."""
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    if not tokens:
        return ""
    # Prefix terms enable partial matches; quoting avoids operator parsing.
    return " ".join(f'"{t}"*' for t in tokens[:20])


def _row_to_result(row, score: float, strategy: str) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "item_id": row["item_id"],
        "document_key": row["document_key"],
        "item_type": row["item_type"],
        "item_type_name": row["item_type_name"],
        "name": row["name"],
        "status": row["status"],
        "section": row["section"],
        "modified_date": row["modified_date"],
        "text": row["text"],
        "score": round(float(score), 6),
        "strategy": strategy,
    }
