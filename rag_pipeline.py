"""RAG pipeline: chunking, embeddings, Multi-Query, hybrid recall, RRF, rerank.

Retrieval chain (``search`` method):
1. Multi-Query   - expand the user query into N sub-queries (LLM if available,
                   otherwise deterministic lexical variants).
2. Hybrid recall - for each sub-query: vector recall (sqlite-vec) + keyword
                   recall (FTS5), each limited to ``candidate_k``.
3. RRF fusion    - Reciprocal Rank Fusion merges all candidate lists into one
                   ranked list of <= ``candidate_k`` unique chunks.
4. Rerank        - local Qwen3-Reranker-0.6B scores (query, chunk) pairs; the
                   top ``top_k`` are returned. If the model is unavailable and
                   ``allow_fallback`` is set, RRF scores are used directly.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

import requests

from config import settings
from db_setup import (fts_search, get_connection, vector_search,
                      fetch_chunks_by_ids)

log = logging.getLogger(__name__)

# LlamaIndex is the primary RAG framework: it provides the recursive splitter
# (SentenceSplitter), the Document/TextNode document model, the prompt template
# engine and the LLM abstraction used for Multi-Query expansion.
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document, TextNode

# Multi-Query prompt (LlamaIndex PromptTemplate). Produces N diverse
# sub-queries for RRF fusion to maximize recall.
MULTI_QUERY_PROMPT = PromptTemplate(
    "You are an expert search query rewriter for a requirements management "
    "system (Jama). Rewrite the user's query into {n} diverse search "
    "sub-queries that capture different semantic angles, to maximize recall. "
    "Return ONLY a JSON array of strings, no commentary.\n\n"
    "Original query: {query}"
)


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


# --------------------------------------------------------------------------- #
# Multi-Query expansion (LlamaIndex LLM + PromptTemplate)
# --------------------------------------------------------------------------- #
class MultiQueryExpander:
    """Expand a query into N sub-queries using LlamaIndex.

    When ``LLM_BASE_URL`` is configured, a LlamaIndex ``OpenAI`` LLM drives the
    expansion via the ``MULTI_QUERY_PROMPT`` PromptTemplate (the LlamaIndex
    native path). Otherwise it falls back to deterministic lexical variants so
    RRF fusion still benefits from multiple query angles without an LLM.
    """

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self._cfg = settings.llm
        self._llm = None
        if self._cfg.base_url and self._cfg.api_key:
            try:
                from llama_index.llms.openai import OpenAI
                # Point LlamaIndex's OpenAI LLM at the configured gateway.
                # api_key/auth header differ between Azure (api-key) and
                # OpenAI (Bearer); the OpenAI client uses Bearer by default.
                self._llm = OpenAI(
                    model=self._cfg.model,
                    api_key=self._cfg.api_key,
                    api_base=f"{self._cfg.base_url.rstrip('/')}/openai/v1",
                    temperature=0.2,
                    max_tokens=256,
                    timeout=self._cfg.timeout,
                )
            except Exception as exc:
                log.warning("Could not init LlamaIndex LLM (%s); "
                            "using fallback multi-query", exc)
                self._llm = None

    def expand(self, query: str) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []
        if self._llm is not None:
            try:
                return self._llm_expand(query)
            except Exception as exc:
                log.warning("Multi-query LLM failed (%s); using fallback", exc)
        return self._fallback_expand(query)

    def _llm_expand(self, query: str) -> list[str]:
        # LlamaIndex native path: format the PromptTemplate and call the LLM.
        prompt_str = MULTI_QUERY_PROMPT.format(n=self.n, query=query)
        resp = self._llm.complete(prompt_str)
        content = str(resp).strip()
        # Tolerate code-fenced JSON.
        content = re.sub(r"^```(?:json)?|```$", "", content,
                         flags=re.MULTILINE).strip()
        subs = json.loads(content)
        subs = [s for s in subs if isinstance(s, str) and s.strip()]
        if not subs:
            return [query]
        return [query] + subs[: self.n - 1]

    def _fallback_expand(self, query: str) -> list[str]:
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
            self._model = AutoModelForCausalLM.from_pretrained(
                src, trust_remote_code=True,
                torch_dtype=torch.float32).to(self._cfg.device).eval()
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
        # Pre-flight speed test against the mirror (1 MB probe). Aborts with a
        # clear network error if the mirror is too slow.
        speed_test(url, min_bytes_per_sec=cfg.hf_min_bytes_per_sec,
                   timeout=cfg.hf_speed_test_timeout, label="hf-mirror")
        # Download weights (resumable, stall-retried) into a local snapshot dir
        # under the project-local HF cache (user/huggingface/hub/...).
        from config import USER_DIR
        cache_dir = os.environ.get("HF_HOME", str(USER_DIR / "huggingface"))
        dest_dir = os.path.join(cache_dir, "hub",
                                "models--" + cfg.model_name.replace("/", "--"),
                                "snapshots", "manual")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "model.safetensors")
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
                logits = self._model(**tok).logits[:, -1, :]
                yes = torch.softmax(logits, dim=-1)[:, yes_id]
                scores.extend(yes.tolist())
        return scores


# --------------------------------------------------------------------------- #
# Top-level pipeline
# --------------------------------------------------------------------------- #
class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = EmbeddingClient()
        self.expander = MultiQueryExpander(n=3)
        self.reranker = QwenReranker()

    # ----- indexing ------------------------------------------------------ #
    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        return self.embedder.embed_texts([c["text"] for c in chunks])

    # ----- retrieval ----------------------------------------------------- #
    def search(self, project_id: int, query: str, *,
               item_type: int | None = None,
               top_k: int = 5, candidate_k: int = 50,
               modified_after: str | None = None,
               modified_before: str | None = None) -> list[dict]:
        """Full RAG chain -> list of result dicts (best first).

        ``modified_after``/``modified_before`` (ISO-8601, UTC-normalized)
        restrict recall to items modified within the inclusive range; applied
        at the recall layer so RRF fusion and reranking only see in-range
        candidates.
        """
        sub_queries = self.expander.expand(query)
        if not sub_queries:
            sub_queries = [query]

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
