"""Jama REST API client.

Responsibilities
----------------
* OAuth2 client-credentials token management (auto-refresh on expiry).
* Resilient HTTP with retries on transient errors and 429 rate limiting.
* Paginated fetch of project items (including Test Cases with their steps).
* HTML -> plain-text cleaning for the rich-text ``description`` and
  ``testCaseSteps`` fields, using BeautifulSoup4.

Read-only by design: this client only issues GET requests against Jama, so it
can never create, modify or delete data on the Jama instance.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings
from net_guard import NetworkTooSlowError, speed_test

log = logging.getLogger(__name__)

# Jama "Test Case" itemType is global (id 89011) but we detect test cases by
# the presence of ``testCaseSteps`` to be robust across tenants.
_TEST_STEPS_KEY = "testCaseSteps"

# Hard cap on consecutive 429 retries for a single logical request. A
# persistent rate-limit (or a misbehaving gateway) used to drive unbounded
# recursion in ``_get``; this bounds it so the call fails cleanly instead of
# hanging for hours or blowing the stack.
_MAX_429_RETRIES = 5


def _parse_retry_after(value: str | None) -> float:
    """Parse a ``Retry-After`` header into seconds.

    The header is RFC-permitted to be either a non-negative integer (seconds)
    or an HTTP-date. We only honor the integer form (what Jama emits); a
    non-numeric value falls back to a short default instead of crashing the
    request with ``ValueError``.
    """
    if not value:
        return 5.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 5.0


# --------------------------------------------------------------------------- #
# HTML cleaning
# --------------------------------------------------------------------------- #
def clean_html(raw: str | None) -> str:
    """Convert Jama rich-text HTML to clean plain text.

    Handles <p>, <br>, <li>, <strong>/<b>, tables and HTML entities (&nbsp; …).
    Block elements are separated by newlines so the splitter keeps structure.
    """
    if not raw:
        return ""
    # Plain text (Jama sometimes returns already-clean strings).
    if "<" not in raw and ">" not in raw:
        return _normalize_ws(raw)

    soup = BeautifulSoup(raw, "lxml")

    # Turn <br> and block closers into newlines before extracting text.
    for br in soup.find_all(["br"]):
        br.replace_with("\n")
    for tag in soup.find_all(["p", "li", "tr", "div", "h1", "h2", "h3", "h4"]):
        tag.append("\n")

    text = soup.get_text(separator=" ", strip=False)
    return _normalize_ws(text)


def _normalize_ws(text: str) -> str:
    # Normalize non-breaking spaces and other unicode whitespace to ASCII space.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    # Collapse runs of spaces/tabs but keep deliberate newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n[ \n]*", "\n", text)
    # Drop lines that are only whitespace (artifacts of &nbsp; blocks).
    text = "\n".join(ln for ln in text.split("\n") if ln.strip())
    return text.strip()


def render_test_steps(steps: Any) -> str:
    """Render a Test Case's ``testCaseSteps`` list as plain text."""
    if not steps or not isinstance(steps, list):
        return ""
    lines = []
    for i, st in enumerate(steps, 1):
        if not isinstance(st, dict):
            continue
        action = clean_html(st.get("action", ""))
        expected = clean_html(st.get("expectedResult", ""))
        notes = clean_html(st.get("notes", ""))
        parts = [f"Step {i}: {action}".strip()]
        if expected:
            parts.append(f"Expected: {expected}".strip())
        if notes:
            parts.append(f"Notes: {notes}".strip())
        lines.append(" | ".join(p for p in parts if p))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTTP session with retries + rate-limit handling
# --------------------------------------------------------------------------- #
def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=settings.jama.max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    # Size the connection pool to the download concurrency. Without this,
    # HTTPAdapter defaults to pool_maxsize=10 while iter_pages_concurrent runs
    # ``download_concurrency`` (default 16) workers on the SAME session: the
    # excess workers churn connections ("Connection pool is full, discarding
    # connection") and pay TCP+TLS rehandshake on every page — a measured
    # bottleneck on large projects (Lyra, 5076 items). Pooling at the
    # concurrency level keeps every worker on a reused keep-alive connection.
    pool = max(settings.sync.download_concurrency, 10)
    adapter = HTTPAdapter(max_retries=retry,
                          pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", adapter)
    s.mount("http://", HTTPAdapter(max_retries=retry,
                                   pool_connections=pool, pool_maxsize=pool))
    s.headers.update({"Accept": "application/json"})
    return s


class JamaClient:
    """Thread-safe, read-only Jama REST client."""

    def __init__(self) -> None:
        self._s = _build_session()
        self._token: str | None = None
        self._token_exp: float = 0.0
        self._lock = threading.Lock()
        # itemType id -> display name cache (populated lazily).
        self._item_types: dict[int, str] = {}

    # ----- auth ---------------------------------------------------------- #
    def _ensure_token(self) -> str:
        with self._lock:
            # Refresh 60s before expiry to be safe.
            if self._token and time.time() < self._token_exp - 60:
                return self._token
            url = f"{settings.jama.url}/rest/oauth/token"
            r = self._s.post(
                url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": settings.jama.client_id,
                    "client_secret": settings.jama.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=settings.jama.request_timeout,
            )
            r.raise_for_status()
            payload = r.json()
            self._token = payload["access_token"]
            self._token_exp = time.time() + int(payload.get("expires_in", 3600))
            log.debug("Jama token refreshed, expires_in=%ss",
                      payload.get("expires_in"))
            return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._ensure_token()}"}

    # ----- pre-flight network speed test -------------------------------- #
    def preflight_speed_check(self) -> float:
        """Run a speed test against the Jama host before any bulk download.

        Aborts early with ``NetworkTooSlowError`` if throughput is below the
        configured floor, so the caller (init/sync) fails fast with a clear
        network message instead of timing out mid-pagination.
        """
        cfg = settings.jama
        # Fetch a full page (50 projects) so the response body is large enough
        # to measure bandwidth rather than just TLS-handshake latency.
        probe_url = (f"{cfg.url}{cfg.api_prefix}/projects"
                     f"?startAt=0&maxResults={cfg.page_size}")
        return speed_test(
            probe_url, min_bytes_per_sec=cfg.min_bytes_per_sec,
            timeout=cfg.speed_test_timeout, headers=self._headers(),
            label="jama")

    # ----- single GET (with token refresh + 429 backoff) ---------------- #
    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{settings.jama.url}{settings.jama.api_prefix}{path}"
        # 401 refresh is single-shot per call (matches the page-fetch path).
        # 429 uses a BOUNDED loop (not recursion) so a persistent rate-limit
        # fails cleanly after ``_MAX_429_RETRIES`` instead of recursing until
        # the stack overflows.
        refreshed = False
        for _ in range(_MAX_429_RETRIES + 1):
            r = self._s.get(url, headers=self._headers(), params=params or {},
                            timeout=settings.jama.request_timeout)
            # 401 -> token may have been revoked mid-flight; refresh once.
            if r.status_code == 401 and not refreshed:
                with self._lock:
                    self._token = None
                refreshed = True
                r = self._s.get(url, headers=self._headers(), params=params or {},
                                timeout=settings.jama.request_timeout)
            if r.status_code == 429:  # explicit backoff beyond urllib3's handling
                retry_after = _parse_retry_after(r.headers.get("Retry-After"))
                log.warning("Jama 429 rate limit; sleeping %ss", retry_after)
                time.sleep(min(retry_after, 30))
                continue
            r.raise_for_status()
            return r.json()
        # Exhausted 429 retries: raise so callers see a real failure.
        r.raise_for_status()
        return r.json()

    def _get_page_with_stall_retry(self, path: str,
                                   params: dict) -> dict:
        """Fetch one page, retrying on timeout/stall/slow-throughput.

        Measures throughput over the response body; if it drops below
        ``page_min_bytes_per_sec`` or a network timeout occurs, retry with
        backoff up to ``page_max_retries``. This is what makes long paginated
        downloads resilient to mid-stream network hiccups.
        """
        cfg = settings.jama
        url = f"{cfg.url}{cfg.api_prefix}{path}"
        last_err: Exception | None = None
        refreshed = False  # guard: refresh token at most once per page
        for attempt in range(1, cfg.page_max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self._s.get(url, headers=self._headers(), params=params,
                                timeout=cfg.request_timeout)
                if r.status_code == 401 and not refreshed:
                    # Token expired/revoked mid-pagination: clear it so the next
                    # _headers() call refreshes, then retry this page on the
                    # next loop iteration. This consumes one of the
                    # page_max_retries slots, but 401 mid-pagination is rare and
                    # the budget (5) absorbs it.
                    with self._lock:
                        self._token = None
                    refreshed = True
                    continue
                if r.status_code == 429:
                    retry_after = _parse_retry_after(r.headers.get("Retry-After"))
                    log.warning("Jama 429 rate limit; sleeping %ss", retry_after)
                    time.sleep(min(retry_after, 30))
                    continue
                r.raise_for_status()
                body = r.content
                elapsed = max(time.monotonic() - t0, 0.001)
                bps = len(body) / elapsed
                if bps < cfg.page_min_bytes_per_sec and len(body) > 1024:
                    raise NetworkTooSlowError(
                        f"Jama page fetch too slow: {bps:.0f} bytes/s < floor "
                        f"{cfg.page_min_bytes_per_sec} (attempt "
                        f"{attempt}/{cfg.page_max_retries})")
                # JSON parse failure (e.g. a truncated/HTML proxy body on a
                # 200) is retried like a transient stall instead of aborting
                # the whole download.
                return r.json()
            except (requests.Timeout, requests.ConnectionError,
                    NetworkTooSlowError, ValueError) as exc:
                last_err = exc
                log.warning("Jama page fetch attempt %d/%d stalled/errored: "
                            "%s", attempt, cfg.page_max_retries, exc)
                time.sleep(min(2 ** attempt, 10))
        raise NetworkTooSlowError(
            f"Jama page fetch failed after {cfg.page_max_retries} retries: "
            f"{last_err}. Network problem — check connectivity.")

    # ----- generic GET with pagination ---------------------------------- #
    def _paginate(self, path: str, params: dict | None = None,
                  max_pages: int | None = None) -> Iterator[dict]:
        """Yield items across all pages of a list endpoint.

        Each page is fetched via ``_get_page_with_stall_retry`` so mid-stream
        network stalls/timeouts trigger a retry rather than aborting the whole
        download.
        """
        base = dict(params or {})
        page = 0
        start_at = 0
        while True:
            page_params = {**base, "startAt": start_at,
                           "maxResults": settings.jama.page_size}
            data = self._get_page_with_stall_retry(path, page_params)
            items = data.get("data", []) or []
            for it in items:
                yield it
            page_info = data.get("meta", {}).get("pageInfo", {})
            total = int(page_info.get("totalResults", 0) or 0)
            result_count = int(page_info.get("resultCount", 0) or 0)
            start_at += result_count
            page += 1
            if result_count == 0 or start_at >= total:
                break
            if max_pages and page >= max_pages:
                break
            if settings.jama.page_delay:
                time.sleep(settings.jama.page_delay)

    def iter_pages_concurrent(self, path: str, params: dict | None = None,
                              *, concurrency: int = 16,
                              on_total=None) -> Iterator[list[dict]]:
        """Concurrent wave-by-wave pager (borrowed from a prior impl).

        Fetches page 1 first to learn ``totalResults`` (and calls
        ``on_total(total)`` so callers can report real progress), then fetches
        the remaining pages in WAVES of ``concurrency`` concurrent requests.
        Yields one list of items per wave so peak memory stays ~
        ``concurrency * page_size`` items regardless of project size.

        Each page uses ``_get_page_with_stall_retry`` so a mid-wave network
        stall on one page is retried rather than aborting the whole wave.
        Order within a wave is preserved by ``ThreadPoolExecutor.map``.
        """
        from concurrent.futures import ThreadPoolExecutor
        base = dict(params or {})
        page_size = settings.jama.page_size

        # Page 1: learn the total.
        first = self._get_page_with_stall_retry(
            path, {**base, "startAt": 0, "maxResults": page_size})
        page_info = first.get("meta", {}).get("pageInfo", {})
        total = int(page_info.get("totalResults", 0) or 0)
        if on_total is not None:
            try:
                on_total(total)
            except Exception:
                pass  # progress reporting must never break the sync
        page1 = list(first.get("data") or [])
        yield page1
        if total <= page_size:
            return

        # Remaining pages: fetch startAt values in concurrent waves.
        starts = list(range(page_size, total, page_size))

        def _fetch(start_at: int) -> list[dict]:
            return self._get_page_with_stall_retry(
                path, {**base, "startAt": start_at, "maxResults": page_size}
            ).get("data") or []

        with ThreadPoolExecutor(max_workers=concurrency,
                                thread_name_prefix="jama-page") as ex:
            for i in range(0, len(starts), concurrency):
                wave: list[dict] = []
                for rows in ex.map(_fetch, starts[i:i + concurrency]):
                    wave.extend(rows)
                yield wave
                wave = None  # free before next wave

    # ----- item types ---------------------------------------------------- #
    def load_item_types(self) -> dict[int, str]:
        """Cache and return {itemType id -> display name}.

        The cache is populated only on a fully successful fetch: a partial
        result set (pagination that errors mid-way) is NOT cached, otherwise
        the top guard would prevent a retry and missing types would render as
        ``"Type {id}"`` permanently until restart.
        """
        if self._item_types:
            return self._item_types
        found: dict[int, str] = {}
        try:
            for it in self._paginate("/itemtypes", max_pages=5):
                if isinstance(it, dict):
                    found[it.get("id")] = it.get("display") or \
                        it.get("displayPlural") or str(it.get("id"))
        except Exception as exc:
            log.warning("Could not load item types: %s", exc)
            # Return whatever we found WITHOUT caching it, so the next call
            # retries. An empty/partial dict is better than nothing for this
            # call, but must not be frozen into the cache.
            return found
        self._item_types = found
        return self._item_types

    def item_type_name(self, item_type_id: int | None) -> str:
        if item_type_id is None:
            return "Unknown"
        if not self._item_types:
            self.load_item_types()
        return self._item_types.get(item_type_id, f"Type {item_type_id}")

    # ----- public read API ---------------------------------------------- #
    def get_project(self, project_id: int) -> dict | None:
        try:
            data = self._get(f"/projects/{project_id}")
            return data.get("data") if isinstance(data, dict) else None
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    def count_project_items(self, project_id: int) -> int:
        """Cheap total item count for a project (single 1-result probe page).

        Used to set a job's ``total`` up-front so progress is visible during
        the streaming fetch+index, without waiting for the full pagination to
        complete. Reads ``meta.pageInfo.totalResults`` from a maxResults=1
        request; returns 0 if the probe fails (callers fall back to unknown).

        NOTE: this queries ``/items``, which returns top-level content items
        (Requirement, Test Case, Feature, …) — the ones worth semantic
        indexing. It deliberately EXCLUDES Test Runs, Folders, Attachments and
        Test Cycles, which ``/abstractitems`` would include but which carry no
        retrievable text. So this count is intentionally smaller than the
        ``/abstractitems`` total that ``query_jama_native_metadata`` sees.
        """
        try:
            data = self._get("/items", params={"project": project_id,
                                               "startAt": 0, "maxResults": 1})
            return int(data.get("meta", {}).get("pageInfo", {})
                       .get("totalResults", 0) or 0)
        except Exception as exc:
            log.warning("count_project_items(%s) failed: %s", project_id, exc)
            return 0

    def iter_project_items(self, project_id: int,
                           modified_after: str | None = None,
                           max_items: int | None = None,
                           *, concurrency: int = 1,
                           on_total=None) -> Iterator[dict]:
        """Yield normalized item dicts for a project.

        ``modified_after`` (ISO-8601) enables incremental sync: only items
        whose ``modifiedDate`` is strictly greater are yielded. Filtering is
        done client-side because Jama's REST API has no server-side
        modified-date filter.

        ``concurrency > 1`` switches to the concurrent wave-by-wave pager
        (``iter_pages_concurrent``): page 1 is fetched first (calling
        ``on_total(total)`` so progress is reportable immediately), then
        remaining pages are fetched in waves of ``concurrency`` parallel
        requests. With ``concurrency=1`` (default) the serial pager is used.
        """
        self.load_item_types()
        count = 0

        def _emit_wave(raws: list[dict]) -> Iterator[dict]:
            nonlocal count
            for raw in raws:
                norm = self._normalize_item(raw)
                if modified_after and norm["modified_date"] and \
                        norm["modified_date"] <= modified_after:
                    continue
                yield norm
                count += 1
                if max_items and count >= max_items:
                    return

        if concurrency > 1:
            for wave in self.iter_pages_concurrent(
                    "/items", {"project": project_id},
                    concurrency=concurrency, on_total=on_total):
                yield from _emit_wave(wave)
                if max_items and count >= max_items:
                    return
        else:
            for raw in self._paginate("/items", {"project": project_id}):
                yield from _emit_wave([raw])
                if max_items and count >= max_items:
                    return

    def _normalize_item(self, raw: dict) -> dict:
        """Flatten a Jama item payload into our storage shape + cleaned text."""
        fields = raw.get("fields", {}) or {}
        desc = clean_html(fields.get("description", ""))
        steps = render_test_steps(fields.get(_TEST_STEPS_KEY))
        item_type = raw.get("itemType")
        # Status lives under different field keys per type; pick the first known.
        status = (fields.get("status") or fields.get("testCaseStatus")
                  or fields.get("testRunStatus") or "")
        return {
            "item_id": raw.get("id"),
            "project_id": raw.get("project"),
            "document_key": raw.get("documentKey"),
            "global_id": raw.get("globalId"),
            "item_type": item_type,
            "item_type_name": self.item_type_name(item_type),
            "name": (fields.get("name") or "").strip(),
            "status": status,
            "description": desc,
            "test_steps": steps,
            "modified_date": raw.get("modifiedDate"),
            "created_date": raw.get("createdDate"),
            "raw_json": _safe_dumps(raw),
        }

    # ----- native metadata query (for query_jama_native_metadata) ------ #
    def query_items_native(self, project_id: int, *,
                           document_key: str | None = None,
                           item_type: int | None = None,
                           status: str | None = None,
                           keyword: str | None = None,
                           limit: int = 20) -> list[dict]:
        """Direct Jama REST filtering with client-side refinement.

        Uses ``/abstractitems`` because it honours ``itemType``,
        ``contains`` and ``documentKey`` server-side filters (the ``/items``
        endpoint ignores ``itemType``). ``status`` has no server-side filter,
        so it is applied client-side. Pagination is walked until ``limit``
        matches are collected or the well is empty.
        """
        params: dict[str, Any] = {"project": project_id}
        if item_type is not None:
            params["itemType"] = item_type
        if keyword:
            params["contains"] = keyword
        if document_key:
            # server-side exact match (much faster than client-side walk)
            params["documentKey"] = document_key

        results: list[dict] = []
        seen = 0
        for raw in self._paginate("/abstractitems", params, max_pages=50):
            seen += 1
            fields = raw.get("fields", {}) or {}
            doc_key = raw.get("documentKey")
            item_status = (fields.get("status") or fields.get("testCaseStatus")
                           or fields.get("testRunStatus") or "")
            # document_key already filtered server-side; this is a safety net.
            if document_key and (doc_key or "").upper() != document_key.upper():
                continue
            if status and (item_status or "").upper() != status.upper():
                continue
            results.append({
                "item_id": raw.get("id"),
                "document_key": doc_key,
                "global_id": raw.get("globalId"),
                "item_type": raw.get("itemType"),
                "item_type_name": self.item_type_name(raw.get("itemType")),
                "name": (fields.get("name") or "").strip(),
                "status": item_status,
                "modified_date": raw.get("modifiedDate"),
                "description": clean_html(fields.get("description", ""))[:500],
            })
            if len(results) >= limit:
                break
            # Safety: don't walk more than a few thousand raw rows.
            if seen >= 2000:
                break
        return results

    # ----- extended read-only query API (browse Jama beyond items) -------- #
    # Every method here is a thin wrapper around the read-only GET machinery
    # above (OAuth, pagination, retry). They expose the rest of Jama's REST
    # query surface to the MCP client as discrete tools; ``get_raw`` is the
    # power-user escape hatch that supports *any* GET endpoint.

    def get_raw(self, path: str, params: dict | None = None,
                *, max_pages: int | None = 1) -> dict | list:
        """Generic read-only GET against any Jama REST endpoint.

        ``path`` is appended to ``{url}{api_prefix}`` (e.g. ``"/projects"``).
        With ``max_pages=1`` (default) returns the first page's ``data``;
        with ``max_pages=None`` walks all pages and returns a flat list.

        ``path`` is sanitized: it must be a relative REST path starting with
        ``/``. Query strings (``?``) and fragments (``#``) are rejected —
        callers must pass query parameters via ``params`` so they are properly
        URL-encoded — and absolute URLs are rejected to prevent SSRF.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        p = path.strip()
        if not p.startswith("/"):
            raise ValueError("path must start with '/' (a relative REST path)")
        if "://" in p:
            raise ValueError("path must be relative, not an absolute URL")
        if "?" in p or "#" in p:
            raise ValueError("path must not contain '?' or '#'; "
                             "pass query parameters via the `params` argument")
        path = p
        if max_pages == 1:
            data = self._get(path, params=params)
            return data.get("data", []) if isinstance(data, dict) else data
        return list(self._paginate(path, params, max_pages=max_pages))

    def list_projects(self) -> list[dict]:
        """All projects visible to the OAuth client."""
        out = []
        for p in self._paginate("/projects", max_pages=50):
            f = p.get("fields", {}) or {}
            out.append({
                "id": p.get("id"),
                "project_key": p.get("projectKey"),
                "name": f.get("name"),
                "status": f.get("status"),
                "description": clean_html(f.get("description", ""))[:300],
            })
        return out

    def get_item(self, item_id: int) -> dict | None:
        """Full single item by id (cleaned text + key metadata)."""
        try:
            raw = self._get(f"/items/{item_id}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        if not isinstance(raw, dict):
            return None
        return self._normalize_item(raw.get("data") or {})

    def get_item_children(self, item_id: int, limit: int = 50) -> list[dict]:
        """Decomposition children of an item."""
        out = []
        for raw in self._paginate(f"/items/{item_id}/children",
                                  max_pages=10):
            out.append(self._compact_item(raw))
            if len(out) >= limit:
                break
        return out

    def get_item_relationships(self, item_id: int, limit: int = 50) -> list[dict]:
        """Relationships where ``item_id`` is the source (fromItem) or target.

        Delegates to :meth:`list_project_relationships` after resolving the
        item's project. Note: Jama has no server-side per-item relationship
        filter, so this walks the whole project's relationships and filters
        client-side — for large projects prefer
        ``list_project_relationships`` directly, or keep ``limit`` small.
        """
        item = self.get_item(item_id)
        if not item or not item.get("project_id"):
            return []
        return self.list_project_relationships(
            item["project_id"], limit=limit, item_id=item_id)

    def list_project_relationships(self, project_id: int, *, limit: int = 50,
                                   item_id: int | None = None) -> list[dict]:
        """List relationships for a project (cursor-paginated ``/relationships``).

        Jama's ``/relationships`` endpoint requires a ``project`` filter and
        uses ``lastId`` cursor pagination (not ``startAt``). When ``item_id``
        is given, results are filtered client-side to those where the item is
        the source (``fromItem``) or target (``toItem``).

        Args:
            project_id: Jama project id (required by the endpoint).
            limit: max relationships to return.
            item_id: optional item id to filter on (fromItem/toItem match).
        """
        cfg = settings.jama
        url = f"{cfg.url}{cfg.api_prefix}/relationships"
        out: list[dict] = []
        last_id = 0
        pages = 0
        max_pages = 50  # safety cap on cursor walks
        while pages < max_pages:
            params = {"project": project_id, "lastId": last_id,
                      "maxResults": cfg.page_size}
            r = self._s.get(url, headers=self._headers(), params=params,
                            timeout=cfg.request_timeout)
            if r.status_code == 401:
                with self._lock:
                    self._token = None
                r = self._s.get(url, headers=self._headers(), params=params,
                                timeout=cfg.request_timeout)
            r.raise_for_status()
            data = r.json()
            rows = data.get("data", []) or []
            if not rows:
                break
            for raw in rows:
                if item_id is not None and \
                        raw.get("fromItem") != item_id and \
                        raw.get("toItem") != item_id:
                    continue
                out.append(self._compact_relationship(raw))
                if len(out) >= limit:
                    return out
            last_id = rows[-1].get("id", last_id)
            page_info = data.get("meta", {}).get("pageInfo", {})
            pages += 1
            if int(page_info.get("resultCount", 0) or 0) == 0:
                break
            total = int(page_info.get("totalResults", 0) or 0)
            if total and pages * cfg.page_size >= total:
                break
            if cfg.page_delay:
                time.sleep(cfg.page_delay)
        return out

    def list_test_runs(self, *, project_id: int | None = None,
                       test_cycle_id: int | None = None,
                       limit: int = 50) -> list[dict]:
        """Test runs for a project and/or test cycle."""
        if project_id is None and test_cycle_id is None:
            raise ValueError("list_test_runs requires project_id or test_cycle_id")
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project"] = project_id
        if test_cycle_id is not None:
            params["testCycle"] = test_cycle_id
        out = []
        for raw in self._paginate("/testruns", params, max_pages=20):
            f = raw.get("fields", {}) or {}
            out.append({
                "id": raw.get("id"),
                "name": f.get("name"),
                "status": f.get("testRunStatus") or f.get("status"),
                "test_cycle": raw.get("testCycle"),
                "item": raw.get("item"),
                "assigned_to": f.get("assignedTo"),
                "modified_date": raw.get("modifiedDate"),
            })
            if len(out) >= limit:
                break
        return out

    def get_item_comments(self, item_id: int, limit: int = 50) -> list[dict]:
        """Comments threaded on an item (cleaned body).

        Uses the ``/items/{id}/comments`` sub-resource (item-scoped), which is
        the canonical Jama endpoint for item comments.
        """
        out = []
        for raw in self._paginate(f"/items/{item_id}/comments", max_pages=10):
            f = raw.get("fields", {}) or {}
            out.append({
                "id": raw.get("id"),
                "body": clean_html(f.get("description", ""))[:1000],
                "created_by": raw.get("createdBy"),
                "created_date": raw.get("createdDate"),
                "modified_date": raw.get("modifiedDate"),
            })
            if len(out) >= limit:
                break
        return out

    def get_item_attachments(self, item_id: int, limit: int = 50) -> list[dict]:
        """Attachment metadata for an item (no binary download).

        Uses the ``/items/{id}/attachments`` sub-resource (item-scoped); the
        flat ``/attachments`` endpoint does not accept an ``item`` filter.
        """
        out = []
        for raw in self._paginate(f"/items/{item_id}/attachments", max_pages=10):
            f = raw.get("fields", {}) or {}
            out.append({
                "id": raw.get("id"),
                "name": f.get("name"),
                "file_type": f.get("fileType"),
                "file_size": f.get("fileSize"),
                "mime_type": f.get("mimeType"),
                "created_date": raw.get("createdDate"),
                "modified_date": raw.get("modifiedDate"),
            })
            if len(out) >= limit:
                break
        return out

    def list_releases(self, project_id: int, limit: int = 50) -> list[dict]:
        """Releases / versions for a project."""
        out = []
        for raw in self._paginate("/releases", {"project": project_id}, max_pages=10):
            f = raw.get("fields", {}) or {}
            out.append({
                "id": raw.get("id"),
                "name": f.get("name"),
                "release_date": f.get("releaseDate"),
                "status": f.get("status"),
                "description": clean_html(f.get("description", ""))[:300],
                "modified_date": raw.get("modifiedDate"),
            })
            if len(out) >= limit:
                break
        return out

    def list_item_types(self) -> list[dict]:
        """All item types (id -> display) for the tenant."""
        self.load_item_types()
        return [{"id": k, "name": v} for k, v in sorted(self._item_types.items())]

    def find_projects(self, name: str, *, exact: bool = False,
                      limit: int = 20) -> list[dict]:
        """Find projects whose name matches ``name`` (case-insensitive).

        Walks the project list and filters client-side because Jama's
        ``/projects`` endpoint has no server-side name filter. With
        ``exact=True`` only names that equal ``name`` are returned; otherwise
        substring containment is used (so "acre" matches "Acrelec"). Each
        result carries the project id so callers can feed it straight into
        ``init_jama_project`` / ``list_jama_releases`` etc.

        Args:
            name: project name (or fragment) to match.
            exact: require full case-insensitive equality instead of substring.
            limit: max matches to return (default 20).

        Returns:
            list of {id, project_key, name, status, description} dicts.
        """
        needle = (name or "").strip().lower()
        if not needle:
            return []
        out: list[dict] = []
        for p in self._paginate("/projects", max_pages=50):
            f = p.get("fields", {}) or {}
            pname = (f.get("name") or "").strip()
            hay = pname.lower()
            match = (hay == needle) if exact else (needle in hay)
            if not match:
                continue
            out.append({
                "id": p.get("id"),
                "project_key": p.get("projectKey"),
                "name": pname,
                "status": f.get("status"),
                "description": clean_html(f.get("description", ""))[:300],
            })
            if len(out) >= limit:
                break
        return out

    def find_item_types(self, name: str, *, exact: bool = False,
                        limit: int = 20) -> list[dict]:
        """Find item types whose display name matches ``name``.

        Fetches the full ``/itemtypes`` payloads (richer than the cached
        ``{id: display}`` map — includes category, category name, display
        plural and description) and filters by display name. Matching is
        case-insensitive; ``exact=True`` requires equality, otherwise
        substring containment is used.

        Args:
            name: type name (or fragment) to match, e.g. "test", "Requirement".
            exact: require full case-insensitive equality instead of substring.
            limit: max matches to return (default 20).

        Returns:
            list of {id, display, display_plural, category, category_name,
            description} dicts.
        """
        needle = (name or "").strip().lower()
        if not needle:
            return []
        out: list[dict] = []
        for it in self._paginate("/itemtypes", max_pages=10):
            if not isinstance(it, dict):
                continue
            display = (it.get("display") or "").strip()
            hay = display.lower()
            match = (hay == needle) if exact else (needle in hay)
            if not match:
                continue
            out.append({
                "id": it.get("id"),
                "display": display,
                "display_plural": it.get("displayPlural"),
                "category": it.get("category"),
                "category_name": it.get("categoryName"),
                "description": (it.get("description") or "").strip(),
            })
            if len(out) >= limit:
                break
        return out

    # ----- compact shapers for list responses --------------------------- #
    def _compact_item(self, raw: dict) -> dict:
        f = raw.get("fields", {}) or {}
        return {
            "item_id": raw.get("id"),
            "document_key": raw.get("documentKey"),
            "global_id": raw.get("globalId"),
            "item_type": raw.get("itemType"),
            "item_type_name": self.item_type_name(raw.get("itemType")),
            "name": (f.get("name") or "").strip(),
            "status": f.get("status") or f.get("testCaseStatus") or "",
            "modified_date": raw.get("modifiedDate"),
        }

    def _compact_relationship(self, raw: dict) -> dict:
        f = raw.get("fields", {}) or {}
        return {
            "id": raw.get("id"),
            "relationship_type": raw.get("relationshipType"),
            "source_item": raw.get("fromItem"),
            "target_item": raw.get("toItem"),
            "suspect": raw.get("suspect"),
            "name": (f.get("name") or "").strip(),
            "modified_date": raw.get("modifiedDate"),
        }


def _safe_dumps(obj: Any) -> str:
    try:
        import json
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+0000")
