"""Network guards: pre-flight speed tests + resumable downloads with retry.

Two concerns this module addresses:
1. **Pre-flight speed test** — before pulling large data (Jama project dumps,
   HuggingFace model weights), measure throughput against the target host.
   If it's below a configured floor, abort immediately with a clear
   ``NetworkTooSlowError`` instead of starting a download that will time out
   halfway through.
2. **Resumable download with retry** — for downloads that stall mid-stream
   (speed drops below floor) or hit network timeouts/errors, resume from the
   partial byte offset with bounded retries.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


class NetworkTooSlowError(RuntimeError):
    """Raised when a pre-flight speed test or a download stalls below the
    configured minimum throughput. Carries the measured speed so callers can
    surface a clear message to the LLM/user."""


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def speed_test(url: str, *, min_bytes_per_sec: int, timeout: int,
               headers: dict | None = None,
               label: str = "host") -> float:
    """Measure download throughput (bytes/sec) against ``url``.

    Downloads a bounded probe (up to ~1 MB or ``timeout`` seconds, whichever
    comes first) and returns the measured bytes/sec. The throughput clock
    starts when the first byte of the body arrives (so TLS handshake + server
    processing latency don't deflate the measured bandwidth). Raises
    ``NetworkTooSlowError`` if the measured speed is below
    ``min_bytes_per_sec`` or if the request itself errors out.
    """
    sess = _session()
    probe_bytes = 1_048_576  # 1 MB probe
    received = 0
    first_byte_t0: float | None = None
    try:
        with sess.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code >= 400:
                # A 4xx/5xx on the probe URL is a connectivity/auth problem,
                # not a slow-network problem — surface it differently.
                raise NetworkTooSlowError(
                    f"[{label}] speed test failed: HTTP {r.status_code} from "
                    f"{url}. Check network connectivity and credentials.")
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                if first_byte_t0 is None:
                    first_byte_t0 = time.monotonic()
                received += len(chunk)
                if first_byte_t0 is not None and \
                        (time.monotonic() - first_byte_t0) >= timeout:
                    break
                if received >= probe_bytes:
                    break
    except requests.RequestException as exc:
        raise NetworkTooSlowError(
            f"[{label}] speed test failed (network error): {exc}. "
            f"This is likely a network connectivity issue.") from exc

    # If we never received any body bytes, the connection is effectively dead.
    if first_byte_t0 is None or received == 0:
        raise NetworkTooSlowError(
            f"[{label}] speed test failed: no data received within {timeout}s. "
            f"This is likely a network connectivity issue.")
    # Throughput = body bytes / body-transfer time (excludes handshake).
    elapsed = max(time.monotonic() - first_byte_t0, 0.001)
    bps = received / elapsed
    log.info("[%s] speed test: %d bytes in %.2fs (body) = %.0f bytes/s "
             "(floor %d)", label, received, elapsed, bps, min_bytes_per_sec)
    if bps < min_bytes_per_sec:
        raise NetworkTooSlowError(
            f"[{label}] network is too slow: measured {bps:.0f} bytes/s "
            f"(floor {min_bytes_per_sec} bytes/s). Aborting download to avoid "
            f"mid-stream timeouts. Please check the network connection.")
    return bps


def download_with_retry(url: str, dest: str, *, min_bytes_per_sec: int,
                        max_retries: int = 4, chunk_timeout: float = 30.0,
                        headers: dict | None = None,
                        label: str = "download") -> str:
    """Download ``url`` to ``dest`` with resume + stall retry.

    Resumes from the existing partial file (byte offset) on each attempt. If
    throughput during a chunk window drops below ``min_bytes_per_sec`` or a
    network error occurs, the attempt is aborted and retried (up to
    ``max_retries``). Raises ``NetworkTooSlowError`` only if all retries are
    exhausted due to sustained stalls.
    """
    sess = _session()
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    expected = _content_length(url, headers, sess, label)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if expected and have >= expected:
            log.info("[%s] already complete (%d bytes)", label, have)
            return dest
        req_headers = dict(headers or {})
        if have:
            req_headers["Range"] = f"bytes={have}-"
        t0 = time.monotonic()
        received = have
        last_window_bytes = have
        last_window_time = t0
        try:
            with sess.get(url, headers=req_headers, stream=True,
                          timeout=chunk_timeout) as r:
                if r.status_code not in (200, 206):
                    raise NetworkTooSlowError(
                        f"[{label}] HTTP {r.status_code} on download attempt "
                        f"{attempt}/{max_retries}: {r.text[:200]}")
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                    received = 0
                    last_window_bytes = 0
                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        # Stall guard: measure speed over a rolling window.
                        window = now - last_window_time
                        if window >= 5.0:
                            window_bytes = received - last_window_bytes
                            bps = window_bytes / window
                            if bps < min_bytes_per_sec:
                                raise NetworkTooSlowError(
                                    f"[{label}] download stalled at "
                                    f"{received}/{expected or '?'} bytes: "
                                    f"{bps:.0f} bytes/s < floor "
                                    f"{min_bytes_per_sec}. Retrying "
                                    f"(attempt {attempt}/{max_retries}).")
                            last_window_bytes = received
                            last_window_time = now
            # Success if expected is known and reached, or no expected size.
            if not expected or os.path.getsize(dest) >= expected:
                log.info("[%s] complete: %d bytes", label,
                         os.path.getsize(dest))
                return dest
            last_err = NetworkTooSlowError(
                f"[{label}] incomplete after attempt {attempt}: "
                f"{os.path.getsize(dest)}/{expected} bytes.")
        except (requests.RequestException, NetworkTooSlowError) as exc:
            last_err = exc
            have_now = os.path.getsize(dest) if os.path.exists(dest) else 0
            log.warning("[%s] attempt %d/%d failed at %d/%s bytes: %s",
                        label, attempt, max_retries, have_now,
                        expected or "?", exc)
            time.sleep(min(2 ** attempt, 10))
    raise NetworkTooSlowError(
        f"[{label}] download failed after {max_retries} retries: {last_err}. "
        f"This is a network problem — please check connectivity and retry.")


def _content_length(url: str, headers: dict | None, sess: requests.Session,
                    label: str) -> int | None:
    try:
        r = sess.head(url, headers=headers, timeout=15, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception as exc:
        log.debug("[%s] HEAD for content-length failed: %s", label, exc)
        return None
