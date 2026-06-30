"""Network guards: pre-flight speed tests.

This module addresses the case where a large download (a Jama project dump or
HuggingFace model weights) is about to start on a connection too slow to
finish: :func:`speed_test` measures throughput against the target host first
and raises :class:`NetworkTooSlowError` if it's below a configured floor, so
the caller aborts with a clear message instead of timing out halfway through.

(The previous ``download_with_retry`` resume helper was removed — the reranker
now uses ``huggingface_hub.snapshot_download``, which has its own retries, and
no caller passed through this module's transfer layer any more.)
"""
from __future__ import annotations

import logging
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

