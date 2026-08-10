"""A small, polite HTTP client for pulling real sports data.

Standard library only, and deliberately cautious about other people's servers:

* every response is cached on disk with a TTL, so re-running a fetch costs
  nothing and a season file is downloaded once a day at most;
* requests to the same host are spaced out;
* 5xx and connection errors retry with backoff, 4xx never does -- a 404 means
  the player or season doesn't exist and hammering it won't help;
* gzip is detected from the payload's magic bytes rather than trusted headers,
  because several of these feeds serve ``.csv`` files that are gzipped and
  ``.csv.gz`` files that aren't.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Mapping

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
GZIP_MAGIC = b"\x1f\x8b"


class FetchError(RuntimeError):
    """A request failed in a way the caller should hear about."""


class HttpClient:
    def __init__(
        self,
        cache_dir: str | Path = ".cache/ingest",
        ttl: float = 43_200.0,  # 12 hours
        timeout: float = 30.0,
        min_interval: float = 0.5,
        user_agent: str = DEFAULT_USER_AGENT,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.timeout = timeout
        self.min_interval = min_interval
        self.user_agent = user_agent
        self.offline = offline
        self._last_request: dict[str, float] = {}

    # ------------------------------------------------------------------
    def get_bytes(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        ttl: float | None = None,
        attempts: int = 3,
    ) -> bytes:
        ttl = self.ttl if ttl is None else ttl
        cache_file = self._cache_path(url)

        if cache_file.exists() and (ttl <= 0 or time.time() - cache_file.stat().st_mtime < ttl):
            return cache_file.read_bytes()
        if self.offline:
            raise FetchError(f"offline and nothing cached for {url}")

        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, **(headers or {})})
        last_error: Exception | None = None
        for attempt in range(attempts):
            self._throttle(url)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:300].decode(errors="replace")
                if exc.code < 500 and exc.code != 429:
                    raise FetchError(f"HTTP {exc.code} for {url}: {detail}") from exc
                last_error = FetchError(f"HTTP {exc.code} for {url}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = FetchError(f"could not reach {urllib.parse.urlsplit(url).netloc}: {exc}")
            if attempt < attempts - 1:
                time.sleep(2.0**attempt)
        else:
            raise last_error or FetchError(f"gave up on {url}")

        if body.startswith(GZIP_MAGIC):
            body = gzip.decompress(body)
        self._write_cache(cache_file, body)
        return body

    def get_json(self, url: str, headers: Mapping[str, str] | None = None, ttl: float | None = None) -> Any:
        raw = self.get_bytes(url, headers, ttl)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            self._cache_path(url).unlink(missing_ok=True)  # don't cache garbage
            raise FetchError(f"{url} did not return JSON: {raw[:200]!r}") from exc

    def get_csv(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        ttl: float | None = None,
    ) -> Iterator[dict[str, str]]:
        raw = self.get_bytes(url, headers, ttl)
        text = raw.decode("utf-8-sig", errors="replace")
        yield from csv.DictReader(io.StringIO(text))

    # ------------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:32]}.bin"

    @staticmethod
    def _write_cache(path: Path, body: bytes) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)  # atomic: a half-written cache file is worse than none

    def _throttle(self, url: str) -> None:
        host = urllib.parse.urlsplit(url).netloc
        elapsed = time.time() - self._last_request.get(host, 0.0)
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request[host] = time.time()
