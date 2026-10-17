"""HTTP downloads with retries, timeouts and a content-addressed raw cache.

Every downloaded payload is hashed (SHA-256) and kept under cache_dir/<source>/ so that
an ingested vintage can be traced back to the exact bytes it came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chronos.config import Settings
from chronos.errors import SourceError
from chronos.logsetup import get_logger
from chronos.storage.db import utc_now

log = get_logger("ingestion.http")


@dataclass(frozen=True)
class FetchResult:
    url: str
    content: bytes
    sha256: str
    retrieved_utc: str
    etag: str | None
    last_modified: str | None
    cache_path: str | None

    def text(self) -> str:
        return self.content.decode("utf-8-sig")

    def json(self) -> Any:
        return json.loads(self.text())


def _session(settings: Settings) -> requests.Session:
    retry = Retry(total=settings.http_retries, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = settings.user_agent
    return s


def fetch(url: str, settings: Settings, source_id: str, params: dict[str, Any] | None = None,
          suffix: str = ".bin", keep_raw: bool = True) -> FetchResult:
    session = _session(settings)
    try:
        resp = session.get(url, params=params, timeout=settings.http_timeout)
    except requests.exceptions.ProxyError as exc:
        raise SourceError(
            f"Proxy refused the connection to {url}. In a sandbox this usually means the host is not on the "
            f"network allowlist. Detail: {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise SourceError(f"Download failed for {url}: {exc}") from exc
    if resp.status_code != 200:
        raise SourceError(f"HTTP {resp.status_code} for {resp.url}: {resp.text[:300]!r}")
    content = resp.content
    digest = hashlib.sha256(content).hexdigest()
    cache_path = None
    if keep_raw:
        folder = Path(settings.cache_dir) / source_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{digest[:20]}{suffix}"
        if not path.exists():
            path.write_bytes(content)
        cache_path = str(path)
    log.info("downloaded", extra={"ctx": {"url": resp.url, "bytes": len(content), "sha256": digest[:12]}})
    return FetchResult(resp.url, content, digest, utc_now(), resp.headers.get("ETag"),
                       resp.headers.get("Last-Modified"), cache_path)


def read_local(path: str | Path) -> FetchResult:
    p = Path(path)
    if not p.exists():
        raise SourceError(f"File not found: {p}")
    content = p.read_bytes()
    return FetchResult(str(p.resolve()), content, hashlib.sha256(content).hexdigest(), utc_now(), None, None, str(p))
