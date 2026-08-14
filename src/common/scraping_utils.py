"""Shared politeness/reliability utilities for HTTP-based data fetchers
(Screener.in, Trendlyne, and anything else scraped in the future).

Centralizing this in one place means every scraper gets the same behavior for
free: a real User-Agent (some sites 403 the default `python-requests` UA),
retry-with-backoff on transient failures, per-domain rate limiting so a
500-symbol universe run doesn't hammer a site, a robots.txt check done once
per domain, and an on-disk cache so repeated runs during development don't
re-fetch pages that haven't changed.

IMPORTANT — read before running any of this against real sites: scraping a
site's HTML (as opposed to using an official API) is inherently fragile
(breaks silently when the site's markup changes) and may be restricted by
that site's Terms of Service regardless of what robots.txt allows. This
module makes scraping *polite* (rate-limited, cached, robots.txt-aware) but
that is not the same as *permitted* — review each site's ToS yourself before
running this at scale, and prefer an official API/data vendor for anything
beyond personal/academic research use. See docs/data_sourcing_spec.md.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_last_request_time: dict[str, float] = {}  # domain -> monotonic time of last request
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def build_session(user_agent: str = DEFAULT_USER_AGENT, total_retries: int = 3) -> requests.Session:
    """A requests.Session with retry-with-backoff on connection errors and
    5xx/429 responses, and a browser-like User-Agent set by default."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.8"})
    retry = Retry(
        total=total_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def is_allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    """Check robots.txt for ``url``'s domain (cached per-domain per process).
    Fails open (returns True) if robots.txt can't be fetched/parsed — absence
    of a robots.txt doesn't mean scraping is prohibited, but note the
    module-level docstring caveat: this is not a substitute for reading the
    site's actual Terms of Service.
    """
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    if domain not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{domain}/robots.txt")
        try:
            rp.read()
        except Exception as exc:  # noqa: BLE001 - robots.txt fetch failures shouldn't be fatal
            logger.warning("Could not fetch robots.txt for %s (%s) — failing open", domain, exc)
            _robots_cache[domain] = None  # type: ignore[assignment]
            return True
        _robots_cache[domain] = rp
    rp = _robots_cache[domain]
    if rp is None:
        return True
    return rp.can_fetch(user_agent, url)


def rate_limited_get(
    session: requests.Session,
    url: str,
    min_delay_seconds: float = 2.0,
    timeout: int = 15,
    check_robots: bool = True,
) -> requests.Response:
    """GET a URL, enforcing a minimum delay since the last request to the
    same domain (module-level state, so it applies across every call in a
    process — e.g. looping over 500 symbols on the same site).
    """
    if check_robots and not is_allowed_by_robots(url):
        raise PermissionError(f"robots.txt disallows fetching {url}")

    domain = urlparse(url).netloc
    last = _last_request_time.get(domain, 0.0)
    elapsed = time.monotonic() - last
    if elapsed < min_delay_seconds:
        time.sleep(min_delay_seconds - elapsed)

    response = session.get(url, timeout=timeout)
    _last_request_time[domain] = time.monotonic()
    response.raise_for_status()
    return response


class DiskCache:
    """A trivial TTL-based on-disk cache for HTTP response text, keyed by URL.
    Not thread-safe; fine for the sequential batch-fetch scripts this is used in.
    """

    def __init__(self, cache_dir: str = "data/raw/.cache", ttl_days: float = 7.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_days * 86400

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.ttl_seconds:
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        with open(path, "w") as f:
            json.dump({"key": key, "value": value}, f)


def cached_get_text(
    session: requests.Session,
    url: str,
    cache: DiskCache | None = None,
    min_delay_seconds: float = 2.0,
) -> str:
    """rate_limited_get, but check/populate ``cache`` first so re-runs during
    development don't re-fetch (and re-rate-limit-wait for) the same page."""
    if cache is not None:
        cached = cache.get(url)
        if cached is not None:
            logger.debug("Cache hit for %s", url)
            return cached
    response = rate_limited_get(session, url, min_delay_seconds=min_delay_seconds)
    text = response.text
    if cache is not None:
        cache.set(url, text)
    return text
