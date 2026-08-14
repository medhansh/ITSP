"""Scraper for trendlyne.com's public equity page — supplementary data only.

READ THIS BEFORE USING: Trendlyne's most valuable fundamentals data —
Durability score, Valuation score, promoter pledge %, and analyst
estimates/target-price detail — is gated behind a paid "GuruQ"/"StratQ"
subscription. This was confirmed directly by fetching a live Trendlyne
company page during development: the Momentum score, SWOT strength/weakness/
opportunity/threat *counts*, current price/volume, and sector/industry were
visible; Durability score, Valuation score, promoter pledge, and shareholding
changes were explicitly replaced with "requires GuruQ or StratQ subscription"
messaging. This fetcher only extracts what's actually free — it does not
pretend to scrape paywalled data, and returns NaN with a documented reason
for anything gated. If you have a paid Trendlyne subscription, you can pass
an authenticated ``requests.Session`` (with cookies from your own logged-in
browser session — obtained by you, never handled by this codebase) via the
``session`` argument to potentially unlock more fields; this is untested.

Symbol resolution: Trendlyne URLs use an internal numeric ID
(e.g. RELIANCE -> 1127), not the NSE symbol directly. There is no confirmed
public search API for this in the current build (a couple of guessed
endpoints returned 405s during development). The reliable path is a local
mapping file — see ``resolve_trendlyne_id`` below and
docs/data_sourcing_spec.md for how to build one.

Extraction here is regex-over-visible-text rather than CSS-selector-based,
deliberately: without confirmed exact class/id names for this site (see
module note above), matching on label text is more robust to markup
changes than guessing selectors that may simply be wrong.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from src.common.logging_utils import get_logger
from src.common.scraping_utils import DiskCache, build_session, cached_get_text

logger = get_logger(__name__)

PAYWALLED_FIELDS = [
    "durability_score", "valuation_score", "promoter_pledge_pct",
    "analyst_target_price", "shareholding_change_flag",
]


def resolve_trendlyne_id(symbol: str, mapping_path: str = "data/universe/trendlyne_id_map.csv") -> str | None:
    """Look up the Trendlyne numeric ID + URL slug for ``symbol`` from a local
    mapping file (columns: symbol, trendlyne_id, slug). Build this file once —
    e.g. by searching "trendlyne <company name>" for each NIFTY500 constituent
    and recording the ID from the resulting URL
    (trendlyne.com/equity/<id>/<SYMBOL>/<slug>/) — since no reliable public
    resolver endpoint was confirmed during development. Returns None (not an
    exception) if the file or the symbol isn't found, so callers can degrade
    gracefully to "Trendlyne data unavailable for this symbol".
    """
    path = Path(mapping_path)
    if not path.is_absolute():
        # Resolve relative to project root (two levels up from src/fundamental_analysis/data_fetchers/)
        path = Path(__file__).resolve().parents[3] / mapping_path
    if not path.exists():
        logger.warning("Trendlyne ID map not found at %s — cannot resolve %s", path, symbol)
        return None
    df = pd.read_csv(path, comment="#")
    match = df[df["symbol"] == symbol]
    if match.empty:
        return None
    row = match.iloc[0]
    return f"{row['trendlyne_id']}/{symbol}/{row['slug']}"


def _extract_number_near_label(text: str, *label_patterns: str) -> float:
    """Search ``text`` for the first label pattern and return the nearest
    number within ~40 characters after it (handles both "Label: 12.3" and
    "Label 12.3/100" style renderings)."""
    for pattern in label_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            window = text[m.end(): m.end() + 40]
            num_match = re.search(r"-?\d+\.?\d*", window)
            if num_match:
                return float(num_match.group())
    return float("nan")


def _is_paywalled(text: str, *label_patterns: str) -> bool:
    for pattern in label_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            window = text[max(0, m.start() - 30): m.end() + 120]
            if re.search(r"subscription|guruq|stratq|premium|not disclosed", window, re.IGNORECASE):
                return True
    return False


def parse_equity_page(html: str) -> dict[str, Any]:
    """Parse a Trendlyne equity page into the (small) set of free fields,
    plus explicit NaN + a logged reason for paywalled fields. Pure function,
    directly unit-testable against a fixture."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)

    result: dict[str, Any] = {
        "momentum_score": _extract_number_near_label(text, r"momentum\s*score"),
        "swot_strengths": _extract_number_near_label(text, r"strengths?\b"),
        "swot_weaknesses": _extract_number_near_label(text, r"weaknesses?\b"),
        "swot_opportunities": _extract_number_near_label(text, r"opportunit(?:y|ies)"),
        "swot_threats": _extract_number_near_label(text, r"threats?\b"),
        "current_price": _extract_number_near_label(text, r"current\s*price"),
    }

    for field, patterns in {
        "durability_score": [r"durability\s*score"],
        "valuation_score": [r"valuation\s*score"],
        "promoter_pledge_pct": [r"(?:promoter\s*)?pledge"],
        "analyst_target_price": [r"(?:1|one)[\s-]*year\s*(?:price\s*)?target"],
    }.items():
        if _is_paywalled(text, *patterns):
            result[field] = float("nan")
            result[f"{field}_reason"] = "requires Trendlyne GuruQ/StratQ subscription"
        else:
            result[field] = _extract_number_near_label(text, *patterns)
            result[f"{field}_reason"] = None

    return result


def fetch_equity_snapshot(
    symbol: str,
    session=None,
    cache: DiskCache | None = None,
    min_delay_seconds: float = 2.0,
    mapping_path: str = "data/universe/trendlyne_id_map.csv",
) -> dict[str, Any]:
    """Fetch + parse one symbol's Trendlyne page. Returns {"symbol": symbol}
    only (no data) if the symbol can't be resolved to a Trendlyne ID or the
    fetch fails — callers should treat that as "no Trendlyne data available",
    not an error, since coverage here is inherently partial by design."""
    resolved = resolve_trendlyne_id(symbol, mapping_path=mapping_path)
    if resolved is None:
        return {"symbol": symbol}

    session = session or build_session()
    url = f"https://trendlyne.com/equity/{resolved}/"
    try:
        html = cached_get_text(session, url, cache=cache, min_delay_seconds=min_delay_seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Trendlyne page for %s: %s", symbol, exc)
        return {"symbol": symbol}

    fields = parse_equity_page(html)
    fields["symbol"] = symbol
    return fields


def fetch_multiple(
    symbols: list[str],
    session=None,
    cache: DiskCache | None = None,
    min_delay_seconds: float = 2.0,
    mapping_path: str = "data/universe/trendlyne_id_map.csv",
) -> pd.DataFrame:
    session = session or build_session()
    rows = []
    for i, symbol in enumerate(symbols):
        logger.info("Fetching Trendlyne data for %s (%d/%d)", symbol, i + 1, len(symbols))
        rows.append(
            fetch_equity_snapshot(symbol, session=session, cache=cache, min_delay_seconds=min_delay_seconds, mapping_path=mapping_path)
        )
    return pd.DataFrame(rows).set_index("symbol", drop=False)
