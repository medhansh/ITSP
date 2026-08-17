"""Multi-source fundamentals orchestration: Screener + Trendlyne + yfinance,
merged field-by-field into one SNAPSHOT_SCHEMA row per symbol.

This module defines the exact schema the rest of fundamental_analysis expects
(``SNAPSHOT_SCHEMA`` / ``HISTORY_SCHEMA`` below) and ``fetch_fundamentals``,
which runs each configured source and merges the results via merge.py. The
individual source fetchers live in their own modules:
  - screener_fetcher.py — primary source; best financial-statement coverage.
  - yfinance_fetcher.py — fallback; best analyst-estimate coverage; also the
    price-panel source used by regime_detection and backtesting.

See docs/data_sourcing_spec.md for the full per-field coverage table, the
scraping-etiquette notes, and the look-ahead-bias caveat that applies when
using this for backtesting (a single "as of today" snapshot is NOT a
substitute for point-in-time historical fundamentals).
"""
from __future__ import annotations

import pandas as pd

from src.common.logging_utils import get_logger
from src.common.scraping_utils import DiskCache, build_session
from src.fundamental_analysis.data_fetchers import screener_fetcher, yfinance_fetcher
from src.fundamental_analysis.data_fetchers.merge import merge_sources

logger = get_logger(__name__)

# Columns every function in metrics/ (that consumes a snapshot) expects to
# find, even if NaN. Kept in one place so a new data source can be validated
# against it directly.
SNAPSHOT_SCHEMA = [
    "symbol", "sector", "industry",
    "price", "shares_outstanding", "market_cap",
    "eps_ttm", "eps_growth_pct", "book_value_per_share", "dividend_per_share",
    "ebitda", "ebit", "enterprise_value", "revenue", "net_income",
    "total_assets", "total_assets_prior", "total_liabilities", "total_equity",
    "total_debt", "interest_expense", "retained_earnings",
    "current_assets", "current_assets_prior",
    "current_liabilities", "current_liabilities_prior",
    "long_term_debt", "long_term_debt_prior",
    "gross_profit", "gross_profit_prior", "revenue_prior",
    "shares_outstanding_prior", "net_income_prior",
    "cfo", "capex",
    "promoter_holding_pct", "promoter_holding_pct_prior", "promoter_pledge_pct",
    "fii_holding_pct", "fii_holding_pct_prior", "dii_holding_pct", "dii_holding_pct_prior",
    "related_party_transactions_flag", "auditor_changed_flag",
    "actual_eps", "analyst_eps_estimate", "analyst_eps_estimate_30d_ago",
]

HISTORY_SCHEMA = ["symbol", "fiscal_year", "revenue", "net_income", "eps"]

DEFAULT_SOURCE_PRIORITY = ["screener", "yfinance"]


def fetch_fundamentals(
    symbols: list[str],
    sources: list[str] = ("screener", "yfinance"),
    source_priority: list[str] | None = None,
    min_delay_seconds: float = 2.0,
    cache_dir: str = "data/raw/.cache",
    cache_ttl_days: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch fundamentals for ``symbols`` from every source in ``sources``,
    then merge field-by-field using ``source_priority`` (defaults to
    DEFAULT_SOURCE_PRIORITY) as the fallback order.

    Returns (merged_snapshot_df, provenance_df) — provenance has the same
    shape as the snapshot and records which source each cell came from (or
    "missing"), so results are auditable rather than a black box. Both are
    indexed by symbol; snapshot columns are reindexed onto SNAPSHOT_SCHEMA
    (missing ones added as all-NaN) so downstream metrics/ code never KeyErrors.

    This is slow by design for a full NIFTY500 run (Screener/Trendlyne have
    no bulk endpoint, so each is one rate-limited HTTP request per symbol per
    source) — expect ~500 symbols x ~2-3 sec/request x up to 2 scraped
    sources to take well over an hour. Results are cached to disk
    (``cache_dir``), so interrupted/re-run fetches don't repeat completed work.
    """
    session = build_session()
    cache = DiskCache(cache_dir=cache_dir, ttl_days=cache_ttl_days)
    source_priority = source_priority or DEFAULT_SOURCE_PRIORITY

    source_dataframes: dict[str, pd.DataFrame] = {}
    if "screener" in sources:
        logger.info("Fetching fundamentals from Screener.in for %d symbols", len(symbols))
        source_dataframes["screener"] = screener_fetcher.fetch_multiple(
            symbols, session=session, cache=cache, min_delay_seconds=min_delay_seconds
        )
    if "yfinance" in sources:
        logger.info("Fetching fundamentals from yfinance for %d symbols", len(symbols))
        source_dataframes["yfinance"] = yfinance_fetcher.fetch_multiple(symbols)
    # Trendlyne was dropped from the lean build: it is supplementary-only
    # and most of its useful fields are paywalled (see the module docstring
    # this project shipped before the cut). Passing "trendlyne" in `sources`
    # now has no effect rather than raising, so an old config value here is
    # silently harmless instead of a crash.

    merged, provenance = merge_sources(source_dataframes, source_priority)
    merged = merged.reindex(columns=SNAPSHOT_SCHEMA)
    merged.index.name = "symbol"
    return merged, provenance


def fetch_fundamentals_history(
    symbols: list[str],
    session=None,
    cache: DiskCache | None = None,
    min_delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Multi-year revenue/net_income/eps history for the growth dimension,
    sourced from Screener's profit & loss table (the only one of the three
    sources with clean multi-year annual history for free).
    """
    session = session or build_session()
    cache = cache or DiskCache()
    rows = []
    for symbol in symbols:
        html = None
        try:
            from src.common.scraping_utils import cached_get_text

            url = screener_fetcher._company_url(symbol, consolidated=True)
            html = cached_get_text(session, url, cache=cache, min_delay_seconds=min_delay_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch history page for %s: %s", symbol, exc)
            continue

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        pl = screener_fetcher._parse_data_table(soup, screener_fetcher.SECTION_IDS["profit_loss"])
        if pl.empty:
            continue
        sales = screener_fetcher._lookup_row(pl, "sales", "revenue")
        net_profit = screener_fetcher._lookup_row(pl, "net profit")
        eps = screener_fetcher._lookup_row(pl, "eps in rs", "eps")
        if sales is None:
            continue
        for period in sales.index:
            if period.upper() == "TTM":
                continue  # TTM isn't a fiscal year; growth.py wants fiscal-year rows only
            rows.append({
                "symbol": symbol,
                "fiscal_year": period,
                "revenue": sales.get(period, float("nan")) * screener_fetcher.CR_TO_RUPEES,
                "net_income": net_profit.get(period, float("nan")) * screener_fetcher.CR_TO_RUPEES if net_profit is not None else float("nan"),
                "eps": eps.get(period, float("nan")) if eps is not None else float("nan"),
            })
    return pd.DataFrame(rows, columns=HISTORY_SCHEMA)
