"""yfinance-based fetchers: fundamentals fallback + analyst estimates, and
the price panel used by regime_detection and backtesting.

yfinance is the most reliable *free* source for two things Screener/Trendlyne
don't cleanly give us: consistent daily OHLC price history across the whole
universe (via a single batched call), and best-effort analyst
estimates/target price. It's a weaker fundamentals source than Screener for
NSE-listed companies (coverage is inconsistent, several fields are None for
smaller names), which is why it's used as the *fallback* source in the
multi-source merge (see merge.py) rather than the primary one.

Every network-calling function here accepts an optional dependency-injected
callable so the parsing/shaping logic can be unit-tested without hitting
yfinance/the network at all.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

INDIA_SPECIFIC_FIELDS_NOT_IN_YFINANCE = [
    "promoter_holding_pct", "promoter_pledge_pct", "fii_holding_pct", "dii_holding_pct",
    "related_party_transactions_flag", "auditor_changed_flag",
]


def _default_ticker_info(symbol: str) -> dict[str, Any]:
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        return ticker.info or {}
    except ImportError:
        logger.warning("yfinance is not installed — `pip install yfinance` to use this fetcher. Returning empty data.")
        return {}
    except Exception as exc:  # noqa: BLE001 - yfinance raises generic Exceptions on network/parse failure
        logger.warning("yfinance failed to fetch info for %s: %s", symbol, exc)
        return {}


def parse_ticker_info(symbol: str, info: dict[str, Any]) -> dict[str, Any]:
    """Map a yfinance ``Ticker.info`` dict onto our SNAPSHOT_SCHEMA field
    names. Pure function (no network) so it's directly unit-testable against
    a fixture dict shaped like a real yfinance response.
    """
    return {
        "symbol": symbol,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "market_cap": info.get("marketCap"),
        "eps_ttm": info.get("trailingEps"),
        "eps_growth_pct": info.get("earningsGrowth") * 100 if info.get("earningsGrowth") is not None else np.nan,
        "book_value_per_share": info.get("bookValue"),
        "dividend_per_share": info.get("dividendRate"),
        "gross_profit": info.get("grossProfits"),
        "ebitda": info.get("ebitda"),
        # approximation: yfinance doesn't expose EBIT directly for most NSE
        # tickers; proxied as EBITDA less D&A when both are available.
        "ebit": (
            info.get("ebitda") - info.get("depreciationAndAmortization", 0)
            if info.get("ebitda") is not None
            else np.nan
        ),
        "enterprise_value": info.get("enterpriseValue"),
        "revenue": info.get("totalRevenue"),
        "net_income": info.get("netIncomeToCommon"),
        "total_debt": info.get("totalDebt"),
        "current_assets": info.get("totalCurrentAssets"),
        "current_liabilities": info.get("totalCurrentLiabilities"),
        "cfo": info.get("operatingCashflow"),
        "capex": -info.get("capitalExpenditures") if info.get("capitalExpenditures") is not None else np.nan,
        "actual_eps": info.get("trailingEps"),
        "analyst_eps_estimate": info.get("forwardEps"),
        "analyst_target_price": info.get("targetMeanPrice"),
        "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
    }


def fetch_snapshot(
    symbol: str, ticker_info_fn: Callable[[str], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Fetch + parse one symbol via yfinance. ``ticker_info_fn`` defaults to a
    real yfinance call; pass a fake for tests."""
    fn = ticker_info_fn or _default_ticker_info
    info = fn(symbol)
    fields = parse_ticker_info(symbol, info)
    if not info:
        logger.warning("No yfinance data returned for %s", symbol)
    return fields


def fetch_multiple(
    symbols: list[str], ticker_info_fn: Callable[[str], dict[str, Any]] | None = None
) -> pd.DataFrame:
    rows = [fetch_snapshot(sym, ticker_info_fn=ticker_info_fn) for sym in symbols]
    df = pd.DataFrame(rows).set_index("symbol", drop=False)
    logger.warning(
        "yfinance does not provide %s for NSE tickers — these will be NaN from "
        "this source (see docs/data_sourcing_spec.md for where else they might come from).",
        INDIA_SPECIFIC_FIELDS_NOT_IN_YFINANCE,
    )
    return df


def _default_downloader(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        # Unlike per-symbol fundamentals (which degrade gracefully to NaN),
        # a missing price panel makes every downstream pipeline unusable —
        # fail loudly with an actionable message rather than a bare ImportError.
        raise ImportError(
            "yfinance is not installed. Run `pip install yfinance` (it's in "
            "requirements.txt) to fetch price data."
        ) from exc

    return yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)


def fetch_price_panel_ohlc(
    symbols: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    downloader: Callable[[list[str], str, str | None], pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Full daily OHLCV for a list of NSE symbols (appends ``.NS``) — the
    per-symbol counterpart to ``fetch_price_panel`` (which only returns
    close). Needed by indicators that use true intraday high/low, not a
    close-derived proxy — e.g. Ichimoku's Tenkan/Kijun/Senkou-B (see
    ``backtesting/adaptive_ichimoku.py``).

    Returns ``{symbol: DataFrame}``, each with lowercase columns
    ``open``/``high``/``low``/``close``/``volume``, NOT a single merged
    panel — different symbols' OHLC frames genuinely are separate
    2D-per-symbol data, unlike a close-only panel which is naturally one
    wide table.

    ``downloader`` is dependency-injected exactly like ``fetch_price_panel``,
    for the same reason (testable without network).
    """
    fn = downloader or _default_downloader
    tickers = [f"{s}.NS" for s in symbols]
    raw = fn(tickers, start, end)

    if raw.empty:
        logger.warning("yfinance returned no OHLCV data for the requested universe")
        return {}

    result: dict[str, pd.DataFrame] = {}
    rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}

    if isinstance(raw.columns, pd.MultiIndex):
        # yf.download's column levels are (field, ticker) by default.
        field_level = 0 if set(raw.columns.get_level_values(0)) & set(rename_map) else 1
        ticker_level = 1 - field_level
        for ticker in raw.columns.get_level_values(ticker_level).unique():
            symbol = ticker.replace(".NS", "")
            sub = raw.xs(ticker, axis=1, level=ticker_level)
            sub = sub.rename(columns=rename_map)
            sub = sub[[c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]]
            if sub["close"].notna().any() if "close" in sub.columns else False:
                result[symbol] = sub
    else:
        # Single-ticker download returns flat columns.
        sub = raw.rename(columns=rename_map)
        sub = sub[[c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]]
        if tickers:
            result[tickers[0].replace(".NS", "")] = sub

    for df in result.values():
        df.index.name = "date"
    return result


def fetch_price_panel(
    symbols: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    downloader: Callable[[list[str], str, str | None], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Daily close-price panel for a list of NSE symbols (appends ``.NS``),
    columns = symbols, index = date. Uses yfinance's batched multi-ticker
    download (one HTTP round-trip family for the whole list, not one call per
    symbol) — the right approach for a 500-symbol universe.

    ``downloader`` is dependency-injected so this is testable without
    network: pass a callable returning a yfinance-shaped MultiIndex-column
    DataFrame (as yf.download returns for multiple tickers).
    """
    fn = downloader or _default_downloader
    tickers = [f"{s}.NS" for s in symbols]
    raw = fn(tickers, start, end)

    if raw.empty:
        logger.warning("yfinance returned no price data for the requested universe")
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # yf.download's column levels are (field, ticker) by default.
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        # Single-ticker download returns flat columns.
        close = raw[["Close"]]
        close.columns = tickers

    close.columns = [c.replace(".NS", "") for c in close.columns]
    close.index.name = "date"
    return close


def fetch_benchmark_ohlcv(
    benchmark_ticker: str = "^CRSLDX",
    start: str = "2015-01-01",
    end: str | None = None,
    downloader: Callable[[list[str], str, str | None], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Full daily OHLCV for a benchmark index (open/high/low/close/volume,
    lowercase columns) — the richer counterpart to ``fetch_benchmark_series``
    (which only returns close). Feeds ``regime_detection.pipeline.run_pipeline``
    via ``data_loader.load_from_csv``, so the regime feature matrix can pick up
    range-based volatility (Parkinson/Garman-Klass) and volume features
    (volume z-score, OBV trend), not just close-derived ones — see
    ``regime_detection/features.py``. Verify the benchmark ticker's volume
    field is actually populated on Yahoo Finance before relying on the
    volume-derived features; some index/total-return-proxy tickers report
    zero or missing volume, in which case those features degrade gracefully
    to all-NaN and get dropped rather than erroring.
    """
    fn = downloader or _default_downloader
    raw = fn([benchmark_ticker], start, end)
    if raw.empty:
        logger.warning("yfinance returned no price data for benchmark %s", benchmark_ticker)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(benchmark_ticker, axis=1, level=1) if benchmark_ticker in raw.columns.get_level_values(1) else raw.droplevel(1, axis=1)

    ohlcv = raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    ohlcv = ohlcv[[c for c in ("open", "high", "low", "close", "volume") if c in ohlcv.columns]]
    ohlcv.index.name = "date"
    return ohlcv


def fetch_benchmark_series(
    benchmark_ticker: str = "^CRSLDX",
    start: str = "2015-01-01",
    end: str | None = None,
    downloader: Callable[[list[str], str, str | None], pd.DataFrame] | None = None,
) -> pd.Series:
    """Single-series daily close for a benchmark index (default: NIFTY 500
    Total Return proxy on Yahoo Finance — verify this ticker is still valid
    before relying on it; index tickers do occasionally change). Fetches the
    index ticker directly (unlike fetch_price_panel, it does NOT append
    ``.NS`` — index tickers like ``^CRSLDX`` already have their own format).

    Thin wrapper around ``fetch_benchmark_ohlcv`` for callers that only need
    the close series (e.g. the backtest engine's returns panel) — use
    ``fetch_benchmark_ohlcv`` directly if you also want open/high/low/volume
    (e.g. for the regime-detection range/volume features).
    """
    ohlcv = fetch_benchmark_ohlcv(benchmark_ticker, start, end, downloader)
    if ohlcv.empty or "close" not in ohlcv.columns:
        return pd.Series(dtype=float, name="close")
    series = ohlcv["close"].rename("close")
    series.index.name = "date"
    return series


def fetch_india_vix_series(
    vix_ticker: str = "^INDIAVIX",
    start: str = "2015-01-01",
    end: str | None = None,
    downloader: Callable[[list[str], str, str | None], pd.DataFrame] | None = None,
) -> pd.Series:
    """Daily India VIX close level, as a standalone Series (name ``"vix"``).

    This is a core production input: the shipped default regime source
    (``regime_detection.production_regime_source: "vix_bucket_contemporaneous"``,
    see ``src/regime_detection/vix_regime.py`` and
    ``docs/regime_detection_spec.md``'s "VIX-bucket regime" section) is
    built directly from it. Kept separate from
    ``regime_detection.data_loader.load_from_yfinance`` (which bundles VIX
    together with the benchmark index download, for the GMM regime feature
    matrix specifically) so the two consumers can be refreshed
    independently.

    Uses the same dependency-injected ``downloader`` pattern as every other
    fetcher here, and the same ``_download_ticker_ohlcv``-shaped single-
    ticker handling as ``fetch_benchmark_ohlcv`` — reuses that function
    directly rather than duplicating the MultiIndex-column-shape handling.

    Returns an empty float Series (name ``"vix"``, never raises) if yfinance
    has no data for ``vix_ticker`` — callers decide whether that's fatal.
    India VIX has a shorter history than most benchmark indices (available
    on NSE since late 2007, but Yahoo Finance's backfill depth for
    ``^INDIAVIX`` specifically should be spot-checked against ``start``).
    """
    ohlcv = fetch_benchmark_ohlcv(vix_ticker, start, end, downloader)
    if ohlcv.empty or "close" not in ohlcv.columns:
        logger.warning(
            "yfinance returned no data for VIX ticker %s -- the production regime source "
            "(regime_detection.production_regime_source='vix_bucket_contemporaneous') cannot run "
            "without it. Verify the ticker is still valid on Yahoo Finance, or set "
            "production_regime_source: 'gmm' as a temporary fallback.",
            vix_ticker,
        )
        return pd.Series(dtype=float, name="vix")
    series = ohlcv["close"].rename("vix")
    series.index.name = "date"
    return series
