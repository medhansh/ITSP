"""Price/index data loading for regime detection.

Two paths are supported:
  1. ``load_from_csv`` — read a local CSV (date, close[, advances, declines, vix]).
     This is the path used in tests and in the sandbox this scaffold was built in,
     which has no outbound network access.
  2. ``load_from_yfinance`` — pull NIFTY 500 / India VIX history live. Requires
     network access and the ``yfinance`` package (both listed in requirements.txt).
     Run this from your own machine or a server with internet access.
"""
from __future__ import annotations

import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def load_from_csv(path: str) -> pd.DataFrame:
    """Load a local price/breadth/VIX CSV.

    Required: date (parseable), close. Optional, all independently:
    advances, declines, vix, open, high, low, volume — if open/high/low are
    present, range-based volatility (Parkinson; Garman-Klass if open is also
    present) is added to the regime feature matrix; if volume is present,
    volume z-score and OBV-trend features are added too (see
    ``features.build_feature_matrix``). Any other unknown extra columns are
    kept but ignored downstream.
    """
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "close" not in df.columns:
        raise ValueError(f"{path} must contain a 'close' column")
    return df


def _download_ticker_ohlcv(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download one ticker's OHLCV via yfinance and return it with clean,
    flat, lowercase columns (open/high/low/close/volume — whichever are
    present), regardless of which column shape this yfinance version hands
    back for a single-ticker download.

    Why this exists: yfinance has changed its default return shape across
    versions — sometimes flat columns (``Open``, ``Close``, ...), sometimes
    ``(field, ticker)`` MultiIndex columns even for a single ticker. Code
    that assumes one shape silently breaks on the other (e.g.
    ``df["Close"]`` on a MultiIndex-columned frame returns a DataFrame, not
    a Series, which is exactly what caused
    ``pd.DataFrame(frames)`` to raise "If using all scalar values, you must
    pass an index" when every sector ticker's "Close" collapsed to
    something ``pd.DataFrame`` couldn't treat as a column of values).
    Returns an empty DataFrame (never raises) if the download itself came
    back empty for this ticker — callers decide whether that's fatal.
    """
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # (field, ticker) or (ticker, field) — figure out which level has
        # the OHLCV field names and drop the other (single-ticker) level.
        field_names = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        if field_names & set(raw.columns.get_level_values(0)):
            raw = raw.xs(raw.columns.get_level_values(1)[0], axis=1, level=1)
        elif field_names & set(raw.columns.get_level_values(-1)):
            raw = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
        else:
            return pd.DataFrame()

    ohlcv = raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    ohlcv = ohlcv[[c for c in ("open", "high", "low", "close", "volume") if c in ohlcv.columns]]
    ohlcv.index.name = "date"
    return ohlcv


def load_from_yfinance(
    index_ticker: str = "^CRSLDX",  # Nifty500 Total Return proxy ticker on Yahoo Finance
    vix_ticker: str = "^INDIAVIX",
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Pull NIFTY 500-equivalent OHLCV and India VIX history via yfinance.

    NOTE: requires outbound internet access, which this scaffold's build
    environment did not have. Verify the ticker symbols are still valid on
    Yahoo Finance before relying on this in production — index tickers do
    occasionally change. Also verify ``index_ticker``'s volume field is
    actually populated for whatever ticker you use — some India index/
    total-return-proxy tickers on Yahoo report zero or missing volume, in
    which case ``features.build_feature_matrix`` will just skip the
    volume-derived features (graceful degradation, not an error) since
    they'll come back all-NaN and get dropped by ``dropna()``.
    """
    logger.info("Downloading %s from yfinance", index_ticker)
    index_df = _download_ticker_ohlcv(index_ticker, start, end)
    if index_df.empty:
        raise ValueError(
            f"yfinance returned no data for index ticker {index_ticker!r}. Verify the ticker "
            "is still valid on Yahoo Finance (index tickers do occasionally change)."
        )

    logger.info("Downloading %s from yfinance", vix_ticker)
    vix_raw = _download_ticker_ohlcv(vix_ticker, start, end)
    vix_df = vix_raw[["close"]].rename(columns={"close": "vix"}) if "close" in vix_raw.columns else pd.DataFrame()
    if vix_df.empty:
        logger.warning(
            "yfinance returned no data for VIX ticker %s — proceeding without VIX features "
            "(features.build_feature_matrix degrades gracefully without it).", vix_ticker,
        )

    merged = index_df.join(vix_df, how="left") if not vix_df.empty else index_df
    merged.index.name = "date"
    return merged


# NSE sector index tickers as listed on Yahoo Finance. VERIFY THESE before
# relying on them in production — Yahoo's NSE index ticker symbols have
# drifted before (see the ^CRSLDX note above) and are not guaranteed stable.
DEFAULT_SECTOR_TICKERS: dict[str, str] = {
    "IT": "^CNXIT",
    "BANK": "^NSEBANK",
    "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
    "REALTY": "^CNXREALTY",
    "FIN_SERVICE": "^CNXFIN",
}


def load_sector_prices_from_yfinance(
    sector_tickers: dict[str, str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Pull daily close levels for a set of sector indices — the multi-asset
    input the geometric wedge-product crash signal (geometric_signal.py)
    needs (it's undefined for a single series). Requires network access and
    ``yfinance``, same caveat as ``load_from_yfinance`` above.

    Tickers that fail to download (invalid/delisted/rate-limited) are
    skipped with a logged warning rather than crashing the whole batch —
    but at least 2 must succeed, since the wedge product needs multiple
    assets; raises ``ValueError`` with the list of failures if fewer than
    that come through.
    """
    tickers = sector_tickers or DEFAULT_SECTOR_TICKERS
    frames = {}
    failed = []
    for name, ticker in tickers.items():
        logger.info("Downloading sector index %s (%s) from yfinance", name, ticker)
        ohlcv = _download_ticker_ohlcv(ticker, start, end)
        if ohlcv.empty or "close" not in ohlcv.columns:
            logger.warning("yfinance returned no usable close data for sector %s (%s) — skipping", name, ticker)
            failed.append(name)
            continue
        frames[name] = ohlcv["close"]

    if len(frames) < 2:
        raise ValueError(
            f"Only {len(frames)}/{len(tickers)} sector tickers returned usable data "
            f"(failed: {failed}) — need >= 2 for the wedge-product signal to be "
            "computable. Verify the tickers in regime_detection.geometric_signal.sector_tickers "
            "are still valid on Yahoo Finance."
        )
    if failed:
        logger.warning("Proceeding with %d/%d sector tickers (failed: %s)", len(frames), len(tickers), failed)

    merged = pd.DataFrame(frames)
    merged.index.name = "date"
    return merged


def load_sector_prices_from_csv(path: str) -> pd.DataFrame:
    """Load a local sector-price CSV: a ``date`` column plus one column per
    sector. This is the path used in tests/sandbox (no network access) —
    mirrors ``load_from_csv``'s role for the single-index case.
    """
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if df.shape[1] < 2:
        raise ValueError(
            f"{path} must have >= 2 non-date columns (one per sector) for the "
            "geometric wedge-product signal to be computable."
        )
    return df
