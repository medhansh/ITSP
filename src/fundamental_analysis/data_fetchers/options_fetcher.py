"""Options-chain data for the pre-earnings implied-volatility / skew signal
(``metrics/options_earnings.py``).

Data-availability reality check (read before wiring this into a live run):
NSE F&O (futures & options) data for individual stocks is NOT available from
yfinance for NSE-listed names (yfinance's options chain support is
US-market/OCC-symbol-only) and Screener/Trendlyne don't carry options data at
all. A real deployment needs one of:
  - NSE's own historical options-chain archives (bhavcopy / option chain API),
    which change format periodically and aren't a stable free bulk source;
  - a paid options-data vendor (e.g. via a broker API).

Because of that — and because this build sandbox has no network access at
all (same limitation documented throughout this project) — this module
follows the exact pattern already used by ``yfinance_fetcher.py``: every
network-calling function accepts an optional dependency-injected callable, so
the actual signal-computation logic (``metrics/options_earnings.py``) can be
fully unit-tested against a fixture chain without a working scraper. Treat
``fetch_option_chain_snapshot``'s default implementation as a documented stub
that raises NotImplementedError — wire in a real NSE/vendor client before
using this for anything beyond fixture-driven tests.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# Long-format schema every option-chain snapshot (real or fixture) must have.
OPTION_CHAIN_SCHEMA = [
    "symbol", "date", "expiry", "strike", "option_type",  # option_type in {"CE", "PE"}
    "close", "open_interest", "implied_volatility", "underlying_close",
]


def _default_option_chain_fetch(symbol: str, date: str) -> pd.DataFrame:
    raise NotImplementedError(
        "No live options-data source is wired in (see module docstring). "
        "Pass `fetch_fn` explicitly (e.g. a fixture-backed function for tests, "
        "or a real NSE/vendor client for production) rather than relying on "
        "the default."
    )


def fetch_option_chain_snapshot(
    symbol: str,
    date: str,
    fetch_fn: Callable[[str, str], pd.DataFrame] = _default_option_chain_fetch,
) -> pd.DataFrame:
    """Fetch one symbol's full option chain as observed on ``date``.

    Returns a DataFrame shaped like ``OPTION_CHAIN_SCHEMA``. ``fetch_fn`` is
    dependency-injected specifically so tests (and, later, a real data
    source) can supply chain data without this module needing to know where
    it came from.
    """
    try:
        chain = fetch_fn(symbol, date)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Options chain fetch failed for %s on %s: %s", symbol, date, exc)
        return pd.DataFrame(columns=OPTION_CHAIN_SCHEMA)

    missing = set(OPTION_CHAIN_SCHEMA) - set(chain.columns)
    if missing:
        raise ValueError(f"fetch_fn returned a chain missing columns: {missing}")
    return chain


def compute_atm_iv(chain_snapshot: pd.DataFrame) -> float:
    """At-the-money implied volatility: average IV of the CE/PE pair whose
    strike is closest to the underlying close, on a single chain snapshot
    (one symbol, one date, both expiries/strikes present).
    """
    if chain_snapshot.empty:
        return np.nan
    underlying = chain_snapshot["underlying_close"].iloc[0]
    nearest_strike = chain_snapshot.loc[(chain_snapshot["strike"] - underlying).abs().idxmin(), "strike"]
    atm = chain_snapshot[chain_snapshot["strike"] == nearest_strike]
    return atm["implied_volatility"].mean()


def compute_put_call_oi_ratio(chain_snapshot: pd.DataFrame) -> float:
    """Total put open interest / total call open interest for a chain
    snapshot. > 1 = more open-interest sitting on the put side (a commonly
    cited, though noisy, bearish-tilt positioning indicator); < 1 = call-side
    heavy (bullish tilt)."""
    if chain_snapshot.empty:
        return np.nan
    put_oi = chain_snapshot.loc[chain_snapshot["option_type"] == "PE", "open_interest"].sum()
    call_oi = chain_snapshot.loc[chain_snapshot["option_type"] == "CE", "open_interest"].sum()
    if call_oi == 0:
        return np.nan
    return put_oi / call_oi


def compute_atm_straddle_implied_move_pct(chain_snapshot: pd.DataFrame) -> float:
    """ATM straddle price (nearest CE + nearest PE at the same strike) as a
    % of the underlying — the options market's implied magnitude of move,
    informational only (see options_earnings.py for why it's not scored
    directionally)."""
    if chain_snapshot.empty:
        return np.nan
    underlying = chain_snapshot["underlying_close"].iloc[0]
    nearest_strike = chain_snapshot.loc[(chain_snapshot["strike"] - underlying).abs().idxmin(), "strike"]
    atm = chain_snapshot[chain_snapshot["strike"] == nearest_strike]
    call_price = atm.loc[atm["option_type"] == "CE", "close"].mean()
    put_price = atm.loc[atm["option_type"] == "PE", "close"].mean()
    if pd.isna(call_price) or pd.isna(put_price) or underlying == 0:
        return np.nan
    return (call_price + put_price) / underlying
