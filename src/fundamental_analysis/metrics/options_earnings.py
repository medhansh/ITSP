"""Pre-earnings options-market signal: implied volatility and put/call
open-interest skew in the run-up to a company's earnings report.

Rationale: elevated pre-earnings IV (relative to the stock's own trailing
history) prices in expected uncertainty/large-move risk; put/call OI skew is
a commonly-cited (noisy) positioning-tilt indicator. Neither is a
"fundamental" in the balance-sheet sense, but both are forward-looking,
market-implied risk signals that are naturally concentrated right around
earnings — which is why this is scored as its own composite dimension
(``options_earnings``) rather than folded into ``earnings_surprise.py``.

NO-LOOKAHEAD DESIGN — read before changing this file
------------------------------------------------------
This is the part of the request that most directly risks look-ahead bias if
implemented carelessly, so the rules are explicit:

1. Only earnings dates with ``earnings_date <= as_of_date`` are ever used as
   the anchor. We are always looking at "the options market's behavior
   heading into the *most recent already-reported* earnings", never a future
   or not-yet-public earnings date.
2. The pre-earnings options window is strictly *before* that anchor date
   (``earnings_date - pre_earnings_window_days`` trading days, up to
   ``earnings_date - 1``) — so even though the anchor itself is safely in
   the past relative to ``as_of_date``, we still never touch options data
   dated on/after the earnings date itself (post-earnings IV crush would
   otherwise contaminate the pre-earnings signal even without being a
   temporal lookahead bug).
3. The IV percentile normalization only ranks against a symbol's own
   trailing history up to ``as_of_date`` — never against later dates.
4. Imputation on missing data widens the pre-earnings window search
   (up to ``max_lookback_days``) rather than ever reaching forward past the
   earnings date or past ``as_of_date``. If nothing is found even after
   widening, the result is NaN — never a fabricated/interpolated-from-later
   value. This mirrors the "no silent fabrication" convention already used
   by screener_fetcher.py / trendlyne_fetcher.py elsewhere in this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

OUTPUT_COLUMNS = ["pre_earnings_iv_percentile", "pre_earnings_put_call_oi_ratio", "implied_move_pct"]


def _most_recent_known_earnings_date(
    earnings_calendar: pd.DataFrame, symbol: str, as_of_date: pd.Timestamp
) -> pd.Timestamp | None:
    rows = earnings_calendar[
        (earnings_calendar["symbol"] == symbol) & (earnings_calendar["earnings_date"] <= as_of_date)
    ]
    if rows.empty:
        return None
    return rows["earnings_date"].max()


def _pre_earnings_window_mean(
    option_summary_history: pd.DataFrame,
    symbol: str,
    earnings_date: pd.Timestamp,
    window_days: int,
    max_lookback_days: int,
    value_col: str,
) -> float:
    """Mean of ``value_col`` in [earnings_date - window_days, earnings_date).
    If that's empty, widen up to ``max_lookback_days`` (still strictly before
    earnings_date) before giving up and returning NaN.
    """
    symbol_hist = option_summary_history[option_summary_history["symbol"] == symbol]
    for lookback in sorted({window_days, max_lookback_days}):
        start = earnings_date - pd.Timedelta(days=lookback)
        window = symbol_hist[(symbol_hist["date"] >= start) & (symbol_hist["date"] < earnings_date)]
        if not window.empty and window[value_col].notna().any():
            return window[value_col].mean(skipna=True)
    return np.nan


def compute_options_earnings_metrics(
    snapshot: pd.DataFrame,
    option_summary_history: pd.DataFrame,
    earnings_calendar: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    pre_earnings_window_days: int = 5,
    max_lookback_days: int = 10,
    iv_percentile_lookback_days: int = 252,
) -> pd.DataFrame:
    """Compute the pre-earnings options metrics for every symbol in
    ``snapshot``'s index, as of ``as_of_date``.

    Args:
        snapshot: any SNAPSHOT_SCHEMA-shaped DataFrame indexed by symbol —
            only its index is used (defines which symbols to compute for).
        option_summary_history: long format, columns
            [symbol, date, atm_iv, put_call_oi_ratio, implied_move_pct] —
            one row per (symbol, trading date) that options data exists for.
            See ``data_fetchers/options_fetcher.py`` for how a single date's
            snapshot is summarized into one row of this shape.
        earnings_calendar: long format, columns [symbol, earnings_date] —
            only *already-occurred* dates should be relied on (see module
            docstring point 1); this function filters to
            ``earnings_date <= as_of_date`` itself as a defensive guard even
            if the caller passes a calendar that includes future dates.
        as_of_date: the point-in-time date this is being computed for
            (typically a backtest rebalance date).

    Returns:
        DataFrame indexed like ``snapshot``, columns:
          pre_earnings_iv_percentile — ATM IV in the pre-earnings window,
            expressed as a percentile of the symbol's own trailing
            ``iv_percentile_lookback_days`` daily ATM IV (0-1). Comparable
            across symbols/vol regimes, unlike a raw IV level.
          pre_earnings_put_call_oi_ratio — mean put/call OI ratio in the
            pre-earnings window.
          implied_move_pct — mean ATM-straddle-implied move in the
            pre-earnings window. Informational only: it's a magnitude-of-
            uncertainty measure, not a "higher/lower is better" quality
            signal, so it is intentionally NOT in composite_score.py's
            METRIC_DIRECTION / DIMENSION_METRICS — it isn't sign-scored, just
            carried through for reporting/inspection.
    """
    as_of = pd.Timestamp(as_of_date)
    result = pd.DataFrame(index=snapshot.index, columns=OUTPUT_COLUMNS, dtype=float)

    if option_summary_history.empty or earnings_calendar.empty:
        logger.info("No options/earnings-calendar data supplied — options_earnings dimension will be all-NaN")
        return result

    calendar = earnings_calendar[earnings_calendar["earnings_date"] <= as_of]

    for symbol in snapshot.index:
        earnings_date = _most_recent_known_earnings_date(calendar, symbol, as_of)
        if earnings_date is None:
            continue  # no known-past earnings yet for this symbol as of as_of_date

        iv = _pre_earnings_window_mean(
            option_summary_history, symbol, earnings_date,
            pre_earnings_window_days, max_lookback_days, "atm_iv",
        )
        pc_ratio = _pre_earnings_window_mean(
            option_summary_history, symbol, earnings_date,
            pre_earnings_window_days, max_lookback_days, "put_call_oi_ratio",
        )
        implied_move = _pre_earnings_window_mean(
            option_summary_history, symbol, earnings_date,
            pre_earnings_window_days, max_lookback_days, "implied_move_pct",
        )

        if pd.notna(iv):
            trailing_start = as_of - pd.Timedelta(days=iv_percentile_lookback_days)
            trailing = option_summary_history[
                (option_summary_history["symbol"] == symbol)
                & (option_summary_history["date"] >= trailing_start)
                & (option_summary_history["date"] <= as_of)
            ]["atm_iv"].dropna()
            iv_percentile = (trailing < iv).mean() if len(trailing) >= 5 else np.nan
        else:
            iv_percentile = np.nan

        result.loc[symbol, "pre_earnings_iv_percentile"] = iv_percentile
        result.loc[symbol, "pre_earnings_put_call_oi_ratio"] = pc_ratio
        result.loc[symbol, "implied_move_pct"] = implied_move

    return result


def compute_options_earnings_dimension(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Thin adapter matching the ``fn(snapshot) -> DataFrame`` signature every
    other dimension in ``fundamental_analysis/pipeline.py``'s
    ``DIMENSION_COMPUTERS`` uses, unlike ``compute_options_earnings_metrics``
    above (which needs the extra options-history/earnings-calendar/as-of-date
    arguments and so is called *before* ``pipeline.run_pipeline``, with its
    output columns merged onto ``snapshot`` — see
    ``point_in_time.run_pit_fundamental_pipeline`` /
    ``scripts/run_full_pipeline.py`` for where that merge happens).

    This adapter just validates the expected columns are present on
    ``snapshot`` (already merged in) and passes them through unchanged; if
    they weren't merged in upstream, it returns all-NaN rather than raising,
    consistent with every other dimension's graceful-degradation convention.
    """
    out = pd.DataFrame(index=snapshot.index)
    for col in OUTPUT_COLUMNS:
        out[col] = snapshot[col] if col in snapshot.columns else np.nan
    return out
