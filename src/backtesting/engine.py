"""Minimal long-only portfolio simulation engine.

Deliberately simple and dependency-free (no external backtesting library) so
the mechanics are fully auditable: given a daily target-weight matrix and a
daily returns panel, it forward-fills weights between rebalance dates,
computes daily portfolio return as the weight-dot-return, and charges
transaction costs proportional to turnover on rebalance days. This is a
long-only, no-leverage engine — weights are expected to sum to <= 1 per row
(residual is uninvested cash, which earns 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def compute_returns_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily % returns for every column (symbol) in a price panel."""
    return prices.sort_index().pct_change(fill_method=None)


def align_weights_to_returns(
    weights: pd.DataFrame, returns_index: pd.DatetimeIndex, returns_columns: pd.Index
) -> pd.DataFrame:
    """Reindex a (typically sparse, rebalance-date-only) weights DataFrame onto
    the full daily returns index/columns, forward-filling between rebalances.
    Missing symbols get 0 weight; missing dates before the first rebalance also
    get 0 weight (uninvested) rather than NaN.
    """
    w = weights.reindex(columns=returns_columns, fill_value=0.0)
    w = w.reindex(returns_index).ffill().fillna(0.0)
    return w


def run_backtest(
    returns_panel: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    lag_days: int = 1,
) -> dict[str, pd.Series]:
    """Simulate a long-only portfolio.

    Args:
        returns_panel: daily simple returns, index=date, columns=symbols.
        weights: target weights, same shape/columns as returns_panel (already
            aligned via ``align_weights_to_returns`` — call it first if your
            weights are sparse/rebalance-date-only).
        transaction_cost_bps: one-way cost in basis points charged on turnover
            (sum of absolute weight changes) each day the EFFECTIVE (lagged)
            weight changes.
        lag_days: trading days between when a weight is "decided" and when it
            starts earning returns. **Default 1 — do not set to 0 without a
            specific reason (see below).** Every weight-builder in this
            project (technical_signals, adaptive_ichimoku, the regime label,
            the fundamentals composite score) is computed using data up to
            and including a given day's close — e.g. an SMA at day T
            necessarily uses T's own closing price. Applying that weight
            directly to day T's own realized return (``r[T] = close[T]/close[T-1]-1``)
            would mean the backtest "knew" T's close before T's return had
            even happened — impossible to replicate live, since you can't
            observe today's close and be already positioned for today's
            return. This was a real bug found via this exact scenario: a
            single-day +20% price spike, with a signal that only turns
            positive *because* of that spike, was getting 100% of that
            spike's return attributed to it by the pre-fix engine. With the
            default ``lag_days=1``, a weight decided using information
            through day T only starts earning returns from day T+1 onward —
            ``portfolio_return[T] = weights[T-1] * returns[T]``, the standard
            no-look-ahead backtest convention. Set 0 only if you have
            already lagged your weights yourself upstream and want this
            function to use them exactly as given (e.g. low-level unit
            tests of the dot-product mechanics) — see
            ``tests/test_backtesting.py``'s regression test reproducing the
            spike scenario for what goes wrong otherwise.

    Returns:
        dict with:
          "returns"   -- net daily portfolio return series (after costs)
          "gross_returns" -- daily portfolio return before costs
          "turnover"  -- daily sum(|weight change|), computed on the
                         EFFECTIVE (lagged) weight series, so it reflects
                         the day the position actually changed, not the day
                         the underlying signal was computed.
          "equity_curve" -- cumulative growth of 1 unit, net of costs
    """
    common_cols = returns_panel.columns.intersection(weights.columns)
    r = returns_panel[common_cols].fillna(0.0)
    w = weights[common_cols].fillna(0.0)
    if lag_days > 0:
        w = w.shift(lag_days).fillna(0.0)

    gross_returns = (w * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = gross_returns - cost

    equity_curve = (1.0 + net_returns).cumprod()

    return {
        "returns": net_returns,
        "gross_returns": gross_returns,
        "turnover": turnover,
        "equity_curve": equity_curve,
    }
