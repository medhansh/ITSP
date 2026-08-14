"""vectorbt-based portfolio simulation — replaces the hand-rolled loop in
``engine.py`` as the primary backtest engine (per explicit request), while
keeping the exact same output contract (``run_backtest_vbt`` returns the
same ``{"returns", "gross_returns", "turnover", "equity_curve"}`` dict shape
as ``engine.run_backtest``) so ``attribution.py``/``metrics.py``/
``reporting.py`` don't need to know which engine produced it.

Why vectorbt instead of the custom loop: it's a vectorized (numba-jitted)
backtesting library that handles order execution, cash accounting, and
per-asset position sizing from a target-weight matrix directly, which is
both faster on a full NIFTY500-sized universe and exercises a
well-tested/widely-used execution model instead of this project's own
one-off simulation logic — useful as a cross-check that engine.py's simpler
model wasn't quietly wrong somewhere.

**Environment note**: this build sandbox has no PyPI access (same limitation
documented throughout this project for hmmlearn/yfinance/pytest), so
``vectorbt`` could not actually be installed or run here — the import is
guarded and ``run_backtest_vbt`` raises a clear ``ImportError`` if it's
missing, exactly like ``models.py``'s ``hmmlearn`` branch. ``engine.py``'s
custom implementation remains in the codebase specifically as the tested,
dependency-free fallback (``attribution.py`` auto-falls-back to it with a
logged warning if vectorbt isn't installed) — verify ``run_backtest_vbt``
against ``engine.run_backtest`` on a small synthetic panel (they should
produce near-identical net returns up to floating-point/fee-timing
differences) before trusting the vectorbt path in production; see
tests/test_backtesting.py for the comparison this project shipped with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _require_vectorbt():
    try:
        import vectorbt as vbt
    except ImportError as e:  # pragma: no cover — not installable in this sandbox
        raise ImportError(
            "vectorbt is required for the 'vectorbt' backtest engine. "
            "`pip install vectorbt`, or set backtesting.engine: 'custom' in "
            "configs/config.yaml to use the dependency-free fallback in engine.py."
        ) from e
    return vbt


def run_backtest_vbt(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    freq: str = "D",
    init_cash: float = 1.0,
    lag_days: int = 1,
) -> dict[str, pd.Series]:
    """Simulate a long-only, no-leverage portfolio from a daily target-weight
    matrix using ``vectorbt.Portfolio.from_orders`` with
    ``size_type="targetpercent"`` (rebalance-to-target semantics — matches
    ``engine.run_backtest``'s "hold weight until it changes" behavior).

    Args:
        prices: daily close prices, index=date, columns=symbols — REQUIRED
            (unlike engine.run_backtest, which works off returns directly;
            vectorbt's order-based engine needs actual price levels to size
            orders and mark positions to market).
        weights: target weights, same shape/columns as prices (already
            daily-aligned via ``engine.align_weights_to_returns`` — call it
            first if your weights are sparse/rebalance-date-only).
        transaction_cost_bps: one-way cost in basis points, passed to
            vectorbt as a fractional ``fees`` rate.
        freq: pandas frequency string for annualization inside vectorbt's own
            stats (this module recomputes everything through metrics.py
            instead, so this mostly just needs to be a valid alias).
        init_cash: starting portfolio value — 1.0 so ``equity_curve`` reads
            the same way engine.run_backtest's does (growth of 1 unit).
        lag_days: same meaning and same default (1) as
            ``engine.run_backtest``'s ``lag_days`` — see that function's
            docstring for the full explanation and the same-bar look-ahead
            bug this prevents (a signal computed using day T's close cannot
            be acted on until day T+1). Applied here by shifting ``weights``
            by ``lag_days`` BEFORE handing them to vectorbt, rather than
            relying on any particular assumption about vectorbt's own
            same-bar-vs-next-bar order-timing convention — shifting our own
            input is the portable way to guarantee this regardless of
            library internals we can't fully verify without vectorbt
            installed (see module docstring's environment note).

    Returns:
        Same dict shape as ``engine.run_backtest``: returns, gross_returns,
        turnover, equity_curve. ``gross_returns`` is derived by re-running
        the same simulation with zero fees (vectorbt doesn't expose a
        separate "as-if-no-fees" series directly), which is exact for this
        target-percent order model since fees don't change *what* gets
        held, only the cash drag.
    """
    vbt = _require_vectorbt()

    common_cols = prices.columns.intersection(weights.columns)
    p = prices[common_cols].sort_index()
    w = weights[common_cols].reindex(p.index).fillna(0.0)
    if lag_days > 0:
        w = w.shift(lag_days).fillna(0.0)
    fees = transaction_cost_bps / 10_000.0

    portfolio = vbt.Portfolio.from_orders(
        close=p,
        size=w,
        size_type="targetpercent",
        fees=fees,
        cash_sharing=True,
        group_by=True,
        freq=freq,
        init_cash=init_cash,
    )
    net_value = portfolio.value()
    net_returns = net_value.pct_change(fill_method=None).fillna(0.0)
    if isinstance(net_returns, pd.DataFrame):
        net_returns = net_returns.iloc[:, 0]

    if fees > 0:
        portfolio_gross = vbt.Portfolio.from_orders(
            close=p, size=w, size_type="targetpercent",
            fees=0.0, cash_sharing=True, group_by=True, freq=freq, init_cash=init_cash,
        )
        gross_value = portfolio_gross.value()
        gross_returns = gross_value.pct_change(fill_method=None).fillna(0.0)
        if isinstance(gross_returns, pd.DataFrame):
            gross_returns = gross_returns.iloc[:, 0]
    else:
        gross_returns = net_returns

    turnover = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())
    equity_curve = (1.0 + net_returns).cumprod()

    return {
        "returns": net_returns,
        "gross_returns": gross_returns,
        "turnover": turnover,
        "equity_curve": equity_curve,
    }


def run_backtest_with_fallback(
    prices: pd.DataFrame,
    returns_panel: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    engine: str = "vectorbt",
    lag_days: int = 1,
) -> dict[str, pd.Series]:
    """Dispatch to vectorbt if ``engine == "vectorbt"`` and it's installed;
    otherwise (or on any import failure) fall back to ``engine.run_backtest``
    with a logged warning, so the pipeline never hard-fails just because
    vectorbt isn't available in a given environment. ``lag_days`` (default
    1) is forwarded to whichever engine actually runs — see
    ``engine.run_backtest``'s docstring for what this prevents.
    """
    from src.backtesting.engine import run_backtest as run_backtest_custom

    if engine == "vectorbt":
        try:
            return run_backtest_vbt(prices, weights, transaction_cost_bps, lag_days=lag_days)
        except ImportError as exc:
            logger.warning(
                "vectorbt unavailable (%s) — falling back to the custom engine.py "
                "implementation. Results should be numerically close but are not "
                "guaranteed identical; see vbt_engine.py docstring.", exc,
            )
    return run_backtest_custom(returns_panel, weights, transaction_cost_bps, lag_days=lag_days)
