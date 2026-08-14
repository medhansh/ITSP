"""Cross-sectional beta-orthogonalization for the ``technical_momentum``
(Ichimoku conviction) composite-scoring dimension.

**Motivation**: walk-forward validation found the Ichimoku conviction
score's out-of-sample edge correlates with benchmark trend strength
(rho ~ 0.66, see docs/fundamental_analysis_spec.md's technical_momentum
section) but not perfectly -- at least one flat-market fold still showed
real edge. That split points at two things being summed into one score:
systematic market-beta momentum (rides the tape) and idiosyncratic,
stock-specific conviction. This module isolates the second part.

**Why beta only, not sector too**: a natural read of "orthogonalize
against systematic effects" would also residualize against sector
membership. That would be redundant here -- ``scoring/composite_score.py``
already sector-z-scores every dimension (including ``ichimoku_conviction``)
via ``sector_relative_zscore`` before dimension averaging, specifically so
sector-level effects don't leak into the composite score. Residualizing
against sector a second time upstream of that step would just partially
undo/duplicate work already being done downstream. Market beta, by
contrast, is NOT handled anywhere else in this pipeline -- this module
closes that specific, actually-open gap, and nothing more.

**Shape**: one cross-sectional (same-date, across-symbol) OLS regression
per rebalance date -- ``conviction ~ 1 + beta`` -- residual standardized to
zero mean / unit variance. This is a single-point-in-time regression, not
a time series, exactly the same per-date shape as
``sector_relative_zscore``.

**Status: experimental, unvalidated on real data as of writing** -- same
caveat as every other new signal-processing step in this project. Compare
walk-forward results with and without this enabled
(``scripts/walk_forward_technical_momentum.py``) before trusting it; see
``configs/config.yaml``'s ``technical_momentum_beta_orthogonalization``
block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def compute_rolling_beta_panel(
    price_panel_close: dict[str, pd.Series] | pd.DataFrame,
    benchmark_close: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Trailing rolling beta of every symbol vs. ``benchmark_close``, one
    column per symbol, indexed by date.

        beta_t = Cov(r_i, r_m) / Var(r_m)

    computed over the trailing ``window`` trading days ENDING at t
    (inclusive) -- purely backward-looking, same PIT convention as every
    other rolling feature in this project (e.g.
    ``regime_detection.features.compute_realized_vol``). Reading this
    panel at rebalance date T only ever uses price data <= T, by
    construction -- no separate PIT-safety argument needed beyond that.

    ``price_panel_close``: either a dict of ``{symbol: close Series}``
    (e.g. ``{s: ohlc["close"] for s, ohlc in price_panel_ohlc.items()}``,
    reusing the same OHLC panel ``adaptive_ichimoku`` already consumes) or
    an already-wide DataFrame (date index, symbol columns).

    ``min_periods``: minimum non-NaN daily-return observations required
    within the window before a beta is emitted; defaults to half the
    window so a handful of missing days (holidays, short gaps) don't blank
    out an otherwise-computable beta. Symbols/dates without enough history
    are NaN, not zero and not a crash -- same "missing input degrades
    gracefully" convention used throughout this codebase.
    """
    if isinstance(price_panel_close, dict):
        close_wide = pd.DataFrame({s: c for s, c in price_panel_close.items()})
    else:
        close_wide = price_panel_close

    min_periods = min_periods if min_periods is not None else max(window // 2, 2)

    stock_returns = np.log(close_wide).diff()
    bench_returns = np.log(benchmark_close).diff().reindex(stock_returns.index)
    bench_var = bench_returns.rolling(window, min_periods=min_periods).var()

    betas = {}
    for symbol in stock_returns.columns:
        cov = stock_returns[symbol].rolling(window, min_periods=min_periods).cov(bench_returns)
        betas[symbol] = cov / bench_var.replace(0, np.nan)
    return pd.DataFrame(betas, index=stock_returns.index)


def residualize_against_beta(
    scores: pd.Series,
    beta: pd.Series,
    min_obs: int = 10,
) -> pd.Series:
    """Cross-sectional OLS residual of ``scores`` (one rebalance date's
    ``ichimoku_conviction`` values, symbol -> value) against ``beta``
    (same symbols -> trailing beta, e.g. one row of
    ``compute_rolling_beta_panel``), standardized to zero mean / unit
    variance:

        scores_i = a + b * beta_i + resid_i
        Z_i = (resid_i - mean(resid)) / std(resid)

    Symbols missing from ``beta`` (insufficient price history for the beta
    window, a recent IPO, a fetch gap) keep their ORIGINAL, un-
    residualized score rather than being dropped -- same "degrade, don't
    discard" convention as the rest of the fundamentals pipeline. The
    caller (``point_in_time.run_pit_fundamental_pipeline``) logs an
    aggregate count of how many symbols fell back across the full run.

    If fewer than ``min_obs`` symbols have both a score and a beta that
    date, or beta has ~zero cross-sectional variance (can't identify a
    slope -- e.g. very early in the backtest window before enough price
    history has accumulated for any symbol), the entire input is returned
    UNCHANGED for that date and a warning is logged. Silently returning
    the raw signal is safer than either crashing or residualizing against
    noise from a near-degenerate regression.
    """
    aligned = pd.DataFrame({"score": scores, "beta": beta}).dropna()
    if len(aligned) < min_obs:
        logger.warning(
            "residualize_against_beta: only %d symbols have both a score and a beta "
            "(need >= %d) -- returning scores unchanged for this date.",
            len(aligned), min_obs,
        )
        return scores

    beta_var = aligned["beta"].var()
    if not np.isfinite(beta_var) or beta_var < 1e-12:
        logger.warning(
            "residualize_against_beta: beta has ~zero cross-sectional variance this date "
            "(%d symbols) -- can't identify a slope, returning scores unchanged.",
            len(aligned),
        )
        return scores

    x = aligned["beta"].to_numpy()
    y = aligned["score"].to_numpy()
    x_mean, y_mean = x.mean(), y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    resid = y - (intercept + slope * x)

    resid_std = resid.std()
    if resid_std < 1e-12:
        # Degenerate (e.g. every score effectively identical) -- residual
        # is already ~0 everywhere; standardizing would just divide by ~0.
        z_values = np.zeros_like(resid)
    else:
        z_values = (resid - resid.mean()) / resid_std
    z = pd.Series(z_values, index=aligned.index)

    out = scores.copy().astype(float)
    out.loc[z.index] = z

    n_scored = int(scores.notna().sum())
    n_unadjusted = n_scored - len(z)
    if n_unadjusted > 0:
        logger.info(
            "residualize_against_beta: %d/%d symbols had no usable beta this date and kept "
            "their original (un-residualized) score.", n_unadjusted, n_scored,
        )
    return out
