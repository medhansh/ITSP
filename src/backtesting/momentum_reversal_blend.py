# Destination: src/backtesting/momentum_reversal_blend.py  (new file)
"""Regime-dependent blending of MOMENTUM and MEAN-REVERSION signals.

**Why this exists.** ``scripts/diagnose_signal_ic_by_regime.py`` measured
the cross-sectional information coefficient of each signal broken out by
regime, and found a clean sign flip on real data:

    signal                 calm(r0)   elevated(r2)   stress(r3)
    momentum_ichimoku       +0.017       -0.023        -0.125
    momentum_12_1           +0.020       +0.019        -0.095
    reversal_short_term     +0.002       +0.044        +0.145
    reversal_dist_from_ma   +0.009       +0.040        +0.107

Momentum ranks stocks BACKWARDS under stress while reversal ranks them
correctly. That is the premise of regime-dependent switching, and it is
the first time on this project that a proposed mechanism has been
supported by a direct signal measurement before being built.

**Sample-size caveat, and why it drives the design.** The stressed regime
holds only 4 rebalance dates, so its large t-statistics are four dates
agreeing, not evidence. The pattern is however corroborated in the
elevated regime on 25-31 dates (reversal_dist_from_ma +0.040, t=2.12,
positive on 72% of dates), which is what makes the crossover credible.

This is exactly why both a RIGID and a CONTINUOUS blend are provided. The
rigid version puts decisive weight on the thin stressed bucket; the
continuous version reads stress as a scalar and uses every date. Given the
project's separate finding that the feature space is a volatility
CONTINUUM rather than separable states (DBSCAN finds one cluster at every
eps; silhouette prefers n=2 everywhere), the continuous form is also the
better-motivated one. Both are built so the comparison is empirical rather
than assumed.

Blends produce a single panel in the same shape as the Ichimoku conviction
panel, so they drop straight into ``point_in_time.run_pit_fundamental_pipeline``'s
``conviction_panel`` argument with no change to scoring or backtesting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _cross_sectional_z(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (date) across symbols.

    Essential before blending: momentum and reversal signals live on
    completely different scales (Ichimoku conviction is bounded in [-1, 1]
    by a tanh, a log-return difference is not), so blending raw values
    would silently weight one leg far more heavily than the stated blend
    weight implies. This is the same fix that turned
    ``apply_ichimoku_conviction_tilt`` from a no-op into a working
    mechanism.
    """
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def build_momentum_panel(stock_prices: pd.DataFrame, kind: str = "12_1") -> pd.DataFrame:
    """Momentum signal panel. ``kind='12_1'`` is trailing 12-month return
    skipping the most recent month (the standard construction, which skips
    the short-term reversal window that would otherwise contaminate it)."""
    log_px = np.log(stock_prices)
    if kind == "12_1":
        return log_px.shift(21) - log_px.shift(252)
    raise ValueError(f"unknown momentum kind: {kind!r}")


def build_reversal_panel(stock_prices: pd.DataFrame, kind: str = "dist_from_ma",
                         ma_window: int = 63, lookback: int = 252,
                         reversal_window: int = 21) -> pd.DataFrame:
    """Mean-reversion signal panel, signed so that HIGHER = more expected
    bounce (i.e. already negated), matching momentum's convention that
    higher is better.

    ``dist_from_ma`` (default): negative z-score of price relative to its
    own ``ma_window`` moving average. Preferred over ``short_term`` because
    it had the better sample support in the IC diagnostic and because it
    does not share its lookback with the monthly rebalance interval.

    ``short_term``: negated trailing ``reversal_window`` return.
    """
    if kind == "short_term":
        log_px = np.log(stock_prices)
        return -(log_px - log_px.shift(reversal_window))
    if kind == "dist_from_ma":
        ma = stock_prices.rolling(ma_window).mean()
        dist = (stock_prices - ma) / ma
        z = (dist - dist.rolling(lookback).mean()) / dist.rolling(lookback).std()
        return -z
    raise ValueError(f"unknown reversal kind: {kind!r}")


def build_blend(
    momentum: pd.DataFrame,
    reversal: pd.DataFrame,
    stress: pd.Series,
    mode: str = "continuous",
    max_reversal_weight: float = 1.0,
) -> pd.DataFrame:
    """Blend momentum and reversal into one conviction panel.

    ``stress``: per-date stress in [0, 1]; 0 = calmest, 1 = most stressed.
    For the rigid mode this is expected to be the regime label mapped onto
    a 0..1 ladder; for the continuous mode it can be any smooth measure.

    Blend weight, in both modes:

        w_rev(t) = max_reversal_weight * stress(t)
        blend(t) = (1 - w_rev(t)) * z(momentum) + w_rev(t) * z(reversal)

    At ``max_reversal_weight=1.0`` and full stress the blend is PURE
    reversal, which is what the measured sign flip implies. Lower values
    tilt without ever fully inverting -- worth testing, since the full flip
    rests on the thin stressed bucket.

    ``mode`` does not change this formula. It exists to make explicit that
    the difference between the rigid and continuous arms is entirely in how
    ``stress`` was CONSTRUCTED (a step function over discrete regime labels
    vs a smooth transform of realized volatility), not in the blending
    itself. Keeping the blend identical is what makes the two arms a fair
    comparison of the stress representation rather than of two different
    mechanisms.
    """
    if mode not in ("rigid", "continuous"):
        raise ValueError(f"mode must be 'rigid' or 'continuous', got {mode!r}")

    common_idx = momentum.index.intersection(reversal.index)
    common_cols = momentum.columns.intersection(reversal.columns)
    mom = _cross_sectional_z(momentum.loc[common_idx, common_cols])
    rev = _cross_sectional_z(reversal.loc[common_idx, common_cols])

    s = stress.reindex(common_idx).ffill()
    if s.isna().all():
        raise ValueError("stress series has no overlap with the signal panels.")
    s = s.fillna(0.0).clip(0.0, 1.0)

    w_rev = (s * max_reversal_weight).clip(0.0, 1.0)
    blended = mom.mul(1.0 - w_rev, axis=0).add(rev.mul(w_rev, axis=0), fill_value=0.0)

    logger.info(
        "build_momentum_reversal_blend(mode=%s): reversal weight ranges %.2f to %.2f "
        "(mean %.2f) across %d dates.",
        mode, float(w_rev.min()), float(w_rev.max()), float(w_rev.mean()), len(w_rev),
    )
    return blended


def _ladder_from_labels(labels: pd.Series, n_labels: int | None = None) -> dict:
    """Shared helper: evenly-spaced 0..1 ladder from a set of ordinal
    integer labels (0 = calmest .. max = most stressed) -- e.g. GMM regime
    labels or (as of this build) the VIX-bucket regime labels
    ``regime_detection.vix_regime.build_production_vix_regime`` produces.
    ``stress_from_regime`` below is the only current caller, but this stays
    factored out as a general-purpose ladder builder in case a future
    regime source needs the same 0..1 mapping.

    ``n_labels`` (optional): the FULL bucket/regime count the label set was
    fit with, even if not every label was actually observed in ``labels``
    (e.g. a short window that happened not to touch the most stressed
    bucket). When given, the ladder spans ``0..n_labels-1`` rather than
    just the observed min/max, so a ladder built from a calm window and one
    built from a stressed window stay on the SAME 0..1 scale. Defaults to
    the observed unique label set.
    """
    if n_labels is not None:
        distinct = list(range(n_labels))
    else:
        distinct = sorted(pd.Series(labels.dropna().unique()).tolist())
    return ({lab: i / (len(distinct) - 1) for i, lab in enumerate(distinct)}
            if len(distinct) > 1 else {distinct[0]: 0.0})


def stress_from_regime(regime: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """RIGID stress: regime label mapped onto an evenly spaced 0..1 ladder.

    A step function. Its decisive value at the top of the ladder is set by
    the most stressed regime, which in the n=4 fit holds ~3% of days -- so
    this arm's behavior in exactly the situation it is designed for is
    determined by a very small number of observations.
    """
    mapping = _ladder_from_labels(regime.dropna())
    return regime.reindex(index, method="ffill").map(mapping).fillna(0.0)


def stress_from_volatility(benchmark_prices: pd.Series, index: pd.DatetimeIndex,
                           vol_window: int = 21, lookback: int = 504) -> pd.Series:
    """CONTINUOUS stress: percentile rank of trailing realized volatility
    within its own trailing history, giving a smooth value in [0, 1].

    Uses a ROLLING percentile rather than a full-sample one, so the value at
    date t depends only on volatility observed up to t. A full-sample
    percentile would leak the future distribution into every historical
    date -- the exact look-ahead this project's PIT architecture exists to
    prevent, and easy to introduce accidentally here.

    Every date contributes, so this arm does not depend on the 4-date
    stressed bucket that the rigid arm hinges on.
    """
    ret = np.log(benchmark_prices).diff()
    vol = ret.rolling(vol_window, min_periods=max(vol_window // 2, 5)).std()
    pct = vol.rolling(lookback, min_periods=lookback // 4).apply(
        lambda w: (w[-1] >= w).mean(), raw=True
    )
    return pct.reindex(index).ffill().fillna(0.0).clip(0.0, 1.0)
