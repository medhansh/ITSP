"""Geometric-algebra "wedge product volume" crash-risk signal.

Source / credit
----------------
This signal is NOT something the team derived independently — it is an
implementation of an idea from a third-party source the user pointed us at:

  - Article: Agus Sudjianto, "The Geometry of a Crash: Why 'Market Volume'
    Beats Probabilistic Models" (agussudjianto.substack.com, Jan 2026).
  - Video:   "Geometric Regime Detection for Market Crashes | Quant Finance"
    (YouTube, VTlrVSJfvH4).

**Caveat — read before trusting this signal.** The source article claims a
16.7% out-of-sample Sharpe improvement and a 98% transaction-cost reduction
versus an HMM, attributed to "a recent analysis of regime detection methods
(2015-2023)" that is not cited by name, DOI, or link. Those specific numbers
are a single blogger's unverified claim, not a peer-reviewed or independently
reproduced result, and this project has NOT reproduced them on NIFTY data.
Treat this module as an experimental, supplementary structural feature to be
validated empirically against our own universe (see "Validating this
signal" below) — not as a proven outperformer. Don't cite the 16.7%/98%
figures in any report this project produces without redoing the comparison
ourselves.

The underlying mathematical idea, independent of those specific performance
claims, is legitimate and well-known: pairwise correlation matrices only
capture two-asset relationships, but a market-wide "everything sells off
together" event is a simultaneous, systemic collapse across *all* assets at
once. Geometric algebra's wedge product (v1 ^ v2 ^ ... ^ vn) gives the
n-dimensional oriented "volume" spanned by n return vectors:

    Volume = ||v1 ^ ... ^ vn|| = sqrt(det(Gram(V)))   where Gram(V) = V^T V

on direction-normalized (unit) vectors, so the volume isolates *correlation
structure* from *magnitude*. When sector returns point in different
directions (healthy rotation), the volume is large. When everything moves
together (panic/liquidation), the vectors collapse toward a single line and
the volume collapses toward zero — this is a purely geometric, model-free
crisis signal that doesn't require fitting a distributional assumption the
way GMM/HMM do, and (per the source article) is less prone to the
day-to-day "flickering" that HMMs exhibit around single-day outlier returns.

Requires multiple, sufficiently-uncorrelated-in-normal-times asset return
series (e.g. sector indices), not a single index price — see
``regime_detection.data_loader.load_sector_prices_from_yfinance`` /
``load_sector_prices_from_csv`` for the input this module expects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_wedge_volume(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling n-dimensional wedge-product "volume" of a multi-asset returns panel.

    Args:
        returns: daily returns, index=date, columns=assets (e.g. sector
            indices). Needs at least 2 columns (n=1 is degenerate: volume is
            undefined/trivially 1). More columns = more dimensions = a
            geometrically richer (but noisier, needs a longer window) signal.
        window: rolling lookback in trading days used to build each window's
            vectors (one column per asset, one row per day within the window).

    Returns:
        Series indexed like ``returns`` (first ``window`` rows dropped, same
        as a rolling calculation) — volume in [0, 1]-ish range (0 = total
        linear collapse / crisis, higher = more structurally diversified).
        NaN rows (e.g. from missing data) inside a window propagate to NaN.
    """
    if returns.shape[1] < 2:
        raise ValueError(
            f"calculate_wedge_volume needs >= 2 asset columns, got {returns.shape[1]}. "
            "Pass a multi-asset (e.g. sector) returns panel, not a single series."
        )

    values = returns.values
    n_rows = values.shape[0]
    out = np.full(n_rows, np.nan)

    for i in range(window, n_rows):
        window_data = values[i - window : i]
        if np.isnan(window_data).any():
            continue  # leave NaN — don't silently drop/interpolate missing days here
        norms = np.linalg.norm(window_data, axis=0)
        norms = np.where(norms == 0, 1e-12, norms)
        normalized = window_data / norms
        gram = normalized.T @ normalized
        det = np.linalg.det(gram)
        out[i] = np.sqrt(abs(det))

    return pd.Series(out, index=returns.index, name=f"wedge_volume_{window}d")


def compute_geometric_crash_features(
    returns: pd.DataFrame,
    window: int = 60,
    smoothing_window: int = 10,
    percentile_window: int = 252,
    crash_percentile_threshold: float = 0.15,
) -> pd.DataFrame:
    """Build the full feature block: raw wedge volume, a smoothed version, a
    rolling percentile rank (self-relative, so it's comparable across market
    eras without needing a fixed absolute threshold), and a binary crash-risk
    flag when the smoothed volume falls into its own bottom
    ``crash_percentile_threshold`` of trailing ``percentile_window`` days.

    Returns columns: wedge_volume_{window}d, wedge_volume_{window}d_smoothed,
    wedge_volume_percentile_{percentile_window}d, geometric_crash_risk_flag.
    """
    raw = calculate_wedge_volume(returns, window=window)
    smoothed = raw.rolling(smoothing_window).mean()
    smoothed.name = f"{raw.name}_smoothed"

    # Rolling percentile rank of the *current* smoothed value within its own
    # trailing history — this is self-relative (not the article's fixed 15th
    # percentile computed on the *full* sample, which would leak future
    # information if used inside a walk-forward backtest).
    def _pct_rank(x: np.ndarray) -> float:
        if np.isnan(x[-1]):
            return np.nan
        valid = x[~np.isnan(x)]
        if len(valid) < 2:
            return np.nan
        return (valid < x[-1]).mean()

    percentile = smoothed.rolling(percentile_window, min_periods=max(20, percentile_window // 5)).apply(
        _pct_rank, raw=True
    )
    percentile.name = f"wedge_volume_percentile_{percentile_window}d"

    crash_flag = (percentile <= crash_percentile_threshold).astype(float)
    crash_flag.name = "geometric_crash_risk_flag"
    crash_flag[percentile.isna()] = np.nan

    return pd.concat([raw, smoothed, percentile, crash_flag], axis=1)


def validate_against_known_crises(
    features: pd.DataFrame,
    crisis_windows: list[tuple[str, str]],
    flag_col: str = "geometric_crash_risk_flag",
) -> pd.DataFrame:
    """Sanity-check helper: what fraction of days inside each known crisis
    window (e.g. [("2020-02-20","2020-04-07")] for the COVID crash) the flag
    was actually raised, vs. the flag's overall base rate outside those
    windows. This is the minimum bar before trusting the signal — if the
    in-crisis hit rate isn't well above the base rate, the signal isn't
    adding anything over a coin flip and the article's claims don't hold up
    on our data. Use this on real (not synthetic) sector price history before
    wiring the flag into ``strategies.py``.
    """
    rows = []
    all_flagged = features[flag_col].dropna()
    base_rate = all_flagged.mean() if len(all_flagged) else np.nan
    for start, end in crisis_windows:
        window = features.loc[start:end, flag_col].dropna()
        hit_rate = window.mean() if len(window) else np.nan
        rows.append({"start": start, "end": end, "n_days": len(window), "hit_rate": hit_rate})
    result = pd.DataFrame(rows)
    result["base_rate"] = base_rate
    result["lift_over_base_rate"] = result["hit_rate"] - base_rate
    return result
