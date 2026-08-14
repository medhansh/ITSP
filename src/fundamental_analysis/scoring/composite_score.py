"""Combine per-dimension metrics into a single, sector-relative composite score.

Indian sectors trade at structurally different multiples and margins (e.g. IT
services vs. capital goods vs. banks) — a raw cross-sectional z-score would
just rediscover sector membership. So every metric is z-scored *within its
own sector* first, then dimension sub-scores are averaged, then dimensions
are combined with the configurable weights from configs/config.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# For each raw metric, whether a *higher* value is better (True) or a *lower*
# value is better (False, e.g. valuation multiples, leverage, red flags).
METRIC_DIRECTION: dict[str, bool] = {
    # valuation — cheaper is better
    "pe_ratio": False,
    "pb_ratio": False,
    "ev_ebitda": False,
    "peg_ratio": False,
    "dividend_yield": True,
    # profitability & quality
    "roe": True,
    "roce": True,
    "gross_margin": True,
    "net_margin": True,
    "piotroski_f_score": True,
    # growth
    "revenue_cagr": True,
    "net_income_cagr": True,
    "eps_cagr": True,
    "revenue_growth_stability": True,
    "eps_growth_stability": True,
    # leverage & solvency
    "debt_to_equity": False,
    "interest_coverage": True,
    "current_ratio": True,
    "quick_ratio": True,
    "altman_z_score": True,
    # cash-flow quality
    "cfo_to_net_income": True,
    "free_cash_flow": True,
    "fcf_yield": True,
    "accruals_ratio": False,
    # ownership & governance
    "promoter_holding_change": True,
    "institutional_ownership_change": True,
    "promoter_pledge_pct": False,
    "governance_red_flag_count": False,
    # earnings surprise & analyst revisions
    "earnings_surprise_pct": True,
    "estimate_revision_momentum": True,
    # pre-earnings options signal (see metrics/options_earnings.py)
    # Lower pre-earnings IV percentile = market pricing in *less* uncertainty
    # around the last report relative to the stock's own history -> scored
    # as "better" (lower perceived event-risk), same convention as leverage.
    "pre_earnings_iv_percentile": False,
    # Lower put/call OI ratio = more call-side (bullish-tilt) positioning
    # into earnings -> scored as "better". Both directions are a mild,
    # debatable convention on a genuinely noisy positioning indicator, not a
    # strong claim — see docs/fundamental_analysis_spec.md.
    "pre_earnings_put_call_oi_ratio": False,
    # implied_move_pct is deliberately NOT scored (no "higher/lower is
    # better" direction makes sense for an uncertainty magnitude) — it's
    # carried through metrics/options_earnings.py for reporting only.
    # technical momentum (see metrics/technical_momentum.py) — higher
    # Ichimoku conviction is better, same "higher is better" direction as
    # a typical momentum/trend factor.
    "ichimoku_conviction": True,
}

DIMENSION_METRICS: dict[str, list[str]] = {
    "valuation": ["pe_ratio", "pb_ratio", "ev_ebitda", "peg_ratio", "dividend_yield"],
    "profitability_quality": ["roe", "roce", "gross_margin", "net_margin", "piotroski_f_score"],
    "growth": [
        "revenue_cagr", "net_income_cagr", "eps_cagr",
        "revenue_growth_stability", "eps_growth_stability",
    ],
    "leverage_solvency": [
        "debt_to_equity", "interest_coverage", "current_ratio",
        "quick_ratio", "altman_z_score",
    ],
    "cashflow_quality": ["cfo_to_net_income", "free_cash_flow", "fcf_yield", "accruals_ratio"],
    "ownership_governance": [
        "promoter_holding_change", "institutional_ownership_change",
        "promoter_pledge_pct", "governance_red_flag_count",
    ],
    "earnings_surprise": ["earnings_surprise_pct", "estimate_revision_momentum"],
    "options_earnings": ["pre_earnings_iv_percentile", "pre_earnings_put_call_oi_ratio"],
    "technical_momentum": ["ichimoku_conviction"],
}


def sector_relative_zscore(series: pd.Series, sector: pd.Series, min_group_size: int = 5) -> pd.Series:
    """Z-score within sector; sectors with too few members fall back to the
    universe-wide z-score so small sectors don't get a degenerate (0/NaN) score."""
    df = pd.DataFrame({"value": series, "sector": sector})
    group_sizes = df.groupby("sector")["value"].transform("count")
    sector_mean = df.groupby("sector")["value"].transform("mean")
    sector_std = df.groupby("sector")["value"].transform("std")
    z_sector = (df["value"] - sector_mean) / sector_std.replace(0, np.nan)

    universe_mean = df["value"].mean()
    universe_std = df["value"].std()
    z_universe = (df["value"] - universe_mean) / (universe_std if universe_std else np.nan)

    return z_sector.where(group_sizes >= min_group_size, z_universe)


def compute_dimension_scores(
    metrics: pd.DataFrame, sector: pd.Series, sector_relative: bool = True
) -> pd.DataFrame:
    """z-score every known metric (sign-adjusted), then average into one
    0-mean score per dimension in DIMENSION_METRICS."""
    z = pd.DataFrame(index=metrics.index)
    for col in metrics.columns:
        if col not in METRIC_DIRECTION:
            continue
        if sector_relative:
            zscore = sector_relative_zscore(metrics[col], sector)
        else:
            zscore = (metrics[col] - metrics[col].mean()) / metrics[col].std()
        z[col] = zscore if METRIC_DIRECTION[col] else -zscore

    dim_scores = pd.DataFrame(index=metrics.index)
    for dim, cols in DIMENSION_METRICS.items():
        present = [c for c in cols if c in z.columns]
        if present:
            dim_scores[dim] = z[present].mean(axis=1, skipna=True)
    return dim_scores


def compute_composite_score(
    dimension_scores: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    """Weighted average of dimension scores, renormalizing weights over the
    dimensions actually present/non-NaN per row so partial fundamental data
    still yields a usable score instead of NaN."""
    available = [d for d in weights if d in dimension_scores.columns]
    w = pd.Series({d: weights[d] for d in available})

    weighted_sum = pd.Series(0.0, index=dimension_scores.index)
    weight_total = pd.Series(0.0, index=dimension_scores.index)
    for dim in available:
        col = dimension_scores[dim]
        valid = col.notna()
        weighted_sum.loc[valid] += col.loc[valid] * w[dim]
        weight_total.loc[valid] += w[dim]

    return weighted_sum / weight_total.replace(0, np.nan)


def rebalanced_weights(base_weights: dict[str, float], dim: str, new_weight: float) -> dict[str, float]:
    """Rescale every OTHER weight proportionally so the total still sums to
    1.0 after setting ``dim``'s weight to ``new_weight``.

    First recovers what the OTHER dimensions' proportions were relative to
    each other (dividing out ``dim``'s current share of ``base_weights``),
    then reapplies them scaled to fill whatever's left over after the new
    weight — so this correctly generalizes from ANY starting point (not
    just ``dim``'s original value) to any target weight.

    Used both for one-off composite-weight sweeps
    (``scripts/sweep_technical_momentum_weight.py``) and for regime-
    conditional weighting, where ``dim``'s effective weight is recomputed
    fresh at every rebalance date based on the current regime
    (``point_in_time.py``'s ``apply_regime_conditional_weight``) — this
    lives in the core library rather than a script specifically so both
    can share it without a script-importing-into-src dependency direction.
    """
    base_dim_weight = base_weights.get(dim, 0.0)
    others_total = 1.0 - base_dim_weight
    if others_total <= 1e-9:
        raise ValueError(f"Cannot rebalance: base weight for {dim!r} is already ~1.0, nothing to redistribute from.")
    if new_weight >= 1.0 or new_weight < 0.0:
        raise ValueError(f"new_weight must be in [0, 1), got {new_weight}")
    result = {d: w / others_total * (1.0 - new_weight) for d, w in base_weights.items() if d != dim}
    result[dim] = new_weight
    return result
