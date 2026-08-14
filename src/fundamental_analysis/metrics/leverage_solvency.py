"""Leverage & solvency metrics, including the Altman Z-score.

Expects a per-symbol snapshot DataFrame with:
    current_assets, current_liabilities, total_assets, total_liabilities,
    retained_earnings, ebit, market_cap, revenue, total_debt, total_equity,
    interest_expense
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def debt_to_equity(df: pd.DataFrame) -> pd.Series:
    """Debt-to-equity. NEGATIVE (or zero) total_equity means liabilities
    exceed assets -- technically insolvent on a book basis, one of the most
    severe distress signals that exists. Dividing total_debt by negative
    equity produces a NEGATIVE ratio, which "lower debt/equity is better"
    scoring would read as FAVORABLE -- exactly backwards from reality.

    Confirmed as a real, live bug via scripts/diagnose_fundamentals_drawdown.py:
    IDEA (Vodafone Idea) had deeply negative equity yet scored ABOVE
    AVERAGE on leverage_solvency, the best of a group of ten names that
    included several PSU banks mid-NPA-crisis and two heavily-leveraged
    infra companies -- all names that later drove a large chunk of a real
    backtest's excess drawdown.

    Negative-or-zero equity now maps to a large finite sentinel (not inf,
    which would corrupt downstream z-scoring's mean/std) clearly beyond any
    realistic positive-equity ratio, so it scores as the WORST case, not
    the best.
    """
    equity = df["total_equity"]
    ratio = df["total_debt"] / equity.replace(0, np.nan)
    NEGATIVE_EQUITY_SENTINEL = 100.0  # realistic positive-equity D/E is rarely above ~10
    return ratio.where(equity > 0, NEGATIVE_EQUITY_SENTINEL)


def interest_coverage(df: pd.DataFrame) -> pd.Series:
    return df["ebit"] / df["interest_expense"].replace(0, np.nan)


def current_ratio(df: pd.DataFrame) -> pd.Series:
    return df["current_assets"] / df["current_liabilities"].replace(0, np.nan)


def quick_ratio(df: pd.DataFrame) -> pd.Series:
    inventory = df.get("inventory", pd.Series(0, index=df.index)).fillna(0)
    return (df["current_assets"] - inventory) / df["current_liabilities"].replace(0, np.nan)


def altman_z_score(df: pd.DataFrame) -> pd.Series:
    """Classic (1968) Altman Z-score for publicly traded, non-financial firms.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5, where:
      X1 = working capital / total assets
      X2 = retained earnings / total assets
      X3 = EBIT / total assets
      X4 = market cap / total liabilities
      X5 = revenue / total assets
    Z > 2.99 = "safe" zone, 1.81-2.99 = "grey" zone, < 1.81 = "distress" zone.
    Not designed for banks/NBFCs — exclude Financial Services when screening.

    **Partial-data handling (fixed 2026-07-27 after a real bug)**: this
    used to be a plain ``1.2*x1 + 1.4*x2 + ...`` sum, which NaN-propagates
    -- a single missing component (e.g. X1, if ``current_assets``/
    ``current_liabilities`` aren't available, confirmed via
    ``scripts/diagnose_fundamentals_drawdown.py`` to be universally missing
    across the real dataset) silently discarded the ENTIRE score, including
    strongly informative components like X2 that alone can flag severe
    distress (e.g. deeply negative retained earnings). Now: weighted-
    average over whichever components ARE present, then rescaled back up
    to the full 1.2+1.4+3.3+0.6+1.0=7.5 weight scale so partial data
    doesn't mechanically shrink the score toward zero (same "partial data
    still produces a usable score, extrapolated fairly" convention as
    ``scoring/composite_score.compute_composite_score``'s per-row weight
    renormalization). Returns NaN only if EVERY component is missing.
    """
    working_capital = df["current_assets"] - df["current_liabilities"]
    components = {
        "x1": (1.2, working_capital / df["total_assets"].replace(0, np.nan)),
        "x2": (1.4, df["retained_earnings"] / df["total_assets"].replace(0, np.nan)),
        "x3": (3.3, df["ebit"] / df["total_assets"].replace(0, np.nan)),
        "x4": (0.6, df["market_cap"] / df["total_liabilities"].replace(0, np.nan)),
        "x5": (1.0, df["revenue"] / df["total_assets"].replace(0, np.nan)),
    }
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_present = pd.Series(0.0, index=df.index)
    for weight, value in components.values():
        valid = value.notna()
        weighted_sum = weighted_sum.add(value.where(valid, 0.0) * weight, fill_value=0.0)
        weight_present = weight_present.add(valid.astype(float) * weight, fill_value=0.0)
    full_weight = sum(w for w, _ in components.values())  # 7.5
    return (weighted_sum / weight_present.replace(0, np.nan)) * full_weight


def compute_leverage_solvency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["debt_to_equity"] = debt_to_equity(df)
    out["interest_coverage"] = interest_coverage(df)
    out["current_ratio"] = current_ratio(df)
    out["quick_ratio"] = quick_ratio(df)
    out["altman_z_score"] = altman_z_score(df)
    return out