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
    return df["total_debt"] / df["total_equity"].replace(0, np.nan)


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
    """
    working_capital = df["current_assets"] - df["current_liabilities"]
    x1 = working_capital / df["total_assets"].replace(0, np.nan)
    x2 = df["retained_earnings"] / df["total_assets"].replace(0, np.nan)
    x3 = df["ebit"] / df["total_assets"].replace(0, np.nan)
    x4 = df["market_cap"] / df["total_liabilities"].replace(0, np.nan)
    x5 = df["revenue"] / df["total_assets"].replace(0, np.nan)
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def compute_leverage_solvency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["debt_to_equity"] = debt_to_equity(df)
    out["interest_coverage"] = interest_coverage(df)
    out["current_ratio"] = current_ratio(df)
    out["quick_ratio"] = quick_ratio(df)
    out["altman_z_score"] = altman_z_score(df)
    return out
