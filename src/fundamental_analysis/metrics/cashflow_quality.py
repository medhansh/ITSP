"""Cash-flow quality metrics — do reported earnings actually turn into cash?

Expects a per-symbol snapshot DataFrame with:
    cfo (cash flow from operations), net_income, capex, market_cap, total_assets
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cfo_to_net_income(df: pd.DataFrame) -> pd.Series:
    """>1 generally means earnings are backed by cash, not just accruals."""
    return df["cfo"] / df["net_income"].replace(0, np.nan)


def free_cash_flow(df: pd.DataFrame) -> pd.Series:
    return df["cfo"] - df["capex"]


def fcf_yield(df: pd.DataFrame) -> pd.Series:
    return free_cash_flow(df) / df["market_cap"].replace(0, np.nan)


def accruals_ratio(df: pd.DataFrame) -> pd.Series:
    """(Net income - CFO) / total assets. High positive accruals are a classic
    earnings-manipulation / low-quality-earnings red flag (Sloan, 1996)."""
    return (df["net_income"] - df["cfo"]) / df["total_assets"].replace(0, np.nan)


def compute_cashflow_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["cfo_to_net_income"] = cfo_to_net_income(df)
    out["free_cash_flow"] = free_cash_flow(df)
    out["fcf_yield"] = fcf_yield(df)
    out["accruals_ratio"] = accruals_ratio(df)
    return out
