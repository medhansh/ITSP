"""Profitability & quality metrics, including the Piotroski F-score.

The Piotroski F-score (Piotroski, 2000) is a 9-point checklist of YoY
fundamental improvements, originally designed to separate strong from weak
value stocks. It needs current *and* prior fiscal year figures, so this
module expects a snapshot DataFrame with both a current-year column and a
matching ``<col>_prior`` column for every input listed below.

Required current-year columns:
    net_income, total_assets, cfo, long_term_debt, current_assets,
    current_liabilities, shares_outstanding, gross_profit, revenue,
    total_equity, ebit
Required prior-year columns (suffix ``_prior``):
    total_assets_prior, long_term_debt_prior, current_assets_prior,
    current_liabilities_prior, shares_outstanding_prior, gross_profit_prior,
    revenue_prior, net_income_prior (for ROA-improved check)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def return_on_equity(df: pd.DataFrame) -> pd.Series:
    return df["net_income"] / df["total_equity"].replace(0, np.nan)


def return_on_capital_employed(df: pd.DataFrame) -> pd.Series:
    capital_employed = df["total_assets"] - df["current_liabilities"]
    return df["ebit"] / capital_employed.replace(0, np.nan)


def gross_margin(df: pd.DataFrame) -> pd.Series:
    return df["gross_profit"] / df["revenue"].replace(0, np.nan)


def net_margin(df: pd.DataFrame) -> pd.Series:
    return df["net_income"] / df["revenue"].replace(0, np.nan)


def _return_on_assets(df: pd.DataFrame, suffix: str = "") -> pd.Series:
    ni = df[f"net_income{suffix}"]
    ta = df[f"total_assets{suffix}"]
    return ni / ta.replace(0, np.nan)


def _gt(a: pd.Series, b) -> pd.Series:
    """Elementwise a > b that is NaN (not False) wherever either side is NaN.

    Plain pandas comparison treats NaN as False, which would make a missing
    prior-year figure silently count as "did not improve" — this helper
    keeps missing data genuinely missing so ``add()`` can exclude it.
    """
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    result = a > b
    return result.where(a.notna() & b.notna())


def piotroski_f_score(df: pd.DataFrame) -> pd.Series:
    """9-point Piotroski F-score. Each signal contributes 0 or 1; a signal
    that can't be computed (missing current or prior-year data) is skipped
    for that row rather than counted as a failure."""
    score = pd.Series(0, index=df.index, dtype=float)
    n_signals = pd.Series(0, index=df.index, dtype=float)

    def add(signal: pd.Series) -> None:
        nonlocal score, n_signals
        valid = signal.notna()
        score.loc[valid] += signal.loc[valid].astype(int)
        n_signals.loc[valid] += 1

    roa = _return_on_assets(df)
    nan_series = pd.Series(np.nan, index=df.index)
    roa_prior = (
        _return_on_assets(df, "_prior")
        if "net_income_prior" in df and "total_assets_prior" in df
        else nan_series
    )
    cfo = df.get("cfo", nan_series)

    add(_gt(roa, 0))                                                 # 1. profitability
    add(_gt(cfo, 0))                                                 # 2. positive operating cash flow
    add(_gt(roa, roa_prior))                                         # 3. improving ROA
    add(_gt(cfo, df["net_income"]))                                  # 4. accrual quality (CFO > NI)

    if "long_term_debt_prior" in df and "total_assets_prior" in df:
        ltd_ratio = df["long_term_debt"] / df["total_assets"].replace(0, np.nan)
        ltd_ratio_prior = df["long_term_debt_prior"] / df["total_assets_prior"].replace(0, np.nan)
        add(_gt(ltd_ratio_prior, ltd_ratio))                         # 5. decreasing leverage
    if "current_assets_prior" in df and "current_liabilities_prior" in df:
        current_ratio = df["current_assets"] / df["current_liabilities"].replace(0, np.nan)
        current_ratio_prior = df["current_assets_prior"] / df["current_liabilities_prior"].replace(0, np.nan)
        add(_gt(current_ratio, current_ratio_prior))                 # 6. improving liquidity
    if "shares_outstanding_prior" in df:
        add(_gt(df["shares_outstanding_prior"], df["shares_outstanding"] - 1e-9))  # 7. no dilution
    if "gross_profit_prior" in df and "revenue_prior" in df:
        gm = df["gross_profit"] / df["revenue"].replace(0, np.nan)
        gm_prior = df["gross_profit_prior"] / df["revenue_prior"].replace(0, np.nan)
        add(_gt(gm, gm_prior))                                       # 8. improving gross margin
    if "revenue_prior" in df and "total_assets_prior" in df:
        asset_turnover = df["revenue"] / df["total_assets"].replace(0, np.nan)
        asset_turnover_prior = df["revenue_prior"] / df["total_assets_prior"].replace(0, np.nan)
        add(_gt(asset_turnover, asset_turnover_prior))               # 9. improving efficiency

    return score  # 0-9; scoring layer can normalize by n_signals if desired


def compute_profitability_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["roe"] = return_on_equity(df)
    out["roce"] = return_on_capital_employed(df)
    out["gross_margin"] = gross_margin(df)
    out["net_margin"] = net_margin(df)
    out["piotroski_f_score"] = piotroski_f_score(df)
    return out
