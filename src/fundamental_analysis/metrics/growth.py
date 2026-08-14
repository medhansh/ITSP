"""Growth metrics computed from a multi-year financial history.

Unlike the other metric modules (which operate on a single latest snapshot),
growth needs a time series. Expects a ``history`` DataFrame with one row per
(symbol, fiscal_year) and columns: symbol, fiscal_year, revenue, net_income,
eps. Output is one row per symbol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cagr(first: float, last: float, n_years: float) -> float:
    if first is None or last is None or n_years <= 0:
        return np.nan
    if first <= 0 or last <= 0:
        # CAGR is undefined/misleading across a sign change (e.g. loss -> profit).
        return np.nan
    return (last / first) ** (1.0 / n_years) - 1.0


def compute_growth_metrics(history: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """CAGR and YoY-growth stability for revenue, net income, and EPS.

    ``growth_stability`` is the (negative of) the standard deviation of YoY
    growth rates — steadier compounders score higher than lumpy ones with the
    same average growth rate, consistent with a quality-growth screen.
    """
    required = {"symbol", "fiscal_year", "revenue", "net_income", "eps"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {missing}")

    records = []
    for symbol, g in history.sort_values("fiscal_year").groupby("symbol"):
        row = {"symbol": symbol}
        n_years = g["fiscal_year"].nunique() - 1
        if len(g) < min_years:
            records.append(
                {**row, "revenue_cagr": np.nan, "net_income_cagr": np.nan,
                 "eps_cagr": np.nan, "revenue_growth_stability": np.nan,
                 "eps_growth_stability": np.nan}
            )
            continue

        row["revenue_cagr"] = _cagr(g["revenue"].iloc[0], g["revenue"].iloc[-1], n_years)
        row["net_income_cagr"] = _cagr(g["net_income"].iloc[0], g["net_income"].iloc[-1], n_years)
        row["eps_cagr"] = _cagr(g["eps"].iloc[0], g["eps"].iloc[-1], n_years)

        rev_yoy = g["revenue"].pct_change(fill_method=None).dropna()
        eps_yoy = g["eps"].pct_change(fill_method=None).dropna()
        row["revenue_growth_stability"] = -rev_yoy.std() if len(rev_yoy) > 1 else np.nan
        row["eps_growth_stability"] = -eps_yoy.std() if len(eps_yoy) > 1 else np.nan

        records.append(row)

    return pd.DataFrame.from_records(records).set_index("symbol")
