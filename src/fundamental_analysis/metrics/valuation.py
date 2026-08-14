"""Valuation metrics.

Expects a per-symbol snapshot DataFrame with (at minimum, per metric used):
    price, eps_ttm, book_value_per_share, enterprise_value, ebitda,
    eps_growth_pct, dividend_per_share
Extra columns are ignored; missing columns for a given metric simply produce
NaN for that metric rather than raising, so partial data still scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_to_earnings(df: pd.DataFrame) -> pd.Series:
    return df["price"] / df["eps_ttm"].replace(0, np.nan)


def price_to_book(df: pd.DataFrame) -> pd.Series:
    return df["price"] / df["book_value_per_share"].replace(0, np.nan)


def ev_to_ebitda(df: pd.DataFrame) -> pd.Series:
    return df["enterprise_value"] / df["ebitda"].replace(0, np.nan)


def peg_ratio(df: pd.DataFrame) -> pd.Series:
    """PEG = P/E divided by expected EPS growth (%, e.g. 15 for 15%)."""
    pe = price_to_earnings(df)
    return pe / df["eps_growth_pct"].replace(0, np.nan)


def dividend_yield(df: pd.DataFrame) -> pd.Series:
    return df["dividend_per_share"] / df["price"].replace(0, np.nan)


def compute_valuation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of valuation metrics indexed the same as ``df``.

    Lower P/E, P/B, EV/EBITDA and PEG generally indicate "cheaper" — this
    module only computes the raw metrics; direction-of-good is handled in
    scoring/composite_score.py where metrics are sector-relative z-scored
    (and sign-flipped where lower is better) before combination.
    """
    out = pd.DataFrame(index=df.index)
    for name, fn in [
        ("pe_ratio", price_to_earnings),
        ("pb_ratio", price_to_book),
        ("ev_ebitda", ev_to_ebitda),
        ("peg_ratio", peg_ratio),
        ("dividend_yield", dividend_yield),
    ]:
        try:
            out[name] = fn(df)
        except KeyError:
            out[name] = np.nan
    return out
