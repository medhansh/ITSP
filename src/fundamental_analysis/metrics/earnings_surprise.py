"""Earnings surprise & analyst-estimate-revision metrics.

This is the module the original project abstract scoped as the whole of
"fundamental analysis" (an earnings-surprise predictor analyzing promoter
activity and analyst patterns). It's retained here as one dimension among
several — promoter activity now lives in ownership_governance.py since it's
a distinct signal family, not just a driver of earnings surprises.

Expects a per-symbol snapshot DataFrame with:
    actual_eps, analyst_eps_estimate, analyst_eps_estimate_30d_ago
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def earnings_surprise_pct(df: pd.DataFrame) -> pd.Series:
    return (df["actual_eps"] - df["analyst_eps_estimate"]) / df[
        "analyst_eps_estimate"
    ].abs().replace(0, np.nan)


def estimate_revision_momentum(df: pd.DataFrame) -> pd.Series:
    """Positive = analysts have been raising estimates over the last 30 days —
    a well-documented drift signal (post-earnings-announcement drift proxy)."""
    return (df["analyst_eps_estimate"] - df["analyst_eps_estimate_30d_ago"]) / df[
        "analyst_eps_estimate_30d_ago"
    ].abs().replace(0, np.nan)


def compute_earnings_surprise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["earnings_surprise_pct"] = earnings_surprise_pct(df)
    out["estimate_revision_momentum"] = estimate_revision_momentum(df)
    return out
