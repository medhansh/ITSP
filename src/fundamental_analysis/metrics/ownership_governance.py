"""Ownership & governance signals — an India-specific fundamental dimension.

Promoter behavior (holding changes, share pledging) and institutional flows
are unusually informative in Indian markets relative to developed markets,
because promoter families typically retain large, concentrated stakes and
their actions are closely watched as an insider signal. Corporate-governance
red flags (related-party transactions, auditor churn) matter more too, given
several high-profile Indian governance blowups.

Expects a per-symbol snapshot DataFrame with:
    promoter_holding_pct, promoter_holding_pct_prior, promoter_pledge_pct,
    fii_holding_pct, fii_holding_pct_prior, dii_holding_pct, dii_holding_pct_prior,
    related_party_transactions_flag (bool/0-1), auditor_changed_flag (bool/0-1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Empirically, sustained pledging above this level has preceded distress at
# several Indian promoter-led companies; used only as a screening heuristic.
HIGH_PLEDGE_THRESHOLD_PCT = 50.0


def promoter_holding_change(df: pd.DataFrame) -> pd.Series:
    return df["promoter_holding_pct"] - df["promoter_holding_pct_prior"]


def institutional_ownership_change(df: pd.DataFrame) -> pd.Series:
    fii_chg = df["fii_holding_pct"] - df["fii_holding_pct_prior"]
    dii_chg = df["dii_holding_pct"] - df["dii_holding_pct_prior"]
    return fii_chg + dii_chg


def pledge_risk_flag(df: pd.DataFrame) -> pd.Series:
    return (df["promoter_pledge_pct"] >= HIGH_PLEDGE_THRESHOLD_PCT).astype(float)


def governance_red_flag_count(df: pd.DataFrame) -> pd.Series:
    """Count of active governance red flags (0-3): high pledge, promoter
    selling down, and any related-party/auditor-change flag."""
    flags = pd.DataFrame(index=df.index)
    flags["high_pledge"] = pledge_risk_flag(df)
    flags["promoter_selling"] = (promoter_holding_change(df) < -1.0).astype(float)  # >1pp reduction
    rpt_flag = df.get("related_party_transactions_flag", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    auditor_flag = df.get("auditor_changed_flag", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    other = (rpt_flag | auditor_flag).astype(float)
    flags["other_governance_flag"] = other
    return flags.sum(axis=1)


def compute_ownership_governance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["promoter_holding_change"] = promoter_holding_change(df)
    out["institutional_ownership_change"] = institutional_ownership_change(df)
    out["promoter_pledge_pct"] = df["promoter_pledge_pct"]
    out["governance_red_flag_count"] = governance_red_flag_count(df)
    return out
