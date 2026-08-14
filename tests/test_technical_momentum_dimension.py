"""Tests for Ichimoku conviction as an 8th fundamentals composite
dimension (``technical_momentum``) — the selection-time integration built
after two post-selection mechanisms (gating, tilting an already-built
portfolio) were both confirmed negative on real data. See
``docs/backtesting_spec.md`` and ``metrics/technical_momentum.py`` for the
full motivation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.adaptive_ichimoku import build_ichimoku_conviction_panel
from src.fundamental_analysis.data_fetchers.fundamentals_fetcher import SNAPSHOT_SCHEMA
from src.fundamental_analysis.metrics.technical_momentum import compute_technical_momentum_metrics
from src.fundamental_analysis.pipeline import run_pipeline
from src.fundamental_analysis.point_in_time import run_pit_fundamental_pipeline


@pytest.fixture
def synthetic_ohlc_panel():
    rng = np.random.default_rng(3)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    panel = {}
    for sym in ("AAA", "BBB", "CCC"):
        ret = rng.normal(0.0004, 0.012, n)
        close = pd.Series(100 * np.exp(np.cumsum(ret)), index=dates)
        high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
        panel[sym] = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    return panel


def test_build_ichimoku_conviction_panel_not_portfolio_normalized(synthetic_ohlc_panel):
    """The whole point of this function vs build_ichimoku_weights: values
    must be raw [0,1] conviction, NOT divided down to ~1/n_active."""
    panel = build_ichimoku_conviction_panel(synthetic_ohlc_panel, variant="static")
    valid = panel.dropna(how="all")
    assert not valid.empty
    # mean should be a real fraction, not ~1/3 = 0.33 divided further down to noise
    assert valid.values[~np.isnan(valid.values)].mean() > 0.01
    assert (panel.dropna().values <= 1.0).all()
    assert (panel.dropna().values >= 0.0).all()


def test_compute_technical_momentum_metrics_wraps_series_correctly():
    conviction = pd.Series({"AAA": 0.8, "BBB": 0.2})
    result = compute_technical_momentum_metrics(conviction)
    assert list(result.columns) == ["ichimoku_conviction"]
    assert result.loc["AAA", "ichimoku_conviction"] == 0.8
    assert result.loc["BBB", "ichimoku_conviction"] == 0.2


def test_compute_technical_momentum_metrics_none_input_returns_empty_shaped_frame():
    result = compute_technical_momentum_metrics(None)
    assert list(result.columns) == ["ichimoku_conviction"]
    assert result.empty


def test_run_pipeline_technical_momentum_ranks_by_conviction():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    snapshot = pd.DataFrame(index=symbols, columns=SNAPSHOT_SCHEMA, dtype=float)
    snapshot["sector"] = "Tech"
    conviction = pd.Series({"AAA": 0.9, "BBB": 0.1, "CCC": 0.5, "DDD": 0.5})

    config = {
        "sector_relative": False,
        "dimensions": {
            "valuation": False, "profitability_quality": False, "growth": False,
            "leverage_solvency": False, "cashflow_quality": False,
            "ownership_governance": False, "earnings_surprise": False,
            "options_earnings": False, "technical_momentum": True,
        },
        "composite_weights": {"technical_momentum": 1.0},
    }
    result = run_pipeline(config, snapshot, technical_conviction=conviction)
    assert result.loc["AAA", "composite_score"] > result.loc["CCC", "composite_score"]
    assert result.loc["CCC", "composite_score"] > result.loc["BBB", "composite_score"]


def test_run_pipeline_technical_momentum_missing_conviction_is_nan_not_crash():
    symbols = ["AAA", "BBB"]
    snapshot = pd.DataFrame(index=symbols, columns=SNAPSHOT_SCHEMA, dtype=float)
    snapshot["sector"] = "Tech"
    config = {
        "sector_relative": False,
        "dimensions": {
            "valuation": False, "profitability_quality": False, "growth": False,
            "leverage_solvency": False, "cashflow_quality": False,
            "ownership_governance": False, "earnings_surprise": False,
            "options_earnings": False, "technical_momentum": True,
        },
        "composite_weights": {"technical_momentum": 1.0},
    }
    # technical_conviction not provided at all
    result = run_pipeline(config, snapshot)
    assert result["composite_score"].isna().all()  # honest gap, not a crash


def test_run_pit_fundamental_pipeline_conviction_is_pit_safe_and_ffilled():
    """A rebalance date not exactly matching a trading day in the
    conviction panel must use the most recent PRIOR value (ffill), never a
    future one -- reindex(method='ffill') gives us this, but pin it down
    with a real assertion rather than trusting the pandas call blindly."""
    symbols = ["AAA", "BBB"]
    # build_pit_panel returns zero rows for ALL dates if history_long is
    # completely empty (nothing to groupby), so give it minimal real
    # quarterly data -- otherwise the per-date scoring loop never runs at
    # all regardless of what rebalance_dates were requested.
    quarterly = pd.DataFrame([
        {"symbol": s, "period_end": pd.Timestamp("2019-12-31"), "known_date": pd.Timestamp("2020-01-01"),
         "field": "revenue", "value": 100.0}
        for s in symbols
    ])
    snapshot = pd.DataFrame(index=symbols, columns=SNAPSHOT_SCHEMA, dtype=float)
    snapshot["sector"] = "Tech"

    # conviction changes value partway through -- a rebalance date landing
    # in the middle of a gap must see the OLD value, not a later one.
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    conviction_panel = pd.DataFrame({"AAA": 0.2, "BBB": 0.2}, index=dates)
    conviction_panel.loc["2020-01-08":, "AAA"] = 0.9  # jumps up on Jan 8

    config = {
        "sector_relative": False,
        "dimensions": {
            "valuation": False, "profitability_quality": False, "growth": False,
            "leverage_solvency": False, "cashflow_quality": False,
            "ownership_governance": False, "earnings_surprise": False,
            "options_earnings": False, "technical_momentum": True,
        },
        "composite_weights": {"technical_momentum": 1.0},
    }
    # rebalance date BEFORE the jump
    scores_before = run_pit_fundamental_pipeline(
        config, snapshot, quarterly, [pd.Timestamp("2020-01-06")], conviction_panel=conviction_panel
    )
    aaa_before = scores_before.set_index("symbol").loc["AAA", "ichimoku_conviction"]
    assert aaa_before == pytest.approx(0.2)  # must NOT see the future 0.9 value

    # rebalance date AFTER the jump
    scores_after = run_pit_fundamental_pipeline(
        config, snapshot, quarterly, [pd.Timestamp("2020-01-09")], conviction_panel=conviction_panel
    )
    aaa_after = scores_after.set_index("symbol").loc["AAA", "ichimoku_conviction"]
    assert aaa_after == pytest.approx(0.9)


def test_run_pit_fundamental_pipeline_without_conviction_panel_unchanged():
    """None (default) must reproduce the exact pre-Ichimoku-dimension
    behavior -- technical_momentum simply comes back NaN, nothing crashes,
    and other enabled dimensions are completely unaffected."""
    symbols = ["AAA", "BBB"]
    quarterly = pd.DataFrame([
        {"symbol": s, "period_end": pd.Timestamp("2019-12-31"), "known_date": pd.Timestamp("2020-01-01"),
         "field": "revenue", "value": 100.0}
        for s in symbols
    ])
    snapshot = pd.DataFrame(index=symbols, columns=SNAPSHOT_SCHEMA, dtype=float)
    snapshot["sector"] = "Tech"

    config = {
        "sector_relative": False,
        "dimensions": {"technical_momentum": True},
        "composite_weights": {"technical_momentum": 1.0},
    }
    scores = run_pit_fundamental_pipeline(config, snapshot, quarterly, [pd.Timestamp("2020-01-06")])
    assert len(scores) == 2  # both symbols still scored (as NaN), not dropped or crashed
    assert "ichimoku_conviction" not in scores.columns or scores["ichimoku_conviction"].isna().all()
    assert scores["composite_score"].isna().all()  # no data at all for the only enabled dimension -> honest NaN
