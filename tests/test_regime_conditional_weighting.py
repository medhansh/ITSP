"""Tests for regime-conditional technical_momentum weighting
(``point_in_time.apply_regime_conditional_weight`` and its wiring into
``run_pit_fundamental_pipeline``) -- built after walk-forward validation
showed technical_momentum's edge was concentrated in trending conditions.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.fundamental_analysis.data_fetchers.fundamentals_fetcher import SNAPSHOT_SCHEMA
from src.fundamental_analysis.point_in_time import (
    apply_regime_conditional_weight,
    run_pit_fundamental_pipeline,
)

MULTIPLIERS = {"low_vol_calm": 1.0, "moderate_vol": 0.6, "elevated_vol": 0.3, "high_vol_stress": 0.0}
BASE_WEIGHTS = {"valuation": 0.5, "technical_momentum": 0.4, "growth": 0.1}


def test_apply_regime_conditional_weight_full_multiplier_unchanged():
    result = apply_regime_conditional_weight(BASE_WEIGHTS, "technical_momentum", "low_vol_calm", MULTIPLIERS)
    assert result["technical_momentum"] == pytest.approx(0.4)
    assert sum(result.values()) == pytest.approx(1.0)


def test_apply_regime_conditional_weight_zero_multiplier_zeroes_dimension():
    result = apply_regime_conditional_weight(BASE_WEIGHTS, "technical_momentum", "high_vol_stress", MULTIPLIERS)
    assert result["technical_momentum"] == pytest.approx(0.0)
    assert sum(result.values()) == pytest.approx(1.0)
    # the freed-up weight should be redistributed proportionally to the other two
    assert result["valuation"] / result["growth"] == pytest.approx(BASE_WEIGHTS["valuation"] / BASE_WEIGHTS["growth"])


def test_apply_regime_conditional_weight_partial_multiplier():
    result = apply_regime_conditional_weight(BASE_WEIGHTS, "technical_momentum", "elevated_vol", MULTIPLIERS)
    assert result["technical_momentum"] == pytest.approx(0.4 * 0.3)
    assert sum(result.values()) == pytest.approx(1.0)


def test_apply_regime_conditional_weight_unknown_regime_falls_back(caplog):
    with caplog.at_level("WARNING"):
        result = apply_regime_conditional_weight(BASE_WEIGHTS, "technical_momentum", "made_up_regime", MULTIPLIERS)
    assert result["technical_momentum"] == pytest.approx(0.4)  # default_multiplier=1.0 -> unchanged
    assert any("not found in configured multipliers" in r.message for r in caplog.records)


def test_apply_regime_conditional_weight_none_regime_falls_back(caplog):
    with caplog.at_level("WARNING"):
        result = apply_regime_conditional_weight(BASE_WEIGHTS, "technical_momentum", None, MULTIPLIERS)
    assert result["technical_momentum"] == pytest.approx(0.4)
    assert any("no regime label available" in r.message for r in caplog.records)


def test_apply_regime_conditional_weight_custom_default_multiplier():
    result = apply_regime_conditional_weight(
        BASE_WEIGHTS, "technical_momentum", "unknown", MULTIPLIERS, default_multiplier=0.5
    )
    assert result["technical_momentum"] == pytest.approx(0.2)


def test_run_pit_fundamental_pipeline_regime_conditioning_changes_score_by_date():
    """The core end-to-end check: the SAME conviction values should
    produce a MORE differentiated composite score on a calm-regime date
    than on a high-vol-stress date, since the latter should have
    technical_momentum's weight zeroed out."""
    symbols = ["AAA", "BBB"]
    quarterly = pd.DataFrame([
        {"symbol": s, "period_end": pd.Timestamp("2019-12-31"), "known_date": pd.Timestamp("2020-01-01"),
         "field": "revenue", "value": 100.0}
        for s in symbols
    ])
    snapshot = pd.DataFrame(index=symbols, columns=SNAPSHOT_SCHEMA, dtype=float)
    snapshot["sector"] = "Tech"

    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    conviction_panel = pd.DataFrame({"AAA": 0.8, "BBB": 0.2}, index=dates)  # constant, deliberately

    regime = pd.Series("low_vol_calm", index=dates)
    regime.loc["2020-04-01":] = "high_vol_stress"

    config = {
        "sector_relative": False,
        "dimensions": {"technical_momentum": True, "growth": False},
        "composite_weights": {"technical_momentum": 0.4, "growth": 0.6},
    }

    scores = run_pit_fundamental_pipeline(
        config, snapshot, quarterly, [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-01")],
        conviction_panel=conviction_panel, regime=regime, regime_weight_multipliers=MULTIPLIERS,
    )
    scores = scores.set_index(["date", "symbol"])

    calm_aaa = scores.loc[(pd.Timestamp("2020-02-01"), "AAA"), "composite_score"]
    calm_bbb = scores.loc[(pd.Timestamp("2020-02-01"), "BBB"), "composite_score"]
    assert calm_aaa != calm_bbb  # calm regime: technical_momentum weight active -> differentiated

    # stress regime: technical_momentum weight forced to 0, growth is the
    # only other (disabled) dimension -> honest NaN, not a crash
    stress_aaa = scores.loc[(pd.Timestamp("2020-05-01"), "AAA"), "composite_score"]
    stress_bbb = scores.loc[(pd.Timestamp("2020-05-01"), "BBB"), "composite_score"]
    assert pd.isna(stress_aaa) and pd.isna(stress_bbb)


def test_run_pit_fundamental_pipeline_without_regime_conditioning_unchanged():
    """None (default) for regime/regime_weight_multipliers must reproduce
    exact pre-regime-conditioning behavior -- static weight throughout."""
    symbols = ["AAA", "BBB"]
    quarterly = pd.DataFrame([
        {"symbol": s, "period_end": pd.Timestamp("2019-12-31"), "known_date": pd.Timestamp("2020-01-01"),
         "field": "revenue", "value": 100.0}
        for s in symbols
    ])
    snapshot = pd.DataFrame(index=symbols, columns=SNAPSHOT_SCHEMA, dtype=float)
    snapshot["sector"] = "Tech"
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    conviction_panel = pd.DataFrame({"AAA": 0.8, "BBB": 0.2}, index=dates)

    config = {
        "sector_relative": False,
        "dimensions": {"technical_momentum": True},
        "composite_weights": {"technical_momentum": 1.0},
    }
    # No regime / multipliers passed at all
    scores = run_pit_fundamental_pipeline(
        config, snapshot, quarterly, [pd.Timestamp("2020-02-01")], conviction_panel=conviction_panel,
    )
    scores = scores.set_index("symbol")
    assert scores.loc["AAA", "composite_score"] != scores.loc["BBB", "composite_score"]
    assert scores.loc["AAA", "composite_score"] > scores.loc["BBB", "composite_score"]
