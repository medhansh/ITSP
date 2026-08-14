"""Tests for wiring the consensus-governor's ``active_regime`` into the
backtest (``strategies.build_regime_exposure_weights``'s mixed-label
handling + ``attribution.run_component_backtests``'s governed_regime_only /
governed_combined components).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.attribution import compute_return_decomposition, run_component_backtests
from src.backtesting.strategies import build_regime_exposure_weights


def test_build_regime_exposure_weights_handles_transitional_string_label():
    idx = pd.bdate_range("2020-01-01", periods=6)
    regime = pd.Series([0, 1, "transitional", 0, "transitional", 1], index=idx)
    weights = build_regime_exposure_weights(regime, {0: 1.0, 1: 0.5, "transitional": 0.25})
    assert weights["benchmark"].tolist() == [1.0, 0.5, 0.25, 1.0, 0.25, 0.5]


def test_build_regime_exposure_weights_warns_and_defaults_when_transitional_unmapped(caplog):
    idx = pd.bdate_range("2020-01-01", periods=3)
    regime = pd.Series([0, "transitional", 0], index=idx)
    with caplog.at_level("WARNING"):
        weights = build_regime_exposure_weights(regime, {0: 1.0})
    assert weights["benchmark"].tolist() == [1.0, 1.0, 1.0]  # unmapped -> default full exposure
    assert any("transitional" in rec.message and "not in exposure_by_regime" in rec.message for rec in caplog.records)


@pytest.fixture
def synthetic_backtest_inputs():
    rng = np.random.default_rng(11)
    n = 300
    dates = pd.bdate_range("2021-01-01", periods=n)
    symbols = ["A", "B", "C"]

    bench_returns = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    benchmark_prices = 100 * np.exp(np.cumsum(bench_returns))

    stock_returns = pd.DataFrame(
        {s: bench_returns.values * 0.7 + rng.normal(0, 0.006, n) for s in symbols}, index=dates
    )
    stock_prices = 100 * np.exp(stock_returns.cumsum())

    regime = pd.Series(rng.integers(0, 3, n), index=dates)
    # active_regime: same as regime, but with a chunk flagged transitional,
    # simulating the governor damping down a flickery stretch
    active_regime = regime.copy().astype(object)
    active_regime.iloc[50:70] = "transitional"

    rebalance_dates = pd.date_range(dates.min(), dates.max(), freq="MS")
    rows = [
        {"date": d, "symbol": s, "composite_score": rng.normal(0, 1)}
        for d in rebalance_dates
        for s in symbols
    ]
    scores_by_date = pd.DataFrame(rows)

    exposure_by_regime = {0: 1.0, 1: 0.6, 2: 0.3, "transitional": 0.25}
    return dict(
        stock_returns=stock_returns,
        benchmark_returns=bench_returns,
        scores_by_date=scores_by_date,
        regime=regime,
        active_regime=active_regime,
        exposure_by_regime=exposure_by_regime,
        stock_prices=stock_prices,
        benchmark_prices=benchmark_prices,
    )


def test_run_component_backtests_without_active_regime_unchanged(synthetic_backtest_inputs):
    kwargs = {k: v for k, v in synthetic_backtest_inputs.items() if k != "active_regime"}
    results = run_component_backtests(engine="custom", **kwargs)
    assert set(results.keys()) == {"benchmark", "regime_only", "fundamentals_only", "combined"}


def test_run_component_backtests_with_active_regime_adds_governed_components(synthetic_backtest_inputs):
    results = run_component_backtests(engine="custom", **synthetic_backtest_inputs)
    assert {"governed_regime_only", "governed_combined"} <= set(results.keys())
    # raw components must be present and untouched (same keys as before)
    assert {"benchmark", "regime_only", "fundamentals_only", "combined"} <= set(results.keys())


def test_governed_components_differ_from_raw_when_governor_actually_changes_exposure(synthetic_backtest_inputs):
    results = run_component_backtests(engine="custom", **synthetic_backtest_inputs)
    raw_returns = results["regime_only"]["returns"]
    governed_returns = results["governed_regime_only"]["returns"]
    # the injected transitional stretch (days 50-70) should cause SOME
    # divergence between raw and governed exposure -- they must not be
    # numerically identical series
    assert not raw_returns.equals(governed_returns)


def test_compute_return_decomposition_includes_governed_fields_only_when_present(synthetic_backtest_inputs):
    kwargs_no_gov = {k: v for k, v in synthetic_backtest_inputs.items() if k != "active_regime"}
    results_no_gov = run_component_backtests(engine="custom", **kwargs_no_gov)
    decomp_no_gov = compute_return_decomposition(results_no_gov)
    assert "governed_regime_contribution" not in decomp_no_gov
    assert "governed_vs_raw_combined_delta" not in decomp_no_gov

    results_gov = run_component_backtests(engine="custom", **synthetic_backtest_inputs)
    decomp_gov = compute_return_decomposition(results_gov)
    assert "governed_regime_contribution" in decomp_gov
    assert "governed_vs_raw_combined_delta" in decomp_gov
    assert decomp_gov["governed_combined_cagr"] == pytest.approx(
        decomp_gov["combined_cagr"] + decomp_gov["governed_vs_raw_combined_delta"], abs=1e-9
    )
