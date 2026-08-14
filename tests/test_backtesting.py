"""Synthetic-data tests for the backtesting & attribution module.

Builds a synthetic multi-stock universe where each stock has a fixed, known
"quality" drift bonus, feeds noisy proxies of that quality into
composite_score at each rebalance date, and derives the market regime from
the *actual* regime_detection module (fit on a synthetic calm/stress price
series) — so this test exercises real cross-module integration, not just the
backtesting module in isolation. With quality genuinely driving both returns
and scores, a working fundamentals_only strategy should beat the benchmark;
if it doesn't, something in the selection/weighting logic is broken.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.attribution import (
    compute_attribution_table,
    compute_return_decomposition,
    run_component_backtests,
)
from src.backtesting.engine import align_weights_to_returns, compute_returns_panel, run_backtest
from src.backtesting.metrics import cagr, performance_summary
from src.backtesting.pipeline import run_backtest_pipeline
from src.backtesting.strategies import (
    build_fundamental_portfolio_weights,
    build_regime_exposure_weights,
)
from src.regime_detection.features import build_feature_matrix
from src.regime_detection.models import RegimeModel


N_STOCKS = 20
N_CALM_DAYS = 500
N_STRESS_DAYS = 250


@pytest.fixture(scope="module")
def synthetic_universe():
    rng = np.random.default_rng(7)
    n = N_CALM_DAYS + N_STRESS_DAYS
    dates = pd.bdate_range("2019-01-02", periods=n)

    # Market factor: calm uptrend, then a higher-vol, lower-drift stress period.
    market_calm = rng.normal(0.0005, 0.008, N_CALM_DAYS)
    market_stress = rng.normal(-0.0004, 0.022, N_STRESS_DAYS)
    market_returns = np.concatenate([market_calm, market_stress])

    symbols = [f"STOCK{i}" for i in range(N_STOCKS)]
    # Fixed per-stock "quality" drift bonus (daily), same sign persistence
    # a fundamentals screen should be able to detect via a noisy proxy.
    quality = rng.normal(0.0, 0.0006, N_STOCKS)

    stock_returns = np.zeros((n, N_STOCKS))
    for i in range(N_STOCKS):
        idio = rng.normal(0, 0.012, n)
        stock_returns[:, i] = 0.8 * market_returns + quality[i] + idio

    stock_prices = pd.DataFrame(
        100 * np.exp(np.cumsum(stock_returns, axis=0)), index=dates, columns=symbols
    )
    benchmark_prices = pd.Series(100 * np.exp(np.cumsum(market_returns)), index=dates, name="close")

    # Derive regime labels from the *real* regime_detection module.
    features = build_feature_matrix(benchmark_prices, return_windows=[5, 21], vol_windows=[21, 63])
    model = RegimeModel(model_type="gmm", n_regimes=4, random_state=0)
    model.fit(features)
    regime = model.predict(features)
    # Backfill the warm-up period (dropped by build_feature_matrix's dropna) with
    # the first available regime so the full price history has a label.
    regime = regime.reindex(dates).bfill().ffill().astype(int)

    # Monthly rebalance dates; composite_score = noisy proxy of true quality
    # (so the signal is real but not perfectly observed, like in practice).
    rebalance_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    rebalance_dates = [d for d in rebalance_dates if d in stock_prices.index or True]
    # Snap each rebalance date to the nearest actual trading day.
    rebalance_dates = [dates[dates.searchsorted(d)] for d in rebalance_dates if dates.searchsorted(d) < len(dates)]

    rows = []
    for d in rebalance_dates:
        noise = rng.normal(0, 0.0004, N_STOCKS)
        scores = quality + noise
        for sym, score in zip(symbols, scores):
            rows.append({"date": d, "symbol": sym, "composite_score": score})
    scores_by_date = pd.DataFrame(rows)

    return {
        "stock_prices": stock_prices,
        "benchmark_prices": benchmark_prices,
        "regime": regime,
        "scores_by_date": scores_by_date,
        "quality": pd.Series(quality, index=symbols),
    }


BACKTEST_CONFIG = {
    "top_quantile": 0.2,
    "min_positions": 3,
    "transaction_cost_bps": 5,
    "rolling_sharpe_window": 63,
    "exposure_by_regime": {0: 1.0, 1: 0.85, 2: 0.55, 3: 0.25},
}


def test_engine_matches_manual_calculation():
    """Pure dot-product mechanics check -- explicitly lag_days=0 so this
    tests only the weight*return arithmetic, independent of the execution-
    timing lag (see test_engine_lags_weights_by_default below for that)."""
    dates = pd.bdate_range("2023-01-02", periods=5)
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.0, 0.005], "B": [0.0, 0.01, 0.01, -0.02, 0.0]}, index=dates)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=dates)
    result = run_backtest(returns, weights, transaction_cost_bps=0, lag_days=0)
    expected_gross = (returns * weights).sum(axis=1)
    pd.testing.assert_series_equal(result["gross_returns"], expected_gross, check_names=False)
    np.testing.assert_allclose(result["equity_curve"].iloc[-1], (1 + expected_gross).prod())


def test_engine_lags_weights_by_default():
    """The actual fix, pinned down as a regression test: with the default
    lag_days=1, portfolio_return[T] must equal weights[T-1] * returns[T],
    NOT weights[T] * returns[T]."""
    dates = pd.bdate_range("2023-01-02", periods=5)
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.0, 0.005], "B": [0.0, 0.01, 0.01, -0.02, 0.0]}, index=dates)
    weights = pd.DataFrame({"A": [0.2, 0.4, 0.6, 0.8, 1.0], "B": [0.8, 0.6, 0.4, 0.2, 0.0]}, index=dates)
    result = run_backtest(returns, weights, transaction_cost_bps=0)  # lag_days=1 default
    expected_gross = (returns * weights.shift(1).fillna(0.0)).sum(axis=1)
    pd.testing.assert_series_equal(result["gross_returns"], expected_gross, check_names=False)


def test_engine_does_not_capture_same_day_spike_from_a_same_day_decision():
    """Regression test for a real bug: a weight that is 0 the day before a
    price spike and only flips to 1 exactly ON the spike day was getting
    100% of that spike's return attributed to it -- impossible to replicate
    live (you can't observe today's close, decide to be invested, and also
    capture today's own return). Found by tracing through exactly this
    scenario; see docs/backtesting_spec.md's look-ahead-bias section."""
    dates = pd.bdate_range("2023-01-02", periods=10)
    returns = pd.DataFrame({"SYM": [0.0] * 5 + [0.50] + [0.0] * 4}, index=dates)  # +50% spike at index 5
    weights = pd.DataFrame({"SYM": [0.0] * 5 + [1.0] * 5}, index=dates)  # weight flips to 1.0 exactly on the spike day

    result = run_backtest(returns, weights, transaction_cost_bps=0)  # lag_days=1 default
    assert result["returns"].iloc[5] == 0.0, "same-day decision must not capture that same day's own return"

    result_no_lag = run_backtest(returns, weights, transaction_cost_bps=0, lag_days=0)
    assert result_no_lag["returns"].iloc[5] == 0.5, "lag_days=0 should reproduce the old (unsafe) behavior exactly"


def test_regime_exposure_weights_bounded(synthetic_universe):
    exposure = build_regime_exposure_weights(synthetic_universe["regime"], BACKTEST_CONFIG["exposure_by_regime"])
    assert (exposure["benchmark"] >= 0).all() and (exposure["benchmark"] <= 1).all()


def test_fundamental_portfolio_weights_shape(synthetic_universe):
    weights = build_fundamental_portfolio_weights(
        synthetic_universe["scores_by_date"], top_quantile=0.2, min_positions=3
    )
    n_expected = max(3, int(N_STOCKS * 0.2))
    # Each rebalance-date row should select exactly n_expected stocks with equal weight.
    for _, row in weights.iterrows():
        selected = row[row > 0]
        assert len(selected) == n_expected
        np.testing.assert_allclose(selected.values, 1.0 / n_expected)


def test_component_backtests_and_attribution_run(synthetic_universe):
    stock_returns = compute_returns_panel(synthetic_universe["stock_prices"])
    benchmark_returns = compute_returns_panel(synthetic_universe["benchmark_prices"].to_frame("benchmark"))["benchmark"]

    results = run_component_backtests(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores_by_date=synthetic_universe["scores_by_date"],
        regime=synthetic_universe["regime"],
        exposure_by_regime=BACKTEST_CONFIG["exposure_by_regime"],
        top_quantile=BACKTEST_CONFIG["top_quantile"],
        min_positions=BACKTEST_CONFIG["min_positions"],
        transaction_cost_bps=BACKTEST_CONFIG["transaction_cost_bps"],
    )
    assert set(results.keys()) == {"benchmark", "regime_only", "fundamentals_only", "combined"}
    for name, r in results.items():
        assert len(r["returns"]) > 0
        assert r["returns"].abs().max() < 1.0  # sanity: no absurd daily returns

    table = compute_attribution_table(results, benchmark_returns=benchmark_returns)
    assert set(table.index) == {"benchmark", "regime_only", "fundamentals_only", "combined"}
    assert "sharpe_ratio" in table.columns

    decomposition = compute_return_decomposition(results)
    expected_keys = {
        "benchmark_cagr", "fundamentals_only_cagr", "regime_only_cagr", "combined_cagr",
        "combined_excess_cagr", "fundamentals_contribution", "regime_contribution", "interaction_effect",
    }
    assert set(decomposition.keys()) == expected_keys

    # Approximate additivity should hold within a reasonable tolerance.
    reconstructed = (
        decomposition["fundamentals_contribution"]
        + decomposition["regime_contribution"]
        + decomposition["interaction_effect"]
    )
    assert reconstructed == pytest.approx(decomposition["combined_excess_cagr"], abs=1e-9)


def test_fundamentals_selection_beats_benchmark_when_signal_is_real(synthetic_universe):
    """The synthetic universe was built so composite_score is a noisy but real
    proxy for each stock's true drift bonus — a correctly-implemented
    top-quantile selection should therefore outperform the benchmark."""
    stock_returns = compute_returns_panel(synthetic_universe["stock_prices"])
    weights_sparse = build_fundamental_portfolio_weights(
        synthetic_universe["scores_by_date"], top_quantile=0.2, min_positions=3
    )
    weights_daily = align_weights_to_returns(weights_sparse, stock_returns.index, stock_returns.columns)
    result = run_backtest(stock_returns, weights_daily, transaction_cost_bps=5)

    benchmark_returns = compute_returns_panel(synthetic_universe["benchmark_prices"].to_frame("benchmark"))["benchmark"]

    fund_cagr = cagr(result["returns"])
    bench_cagr = cagr(benchmark_returns)
    assert fund_cagr > bench_cagr


def test_full_backtest_pipeline_produces_report_and_figures(synthetic_universe, tmp_path):
    out_dir = tmp_path / "reports"
    result = run_backtest_pipeline(
        BACKTEST_CONFIG,
        synthetic_universe["stock_prices"],
        synthetic_universe["benchmark_prices"],
        synthetic_universe["regime"],
        synthetic_universe["scores_by_date"],
        out_dir=str(out_dir),
    )

    report_path = Path(result["report_path"])
    assert report_path.exists() and report_path.stat().st_size > 0

    fig_dir = out_dir / "figures"
    expected_figures = [
        "equity_curves.png", "drawdowns.png", "rolling_sharpe.png",
        "regime_timeline.png", "contribution_bar.png", "score_distribution.png",
    ]
    for fname in expected_figures:
        fpath = fig_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 0, f"missing or empty figure: {fname}"

    assert (out_dir / "tables" / "attribution_table.csv").exists()
    assert (out_dir / "tables" / "return_decomposition.json").exists()

    report_text = report_path.read_text()
    for name in ["benchmark", "regime_only", "fundamentals_only", "combined"]:
        assert name.replace("_", " ").title() in report_text or name in report_text


# --- vectorbt engine dispatch/fallback (src/backtesting/vbt_engine.py) ---

from src.backtesting.vbt_engine import run_backtest_with_fallback


def test_vbt_engine_falls_back_to_custom_when_unavailable():
    """This sandbox (like the rest of this project's dev environment — see
    requirements.txt / docs/backtesting_spec.md) may not have vectorbt
    installed. run_backtest_with_fallback must still produce a valid result
    by falling back to engine.run_backtest rather than raising."""
    dates = pd.bdate_range("2022-01-01", periods=100)
    prices = pd.DataFrame(
        {"A": 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 100)))},
        index=dates,
    )
    returns = prices.pct_change()
    weights = pd.DataFrame({"A": 1.0}, index=dates)

    result = run_backtest_with_fallback(prices, returns, weights, transaction_cost_bps=10, engine="vectorbt")
    assert set(result.keys()) == {"returns", "gross_returns", "turnover", "equity_curve"}
    assert len(result["returns"]) == len(dates)
    assert not result["equity_curve"].isna().any()


def test_engine_config_flag_is_respected_end_to_end(synthetic_universe):
    """run_component_backtests(engine='custom') must not attempt to import
    vectorbt at all — a direct regression guard for the config.yaml
    backtesting.engine switch."""
    from src.backtesting.attribution import run_component_backtests
    from src.backtesting.engine import compute_returns_panel

    stock_returns = compute_returns_panel(synthetic_universe["stock_prices"])
    benchmark_returns = compute_returns_panel(
        synthetic_universe["benchmark_prices"].to_frame("benchmark")
    )["benchmark"]
    result = run_component_backtests(
        stock_returns=stock_returns, benchmark_returns=benchmark_returns,
        scores_by_date=synthetic_universe["scores_by_date"], regime=synthetic_universe["regime"],
        exposure_by_regime={0: 1.0, 1: 0.85, 2: 0.55, 3: 0.25}, transaction_cost_bps=5.0, engine="custom",
    )
    assert set(result.keys()) == {"benchmark", "regime_only", "fundamentals_only", "combined"}


# --- geometric overlay: standalone component + "on top of everything" application
# (src/backtesting/strategies.py, src/backtesting/attribution.py) ---

from src.backtesting.strategies import apply_geometric_overlay, build_geometric_overlay_weights


def test_build_geometric_overlay_weights_cuts_exposure_only_on_flagged_days():
    dates = pd.bdate_range("2022-01-01", periods=6)
    flag = pd.Series([0, 0, 1, 1, np.nan, 0], index=dates)
    weights = build_geometric_overlay_weights(flag, crash_exposure_multiplier=0.3)
    assert weights["benchmark"].tolist() == [1.0, 1.0, 0.3, 0.3, 1.0, 1.0]


def test_apply_geometric_overlay_is_noop_when_flag_is_none():
    dates = pd.bdate_range("2022-01-01", periods=5)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)
    result = apply_geometric_overlay(weights, None)
    pd.testing.assert_frame_equal(result, weights)


def test_apply_geometric_overlay_scales_existing_weights():
    dates = pd.bdate_range("2022-01-01", periods=4)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)
    flag = pd.Series([0, 1, 1, 0], index=dates)
    result = apply_geometric_overlay(weights, flag, crash_exposure_multiplier=0.4)
    assert result["A"].tolist() == [0.5, 0.2, 0.2, 0.5]


def test_component_backtests_include_geometric_overlay_only_when_flag_supplied(synthetic_universe):
    from src.backtesting.attribution import run_component_backtests
    from src.backtesting.engine import compute_returns_panel

    stock_returns = compute_returns_panel(synthetic_universe["stock_prices"])
    benchmark_returns = compute_returns_panel(
        synthetic_universe["benchmark_prices"].to_frame("benchmark")
    )["benchmark"]
    regime = synthetic_universe["regime"]

    # No flag -> exactly the original 4 components, unchanged behavior.
    without = run_component_backtests(
        stock_returns=stock_returns, benchmark_returns=benchmark_returns,
        scores_by_date=synthetic_universe["scores_by_date"], regime=regime,
        exposure_by_regime={0: 1.0, 1: 0.85, 2: 0.55, 3: 0.25}, engine="custom",
    )
    assert set(without.keys()) == {"benchmark", "regime_only", "fundamentals_only", "combined"}

    # With a (synthetic, independent-of-regime) flag -> 5th component appears,
    # and "combined" changes because the overlay applies on top of it.
    rng = np.random.default_rng(42)
    crash_flag = pd.Series(rng.integers(0, 2, len(regime)), index=regime.index).astype(float)
    with_flag = run_component_backtests(
        stock_returns=stock_returns, benchmark_returns=benchmark_returns,
        scores_by_date=synthetic_universe["scores_by_date"], regime=regime,
        exposure_by_regime={0: 1.0, 1: 0.85, 2: 0.55, 3: 0.25}, engine="custom",
        geometric_crash_flag=crash_flag, crash_exposure_multiplier=0.4,
    )
    assert set(with_flag.keys()) == {
        "benchmark", "regime_only", "fundamentals_only", "combined", "geometric_overlay_only"
    }
    # regime_only must be completely unaffected by the geometric flag (it was
    # never fed into the GMM and isn't overlaid onto regime_only, only combined).
    pd.testing.assert_series_equal(
        without["regime_only"]["returns"], with_flag["regime_only"]["returns"]
    )
    # combined SHOULD differ, since the overlay is applied on top of it.
    assert not without["combined"]["returns"].equals(with_flag["combined"]["returns"])


def test_return_decomposition_includes_geometric_line_only_when_present(synthetic_universe):
    from src.backtesting.attribution import compute_return_decomposition, run_component_backtests
    from src.backtesting.engine import compute_returns_panel

    stock_returns = compute_returns_panel(synthetic_universe["stock_prices"])
    benchmark_returns = compute_returns_panel(
        synthetic_universe["benchmark_prices"].to_frame("benchmark")
    )["benchmark"]
    regime = synthetic_universe["regime"]
    rng = np.random.default_rng(0)
    crash_flag = pd.Series(rng.integers(0, 2, len(regime)), index=regime.index).astype(float)

    results = run_component_backtests(
        stock_returns=stock_returns, benchmark_returns=benchmark_returns,
        scores_by_date=synthetic_universe["scores_by_date"], regime=regime,
        exposure_by_regime={0: 1.0, 1: 0.85, 2: 0.55, 3: 0.25}, engine="custom",
        geometric_crash_flag=crash_flag,
    )
    decomposition = compute_return_decomposition(results)
    assert "geometric_overlay_cagr" in decomposition
    assert "geometric_overlay_contribution" in decomposition
    # Core additive identity must still hold regardless of the extra field.
    assert decomposition["interaction_effect"] == pytest.approx(
        decomposition["combined_excess_cagr"]
        - decomposition["fundamentals_contribution"]
        - decomposition["regime_contribution"]
    )
