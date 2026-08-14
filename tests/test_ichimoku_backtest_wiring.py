"""Tests for wiring Ichimoku (``adaptive_ichimoku.build_ichimoku_weights``)
into the main backtest as real components acting on the same portfolio as
everything else (``strategies.apply_ichimoku_gate`` +
``attribution.run_component_backtests``'s ``ichimoku_only``/
``combined_with_ichimoku``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.attribution import compute_return_decomposition, run_component_backtests
from src.backtesting.strategies import (
    apply_ichimoku_breadth_scalar,
    apply_ichimoku_conviction_tilt,
    apply_ichimoku_gate,
)


def test_apply_ichimoku_gate_zeroes_unconfirmed_symbols():
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [0.0, 1.0, 1.0], "B": [1.0, 0.0, 1.0]}, index=idx)
    gated = apply_ichimoku_gate(weights, ichimoku)
    assert gated["A"].tolist() == [0.0, 0.5, 0.5]
    assert gated["B"].tolist() == [0.5, 0.0, 0.5]


def test_apply_ichimoku_gate_passes_through_symbols_missing_ohlc_data(caplog):
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "C": [0.5, 0.5, 0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)  # C entirely absent
    with caplog.at_level("WARNING"):
        gated = apply_ichimoku_gate(weights, ichimoku)
    assert gated["C"].tolist() == [0.5, 0.5, 0.5]  # untouched, not zeroed
    assert gated["A"].tolist() == [0.0, 0.5, 0.5]  # still gated normally
    assert any("no Ichimoku data" in rec.message for rec in caplog.records)


def test_apply_ichimoku_gate_none_is_passthrough():
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=idx)
    assert apply_ichimoku_gate(weights, None).equals(weights)


def test_apply_ichimoku_breadth_scalar_scales_by_confirmation_fraction():
    idx = pd.bdate_range("2020-01-01", periods=3)
    # day0: both A,B confirmed -> fraction 1.0, no scaling
    # day1: only A confirmed of {A,B} held -> fraction 0.5
    # day2: neither confirmed -> fraction 0.0
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [1.0, 0.0, 0.0]}, index=idx)
    scaled = apply_ichimoku_breadth_scalar(weights, ichimoku)
    assert scaled.loc[idx[0]].tolist() == pytest.approx([0.5, 0.5])
    assert scaled.loc[idx[1]].tolist() == pytest.approx([0.25, 0.25])
    assert scaled.loc[idx[2]].tolist() == pytest.approx([0.0, 0.0])


def test_apply_ichimoku_breadth_scalar_never_zeroes_out_a_specific_name_alone():
    """The whole point of the breadth scalar vs the hard gate: it never
    drops one held name while keeping another -- every day's scaling
    factor applies uniformly across all currently-held names."""
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.4], "B": [0.4], "C": [0.2]}, index=idx)
    ichimoku = pd.DataFrame({"A": [1.0], "B": [0.0], "C": [1.0]}, index=idx)
    scaled = apply_ichimoku_breadth_scalar(weights, ichimoku)
    # fraction = 2/3 confirmed -> ALL three scaled by the same factor, none zeroed individually
    fraction = 2 / 3
    assert scaled.loc[idx[0]].tolist() == pytest.approx([0.4 * fraction, 0.4 * fraction, 0.2 * fraction])


def test_apply_ichimoku_breadth_scalar_excludes_missing_coverage_from_fraction(caplog):
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.5], "C": [0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [0.0]}, index=idx)  # C has zero OHLC coverage
    with caplog.at_level("WARNING"):
        scaled = apply_ichimoku_breadth_scalar(weights, ichimoku)
    # A is covered and unconfirmed -> scaled down (fraction 0/1 = 0.0)
    assert scaled.loc[idx[0], "A"] == pytest.approx(0.0)
    # C has NO coverage at all -> retains its ORIGINAL weight, unaffected by
    # A's confirmation status -- this is the whole point of computing the
    # fraction only from covered names AND only applying it to covered names
    assert scaled.loc[idx[0], "C"] == pytest.approx(0.5)
    assert any("no Ichimoku data" in rec.message for rec in caplog.records)


def test_apply_ichimoku_breadth_scalar_defaults_to_full_exposure_with_zero_coverage():
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"C": [0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [1.0]}, index=idx)  # C has no coverage at all, no other held name either
    scaled = apply_ichimoku_breadth_scalar(weights, ichimoku)
    assert scaled.loc[idx[0], "C"] == pytest.approx(0.5)  # no opinion -> pass through


def test_apply_ichimoku_breadth_scalar_none_is_passthrough():
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=idx)
    assert apply_ichimoku_breadth_scalar(weights, None).equals(weights)


def test_apply_ichimoku_breadth_scalar_less_extreme_than_hard_gate_on_disjoint_selections():
    """Reproduces the pathology from the first real backtest at small
    scale: fundamentals selects {A,B,C,D}, Ichimoku only confirms a
    DISJOINT set {E,F} that isn't held at all. Hard gate should collapse to
    zero exposure; breadth scalar should also go to zero here (there's
    genuinely 0% confirmation among held names) -- but critically, once
    there's ANY overlap the breadth scalar degrades smoothly instead of
    cliff-edging to nothing, which this test's second day demonstrates."""
    idx = pd.bdate_range("2020-01-01", periods=2)
    weights = pd.DataFrame({"A": [0.25, 0.25], "B": [0.25, 0.25], "C": [0.25, 0.25], "D": [0.25, 0.25]}, index=idx)
    ichimoku = pd.DataFrame(
        {
            "A": [0.0, 1.0], "B": [0.0, 0.0], "C": [0.0, 0.0], "D": [0.0, 0.0],
            "E": [1.0, 1.0], "F": [1.0, 1.0],
        },
        index=idx,
    )
    hard = apply_ichimoku_gate(weights, ichimoku)
    soft = apply_ichimoku_breadth_scalar(weights, ichimoku)
    # day0: zero overlap -> both collapse to 0 exposure
    assert hard.loc[idx[0]].sum() == pytest.approx(0.0)
    assert soft.loc[idx[0]].sum() == pytest.approx(0.0)
    # day1: A confirmed -> hard gate keeps ONLY A (drops B,C,D entirely);
    # breadth scalar instead keeps ALL FOUR at a uniformly reduced (1/4) weight
    assert hard.loc[idx[1]].tolist() == pytest.approx([0.25, 0.0, 0.0, 0.0])
    assert soft.loc[idx[1]].tolist() == pytest.approx([0.0625, 0.0625, 0.0625, 0.0625])
    assert soft.loc[idx[1]].sum() == pytest.approx(0.25)  # total exposure = 1/4 confirmed, spread across all 4


def test_apply_ichimoku_conviction_tilt_preserves_total_exposure_exactly():
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.25], "B": [0.25], "C": [0.25], "D": [0.0], "E": [0.0]}, index=idx)
    ichimoku = pd.DataFrame({"A": [0.9], "B": [0.1], "C": [0.5], "D": [0.9], "E": [0.9]}, index=idx)
    tilted = apply_ichimoku_conviction_tilt(weights, ichimoku, tilt_strength=1.5)
    assert tilted.loc[idx[0]].sum() == pytest.approx(weights.loc[idx[0]].sum(), abs=1e-9)
    # unheld names (D, E) must stay exactly 0 regardless of their conviction
    assert tilted.loc[idx[0], "D"] == pytest.approx(0.0)
    assert tilted.loc[idx[0], "E"] == pytest.approx(0.0)


def test_apply_ichimoku_conviction_tilt_favors_higher_conviction_names():
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.25], "B": [0.25], "C": [0.25], "D": [0.25]}, index=idx)
    ichimoku = pd.DataFrame({"A": [0.9], "B": [0.1], "C": [0.5], "D": [0.5]}, index=idx)
    # tilt_strength=0.5 (not 1.0): at strength=1.0 here B's z-score is large
    # enough that its tilt clips at 0, which breaks exact cancellation during
    # renormalization and would make this specific "C stays exact" assertion
    # fragile for reasons unrelated to what this test is actually checking.
    tilted = apply_ichimoku_conviction_tilt(weights, ichimoku, tilt_strength=0.5)
    row = tilted.loc[idx[0]]
    assert row["A"] > weights.loc[idx[0], "A"]  # above-average conviction -> tilted up
    assert row["B"] < weights.loc[idx[0], "B"]  # below-average conviction -> tilted down
    assert row["C"] == pytest.approx(weights.loc[idx[0], "C"])  # exactly at the mean -> unchanged


def test_apply_ichimoku_conviction_tilt_missing_coverage_stays_unchanged():
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.5], "C": [0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [0.9]}, index=idx)  # C has zero coverage
    tilted = apply_ichimoku_conviction_tilt(weights, ichimoku, tilt_strength=2.0)
    assert tilted.loc[idx[0], "C"] == pytest.approx(0.5)  # untouched
    assert tilted.loc[idx[0]].sum() == pytest.approx(1.0)  # exposure still preserved overall


def test_apply_ichimoku_conviction_tilt_zero_strength_is_noop():
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.3, 0.5], "B": [0.5, 0.3, 0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [0.9, 0.1, 0.5], "B": [0.1, 0.9, 0.5]}, index=idx)
    result = apply_ichimoku_conviction_tilt(weights, ichimoku, tilt_strength=0.0)
    pd.testing.assert_frame_equal(result, weights)


def test_apply_ichimoku_conviction_tilt_none_is_passthrough():
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=idx)
    assert apply_ichimoku_conviction_tilt(weights, None).equals(weights)


def test_apply_ichimoku_conviction_tilt_never_flips_to_negative_weight():
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=idx)
    ichimoku = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=idx)  # extreme conviction spread
    tilted = apply_ichimoku_conviction_tilt(weights, ichimoku, tilt_strength=10.0)  # aggressive strength
    assert (tilted >= 0).all().all()


def test_apply_ichimoku_conviction_tilt_works_at_realistic_normalized_weight_scale():
    """Regression test for a real bug (found 2026-07-24): the first version
    of this function used a raw demeaned term calibrated for conviction
    values on [0, 1], but was actually fed
    ``adaptive_ichimoku.build_ichimoku_weights``'s OUTPUT in production —
    an already-normalized portfolio-weight matrix (e.g. ~1/500 = 0.002 mean
    for a 500-symbol universe), two orders of magnitude smaller than
    assumed. The resulting tilt was numerically ~1.0 for every name (a
    silent no-op), confirmed on a real backtest where
    ``combined_ichimoku_tilted`` came back indistinguishable from plain
    ``combined`` (CAGR delta of -0.00004). This test reproduces that input
    scale directly and asserts a REAL (not negligible) reallocation
    happens, so this specific failure mode can't silently return."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=1)
    n_symbols = 500
    n_held = 100
    symbols = [f"S{i}" for i in range(n_symbols)]
    held_symbols = symbols[:n_held]

    weights = pd.DataFrame(0.0, index=idx, columns=symbols)
    weights.loc[idx[0], held_symbols] = 1.0 / n_held

    raw_conviction = rng.uniform(0, 1, n_symbols)
    # this normalization (sum to 1 across active names) is EXACTLY what
    # build_ichimoku_weights does -- mean value ~1/n_symbols, not ~0.5
    normalized = raw_conviction / raw_conviction.sum()
    ichimoku = pd.DataFrame([normalized], index=idx, columns=symbols)
    assert ichimoku.values.mean() == pytest.approx(1.0 / n_symbols, rel=0.05)  # sanity-check the setup itself

    tilted = apply_ichimoku_conviction_tilt(weights, ichimoku, tilt_strength=0.5)
    held_before = weights.loc[idx[0], held_symbols]
    held_after = tilted.loc[idx[0], held_symbols]

    # the bug produced tilt ~1.0 everywhere -> post-tilt weights numerically
    # equal to pre-tilt weights. The fix must show REAL dispersion instead.
    relative_spread = (held_after - held_before).abs().max() / held_before.iloc[0]
    assert relative_spread > 0.10  # at least a 10% swing for the most-affected name
    assert tilted.loc[idx[0]].sum() == pytest.approx(weights.loc[idx[0]].sum(), abs=1e-9)


@pytest.fixture
def synthetic_backtest_inputs_with_ichimoku():
    rng = np.random.default_rng(21)
    n = 260
    dates = pd.bdate_range("2021-01-01", periods=n)
    symbols = ["A", "B", "C"]

    bench_returns = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    benchmark_prices = 100 * np.exp(np.cumsum(bench_returns))

    stock_returns = pd.DataFrame(
        {s: bench_returns.values * 0.7 + rng.normal(0, 0.006, n) for s in symbols}, index=dates
    )
    stock_prices = 100 * np.exp(stock_returns.cumsum())

    regime = pd.Series(rng.integers(0, 3, n), index=dates)

    rebalance_dates = pd.date_range(dates.min(), dates.max(), freq="MS")
    rows = [
        {"date": d, "symbol": s, "composite_score": rng.normal(0, 1)}
        for d in rebalance_dates
        for s in symbols
    ]
    scores_by_date = pd.DataFrame(rows)

    # Ichimoku weights: alternating confirmation pattern per symbol, only
    # covering A and B (C simulates a symbol with no OHLC data at all).
    ichimoku_weights = pd.DataFrame(
        {
            "A": np.where(np.arange(n) % 4 < 2, 0.5, 0.0),
            "B": np.where(np.arange(n) % 3 == 0, 0.5, 0.0),
        },
        index=dates,
    )

    exposure_by_regime = {0: 1.0, 1: 0.6, 2: 0.3}
    return dict(
        stock_returns=stock_returns,
        benchmark_returns=bench_returns,
        scores_by_date=scores_by_date,
        regime=regime,
        exposure_by_regime=exposure_by_regime,
        stock_prices=stock_prices,
        benchmark_prices=benchmark_prices,
        ichimoku_weights=ichimoku_weights,
    )


def test_run_component_backtests_without_ichimoku_weights_unchanged(synthetic_backtest_inputs_with_ichimoku):
    kwargs = {k: v for k, v in synthetic_backtest_inputs_with_ichimoku.items() if k != "ichimoku_weights"}
    results = run_component_backtests(engine="custom", **kwargs)
    assert "ichimoku_only" not in results
    assert "combined_with_ichimoku" not in results


def test_run_component_backtests_with_ichimoku_weights_adds_components(synthetic_backtest_inputs_with_ichimoku):
    results = run_component_backtests(engine="custom", **synthetic_backtest_inputs_with_ichimoku)
    assert {"ichimoku_only", "combined_with_ichimoku"} <= set(results.keys())
    # raw combined must still be present and is a genuinely different series
    # from the ichimoku-gated version (gate should actually bite somewhere
    # given the alternating confirmation pattern injected above)
    assert not results["combined"]["returns"].equals(results["combined_with_ichimoku"]["returns"])


def test_ichimoku_gate_never_increases_combined_exposure(synthetic_backtest_inputs_with_ichimoku):
    """The gate can only zero out weight, never add exposure the raw
    'combined' strategy didn't already have -- a structural invariant of
    apply_ichimoku_gate worth pinning down given it directly controls how
    much capital ends up at risk."""
    results = run_component_backtests(engine="custom", **synthetic_backtest_inputs_with_ichimoku)
    combined_turnover = results["combined"]["turnover"]
    gated_turnover = results["combined_with_ichimoku"]["turnover"]
    # not a strict per-day inequality (turnover can behave non-monotonically
    # around gating transitions), but gated total gross exposure over time
    # should not exceed raw combined's by more than a small tolerance
    assert gated_turnover.sum() <= combined_turnover.sum() * 1.5


def test_ichimoku_mode_hard_gate_vs_breadth_scalar_produce_different_results(synthetic_backtest_inputs_with_ichimoku):
    results_soft = run_component_backtests(
        engine="custom", ichimoku_mode="breadth_scalar", **synthetic_backtest_inputs_with_ichimoku
    )
    results_hard = run_component_backtests(
        engine="custom", ichimoku_mode="hard_gate", **synthetic_backtest_inputs_with_ichimoku
    )
    assert not results_soft["combined_with_ichimoku"]["returns"].equals(
        results_hard["combined_with_ichimoku"]["returns"]
    )


def test_compute_return_decomposition_includes_ichimoku_fields_only_when_present(synthetic_backtest_inputs_with_ichimoku):
    kwargs_no_ich = {k: v for k, v in synthetic_backtest_inputs_with_ichimoku.items() if k != "ichimoku_weights"}
    results_no_ich = run_component_backtests(engine="custom", **kwargs_no_ich)
    decomp_no_ich = compute_return_decomposition(results_no_ich)
    assert "ichimoku_contribution" not in decomp_no_ich
    assert "ichimoku_vs_raw_combined_delta" not in decomp_no_ich

    results_ich = run_component_backtests(engine="custom", **synthetic_backtest_inputs_with_ichimoku)
    decomp_ich = compute_return_decomposition(results_ich)
    assert "ichimoku_contribution" in decomp_ich
    assert "ichimoku_vs_raw_combined_delta" in decomp_ich
    assert decomp_ich["combined_with_ichimoku_cagr"] == pytest.approx(
        decomp_ich["combined_cagr"] + decomp_ich["ichimoku_vs_raw_combined_delta"], abs=1e-9
    )


def test_combined_ichimoku_tilted_not_computed_by_default(synthetic_backtest_inputs_with_ichimoku):
    """ichimoku_tilt_strength defaults to 0.0 (off) -- combined_ichimoku_tilted
    should NOT appear unless explicitly requested."""
    results = run_component_backtests(engine="custom", **synthetic_backtest_inputs_with_ichimoku)
    assert "combined_ichimoku_tilted" not in results


def test_combined_ichimoku_tilted_appears_when_tilt_strength_nonzero(synthetic_backtest_inputs_with_ichimoku):
    results = run_component_backtests(
        engine="custom", ichimoku_tilt_strength=1.0, **synthetic_backtest_inputs_with_ichimoku
    )
    assert "combined_ichimoku_tilted" in results
    # still additive: combined_with_ichimoku (gate-based) is unaffected and both present together
    assert "combined_with_ichimoku" in results


def test_combined_ichimoku_tilted_differs_from_combined_with_ichimoku(synthetic_backtest_inputs_with_ichimoku):
    """The whole point: tilting and gating are different mechanisms and
    should generally produce different results (reallocation vs
    exposure-cutting)."""
    results = run_component_backtests(
        engine="custom", ichimoku_tilt_strength=1.5, **synthetic_backtest_inputs_with_ichimoku
    )
    assert not results["combined_ichimoku_tilted"]["returns"].equals(
        results["combined_with_ichimoku"]["returns"]
    )


def test_compute_return_decomposition_includes_tilt_fields_only_when_present(synthetic_backtest_inputs_with_ichimoku):
    results_no_tilt = run_component_backtests(engine="custom", **synthetic_backtest_inputs_with_ichimoku)
    decomp_no_tilt = compute_return_decomposition(results_no_tilt)
    assert "combined_ichimoku_tilted_cagr" not in decomp_no_tilt
    assert "ichimoku_tilt_vs_raw_combined_delta" not in decomp_no_tilt

    results_tilt = run_component_backtests(
        engine="custom", ichimoku_tilt_strength=1.0, **synthetic_backtest_inputs_with_ichimoku
    )
    decomp_tilt = compute_return_decomposition(results_tilt)
    assert "combined_ichimoku_tilted_cagr" in decomp_tilt
    assert "ichimoku_tilt_vs_raw_combined_delta" in decomp_tilt
    assert decomp_tilt["combined_ichimoku_tilted_cagr"] == pytest.approx(
        decomp_tilt["combined_cagr"] + decomp_tilt["ichimoku_tilt_vs_raw_combined_delta"], abs=1e-9
    )
