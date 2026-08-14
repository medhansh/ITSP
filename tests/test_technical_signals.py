"""Tests for the multi-scale SMA-dispersion trend-strength signal
(src/backtesting/technical_signals.py).

Uses synthetic price paths with clearly engineered regime transitions (flat
-> strong uptrend -> sharp reversal) since the whole point is to verify the
score and entry/exit logic behave sensibly on *known* ground truth — there's
no claim anywhere here about this signal being profitable on real data (see
the module's own docstring caveat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.technical_signals import (
    build_conviction_weighted_signal_weights,
    build_technical_signal_weights,
    compute_dispersion_score,
    compute_signed_normalized_score,
    generate_signal,
)


@pytest.fixture
def regime_shift_price() -> pd.Series:
    """Flat/choppy (0-350) -> strong sustained uptrend (350-500) -> sharp
    reversal downtrend (500-600)."""
    rng = np.random.default_rng(0)
    n = 600
    dates = pd.bdate_range("2021-01-01", periods=n)
    returns = np.zeros(n)
    returns[:350] = rng.normal(0, 0.006, 350)
    returns[350:500] = rng.normal(0.006, 0.006, 150)
    returns[500:] = rng.normal(-0.01, 0.008, 100)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)


def test_dispersion_score_is_nonnegative(regime_shift_price):
    score = compute_dispersion_score(regime_shift_price, t=10)
    assert (score.dropna() >= 0).all()


def test_dispersion_score_higher_during_trend_than_flat_period(regime_shift_price):
    score = compute_dispersion_score(regime_shift_price, t=10)
    flat_mean = score.iloc[300:340].mean()
    trend_mean = score.iloc[420:480].mean()
    assert trend_mean > flat_mean


def test_signed_score_direction_matches_trend_direction(regime_shift_price):
    scores = compute_signed_normalized_score(regime_shift_price, t=10, zscore_window=252)
    assert scores["signed_score"].iloc[420:480].mean() > 0.5   # uptrend -> positive
    assert scores["signed_score"].iloc[550:600].mean() < -0.5  # downtrend -> negative


def test_signed_score_bounded():
    rng = np.random.default_rng(1)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    scores = compute_signed_normalized_score(price, t=10, zscore_window=100)
    valid = scores["signed_score"].dropna()
    assert (valid >= -1).all() and (valid <= 1).all()


# --- generate_signal: parameter validation ---

def test_generate_signal_rejects_invalid_q_entry():
    with pytest.raises(ValueError, match="q_entry"):
        generate_signal(pd.Series([0.1, 0.2]), q_entry=1.5)


def test_generate_signal_rejects_q_exit_not_less_than_q_entry():
    with pytest.raises(ValueError, match="q_exit"):
        generate_signal(pd.Series([0.1, 0.2]), q_entry=0.5, q_exit=0.6)


def test_generate_signal_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        generate_signal(pd.Series([0.1, 0.2]), q_entry=0.5, mode="bogus")


# --- generate_signal: trend mode ---

def test_trend_mode_goes_long_during_uptrend_and_short_during_downtrend(regime_shift_price):
    scores = compute_signed_normalized_score(regime_shift_price, t=10, zscore_window=252)
    position = generate_signal(scores["signed_score"], q_entry=0.5, mode="trend")
    assert (position.iloc[420:480] == 1.0).mean() > 0.9
    assert (position.iloc[550:600] == -1.0).mean() > 0.9


def test_trend_mode_flat_during_quiet_regime(regime_shift_price):
    scores = compute_signed_normalized_score(regime_shift_price, t=10, zscore_window=252)
    position = generate_signal(scores["signed_score"], q_entry=0.5, mode="trend")
    # Well into the flat phase (past zscore warmup), should mostly be flat.
    assert (position.iloc[300:340] == 0.0).mean() >= 0.5


def test_trend_mode_reverses_directly_without_getting_stuck(regime_shift_price):
    """Regression test for a real bug found during development: the original
    state machine only checked the exit condition once in a position, so a
    sharp reversal (score swings from one extreme straight past the other,
    never lingering near zero) left the position stuck on the wrong side —
    here, long, all the way through a subsequent strong downtrend. Fixed by
    allowing a direct reversal whenever the score crosses the opposite
    entry threshold, not just an exit-then-re-entry."""
    scores = compute_signed_normalized_score(regime_shift_price, t=10, zscore_window=252)
    position = generate_signal(scores["signed_score"], q_entry=0.5, mode="trend")
    # By late in the downtrend, position must have actually flipped to short
    # — not still be long from the earlier uptrend.
    assert position.iloc[580] == -1.0


# --- generate_signal: mean-reversion mode is the sign-mirrored version ---

def test_mean_reversion_mode_fades_the_uptrend_and_the_downtrend(regime_shift_price):
    scores = compute_signed_normalized_score(regime_shift_price, t=10, zscore_window=252)
    position = generate_signal(scores["signed_score"], q_entry=0.5, mode="mean_reversion")
    # Fades (shorts) the uptrend, longs the downtrend -- exact mirror of trend mode.
    assert (position.iloc[420:480] == -1.0).mean() > 0.9
    assert (position.iloc[550:600] == 1.0).mean() > 0.9


def test_trend_and_mean_reversion_are_sign_mirrors_when_never_flat(regime_shift_price):
    scores = compute_signed_normalized_score(regime_shift_price, t=10, zscore_window=252)
    trend_pos = generate_signal(scores["signed_score"], q_entry=0.5, mode="trend")
    mr_pos = generate_signal(scores["signed_score"], q_entry=0.5, mode="mean_reversion")
    both_active = (trend_pos != 0) & (mr_pos != 0)
    assert (trend_pos[both_active] == -mr_pos[both_active]).all()


# --- build_technical_signal_weights: portfolio-level construction ---

@pytest.fixture
def synthetic_price_panel() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 500
    dates = pd.bdate_range("2021-01-01", periods=n)
    panel = {}
    for i, drift in enumerate([0.0008, -0.0006, 0.0, 0.0004]):
        panel[f"SYM{i}"] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(panel, index=dates)


def test_build_technical_weights_long_only_never_negative(synthetic_price_panel):
    weights = build_technical_signal_weights(synthetic_price_panel, t=10, q_entry=0.5, long_only=True)
    assert (weights >= 0).all().all()


def test_build_technical_weights_rows_sum_to_at_most_one(synthetic_price_panel):
    weights = build_technical_signal_weights(synthetic_price_panel, t=10, q_entry=0.5)
    assert (weights.sum(axis=1) <= 1.0001).all()


def test_build_technical_weights_equal_weights_among_active_positions(synthetic_price_panel):
    weights = build_technical_signal_weights(synthetic_price_panel, t=10, q_entry=0.5)
    for _, row in weights.iterrows():
        active = row[row > 0]
        if len(active) > 1:
            assert np.allclose(active.values, active.values[0])


def test_build_technical_weights_plugs_into_backtest_engine(synthetic_price_panel):
    """End-to-end smoke test: the weights this produces must be directly
    usable by the existing engine.run_backtest without any extra alignment
    step (unlike the sparse rebalance-date fundamentals weights)."""
    from src.backtesting.engine import compute_returns_panel, run_backtest

    weights = build_technical_signal_weights(synthetic_price_panel, t=10, q_entry=0.5)
    returns_panel = compute_returns_panel(synthetic_price_panel)
    result = run_backtest(returns_panel, weights, transaction_cost_bps=10)
    assert set(result.keys()) == {"returns", "gross_returns", "turnover", "equity_curve"}
    assert not result["equity_curve"].isna().any()


# --- build_conviction_weighted_signal_weights: continuous sizing, no hard gate ---

def test_conviction_weights_rejects_invalid_mode(synthetic_price_panel):
    with pytest.raises(ValueError, match="mode"):
        build_conviction_weighted_signal_weights(synthetic_price_panel, mode="bogus")


def test_conviction_weights_long_only_never_negative(synthetic_price_panel):
    weights = build_conviction_weighted_signal_weights(synthetic_price_panel, t=10, long_only=True)
    assert (weights >= 0).all().all()


def test_conviction_weights_bounded_gross_exposure(synthetic_price_panel):
    weights = build_conviction_weighted_signal_weights(synthetic_price_panel, t=10, long_only=True)
    assert (weights.sum(axis=1) <= 1.0001).all()

    weights_short = build_conviction_weighted_signal_weights(synthetic_price_panel, t=10, long_only=False)
    assert (weights_short.abs().sum(axis=1) <= 1.0001).all()


def test_conviction_weights_are_genuinely_continuous_not_binary(synthetic_price_panel):
    """The whole point of this function: exposure should vary smoothly with
    conviction, not snap between a small fixed set of values the way
    build_technical_signal_weights's equal-weight-among-active scheme does."""
    weights = build_conviction_weighted_signal_weights(synthetic_price_panel, t=10, long_only=True)
    symbol = synthetic_price_panel.columns[0]
    n_distinct = weights[symbol].round(4).nunique()
    # A threshold/equal-weight scheme over a small panel only ever produces a
    # handful of distinct values (0, 1/1, 1/2, 1/3, ...); continuous sizing
    # should produce far more since every distinct conviction level maps to
    # its own weight.
    assert n_distinct > 20


def test_conviction_weights_trend_and_mean_reversion_are_sign_mirrors(synthetic_price_panel):
    trend = build_conviction_weighted_signal_weights(synthetic_price_panel, t=10, mode="trend", long_only=False)
    mean_rev = build_conviction_weighted_signal_weights(synthetic_price_panel, t=10, mode="mean_reversion", long_only=False)
    pd.testing.assert_frame_equal(trend, -mean_rev)


def test_conviction_weights_scale_with_signed_score_directly(regime_shift_price):
    """Weight for a single symbol at time t should equal
    signed_score(t) / N (long-only clipped) -- verify the actual formula,
    not just aggregate properties."""
    panel = pd.DataFrame({"ONLY": regime_shift_price})
    weights = build_conviction_weighted_signal_weights(panel, t=10, zscore_window=252, long_only=True)
    scores = compute_signed_normalized_score(regime_shift_price, t=10, zscore_window=252)
    expected = scores["signed_score"].clip(lower=0.0) / 1  # N=1 for a single-symbol panel
    valid = expected.dropna().index.intersection(weights.dropna().index)
    assert np.allclose(weights.loc[valid, "ONLY"], expected.loc[valid])


def test_conviction_weights_plugs_into_backtest_engine(synthetic_price_panel):
    from src.backtesting.engine import compute_returns_panel, run_backtest

    weights = build_conviction_weighted_signal_weights(synthetic_price_panel, t=10)
    returns_panel = compute_returns_panel(synthetic_price_panel)
    result = run_backtest(returns_panel, weights, transaction_cost_bps=10)
    assert set(result.keys()) == {"returns", "gross_returns", "turnover", "equity_curve"}
    assert not result["equity_curve"].isna().any()
