"""Tests for src/backtesting/adaptive_ichimoku.py (proper OHLC-based
implementation: true high/low, forward-shifted cloud, Chikou Span
confirmation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.adaptive_ichimoku import (
    _scatter_forward,
    _variable_lag_lookup,
    build_ichimoku_weights,
    compute_adaptive_ichimoku,
    compute_adaptive_periods,
    compute_ichimoku_conviction_score,
    compute_static_ichimoku,
    generate_ichimoku_signal,
)
from src.backtesting.technical_signals import compute_signed_normalized_score


@pytest.fixture
def synthetic_ohlc() -> pd.DataFrame:
    """Flat/choppy (0-350) -> strong sustained uptrend (350-550) -> sharp
    reversal downtrend (550-700), with real high/low around the close."""
    rng = np.random.default_rng(3)
    n = 700
    dates = pd.bdate_range("2021-01-01", periods=n)
    returns = np.zeros(n)
    returns[:350] = rng.normal(0, 0.006, 350)
    returns[350:550] = rng.normal(0.006, 0.006, 200)
    returns[550:] = rng.normal(-0.008, 0.007, 150)
    close = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)


@pytest.fixture
def magnitude(synthetic_ohlc) -> pd.Series:
    scores = compute_signed_normalized_score(synthetic_ohlc["close"], t=10, zscore_window=252)
    return scores["magnitude"]


# --- _scatter_forward: the mechanism behind the adaptive forward-shifted cloud ---

def test_scatter_forward_collision_most_recent_wins():
    raw = pd.Series([10.0, 20.0, np.nan, 30.0])
    offsets = pd.Series([2.0, 1.0, np.nan, 0.0])
    # i=0 -> target 2 (10.0), i=1 -> target 2 (20.0): later source wins.
    result = _scatter_forward(raw, offsets)
    assert result.iloc[2] == 20.0


def test_scatter_forward_fills_gaps_from_growing_offset():
    raw = pd.Series([100.0, 200.0, 300.0])
    offsets = pd.Series([0.0, 1.0, 4.0])  # i=2 lands out of bounds and is dropped
    result = _scatter_forward(raw, offsets)
    assert result.iloc[0] == 100.0
    assert result.iloc[1] == 100.0  # position 1 has no direct write -> forward-filled from 0


def test_scatter_forward_never_uses_future_source_values():
    """The defining causal property: value at position T only ever derives
    from raw_values[i] with i <= T."""
    n = 50
    raw = pd.Series(np.arange(n, dtype=float))
    offsets = pd.Series(np.full(n, 3.0))
    result = _scatter_forward(raw, offsets)
    # position T's value must equal raw[T-3] (forward-filled), never raw[i] for i>T-3... 
    # concretely: since offset is constant 3, target(i) = i+3, so position T <- raw[T-3].
    for t in range(3, n):
        assert result.iloc[t] == raw.iloc[t - 3]


# --- _variable_lag_lookup: Chikou Span reference (backward only, no ambiguity) ---

def test_variable_lag_lookup_basic():
    series = pd.Series(np.arange(8, dtype=float))
    lags = pd.Series(np.full(8, 2.0))
    result = _variable_lag_lookup(series, lags)
    assert result.iloc[5] == series.iloc[3]
    assert pd.isna(result.iloc[1])  # 1-2 = -1, out of bounds


# --- compute_adaptive_ichimoku: no-lookahead + agreement with compute_static_ichimoku ---

def test_adaptive_ichimoku_no_lookahead(synthetic_ohlc):
    """Truncating the series must not change already-computed historical
    values, regardless of the variable-offset scatter/gap-fill machinery."""
    periods = pd.DataFrame(
        {"tenkan_period": 5.0, "kijun_period": 10.0, "senkou_b_period": 20.0},
        index=synthetic_ohlc.index,
    )
    full = compute_adaptive_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"], periods)
    cutoff = 400
    truncated = compute_adaptive_ichimoku(
        synthetic_ohlc["high"].iloc[:cutoff], synthetic_ohlc["low"].iloc[:cutoff],
        synthetic_ohlc["close"].iloc[:cutoff], periods.iloc[:cutoff],
    )
    check_point = 300  # comfortably before cutoff, away from ffill boundary effects
    for col in ["tenkan", "kijun", "cloud_top", "cloud_bottom", "chikou_reference"]:
        assert np.isclose(full[col].iloc[check_point], truncated[col].iloc[check_point], equal_nan=True)


def test_adaptive_ichimoku_with_constant_periods_matches_static(synthetic_ohlc):
    """The general variable-window/scatter/lag machinery, given CONSTANT
    periods, must exactly reproduce the simple vectorized static
    implementation -- this is the strongest correctness check available
    (two independent code paths, same answer)."""
    n = len(synthetic_ohlc)
    periods = pd.DataFrame(
        {"tenkan_period": 9.0, "kijun_period": 26.0, "senkou_b_period": 52.0},
        index=synthetic_ohlc.index,
    )
    adaptive_const = compute_adaptive_ichimoku(
        synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"], periods
    )
    static = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"], 9, 26, 52)

    for col in ["tenkan", "kijun", "cloud_top", "cloud_bottom", "chikou_reference"]:
        a = adaptive_const[col].dropna()
        b = static[col].reindex(a.index).dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 100  # sanity: didn't just match on an empty overlap
        assert np.allclose(a.loc[common], b.loc[common])


def test_cloud_top_geq_bottom_static_and_adaptive(synthetic_ohlc, magnitude):
    static = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    valid = static.dropna()
    assert (valid["cloud_top"] >= valid["cloud_bottom"]).all()

    periods = compute_adaptive_periods(magnitude, direction="shrink_when_low")
    adaptive = compute_adaptive_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"], periods)
    valid_a = adaptive.dropna()
    assert (valid_a["cloud_top"] >= valid_a["cloud_bottom"]).all()


# --- compute_adaptive_periods (unchanged logic, still worth covering here) ---

def test_period_ordering_always_preserved(magnitude):
    for direction in ("shrink_when_high", "shrink_when_low"):
        periods = compute_adaptive_periods(magnitude, direction=direction).dropna()
        assert (periods["tenkan_period"] <= periods["kijun_period"]).all()
        assert (periods["kijun_period"] <= periods["senkou_b_period"]).all()


def test_rejects_invalid_direction(magnitude):
    with pytest.raises(ValueError, match="direction"):
        compute_adaptive_periods(magnitude, direction="bogus")


# --- generate_ichimoku_signal: triple confirmation ---

def test_signal_requires_all_three_confirmations_to_agree():
    """Construct a case where price is above the cloud and Tenkan>Kijun
    (2 of 3 bullish) but Chikou disagrees (today's close below price from
    kijun_period ago) -- must NOT go long, since all three must agree."""
    dates = pd.bdate_range("2022-01-01", periods=5)
    ichimoku = pd.DataFrame(
        {
            "tenkan": [10, 11, 12, 13, 14],
            "kijun": [9, 9, 9, 9, 9],           # tenkan > kijun throughout: bullish
            "cloud_top": [8, 8, 8, 8, 8],        # close will be above this: bullish
            "cloud_bottom": [6, 6, 6, 6, 6],
            "chikou_reference": [50, 50, 50, 50, 50],  # deliberately way above close -> Chikou disagrees
        },
        index=dates,
    )
    close = pd.Series([9, 9, 9, 9, 9], index=dates, dtype=float)
    position = generate_ichimoku_signal(ichimoku, close, long_only=True)
    assert (position == 0.0).all(), "Should stay flat -- Chikou confirmation disagreed with the other two"


def test_signal_goes_long_when_all_three_confirmations_agree():
    dates = pd.bdate_range("2022-01-01", periods=5)
    ichimoku = pd.DataFrame(
        {
            "tenkan": [10, 11, 12, 13, 14],
            "kijun": [9, 9, 9, 9, 9],
            "cloud_top": [8, 8, 8, 8, 8],
            "cloud_bottom": [6, 6, 6, 6, 6],
            "chikou_reference": [5, 5, 5, 5, 5],  # below close -> Chikou agrees (bullish)
        },
        index=dates,
    )
    close = pd.Series([9, 9, 9, 9, 9], index=dates, dtype=float)
    position = generate_ichimoku_signal(ichimoku, close, long_only=True)
    assert (position == 1.0).all()


def test_signal_long_only_never_negative(synthetic_ohlc):
    ichimoku = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    position = generate_ichimoku_signal(ichimoku, synthetic_ohlc["close"], long_only=True)
    assert (position >= 0).all()


def test_signal_allows_short_when_not_long_only(synthetic_ohlc):
    ichimoku = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    position = generate_ichimoku_signal(ichimoku, synthetic_ohlc["close"], long_only=False)
    assert (position.iloc[550:] < 0).any()


def test_signal_goes_long_during_confirmed_uptrend(synthetic_ohlc):
    ichimoku = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    position = generate_ichimoku_signal(ichimoku, synthetic_ohlc["close"], long_only=True)
    assert (position.iloc[480:550] == 1.0).mean() > 0.5


# --- build_ichimoku_weights: OHLC-dict-based portfolio construction ---

@pytest.fixture
def synthetic_ohlc_panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(9)
    n = 700
    dates = pd.bdate_range("2020-01-01", periods=n)
    panel = {}
    for i, drift in enumerate([0.0007, -0.0004, 0.0003]):
        close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
        high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
        panel[f"SYM{i}"] = pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)
    return panel


def test_rejects_invalid_variant(synthetic_ohlc_panel):
    with pytest.raises(ValueError, match="variant"):
        build_ichimoku_weights(synthetic_ohlc_panel, variant="bogus")


def test_rejects_symbol_missing_ohlc_columns_gracefully(synthetic_ohlc_panel):
    bad_panel = dict(synthetic_ohlc_panel)
    bad_panel["BAD"] = pd.DataFrame({"close": [1, 2, 3]})  # missing high/low
    weights = build_ichimoku_weights(bad_panel, t=10, variant="static")
    assert "BAD" not in weights.columns


@pytest.mark.parametrize("variant", ["static", "shrink_when_high", "shrink_when_low"])
def test_build_ichimoku_weights_valid_shape_all_variants(variant, synthetic_ohlc_panel):
    weights = build_ichimoku_weights(synthetic_ohlc_panel, t=10, variant=variant)
    assert (weights >= 0).all().all()
    assert (weights.sum(axis=1) <= 1.0001).all()


def test_build_ichimoku_weights_plugs_into_backtest_engine(synthetic_ohlc_panel):
    from src.backtesting.engine import compute_returns_panel, run_backtest

    weights = build_ichimoku_weights(synthetic_ohlc_panel, t=10, variant="static")
    close_panel = pd.DataFrame({s: df["close"] for s, df in synthetic_ohlc_panel.items()})
    returns_panel = compute_returns_panel(close_panel)
    weights = weights.reindex(columns=close_panel.columns, fill_value=0.0)
    result = run_backtest(returns_panel, weights, transaction_cost_bps=10)
    assert set(result.keys()) == {"returns", "gross_returns", "turnover", "equity_curve"}
    assert not result["equity_curve"].isna().any()


def test_compute_ichimoku_conviction_score_bounded_and_valid(synthetic_ohlc):
    ichimoku = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    conviction = compute_ichimoku_conviction_score(ichimoku, synthetic_ohlc["close"])
    valid = conviction.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()  # long_only default clips negative to 0
    assert (valid <= 1.0).all()  # tanh bound
    assert not valid.isna().any()


def test_compute_ichimoku_conviction_score_signed_when_not_long_only(synthetic_ohlc):
    ichimoku = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    conviction = compute_ichimoku_conviction_score(ichimoku, synthetic_ohlc["close"], long_only=False)
    assert conviction.min() >= -1.0
    assert conviction.max() <= 1.0
    assert (conviction < 0).any()  # the synthetic fixture has a sharp downtrend segment


def test_compute_ichimoku_conviction_score_higher_during_strong_uptrend(synthetic_ohlc):
    """The fixture's 350-550 segment is a strong sustained uptrend -- mean
    conviction there should clearly exceed the flat/choppy 0-350 segment."""
    ichimoku = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    conviction = compute_ichimoku_conviction_score(ichimoku, synthetic_ohlc["close"])
    choppy_mean = conviction.iloc[100:300].mean()  # skip warmup at the very start
    uptrend_mean = conviction.iloc[400:550].mean()
    assert uptrend_mean > choppy_mean


def test_compute_ichimoku_conviction_score_hit_rate_at_least_as_high_as_binary(synthetic_ohlc):
    """The whole point of the additive score: it shouldn't be MORE
    restrictive than the strict AND it's replacing."""
    ichimoku = compute_static_ichimoku(synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"])
    conviction = compute_ichimoku_conviction_score(ichimoku, synthetic_ohlc["close"])
    binary = generate_ichimoku_signal(ichimoku, synthetic_ohlc["close"])
    assert (conviction > 0).mean() >= (binary > 0).mean()


def test_build_ichimoku_weights_conviction_mode_is_not_equal_weighted(synthetic_ohlc_panel):
    """Distinguishing behavior vs triple_confirmation: conviction mode
    should size positions by relative conviction, not split equally among
    active names (unless conviction happens to tie, which a real multi-
    symbol synthetic panel essentially never will across an entire
    history)."""
    weights = build_ichimoku_weights(synthetic_ohlc_panel, variant="static", signal_mode="conviction_score")
    active_rows = weights[weights.abs().sum(axis=1) > 0]
    # at least some active day must have UNEQUAL weights across the symbols held that day
    n_symbols = weights.shape[1]
    equal_weight = 1.0 / n_symbols
    any_unequal = (active_rows.apply(lambda row: row[row > 0].nunique(), axis=1) > 1).any()
    assert any_unequal


def test_build_ichimoku_weights_conviction_mode_rows_sum_to_one_or_zero(synthetic_ohlc_panel):
    weights = build_ichimoku_weights(synthetic_ohlc_panel, variant="static", signal_mode="conviction_score")
    row_sums = weights.sum(axis=1)
    active = row_sums[row_sums > 1e-9]
    assert np.allclose(active, 1.0, atol=1e-6)


def test_build_ichimoku_weights_rejects_unknown_signal_mode(synthetic_ohlc_panel):
    with pytest.raises(ValueError, match="signal_mode"):
        build_ichimoku_weights(synthetic_ohlc_panel, variant="static", signal_mode="bogus_mode")
