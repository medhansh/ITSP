"""Synthetic-data tests for the regime detection module.

No real market data is required — a synthetic two-regime price series (a
calm low-vol uptrend followed by a stressed high-vol drawdown) is enough to
verify the feature engineering and model pipeline run end-to-end and produce
sensible, temporally-clustered regime labels.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from src.regime_detection.features import build_feature_matrix
from src.regime_detection.models import RegimeModel


@pytest.fixture
def synthetic_prices() -> pd.Series:
    rng = np.random.default_rng(42)
    n_calm, n_stress = 500, 300
    dates = pd.bdate_range("2020-01-01", periods=n_calm + n_stress)

    calm_returns = rng.normal(loc=0.0006, scale=0.006, size=n_calm)
    stress_returns = rng.normal(loc=-0.0010, scale=0.025, size=n_stress)
    returns = np.concatenate([calm_returns, stress_returns])

    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="close")


def test_build_feature_matrix_shape(synthetic_prices):
    features = build_feature_matrix(
        synthetic_prices, return_windows=[5, 21], vol_windows=[21, 63]
    )
    assert not features.empty
    assert {"return_5d", "return_21d", "realized_vol_21d", "realized_vol_63d", "drawdown"} <= set(
        features.columns
    )
    assert features.isna().sum().sum() == 0  # dropna() should have cleared warm-up NaNs


@pytest.mark.parametrize("model_type", ["gmm", "kmeans"])
def test_regime_model_fit_predict(synthetic_prices, model_type):
    features = build_feature_matrix(
        synthetic_prices, return_windows=[5, 21], vol_windows=[21, 63]
    )
    model = RegimeModel(model_type=model_type, n_regimes=2, random_state=0)
    model.fit(features)

    labels = model.predict(features)
    assert set(labels.unique()) <= {0, 1}
    assert len(labels) == len(features)

    proba = model.predict_proba(features)
    assert proba.shape == (len(features), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # The high-volatility synthetic period should be dominated by the
    # higher-labeled (vol-sorted) regime.
    stress_period_labels = labels.loc[labels.index >= "2022-01-01"]
    assert stress_period_labels.mean() > 0.5


def test_regime_labels_are_volatility_ordered(synthetic_prices):
    features = build_feature_matrix(
        synthetic_prices, return_windows=[5, 21], vol_windows=[21, 63]
    )
    model = RegimeModel(model_type="gmm", n_regimes=2, random_state=0)
    model.fit(features)
    labels = model.predict(features)

    vol_by_label = features.groupby(labels)["realized_vol_21d"].mean()
    assert vol_by_label.loc[0] < vol_by_label.loc[1]


# --- geometric wedge-product crash-risk signal (src/regime_detection/geometric_signal.py) ---

from src.regime_detection.geometric_signal import (
    calculate_wedge_volume,
    compute_geometric_crash_features,
    validate_against_known_crises,
)


@pytest.fixture
def synthetic_sector_returns() -> pd.DataFrame:
    """5 sector return series, uncorrelated in "normal" times, forced to
    move identically (rank-collapse) during a synthetic 30-day crisis window
    so the wedge-volume signal has an unambiguous ground truth to detect."""
    rng = np.random.default_rng(11)
    n = 400
    dates = pd.bdate_range("2022-01-01", periods=n)
    base = rng.normal(0, 0.01, size=(n, 5))
    crisis = slice(220, 250)
    common_shock = rng.normal(0, 0.03, size=(30,))
    for c in range(5):
        base[crisis, c] = common_shock
    return pd.DataFrame(base, index=dates, columns=[f"sector{i}" for i in range(5)]), crisis


def test_wedge_volume_requires_multiple_assets():
    single = pd.DataFrame({"a": np.random.normal(size=100)})
    with pytest.raises(ValueError):
        calculate_wedge_volume(single, window=20)


def test_wedge_volume_collapses_during_synthetic_crisis(synthetic_sector_returns):
    returns, crisis = synthetic_sector_returns
    volume = calculate_wedge_volume(returns, window=60)
    crisis_mean = volume.iloc[crisis].mean()
    overall_mean = volume.mean()
    # Volume should collapse well below its own overall average during the
    # engineered rank-collapse window — this is the core claim the signal
    # makes (see geometric_signal.py's docstring on what it's measuring).
    assert crisis_mean < 0.5 * overall_mean


def test_crash_flag_fires_more_often_inside_known_crisis(synthetic_sector_returns):
    returns, crisis = synthetic_sector_returns
    features = compute_geometric_crash_features(
        returns, window=60, smoothing_window=10, percentile_window=120, crash_percentile_threshold=0.15
    )
    dates = returns.index
    report = validate_against_known_crises(
        features, [(str(dates[crisis.start].date()), str(dates[crisis.stop - 1].date()))]
    )
    assert report.loc[0, "lift_over_base_rate"] > 0


# --- OHLCV features: range volatility + volume (src/regime_detection/features.py) ---

from src.regime_detection.features import (
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_volume_features,
)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 300
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=dates)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, n))
    high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, n).astype(float), index=dates)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_range_volatility_is_positive_and_roughly_tracks_close_to_close_vol(synthetic_ohlcv):
    df = synthetic_ohlcv
    pk = compute_parkinson_volatility(df["high"], df["low"], [21])
    gk = compute_garman_klass_volatility(df["open"], df["high"], df["low"], df["close"], [21])
    assert (pk["parkinson_vol_21d"].dropna() >= 0).all()
    assert (gk["garman_klass_vol_21d"].dropna() >= 0).all()
    # Both are range-based vol estimates on the same underlying series — should
    # be the same order of magnitude, not wildly different.
    ratio = (pk["parkinson_vol_21d"].dropna().mean() / gk["garman_klass_vol_21d"].dropna().mean())
    assert 0.3 < ratio < 3.0


def test_obv_trend_is_strongly_negative_on_a_down_move_with_volume_spike(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    rng = np.random.default_rng(1)
    # Inject a sharp, sustained down move on a large volume spike.
    df.loc[df.index[150:160], "volume"] *= 8
    down_move = df["close"].iloc[149] * np.exp(np.cumsum(rng.normal(-0.02, 0.005, 10)))
    df.loc[df.index[150:160], "close"] = down_move

    features = compute_volume_features(df["close"], df["volume"], zscore_windows=[21], obv_trend_windows=[10])
    obv_trend_at_end_of_move = features["obv_trend_10d"].iloc[159]
    # Every day in the 10-day window was a down day on outsized volume, so
    # net directional volume pressure should be at (or very near) -1.
    assert obv_trend_at_end_of_move < -0.9


def test_volume_zscore_flags_the_spike(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    df.loc[df.index[150:160], "volume"] *= 8
    features = compute_volume_features(df["close"], df["volume"], zscore_windows=[21], obv_trend_windows=[21])
    assert features["volume_zscore_21d"].iloc[150:160].max() > 1.5


def test_build_feature_matrix_degrades_gracefully_without_ohlcv(synthetic_prices):
    """Passing no open/high/low/volume must behave exactly as before — no
    new columns, no errors — this is the graceful-degradation contract the
    rest of the pipeline (and every existing caller) relies on."""
    features = build_feature_matrix(
        synthetic_prices, return_windows=[5, 21], vol_windows=[21, 63]
    )
    range_or_volume_cols = [
        c for c in features.columns
        if c.startswith(("parkinson_", "garman_klass_", "volume_zscore_", "obv_trend_"))
    ]
    assert range_or_volume_cols == []


def test_build_feature_matrix_adds_ohlcv_features_when_available(synthetic_ohlcv):
    features = build_feature_matrix(
        synthetic_ohlcv["close"], return_windows=[5, 21], vol_windows=[21, 63],
        open_=synthetic_ohlcv["open"], high=synthetic_ohlcv["high"],
        low=synthetic_ohlcv["low"], volume=synthetic_ohlcv["volume"],
    )
    for expected in ["parkinson_vol_21d", "garman_klass_vol_21d", "volume_zscore_21d", "obv_trend_21d"]:
        assert expected in features.columns
    assert features.isna().sum().sum() == 0


# --- regression: geometric signal must NOT reach the GMM/KMeans/HMM clustering
# (src/regime_detection/pipeline.py) ---

from src.regime_detection.pipeline import run_pipeline


def test_geometric_signal_never_reaches_the_clustering_model(tmp_path, synthetic_sector_returns):
    """Explicit design requirement: the geometric wedge-product crash-risk
    signal is computed and joined onto the regime history AFTER the model has
    already been fit/predicted — it must never appear in
    RegimeModel.feature_names_ (what the model actually saw), regardless of
    whether sector prices / geometric_signal.enabled are configured."""
    rng = np.random.default_rng(3)
    sector_returns, _ = synthetic_sector_returns
    dates = sector_returns.index  # same index for both, so nothing is silently misaligned/NaN
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(dates))))
    price_csv = tmp_path / "price.csv"
    pd.DataFrame({"date": dates, "close": close}).to_csv(price_csv, index=False)

    sector_prices = 100 * np.exp(np.log1p(sector_returns.fillna(0)).cumsum())
    sector_prices.index.name = "date"
    sector_csv = tmp_path / "sectors.csv"
    sector_prices.to_csv(sector_csv)

    cfg = {
        "feature_windows": {"returns": [5, 21], "realized_vol": [21, 63], "breadth": [21]},
        "model": {"type": "gmm", "n_regimes": 4, "random_state": 42},
        "geometric_signal": {
            "enabled": True, "window": 60, "smoothing_window": 10,
            "percentile_window": 120, "crash_percentile_threshold": 0.15,
        },
    }
    result, model = run_pipeline(cfg, str(price_csv), sector_price_csv=str(sector_csv))

    assert not any("wedge" in c or "crash" in c for c in model.feature_names_), (
        f"Geometric signal leaked into the clustering feature set: {model.feature_names_}"
    )
    # But the columns must still be present on the output for downstream use
    # (the standalone overlay in backtesting/strategies.py).
    assert "geometric_crash_risk_flag" in result.columns
    assert "wedge_volume_60d" in result.columns


def test_run_pipeline_without_sector_data_has_no_geometric_columns(synthetic_prices):
    """No sector_price_csv given (and geometric_signal not enabled in config)
    -> no geometric columns at all, and behavior identical to before the
    signal existed."""
    dates = synthetic_prices.index
    cfg = {
        "feature_windows": {"returns": [5, 21], "realized_vol": [21, 63], "breadth": [21]},
        "model": {"type": "gmm", "n_regimes": 4, "random_state": 42},
    }
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        price_csv = os.path.join(d, "price.csv")
        pd.DataFrame({"date": dates, "close": synthetic_prices.values}).to_csv(price_csv, index=False)
        result, model = run_pipeline(cfg, price_csv)

    assert not any("wedge" in c or "crash" in c for c in result.columns)


# --- regression: yfinance sector-price loader must handle MultiIndex columns,
# empty per-ticker results, and give a clear error instead of a cryptic pandas
# "must pass an index" crash (src/regime_detection/data_loader.py) ---

from src.regime_detection.data_loader import load_from_yfinance, load_sector_prices_from_yfinance


def _install_fake_yfinance(monkeypatch, download_fn):
    import types
    fake_module = types.ModuleType("yfinance")
    fake_module.download = download_fn
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)


def _make_ohlcv(ticker: str, multiindex: bool = False, n: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-01", periods=n)
    df = pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": np.linspace(100, 150, n), "Volume": 1000.0},
        index=dates,
    )
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df


def test_sector_loader_handles_multiindex_columns(monkeypatch):
    """This is the exact bug pattern that broke on a real run: newer yfinance
    versions return (field, ticker) MultiIndex columns even for a
    single-ticker download, which made df["Close"] return something
    pd.DataFrame() couldn't treat as a Series, and the whole batch crashed
    with a cryptic 'must pass an index' error deep in pandas internals."""
    def fake_download(ticker, start=None, end=None, progress=False):
        return _make_ohlcv(ticker, multiindex=True)

    _install_fake_yfinance(monkeypatch, fake_download)
    result = load_sector_prices_from_yfinance({"A": "T1", "B": "T2", "C": "T3"})
    assert result.shape == (30, 3)
    assert list(result.columns) == ["A", "B", "C"]


def test_sector_loader_handles_flat_columns(monkeypatch):
    def fake_download(ticker, start=None, end=None, progress=False):
        return _make_ohlcv(ticker, multiindex=False)

    _install_fake_yfinance(monkeypatch, fake_download)
    result = load_sector_prices_from_yfinance({"A": "T1", "B": "T2"})
    assert result.shape == (30, 2)


def test_sector_loader_skips_failed_tickers_without_crashing(monkeypatch):
    def fake_download(ticker, start=None, end=None, progress=False):
        if ticker == "BAD":
            return pd.DataFrame()
        return _make_ohlcv(ticker, multiindex=True)

    _install_fake_yfinance(monkeypatch, fake_download)
    result = load_sector_prices_from_yfinance({"A": "T1", "B": "BAD", "C": "T3"})
    assert result.shape == (30, 2)
    assert "B" not in result.columns


def test_sector_loader_raises_clear_error_when_too_few_succeed(monkeypatch):
    def fake_download(ticker, start=None, end=None, progress=False):
        return pd.DataFrame()  # every ticker fails

    _install_fake_yfinance(monkeypatch, fake_download)
    with pytest.raises(ValueError, match="need >= 2"):
        load_sector_prices_from_yfinance({"A": "T1", "B": "T2"})


def test_load_from_yfinance_handles_multiindex_index_and_vix(monkeypatch):
    def fake_download(ticker, start=None, end=None, progress=False):
        return _make_ohlcv(ticker, multiindex=True)

    _install_fake_yfinance(monkeypatch, fake_download)
    result = load_from_yfinance(index_ticker="IDX", vix_ticker="VIX")
    assert {"open", "high", "low", "close", "volume", "vix"} <= set(result.columns)
