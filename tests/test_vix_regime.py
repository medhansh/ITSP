# Destination: tests/test_vix_regime.py  (new file)
"""Synthetic-data tests for src/regime_detection/vix_regime.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_detection.vix_regime import (
    apply_bucket_hysteresis,
    bucket_labels_to_regime_series,
    build_production_vix_regime,
    choose_bucket_count,
    describe_bucket_edges,
    fit_vix_buckets,
    sweep_bucket_counts,
)


@pytest.fixture
def synthetic_vix() -> pd.Series:
    """Three visually-separated VIX regimes: calm (~12), elevated (~20),
    stressed (~35), so bucket count / ordering is easy to sanity check."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-01", periods=900)
    calm = rng.normal(12, 1.5, 500)
    elevated = rng.normal(20, 2.0, 300)
    stressed = rng.normal(35, 4.0, 100)
    values = np.concatenate([calm, elevated, stressed]).clip(min=5)
    return pd.Series(values, index=dates, name="vix")


def test_sweep_bucket_counts_shape(synthetic_vix):
    result = sweep_bucket_counts(synthetic_vix, k_range=range(2, 6))
    assert list(result.index) == [2, 3, 4, 5]
    assert {"bic", "silhouette"} <= set(result.columns)
    assert result["bic"].notna().all()
    # k=1 excluded from k_range here, but every k>1 tested must have a silhouette score
    assert result["silhouette"].notna().all()


def test_choose_bucket_count_bic_and_silhouette(synthetic_vix):
    sweep = sweep_bucket_counts(synthetic_vix, k_range=range(2, 6))
    bic_choice = choose_bucket_count(sweep, criterion="bic")
    sil_choice = choose_bucket_count(sweep, criterion="silhouette")
    assert bic_choice in sweep.index
    assert sil_choice in sweep.index


def test_choose_bucket_count_invalid_criterion(synthetic_vix):
    sweep = sweep_bucket_counts(synthetic_vix, k_range=range(2, 4))
    with pytest.raises(ValueError):
        choose_bucket_count(sweep, criterion="not_a_real_criterion")


def test_fit_vix_buckets_calm_to_stressed_ordering(synthetic_vix):
    """Bucket 0 must be the calmest bucket (lowest mean VIX) -- this is the
    whole point of clustering on a single vix_level column, per the module
    docstring's "trivially correct" claim."""
    model = fit_vix_buckets(synthetic_vix, n_buckets=3)
    edges = describe_bucket_edges(model, synthetic_vix)
    assert list(edges.index) == sorted(edges.index)
    # mean VIX must be monotonically increasing with bucket id
    means = edges["mean"].values
    assert np.all(np.diff(means) > 0)


def test_fit_vix_buckets_n_buckets_respected(synthetic_vix):
    for k in (2, 3, 4):
        model = fit_vix_buckets(synthetic_vix, n_buckets=k)
        labels = model.predict(synthetic_vix)
        assert set(labels.unique()) <= set(range(k))
        assert model.n_buckets == k


def test_log_transform_runs_and_still_orders_correctly(synthetic_vix):
    model = fit_vix_buckets(synthetic_vix, n_buckets=3, log_transform=True)
    edges = describe_bucket_edges(model, synthetic_vix)
    assert np.all(np.diff(edges["mean"].values) > 0)


def test_predict_drops_nan_rows():
    dates = pd.bdate_range("2020-01-01", periods=50)
    vix = pd.Series(np.linspace(10, 30, 50), index=dates)
    vix.iloc[5:8] = np.nan
    model = fit_vix_buckets(vix, n_buckets=2)
    labels = model.predict(vix)
    assert len(labels) == 47


def test_apply_bucket_hysteresis_disabled_is_noop():
    dates = pd.bdate_range("2021-01-01", periods=6)
    labels = pd.Series([0, 1, 0, 2, 1, 0], index=dates)
    result = apply_bucket_hysteresis(labels, min_days_to_downgrade=0)
    pd.testing.assert_series_equal(result, labels)


def test_apply_bucket_hysteresis_upgrade_is_instant():
    dates = pd.bdate_range("2021-01-01", periods=4)
    labels = pd.Series([0, 0, 3, 3], index=dates)  # jump straight to bucket 3 on day 2
    result = apply_bucket_hysteresis(labels, min_days_to_downgrade=5)
    assert result.iloc[2] == 3  # accepted immediately, no confirmation delay


def test_apply_bucket_hysteresis_downgrade_requires_consecutive_days():
    dates = pd.bdate_range("2021-01-01", periods=7)
    # confirmed=2 from day0, drops to 0 on day1 but comes back to 2 on day3 (breaks the streak),
    # then stays at 0 for 3 consecutive days (4,5,6) -- should only downgrade on day6 with min=3
    labels = pd.Series([2, 0, 0, 2, 0, 0, 0], index=dates)
    result = apply_bucket_hysteresis(labels, min_days_to_downgrade=3)
    assert result.iloc[0] == 2
    assert result.iloc[1] == 2  # streak=1, not yet confirmed
    assert result.iloc[2] == 2  # streak=2, not yet confirmed
    assert result.iloc[3] == 2  # raw returned to confirmed level -- streak resets, no downgrade
    assert result.iloc[4] == 2  # streak=1 again
    assert result.iloc[5] == 2  # streak=2
    assert result.iloc[6] == 0  # streak=3 -- downgrade finally accepted


def test_apply_bucket_hysteresis_never_delays_a_further_upgrade_mid_pending_downgrade():
    dates = pd.bdate_range("2021-01-01", periods=4)
    labels = pd.Series([2, 0, 0, 3], index=dates)  # pending downgrade interrupted by a bigger upgrade
    result = apply_bucket_hysteresis(labels, min_days_to_downgrade=5)
    assert result.iloc[3] == 3  # upgrade always instant, regardless of pending downgrade state


def test_apply_bucket_hysteresis_causal_no_lookahead():
    """Perturbing values strictly after date t must not change the
    confirmed value AT OR BEFORE t -- the state machine only ever looks
    backward."""
    dates = pd.bdate_range("2021-01-01", periods=10)
    labels_a = pd.Series([2, 0, 0, 0, 1, 2, 0, 0, 3, 1], index=dates)
    labels_b = labels_a.copy()
    labels_b.iloc[6:] = [3, 3, 0, 0]  # change everything from index 6 onward
    result_a = apply_bucket_hysteresis(labels_a, min_days_to_downgrade=3)
    result_b = apply_bucket_hysteresis(labels_b, min_days_to_downgrade=3)
    pd.testing.assert_series_equal(result_a.iloc[:6], result_b.iloc[:6])


# ----------------------------------------------------------------------
# build_production_vix_regime -- the actual production entry point
# ----------------------------------------------------------------------

def test_build_production_vix_regime_bounded_and_full_length(synthetic_vix):
    regime = build_production_vix_regime(
        synthetic_vix, synthetic_vix.index, n_buckets=3, min_days_to_downgrade=0,
    )
    assert regime.dtype == int
    assert set(regime.unique()) <= set(range(3))
    assert regime.index.equals(synthetic_vix.index)


def test_build_production_vix_regime_hysteresis_reduces_transitions(synthetic_vix):
    def n_transitions(s):
        return int((s.diff().fillna(0) != 0).sum())

    off = build_production_vix_regime(synthetic_vix, synthetic_vix.index, n_buckets=3, min_days_to_downgrade=0)
    on = build_production_vix_regime(synthetic_vix, synthetic_vix.index, n_buckets=3, min_days_to_downgrade=5)
    assert n_transitions(on) <= n_transitions(off)


def test_build_production_vix_regime_calmest_ordering(synthetic_vix):
    """Bucket 0 must correspond to the lowest VIX levels -- production
    exposure_by_regime/beta_rotation config assumes this ordering."""
    regime = build_production_vix_regime(synthetic_vix, synthetic_vix.index, n_buckets=3)
    aligned_vix = synthetic_vix.reindex(regime.index)
    means = aligned_vix.groupby(regime).mean()
    assert means.loc[means.index.min()] < means.loc[means.index.max()]


def test_bucket_labels_to_regime_series_ffill_and_warmup_default():
    dates = pd.bdate_range("2021-01-01", periods=6)
    sparse = pd.Series([2, 1], index=[dates[2], dates[4]])
    regime = bucket_labels_to_regime_series(sparse, dates)
    assert regime.dtype == int
    assert regime.iloc[0] == 0 and regime.iloc[1] == 0  # warm-up defaults to calmest
    assert regime.iloc[2] == 2 and regime.iloc[3] == 2  # ffill
    assert regime.iloc[4] == 1 and regime.iloc[5] == 1
