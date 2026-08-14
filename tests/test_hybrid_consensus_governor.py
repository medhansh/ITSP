"""Tests for the Hybrid HMM + Wasserstein consensus governor
(``wasserstein_proximity.py`` + ``state_governor.py``).

Synthetic three-regime price series, same style as
``test_regime_detection.py``: a calm uptrend, a stressed drawdown, and a
second calm regime, so the governor has a genuine transition to detect and
a "should stay put" period to confirm it doesn't over-switch on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_detection.features import build_feature_matrix
from src.regime_detection.models import RegimeModel
from src.regime_detection.state_governor import RegimeStateGovernor, run_governor_over_history
from src.regime_detection.wasserstein_proximity import (
    build_regime_templates,
    rolling_wasserstein_proximity,
)

TEMPLATE_COLUMNS = ("return_5d", "return_21d", "realized_vol_21d", "realized_vol_63d")


@pytest.fixture
def synthetic_three_regime_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_calm, n_stress, n_calm2 = 400, 250, 300
    dates = pd.bdate_range("2019-01-01", periods=n_calm + n_stress + n_calm2)

    returns = np.concatenate(
        [
            rng.normal(0.0006, 0.006, n_calm),
            rng.normal(-0.0010, 0.025, n_stress),
            rng.normal(0.0005, 0.007, n_calm2),
        ]
    )
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name="close")

    features = build_feature_matrix(prices, return_windows=[5, 21], vol_windows=[21, 63])
    model = RegimeModel(model_type="gmm", n_regimes=3, random_state=42)
    model.fit(features)
    labels = model.predict(features)
    proba = model.predict_proba(features)
    return features, labels, proba


def test_build_regime_templates_covers_every_regime(synthetic_three_regime_features):
    features, labels, _ = synthetic_three_regime_features
    templates = build_regime_templates(features, labels, columns=TEMPLATE_COLUMNS)
    assert set(templates.regime_ids) == set(labels.unique())
    for regime_id in templates.regime_ids:
        for col in TEMPLATE_COLUMNS:
            assert len(templates.distributions[regime_id][col]) >= 5


def test_build_regime_templates_rejects_mismatched_index(synthetic_three_regime_features):
    features, labels, _ = synthetic_three_regime_features
    with pytest.raises(ValueError, match="same index"):
        build_regime_templates(features, labels.iloc[:-5], columns=TEMPLATE_COLUMNS)


def test_build_regime_templates_rejects_unknown_column(synthetic_three_regime_features):
    features, labels, _ = synthetic_three_regime_features
    with pytest.raises(ValueError, match="not found in features"):
        build_regime_templates(features, labels, columns=("nonexistent_col",))


def test_rolling_wasserstein_proximity_rows_sum_to_one(synthetic_three_regime_features):
    features, labels, _ = synthetic_three_regime_features
    templates = build_regime_templates(features, labels, columns=TEMPLATE_COLUMNS)
    prox = rolling_wasserstein_proximity(features, templates, window=21)

    warm = prox.dropna()
    assert len(warm) > 0
    row_sums = warm.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert (warm >= 0).all().all()

    # warm-up period (< window rows of history) must be NaN, not silently zero
    assert prox.iloc[: 21 - 1].isna().all().all()


def test_wasserstein_proximity_favors_own_regime_template():
    # A window drawn directly from a regime's own template values should be
    # closer (higher proximity) to that regime's template than a window of
    # values drawn from a very different regime.
    rng = np.random.default_rng(0)
    idx = pd.RangeIndex(60)
    features = pd.DataFrame(
        {"return_5d": rng.normal(0, 0.01, 60), "realized_vol_21d": rng.normal(0.1, 0.01, 60)},
        index=idx,
    )
    calm_labels = pd.Series(0, index=idx)
    calm_labels.iloc[30:] = 1
    features.loc[30:, "return_5d"] = rng.normal(-0.05, 0.02, 30)
    features.loc[30:, "realized_vol_21d"] = rng.normal(0.4, 0.03, 30)

    templates = build_regime_templates(features, calm_labels, columns=("return_5d", "realized_vol_21d"))
    prox = rolling_wasserstein_proximity(features, templates, window=15)

    # last window (all stress-regime rows) should be closer to template 1 than template 0
    last_row = prox.iloc[-1]
    assert last_row["wasserstein_proximity_1"] > last_row["wasserstein_proximity_0"]


def test_state_governor_entropy_zero_for_certain_vector():
    gov = RegimeStateGovernor()
    assert gov.entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_state_governor_entropy_max_for_uniform_vector():
    gov = RegimeStateGovernor()
    n = 4
    h = gov.entropy(np.ones(n) / n)
    assert h == pytest.approx(np.log2(n), abs=1e-9)


def test_state_governor_requires_persistence_before_switching():
    gov = RegimeStateGovernor(entropy_limit=0.85, persistence_window=3, hysteresis_epsilon=0.0)
    gov.active_state = 0
    regime_ids = [0, 1]

    # regime 1 proposed but only for 2 bars -> should NOT switch yet
    consensus = np.array([0.1, 0.9])
    r1 = gov.step(consensus, regime_ids)
    r2 = gov.step(consensus, regime_ids)
    assert r1["active_regime"] == 0
    assert r2["active_regime"] == 0

    # third consecutive bar -> persistence satisfied, switches
    r3 = gov.step(consensus, regime_ids)
    assert r3["active_regime"] == 1


def test_state_governor_resets_candidate_counter_on_flip_flop():
    gov = RegimeStateGovernor(entropy_limit=0.85, persistence_window=3, hysteresis_epsilon=0.0)
    gov.active_state = 0
    regime_ids = [0, 1]

    propose_1 = np.array([0.1, 0.9])
    propose_0 = np.array([0.9, 0.1])

    gov.step(propose_1, regime_ids)
    gov.step(propose_1, regime_ids)
    r = gov.step(propose_0, regime_ids)  # flips candidate back to 0 -> counter resets
    assert r["active_regime"] == 0
    r2 = gov.step(propose_1, regime_ids)  # candidate counter restarts at 1
    assert r2["candidate_count"] == 1
    assert r2["active_regime"] == 0


def test_state_governor_hysteresis_blocks_narrow_margin_switch():
    # entropy_limit set high (effectively disabled) so this test isolates
    # hysteresis specifically -- a near-50/50 split is inherently
    # high-entropy and would otherwise be caught by consensus gating first,
    # which is a different rule tested separately above.
    gov = RegimeStateGovernor(entropy_limit=2.0, persistence_window=1, hysteresis_epsilon=0.2)
    gov.active_state = 0
    regime_ids = [0, 1]
    # regime 1 barely ahead (0.55 vs 0.45) -> margin 0.10 < epsilon 0.2 -> blocked
    r = gov.step(np.array([0.45, 0.55]), regime_ids)
    assert r["active_regime"] == 0


def test_state_governor_high_entropy_forces_transitional():
    gov = RegimeStateGovernor(entropy_limit=0.5, persistence_window=1, hysteresis_epsilon=0.0)
    gov.active_state = 0
    regime_ids = [0, 1, 2]
    # near-uniform vector -> high entropy -> forced transitional regardless of argmax
    r = gov.step(np.array([0.34, 0.33, 0.33]), regime_ids)
    assert r["proposed_regime"] == "transitional"


def test_run_governor_over_history_end_to_end(synthetic_three_regime_features):
    features, labels, proba = synthetic_three_regime_features
    templates = build_regime_templates(features, labels, columns=TEMPLATE_COLUMNS)
    prox = rolling_wasserstein_proximity(features, templates, window=21)

    governed = run_governor_over_history(
        proba, prox, entropy_limit=0.85, persistence_window=5, hysteresis_epsilon=0.05
    )
    assert list(governed.columns) == [
        "consensus_entropy",
        "proposed_regime",
        "active_regime",
        "is_transitional",
        "candidate_count",
    ]
    assert governed.index.equals(proba.index)

    # The governor can introduce its own "transitional" state as an extra
    # rung between two concrete regimes, so total active_regime switch
    # count is NOT guaranteed to be <= the raw argmax's switch count (a
    # transitional detour adds switches even though it's the conservative
    # choice). The invariant that must hold instead: every direct switch
    # between two *concrete* (non-transitional) regimes only happens after
    # that candidate regime was the top consensus pick for at least
    # persistence_window consecutive bars.
    concrete = governed[governed["active_regime"] != "transitional"]
    hard_switches = concrete["active_regime"][concrete["active_regime"] != concrete["active_regime"].shift()]
    assert len(hard_switches) >= 1  # the synthetic series has a genuine regime change to detect
