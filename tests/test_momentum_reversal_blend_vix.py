# Destination: tests/test_momentum_reversal_blend_vix.py  (modified -- trimmed)
"""Tests for the ladder helpers in src/backtesting/momentum_reversal_blend.py:
``_ladder_from_labels`` and ``stress_from_regime``. Both are source-agnostic
(they don't care whether the label Series came from GMM or the VIX-bucket
production regime) so they're tested here independent of either.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.momentum_reversal_blend import _ladder_from_labels, stress_from_regime


def test_ladder_from_labels_default_matches_observed_range():
    labels = pd.Series([0, 1, 2, 1, 0])
    mapping = _ladder_from_labels(labels)
    assert mapping == {0: 0.0, 1: 0.5, 2: 1.0}


def test_ladder_from_labels_single_label():
    labels = pd.Series([2, 2, 2])
    mapping = _ladder_from_labels(labels)
    assert mapping == {2: 0.0}


def test_ladder_from_labels_n_labels_overrides_observed_range():
    """If a window only ever observed buckets {0, 1} but the full
    configured count is 4, the ladder must still span 0..3 -- otherwise a
    calm-only window's bucket 1 would incorrectly map to stress=1.0
    (maximum) instead of a middling value."""
    labels = pd.Series([0, 1, 1, 0])
    mapping = _ladder_from_labels(labels, n_labels=4)
    assert mapping[0] == 0.0
    assert mapping[1] == pytest.approx(1 / 3)
    assert 2 in mapping and 3 in mapping
    assert mapping[3] == 1.0


def test_stress_from_regime_ffill_between_sparse_labels():
    """``regime`` is typically only defined on a sparser index than the
    daily index it gets reindexed onto -- ffill should carry the last-known
    label forward across the gaps."""
    dates = pd.bdate_range("2021-01-01", periods=5)
    sparse_regime = pd.Series([0, 2], index=[dates[0], dates[3]])
    stress = stress_from_regime(sparse_regime, dates)
    assert stress.iloc[1] == 0.0
    assert stress.iloc[2] == 0.0
    assert stress.iloc[4] == 1.0


def test_stress_from_regime_bounded_unit_interval():
    dates = pd.bdate_range("2021-01-01", periods=100)
    regime = pd.Series([i % 4 for i in range(100)], index=dates)
    stress = stress_from_regime(regime, dates)
    assert (stress >= 0).all() and (stress <= 1).all()
