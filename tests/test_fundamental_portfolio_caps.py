"""Tests for sector/position weight caps in
``strategies.build_fundamental_portfolio_weights`` -- added after
fundamentals_only's max drawdown was repeatedly worse than the raw
benchmark's across every real backtest run, a sector-concentration
signature (confirmed further when regime-conditional technical_momentum
weighting's fallback -- the other 8 dimensions -- inherited the same
worse-than-benchmark drawdown standalone).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.strategies import build_fundamental_portfolio_weights

DATE = pd.Timestamp("2020-01-01")


def _scores_with_dominant_sector():
    rows = []
    for i in range(8):
        rows.append({"date": DATE, "symbol": f"TECH{i}", "sector": "Tech", "composite_score": 2.0 - i * 0.05})
    for i in range(4):
        rows.append({"date": DATE, "symbol": f"FIN{i}", "sector": "Financials", "composite_score": 1.5 - i * 0.05})
    for i in range(4):
        rows.append({"date": DATE, "symbol": f"ENE{i}", "sector": "Energy", "composite_score": 1.0 - i * 0.05})
    for i in range(4):
        rows.append({"date": DATE, "symbol": f"HLT{i}", "sector": "Healthcare", "composite_score": 0.5 - i * 0.05})
    return pd.DataFrame(rows)


def test_no_cap_produces_fully_concentrated_selection_when_scores_favor_one_sector():
    """Sanity check on the fixture itself: without a cap, the top 8 by
    score really are all Tech -- otherwise the cap tests below wouldn't be
    testing anything."""
    scores = _scores_with_dominant_sector()
    weights = build_fundamental_portfolio_weights(scores, top_quantile=0.4, min_positions=5)
    selected = weights.loc[DATE]
    selected = selected[selected > 0]
    sectors = scores.set_index("symbol").loc[selected.index, "sector"]
    assert (sectors == "Tech").all()


def test_sector_cap_diversifies_and_still_fills_the_portfolio():
    scores = _scores_with_dominant_sector()
    weights = build_fundamental_portfolio_weights(scores, top_quantile=0.4, min_positions=5, max_sector_weight=0.3)
    selected = weights.loc[DATE]
    selected = selected[selected > 0]
    sectors = scores.set_index("symbol").loc[selected.index, "sector"]
    sector_counts = sectors.value_counts()

    assert len(selected) == 8  # portfolio still fully filled despite the cap
    assert selected.sum() == pytest.approx(1.0)  # fully invested, not shrunk
    assert sector_counts.max() <= max(1, int(8 * 0.3))  # no sector exceeds the cap
    assert sector_counts.nunique() > 1 or len(sector_counts) > 1  # genuinely diversified, not still one sector


def test_sector_cap_prefers_next_best_candidate_over_shrinking_portfolio():
    """The specific 'skip and backfill from elsewhere' behavior, not just
    the aggregate diversification outcome -- directly checks that TECH2
    (rank 3 within Tech, which the cap should exclude) is replaced by
    FIN0 (the next-best candidate overall), not simply dropped."""
    scores = _scores_with_dominant_sector()
    weights = build_fundamental_portfolio_weights(scores, top_quantile=0.4, min_positions=5, max_sector_weight=0.3)
    selected = set(weights.loc[DATE][weights.loc[DATE] > 0].index)
    assert "TECH0" in selected and "TECH1" in selected  # top 2 Tech names still make it (cap allows 2)
    assert "TECH2" not in selected  # 3rd Tech name excluded by the cap
    assert "FIN0" in selected  # backfilled from the next-best sector instead


def test_max_sector_weight_missing_sector_column_warns_and_no_ops(caplog):
    scores = pd.DataFrame([
        {"date": DATE, "symbol": f"S{i}", "composite_score": 1.0 - i * 0.1} for i in range(10)
    ])
    with caplog.at_level("WARNING"):
        weights = build_fundamental_portfolio_weights(scores, top_quantile=0.5, min_positions=3, max_sector_weight=0.3)
    selected = weights.loc[DATE]
    assert (selected[selected > 0] == pytest.approx(0.2)).all()  # normal equal-weight top-5, cap had no effect
    assert any("no 'sector' column" in r.message for r in caplog.records)


def test_max_position_weight_caps_and_shrinks_exposure_rather_than_renormalizing():
    scores = pd.DataFrame([
        {"date": DATE, "symbol": f"S{i}", "sector": "X", "composite_score": 1.0 - i * 0.1} for i in range(3)
    ])
    weights = build_fundamental_portfolio_weights(scores, top_quantile=1.0, min_positions=3, max_position_weight=0.20)
    selected = weights.loc[DATE]
    selected = selected[selected > 0]
    assert (selected == pytest.approx(0.20)).all()
    assert selected.sum() == pytest.approx(0.60)  # NOT renormalized back to 1.0


def test_max_position_weight_none_default_unchanged():
    scores = pd.DataFrame([
        {"date": DATE, "symbol": f"S{i}", "sector": "X", "composite_score": 1.0 - i * 0.1} for i in range(3)
    ])
    weights = build_fundamental_portfolio_weights(scores, top_quantile=1.0, min_positions=3)
    selected = weights.loc[DATE]
    selected = selected[selected > 0]
    assert selected.sum() == pytest.approx(1.0)


def test_no_caps_reproduces_exact_prior_behavior():
    """Both caps default to None -- must reproduce identical output to
    before caps existed, for anyone not opting in."""
    scores = _scores_with_dominant_sector()
    weights_a = build_fundamental_portfolio_weights(scores, top_quantile=0.4, min_positions=5)
    weights_b = build_fundamental_portfolio_weights(
        scores, top_quantile=0.4, min_positions=5, max_sector_weight=None, max_position_weight=None
    )
    pd.testing.assert_frame_equal(weights_a, weights_b)


def test_sector_cap_handles_insufficient_diversity_by_backfilling_anyway():
    """If EVERY candidate is the same sector, the cap can't be satisfied --
    must still fill the portfolio (not shrink it to just the capped
    amount), since a cap enforces diversification among viable
    alternatives, not portfolio size when there isn't any alternative."""
    scores = pd.DataFrame([
        {"date": DATE, "symbol": f"S{i}", "sector": "OnlySector", "composite_score": 1.0 - i * 0.1}
        for i in range(6)
    ])
    weights = build_fundamental_portfolio_weights(scores, top_quantile=1.0, min_positions=6, max_sector_weight=0.3)
    selected = weights.loc[DATE]
    selected = selected[selected > 0]
    assert len(selected) == 6  # fully filled despite cap being technically unsatisfiable
    assert selected.sum() == pytest.approx(1.0)
