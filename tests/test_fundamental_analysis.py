"""Synthetic-data tests for the fundamental analysis module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamental_analysis.metrics.cashflow_quality import compute_cashflow_quality_metrics
from src.fundamental_analysis.metrics.earnings_surprise import compute_earnings_surprise_metrics
from src.fundamental_analysis.metrics.growth import compute_growth_metrics
from src.fundamental_analysis.metrics.leverage_solvency import compute_leverage_solvency_metrics
from src.fundamental_analysis.metrics.ownership_governance import (
    compute_ownership_governance_metrics,
)
from src.fundamental_analysis.metrics.profitability_quality import (
    compute_profitability_quality_metrics,
    piotroski_f_score,
)
from src.fundamental_analysis.metrics.valuation import compute_valuation_metrics
from src.fundamental_analysis.pipeline import run_pipeline
from src.fundamental_analysis.scoring.composite_score import (
    compute_composite_score,
    compute_dimension_scores,
)


@pytest.fixture
def synthetic_snapshot() -> pd.DataFrame:
    symbols = [f"STOCK{i}" for i in range(12)]
    rng = np.random.default_rng(1)
    n = len(symbols)
    df = pd.DataFrame(
        {
            "symbol": symbols,
            "sector": (["IT"] * 6 + ["BANKING"] * 6),
            "industry": (["Software"] * 6 + ["Private Bank"] * 6),
            "price": rng.uniform(100, 3000, n),
            "eps_ttm": rng.uniform(5, 150, n),
            "eps_growth_pct": rng.uniform(5, 25, n),
            "book_value_per_share": rng.uniform(50, 800, n),
            "dividend_per_share": rng.uniform(0, 40, n),
            "ebitda": rng.uniform(1e8, 5e9, n),
            "ebit": rng.uniform(8e7, 4.5e9, n),
            "enterprise_value": rng.uniform(1e9, 5e10, n),
            "revenue": rng.uniform(5e8, 2e10, n),
            "revenue_prior": rng.uniform(4e8, 1.9e10, n),
            "net_income": rng.uniform(5e7, 3e9, n),
            "net_income_prior": rng.uniform(4e7, 2.8e9, n),
            "total_assets": rng.uniform(1e9, 4e10, n),
            "total_assets_prior": rng.uniform(9e8, 3.8e10, n),
            "total_liabilities": rng.uniform(5e8, 2e10, n),
            "total_equity": rng.uniform(5e8, 2e10, n),
            "total_debt": rng.uniform(0, 1e10, n),
            "interest_expense": rng.uniform(1e6, 5e8, n),
            "retained_earnings": rng.uniform(1e8, 1e10, n),
            "current_assets": rng.uniform(5e8, 1e10, n),
            "current_assets_prior": rng.uniform(4.5e8, 9.5e9, n),
            "current_liabilities": rng.uniform(2e8, 5e9, n),
            "current_liabilities_prior": rng.uniform(1.8e8, 4.8e9, n),
            "long_term_debt": rng.uniform(0, 5e9, n),
            "long_term_debt_prior": rng.uniform(0, 5.5e9, n),
            "gross_profit": rng.uniform(2e8, 1.5e10, n),
            "gross_profit_prior": rng.uniform(1.8e8, 1.4e10, n),
            "shares_outstanding": rng.uniform(1e7, 5e8, n),
            "shares_outstanding_prior": rng.uniform(1e7, 5e8, n),
            "cfo": rng.uniform(5e7, 4e9, n),
            "capex": rng.uniform(1e6, 1e9, n),
            "market_cap": rng.uniform(1e9, 5e10, n),
            "promoter_holding_pct": rng.uniform(20, 75, n),
            "promoter_holding_pct_prior": rng.uniform(20, 75, n),
            "promoter_pledge_pct": rng.uniform(0, 60, n),
            "fii_holding_pct": rng.uniform(5, 40, n),
            "fii_holding_pct_prior": rng.uniform(5, 40, n),
            "dii_holding_pct": rng.uniform(5, 30, n),
            "dii_holding_pct_prior": rng.uniform(5, 30, n),
            "related_party_transactions_flag": rng.integers(0, 2, n),
            "auditor_changed_flag": rng.integers(0, 2, n),
            "actual_eps": rng.uniform(5, 150, n),
            "analyst_eps_estimate": rng.uniform(5, 150, n),
            "analyst_eps_estimate_30d_ago": rng.uniform(5, 150, n),
        }
    )
    return df.set_index("symbol", drop=False)


@pytest.fixture
def synthetic_history() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for i in range(12):
        symbol = f"STOCK{i}"
        revenue = 1e9
        net_income = 1e8
        eps = 20.0
        for year in range(2019, 2024):
            revenue *= 1 + rng.uniform(0.03, 0.20)
            net_income *= 1 + rng.uniform(-0.05, 0.25)
            eps *= 1 + rng.uniform(-0.05, 0.25)
            rows.append(
                {"symbol": symbol, "fiscal_year": year, "revenue": revenue,
                 "net_income": net_income, "eps": eps}
            )
    return pd.DataFrame(rows)


def test_each_metric_module_runs(synthetic_snapshot):
    for fn in [
        compute_valuation_metrics,
        compute_profitability_quality_metrics,
        compute_leverage_solvency_metrics,
        compute_cashflow_quality_metrics,
        compute_ownership_governance_metrics,
        compute_earnings_surprise_metrics,
    ]:
        out = fn(synthetic_snapshot)
        assert len(out) == len(synthetic_snapshot)
        assert out.index.equals(synthetic_snapshot.index)


def test_growth_metrics(synthetic_history):
    out = compute_growth_metrics(synthetic_history)
    assert len(out) == synthetic_history["symbol"].nunique()
    assert (out["revenue_cagr"] > -1).all()  # sanity: not garbage


def test_piotroski_score_bounds(synthetic_snapshot):
    score = piotroski_f_score(synthetic_snapshot)
    assert (score >= 0).all() and (score <= 9).all()


def test_piotroski_missing_prior_data_not_penalized():
    """A stock with zero prior-year data should score purely on the four
    signals computable from current-year data alone (0-4), never lower just
    because five signals were un-computable."""
    df = pd.DataFrame(
        {
            "net_income": [100.0],
            "total_assets": [1000.0],
            "cfo": [150.0],
            "total_equity": [500.0],
            "ebit": [200.0],
            "current_liabilities": [200.0],
        },
        index=["STOCK_NO_HISTORY"],
    )
    score = piotroski_f_score(df)
    assert score.iloc[0] <= 4
    assert score.iloc[0] >= 0


def test_composite_score_pipeline(synthetic_snapshot, synthetic_history):
    config = {
        "sector_relative": True,
        "dimensions": {
            "valuation": True,
            "profitability_quality": True,
            "growth": True,
            "leverage_solvency": True,
            "cashflow_quality": True,
            "ownership_governance": True,
            "earnings_surprise": True,
        },
        "composite_weights": {
            "valuation": 0.15,
            "profitability_quality": 0.20,
            "growth": 0.15,
            "leverage_solvency": 0.15,
            "cashflow_quality": 0.15,
            "ownership_governance": 0.10,
            "earnings_surprise": 0.10,
        },
    }
    result = run_pipeline(config, synthetic_snapshot, synthetic_history)
    assert "composite_score" in result.columns
    assert len(result) == len(synthetic_snapshot)
    # Ranked descending by composite score
    assert (result["composite_score"].dropna().diff().dropna() <= 1e-9).all()


def test_sector_relative_scoring_uses_sector_not_universe(synthetic_snapshot):
    metrics = compute_valuation_metrics(synthetic_snapshot)
    dim_scores = compute_dimension_scores(metrics, synthetic_snapshot["sector"])
    assert "valuation" in dim_scores.columns
    # Each sector's scores should be roughly centered near 0 independently.
    it_mean = dim_scores.loc[synthetic_snapshot["sector"] == "IT", "valuation"].mean()
    bank_mean = dim_scores.loc[synthetic_snapshot["sector"] == "BANKING", "valuation"].mean()
    assert abs(it_mean) < 1.5
    assert abs(bank_mean) < 1.5


# --- point-in-time fundamentals (src/fundamental_analysis/point_in_time.py) ---

from src.fundamental_analysis.point_in_time import build_pit_panel, build_pit_snapshot


@pytest.fixture
def quarterly_history_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": "AAA", "period_end": pd.Timestamp("2023-03-31"), "known_date": pd.Timestamp("2023-05-15"), "field": "revenue", "value": 100.0},
        {"symbol": "AAA", "period_end": pd.Timestamp("2023-06-30"), "known_date": pd.Timestamp("2023-08-14"), "field": "revenue", "value": 110.0},
        {"symbol": "AAA", "period_end": pd.Timestamp("2023-09-30"), "known_date": pd.Timestamp("2023-11-14"), "field": "revenue", "value": 120.0},
        {"symbol": "BBB", "period_end": pd.Timestamp("2023-03-31"), "known_date": pd.Timestamp("2023-05-10"), "field": "revenue", "value": 50.0},
    ])


def test_pit_snapshot_has_no_data_before_first_known_result(quarterly_history_fixture):
    snapshot = build_pit_snapshot(quarterly_history_fixture, "2023-04-01")
    assert snapshot.empty or snapshot["revenue"].isna().all()


def test_pit_snapshot_forward_fills_between_results_without_lookahead(quarterly_history_fixture):
    # Strictly between AAA's 1st (known 2023-05-15) and 2nd (known 2023-08-14)
    # results: must see the 1st value, must NOT see the 2nd or 3rd.
    snapshot = build_pit_snapshot(quarterly_history_fixture, "2023-07-01")
    assert snapshot.loc["AAA", "revenue"] == 100.0

    # After the 2nd but before the 3rd: must see the 2nd value (110), not the
    # 3rd (120) — this is the core no-look-ahead guarantee.
    snapshot2 = build_pit_snapshot(quarterly_history_fixture, "2023-10-01")
    assert snapshot2.loc["AAA", "revenue"] == 110.0


def test_pit_panel_matches_pit_snapshot_at_each_date(quarterly_history_fixture):
    dates = pd.to_datetime(["2023-04-01", "2023-07-01", "2023-10-01", "2023-12-01"])
    panel = build_pit_panel(quarterly_history_fixture, dates)
    for date in dates:
        expected = build_pit_snapshot(quarterly_history_fixture, date)
        actual = panel[(panel["date"] == date) & (panel["symbol"] == "AAA")]
        actual_value = actual.loc[actual["field"] == "revenue", "value"]
        expected_value = expected["revenue"].get("AAA") if not expected.empty else np.nan
        if pd.isna(expected_value):
            assert actual_value.isna().all()
        else:
            assert actual_value.iloc[0] == expected_value


# --- pre-earnings options signal (src/fundamental_analysis/metrics/options_earnings.py) ---

from src.fundamental_analysis.metrics.options_earnings import compute_options_earnings_metrics


@pytest.fixture
def options_history_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rows = []
    earnings_date = pd.Timestamp("2024-01-15")
    for d in dates:
        iv = 20.0
        if earnings_date - pd.Timedelta(days=5) <= d < earnings_date:
            iv = 45.0  # elevated pre-earnings IV window
        if d >= earnings_date:
            iv = 90.0  # post-earnings spike — must never leak into the pre-earnings read
        rows.append({"symbol": "AAA", "date": d, "atm_iv": iv, "put_call_oi_ratio": 1.1, "implied_move_pct": 0.05})
    history = pd.DataFrame(rows)
    calendar = pd.DataFrame([
        {"symbol": "AAA", "earnings_date": earnings_date},
        {"symbol": "AAA", "earnings_date": pd.Timestamp("2024-02-20")},  # future relative to as_of below
    ])
    snapshot = pd.DataFrame(index=["AAA"])
    return snapshot, history, calendar


def test_options_earnings_all_nan_before_any_known_earnings(options_history_fixture):
    snapshot, history, calendar = options_history_fixture
    result = compute_options_earnings_metrics(snapshot, history, calendar, as_of_date="2024-01-10")
    assert result.loc["AAA"].isna().all()


def test_options_earnings_uses_pre_earnings_window_not_post_earnings_spike(options_history_fixture):
    snapshot, history, calendar = options_history_fixture
    result = compute_options_earnings_metrics(snapshot, history, calendar, as_of_date="2024-01-20")
    # 45 (pre-earnings window) should dominate the percentile calc; the 90
    # post-earnings spike must not have leaked into the pre-earnings mean.
    assert result.loc["AAA", "pre_earnings_iv_percentile"] < 1.0
    assert not pd.isna(result.loc["AAA", "pre_earnings_put_call_oi_ratio"])


def test_options_earnings_ignores_future_earnings_dates(options_history_fixture):
    snapshot, history, calendar = options_history_fixture
    # As of right before the 2nd (future, unannounced-as-of-this-date)
    # earnings date, the anchor must still be the 1st (already-known) one.
    result = compute_options_earnings_metrics(snapshot, history, calendar, as_of_date="2024-02-19")
    assert not result.loc["AAA"].isna().all()
