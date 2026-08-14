"""Tests for ``build_annual_growth_history_pit`` and its wiring into
``run_pit_fundamental_pipeline`` — fixes a real, previously silent gap
where the ``growth`` fundamentals dimension was unconditionally skipped
(``history=None`` hardcoded) at every rebalance date in every real
backtest. See ``point_in_time.py``'s docstring for the full story.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamental_analysis.data_fetchers.fundamentals_fetcher import SNAPSHOT_SCHEMA
from src.fundamental_analysis.metrics.growth import compute_growth_metrics
from src.fundamental_analysis.point_in_time import (
    build_annual_growth_history_pit,
    run_pit_fundamental_pipeline,
)


def _synthetic_quarterly(symbols_and_bases: dict, start_year: int = 2015, end_year: int = 2022, annual_growth: float = 0.08) -> pd.DataFrame:
    rows = []
    for symbol, base in symbols_and_bases.items():
        for year in range(start_year, end_year + 1):
            for month in (3, 6, 9, 12):
                period_end = pd.Timestamp(year=year, month=month, day=28)
                known_date = period_end + pd.Timedelta(days=45)
                factor = (1.0 + annual_growth) ** (year - start_year)
                for field, val in (
                    ("revenue", base * factor),
                    ("net_income", base * 0.15 * factor),
                    ("eps", 2.0 * factor),
                ):
                    rows.append(
                        {"symbol": symbol, "period_end": period_end, "known_date": known_date, "field": field, "value": val}
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def quarterly_history():
    return _synthetic_quarterly({"AAA": 100.0, "BBB": 50.0})


def test_build_annual_growth_history_pit_produces_expected_shape(quarterly_history):
    annual = build_annual_growth_history_pit(quarterly_history, "2022-06-30")
    assert set(annual.columns) == {"symbol", "fiscal_year", "revenue", "net_income", "eps"}
    assert set(annual["symbol"].unique()) == {"AAA", "BBB"}
    # 2015..2022 inclusive = 8 fiscal years per symbol (once TTM windows exist)
    assert annual[annual["symbol"] == "AAA"]["fiscal_year"].tolist() == sorted(
        annual[annual["symbol"] == "AAA"]["fiscal_year"].tolist()
    )


def test_build_annual_growth_history_pit_respects_pit_discipline(quarterly_history):
    # As of the very start, there's not even one full TTM window yet -> empty
    early = build_annual_growth_history_pit(quarterly_history, "2015-03-01")
    assert early.empty

    # As of a date only 2 quarters in, still not enough for a TTM window
    still_early = build_annual_growth_history_pit(quarterly_history, "2015-09-01")
    assert still_early.empty

    # As of a date with a full 4 quarters known (Dec-2015 quarter's known_date
    # is period_end + 45 days = 2016-02-11), exactly one annual point appears
    one_year_in = build_annual_growth_history_pit(quarterly_history, "2016-03-01")
    assert not one_year_in.empty
    assert one_year_in["fiscal_year"].max() <= 2016


def test_build_annual_growth_history_pit_never_uses_future_quarters(quarterly_history):
    """A later as_of_date must never produce FEWER or DIFFERENT early-year
    annual figures than an earlier as_of_date already saw -- i.e. later
    information can only extend the series forward, never rewrite the past
    (that would be look-ahead leaking backward, which shouldn't be possible
    given the known_date <= as_of filter, but is worth pinning down)."""
    snapshot_2018 = build_annual_growth_history_pit(quarterly_history, "2018-06-30")
    snapshot_2022 = build_annual_growth_history_pit(quarterly_history, "2022-06-30")

    early_2018 = snapshot_2018[snapshot_2018["fiscal_year"] == 2016].set_index("symbol")
    early_2022 = snapshot_2022[snapshot_2022["fiscal_year"] == 2016].set_index("symbol")
    pd.testing.assert_frame_equal(early_2018.sort_index(), early_2022.sort_index())


def test_build_annual_growth_history_pit_feeds_compute_growth_metrics_correctly(quarterly_history):
    annual = build_annual_growth_history_pit(quarterly_history, "2022-06-30")
    metrics = compute_growth_metrics(annual, min_years=3)
    assert "revenue_cagr" in metrics.columns
    assert metrics.loc["AAA", "revenue_cagr"] == pytest.approx(0.08, abs=0.02)  # ~8% synthetic growth rate
    assert metrics["revenue_cagr"].notna().all()


def test_build_annual_growth_history_pit_empty_input_returns_empty_shaped_frame():
    empty = pd.DataFrame(columns=["symbol", "period_end", "known_date", "field", "value"])
    result = build_annual_growth_history_pit(empty, "2022-01-01")
    assert result.empty
    assert set(result.columns) == {"symbol", "fiscal_year", "revenue", "net_income", "eps"}


def test_run_pit_fundamental_pipeline_actually_computes_growth_not_skipped(quarterly_history, caplog):
    """The core regression test for the bug: growth columns must be
    present and non-null for at least the later rebalance dates, not
    silently all-NaN/absent because history=None was hardcoded."""
    snapshot = pd.DataFrame(index=["AAA", "BBB"], columns=SNAPSHOT_SCHEMA, dtype=float)
    snapshot["sector"] = "Tech"

    config = {
        "sector_relative": False,
        "dimensions": {
            "valuation": True, "profitability_quality": True, "growth": True,
            "leverage_solvency": True, "cashflow_quality": True,
            "ownership_governance": True, "earnings_surprise": True, "options_earnings": False,
        },
        "composite_weights": {
            "valuation": 0.1425, "profitability_quality": 0.19, "growth": 0.1425,
            "leverage_solvency": 0.1425, "cashflow_quality": 0.1425,
            "ownership_governance": 0.1425, "earnings_surprise": 0.09,
        },
    }
    rebalance_dates = pd.date_range("2015-01-01", "2022-12-31", freq="MS")

    with caplog.at_level("WARNING"):
        scores = run_pit_fundamental_pipeline(config, snapshot, quarterly_history, rebalance_dates)

    assert "revenue_cagr" in scores.columns
    assert scores["revenue_cagr"].notna().sum() > 0
    # the old bug logged this on EVERY rebalance date; it should now only
    # ever fire (if at all) for the earliest dates lacking a full TTM window
    growth_skipped_warnings = [r for r in caplog.records if "growth dimension enabled but no" in r.message]
    n_dates = scores["date"].nunique()
    assert len(growth_skipped_warnings) < n_dates * len(["AAA", "BBB"])
