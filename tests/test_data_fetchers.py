"""Tests for the Screener / Trendlyne / yfinance data fetchers and the
multi-source merge layer.

None of this hits the network — the build sandbox this was developed in has
no outbound internet access at all (see docs/data_sourcing_spec.md). Instead:
  - screener_fetcher / trendlyne_fetcher are tested against fixture HTML
    (tests/fixtures/screener_sample.html, and inline HTML for Trendlyne)
    that mimics each site's known structure, so the *parsing* logic is
    verified even though the *scraping* (network I/O) isn't.
  - yfinance_fetcher and merge.py are tested with dependency-injected fake
    data (a fixture info dict, a fake downloader callable) rather than a
    live yfinance call.
Both fetch_company_snapshot/fetch_equity_snapshot (network) and
parse_company_page/parse_equity_page (pure parsing) are exposed separately
in the fetcher modules specifically so this split is possible.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.scraping_utils import DiskCache, rate_limited_get
from src.fundamental_analysis.data_fetchers import screener_fetcher, trendlyne_fetcher, yfinance_fetcher
from src.fundamental_analysis.data_fetchers.merge import merge_field_records, merge_sources

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# Screener
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def screener_fields():
    html = (FIXTURES_DIR / "screener_sample.html").read_text()
    return screener_fetcher.parse_company_page(html)


def test_screener_top_ratios(screener_fields):
    assert screener_fields["price"] == pytest.approx(1500.0)
    assert screener_fields["market_cap"] == pytest.approx(120000 * 1e7)
    assert screener_fields["book_value_per_share"] == pytest.approx(450.0)
    assert screener_fields["eps_ttm"] == pytest.approx(58.80)
    assert screener_fields["roe"] == pytest.approx(15.10)
    assert screener_fields["roce"] == pytest.approx(18.20)


def test_screener_uses_annual_not_ttm_for_current_prior(screener_fields):
    # Fixture P&L: Sales = [8000, 9200, 10500, TTM=11000] (Cr) — TTM must be
    # excluded so "current" is Mar 2024 (10500), not the rolling TTM (11000).
    assert screener_fields["revenue"] == pytest.approx(10500 * 1e7)
    assert screener_fields["revenue_prior"] == pytest.approx(9200 * 1e7)
    assert screener_fields["net_income"] == pytest.approx(1433 * 1e7)
    assert screener_fields["net_income_prior"] == pytest.approx(1118 * 1e7)


def test_screener_balance_sheet_and_derived_liabilities(screener_fields):
    assert screener_fields["total_assets"] == pytest.approx(10440 * 1e7)
    assert screener_fields["total_assets_prior"] == pytest.approx(9240 * 1e7)
    # total_equity = equity_capital + reserves = 240 + 6300 = 6540 (Cr)
    assert screener_fields["total_equity"] == pytest.approx(6540 * 1e7)
    # total_liabilities is DERIVED (total_assets - total_equity), not scraped
    # from Screener's misleading "Total Liabilities" row — see module docstring.
    expected_total_liabilities = (10440 - 6540) * 1e7
    assert screener_fields["total_liabilities"] == pytest.approx(expected_total_liabilities)
    assert screener_fields["total_debt"] == pytest.approx(1400 * 1e7)


def test_screener_cash_flow_and_approximated_fields(screener_fields):
    assert screener_fields["cfo"] == pytest.approx(1600 * 1e7)
    # capex approximated as -(investing cash flow) when investing CF is an outflow
    assert screener_fields["capex"] == pytest.approx(850 * 1e7)
    # ebit approximated as PBT + interest = 1910 + 190 = 2100 (Cr)
    assert screener_fields["ebit"] == pytest.approx(2100 * 1e7)
    # ebitda approximated as ebit + depreciation = 2100 + 340 = 2440 (Cr)
    assert screener_fields["ebitda"] == pytest.approx(2440 * 1e7)
    # dividend_per_share approximated as price * yield% = 1500 * 0.005
    assert screener_fields["dividend_per_share"] == pytest.approx(7.5)


def test_screener_shareholding_pattern(screener_fields):
    assert screener_fields["promoter_holding_pct"] == pytest.approx(54.80)
    assert screener_fields["promoter_holding_pct_prior"] == pytest.approx(55.00)
    assert screener_fields["fii_holding_pct"] == pytest.approx(18.90)
    assert screener_fields["dii_holding_pct"] == pytest.approx(15.60)


def test_screener_honestly_reports_unavailable_fields_as_nan(screener_fields):
    # These are NOT on Screener's free page at all — must stay NaN, not a
    # fabricated proxy value.
    for field in ["current_assets", "current_liabilities", "gross_profit", "promoter_pledge_pct"]:
        assert math.isnan(screener_fields[field]), f"{field} should be NaN (not available from Screener)"


def test_screener_history_table_parses_multiple_years():
    html = (FIXTURES_DIR / "screener_sample.html").read_text()
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    pl = screener_fetcher._parse_data_table(soup, screener_fetcher.SECTION_IDS["profit_loss"])
    assert list(pl.columns) == ["Mar 2022", "Mar 2023", "Mar 2024", "TTM"]
    sales = screener_fetcher._lookup_row(pl, "sales")
    assert sales.tolist() == pytest.approx([8000, 9200, 10500, 11000])


def test_screener_missing_section_degrades_to_empty_not_exception():
    minimal_html = "<html><body><div id='top-ratios'></div></body></html>"
    fields = screener_fetcher.parse_company_page(minimal_html)
    assert math.isnan(fields["revenue"])
    assert math.isnan(fields["price"])


# --- period-label parsing for point-in-time fundamentals (screener_fetcher._parse_period_end) ---
# Regression coverage for a real bug: a live run against actual Screener pages
# came back with an entirely empty quarterly PIT history (and therefore a
# flat 0%-return backtest) because the original regex only accepted the
# annual-table style "Mon YYYY" and Screener's quarterly table uses other
# variants. _parse_period_end was loosened to handle all of these.

@pytest.mark.parametrize(
    "label,expected",
    [
        ("Mar 2024", pd.Timestamp("2024-03-31")),
        ("Mar-24", pd.Timestamp("2024-03-31")),
        ("Mar'24", pd.Timestamp("2024-03-31")),
        ("Sep\xa02023", pd.Timestamp("2023-09-30")),  # non-breaking space
        ("Sept 2023", pd.Timestamp("2023-09-30")),
        ("Dec 2020", pd.Timestamp("2020-12-31")),
        ("Jun-23", pd.Timestamp("2023-06-30")),
    ],
)
def test_parse_period_end_handles_real_world_label_variants(label, expected):
    assert screener_fetcher._parse_period_end(label) == expected


@pytest.mark.parametrize("label", ["TTM", "garbage", "", "Q1 FY24"])
def test_parse_period_end_returns_none_for_non_period_labels(label):
    assert screener_fetcher._parse_period_end(label) is None


def test_fetch_multiple_quarterly_history_warns_on_near_total_failure(monkeypatch, caplog):
    """If every symbol's quarterly fetch comes back empty (e.g. a markup or
    label-format change), this must log a loud warning immediately rather
    than silently returning an empty panel with no signal until the backtest
    quietly shows a flat 0% return many steps later."""
    def fake_fetch_quarterly_history(symbol, **kwargs):
        return pd.DataFrame(columns=["symbol", "period_end", "known_date", "field", "value"])

    monkeypatch.setattr(screener_fetcher, "fetch_quarterly_history", fake_fetch_quarterly_history)

    with caplog.at_level("WARNING"):
        result = screener_fetcher.fetch_multiple_quarterly_history(["AAA", "BBB", "CCC"])

    assert result.empty
    assert any("only 0/3 symbols yielded" in rec.message for rec in caplog.records)


# --- regression test for a real bug found against live data: fetch_quarterly_history
# was iterating quarters.index (row labels like "Sales", "Net Profit") instead of
# quarters.columns (period labels like "Mar 2024") when looking for periods to parse,
# so it silently extracted zero rows from a perfectly well-formed table — this is
# exactly what _parse_data_table's own docstring says the shape is ("indexed by row
# label, columns = period labels"), so this is now pinned down with a fixture that
# mirrors the exact shape confirmed against a live Screener page (see the
# fetch_raw_quarters_table probe output in the conversation this was diagnosed from).

QUARTERS_FIXTURE_HTML = """
<html><body>
  <section id="quarters" class="card card-large">
    <table class="data-table">
      <thead>
        <tr><th class="text"></th><th>Mar 2023</th><th>Jun 2023</th><th>Sep 2023</th></tr>
      </thead>
      <tbody>
        <tr><td class="text">Sales <button>+</button></td><td>2,400</td><td>2,600</td><td>2,750</td></tr>
        <tr><td class="text">Expenses <button>+</button></td><td>1,900</td><td>2,000</td><td>2,100</td></tr>
        <tr><td class="text">Operating Profit</td><td>500</td><td>600</td><td>650</td></tr>
        <tr><td class="text">OPM %</td><td>21</td><td>23</td><td>24</td></tr>
        <tr><td class="text">Net Profit <button>+</button></td><td>310</td><td>360</td><td>390</td></tr>
        <tr><td class="text">EPS in Rs</td><td>12.40</td><td>14.10</td><td>15.20</td></tr>
      </tbody>
    </table>
  </section>
</body></html>
"""


def test_fetch_raw_quarters_table_parses_period_columns_not_row_labels():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(QUARTERS_FIXTURE_HTML, "lxml")
    quarters = screener_fetcher._parse_data_table(soup, screener_fetcher.SECTION_IDS["quarters"])
    assert list(quarters.index) == ["Sales", "Expenses", "Operating Profit", "OPM %", "Net Profit", "EPS in Rs"]
    assert list(quarters.columns) == ["Mar 2023", "Jun 2023", "Sep 2023"]


def test_fetch_quarterly_history_extracts_rows_from_real_shaped_table(monkeypatch):
    """The actual regression guard: given a well-formed quarters table (row
    labels x period columns, same shape confirmed against a live Screener
    page), fetch_quarterly_history must produce one row per (period, field),
    NOT an empty result."""
    def fake_cached_get_text(session, url, cache, min_delay_seconds):
        return QUARTERS_FIXTURE_HTML

    monkeypatch.setattr(
        "src.common.scraping_utils.cached_get_text", fake_cached_get_text
    )

    history = screener_fetcher.fetch_quarterly_history("TESTCO", reporting_lag_days=45)

    assert not history.empty, "fetch_quarterly_history returned nothing from a well-formed quarters table"
    assert set(history["field"].unique()) <= {"revenue", "net_income", "eps", "operating_profit_margin_pct"}
    assert set(history["period_end"].dt.strftime("%Y-%m-%d")) == {"2023-03-31", "2023-06-30", "2023-09-30"}

    revenue_rows = history[(history["symbol"] == "TESTCO") & (history["field"] == "revenue")]
    revenue_rows = revenue_rows.sort_values("period_end")
    # Sales 2,400 / 2,600 / 2,750 (in Cr) -> canonicalized to absolute rupees
    assert revenue_rows["value"].tolist() == [2400 * screener_fetcher.CR_TO_RUPEES,
                                               2600 * screener_fetcher.CR_TO_RUPEES,
                                               2750 * screener_fetcher.CR_TO_RUPEES]
    # known_date must be period_end + reporting_lag_days, never before it
    for _, row in revenue_rows.iterrows():
        assert row["known_date"] == row["period_end"] + pd.Timedelta(days=45)


# --------------------------------------------------------------------------
# Trendlyne
# --------------------------------------------------------------------------

TRENDLYNE_FREE_HTML = """
<html><body>
<div>Momentum Score: 39.4/100</div>
<div>Strengths 10</div>
<div>Weaknesses 7</div>
<div>Opportunities 5</div>
<div>Threats 0</div>
<div>Current Price Rs 1318.10</div>
<div>Durability Score: Not disclosed (requires GuruQ or StratQ subscription)</div>
<div>Valuation Score: Not disclosed (requires GuruQ or StratQ subscription)</div>
<div>Promoter pledge data is a premium feature (StratQ subscription)</div>
<div>1-Year Price Target: premium feature, requires subscription</div>
</body></html>
"""


def test_trendlyne_free_fields_extracted():
    fields = trendlyne_fetcher.parse_equity_page(TRENDLYNE_FREE_HTML)
    assert fields["momentum_score"] == pytest.approx(39.4)
    assert fields["swot_strengths"] == pytest.approx(10)
    assert fields["swot_weaknesses"] == pytest.approx(7)
    assert fields["swot_opportunities"] == pytest.approx(5)
    assert fields["swot_threats"] == pytest.approx(0)


def test_trendlyne_paywalled_fields_are_nan_with_reason():
    fields = trendlyne_fetcher.parse_equity_page(TRENDLYNE_FREE_HTML)
    for field in ["durability_score", "valuation_score", "promoter_pledge_pct", "analyst_target_price"]:
        assert math.isnan(fields[field]), f"{field} should be NaN (paywalled)"
        assert fields[f"{field}_reason"] is not None
        assert "subscription" in fields[f"{field}_reason"].lower()


def test_resolve_trendlyne_id_missing_file_returns_none(tmp_path):
    result = trendlyne_fetcher.resolve_trendlyne_id("RELIANCE", mapping_path=str(tmp_path / "nonexistent.csv"))
    assert result is None


def test_resolve_trendlyne_id_from_mapping(tmp_path):
    mapping = tmp_path / "map.csv"
    mapping.write_text("symbol,trendlyne_id,slug\nRELIANCE,1127,reliance-industries-ltd\n")
    result = trendlyne_fetcher.resolve_trendlyne_id("RELIANCE", mapping_path=str(mapping))
    assert result == "1127/RELIANCE/reliance-industries-ltd"

    missing = trendlyne_fetcher.resolve_trendlyne_id("TCS", mapping_path=str(mapping))
    assert missing is None


# --------------------------------------------------------------------------
# yfinance
# --------------------------------------------------------------------------

FAKE_YFINANCE_INFO = {
    "sector": "Energy",
    "industry": "Oil & Gas Refining",
    "currentPrice": 1490.0,
    "sharesOutstanding": 6_765_000_000,
    "marketCap": 1_500_000_000_000,
    "trailingEps": 62.5,
    "earningsGrowth": 0.12,
    "bookValue": 470.0,
    "dividendRate": 8.0,
    "grossProfits": 200_000_000_000,
    "ebitda": 300_000_000_000,
    "depreciationAndAmortization": 40_000_000_000,
    "enterpriseValue": 1_600_000_000_000,
    "totalRevenue": 900_000_000_000,
    "netIncomeToCommon": 70_000_000_000,
    "totalDebt": 250_000_000_000,
    "totalCurrentAssets": 400_000_000_000,
    "totalCurrentLiabilities": 300_000_000_000,
    "operatingCashflow": 150_000_000_000,
    "capitalExpenditures": -80_000_000_000,
    "forwardEps": 68.0,
    "targetMeanPrice": 1650.0,
    "numberOfAnalystOpinions": 34,
}


def test_yfinance_parse_ticker_info():
    fields = yfinance_fetcher.parse_ticker_info("RELIANCE", FAKE_YFINANCE_INFO)
    assert fields["price"] == 1490.0
    assert fields["eps_growth_pct"] == pytest.approx(12.0)
    assert fields["ebit"] == pytest.approx(260_000_000_000)  # ebitda - D&A
    assert fields["capex"] == pytest.approx(80_000_000_000)  # -(-80e9)
    assert fields["analyst_eps_estimate"] == 68.0


def test_yfinance_parse_ticker_info_handles_empty_response():
    fields = yfinance_fetcher.parse_ticker_info("UNKNOWN", {})
    assert fields["symbol"] == "UNKNOWN"
    assert fields["price"] is None


def test_yfinance_fetch_snapshot_with_injected_fn():
    fields = yfinance_fetcher.fetch_snapshot("RELIANCE", ticker_info_fn=lambda s: FAKE_YFINANCE_INFO)
    assert fields["market_cap"] == 1_500_000_000_000


def test_yfinance_fetch_price_panel_with_injected_downloader():
    dates = pd.bdate_range("2023-01-02", periods=5)
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["AAA.NS", "BBB.NS"]])
    fake_data = pd.DataFrame(
        np.arange(len(dates) * 4).reshape(len(dates), 4), index=dates, columns=columns
    )

    def fake_downloader(tickers, start, end):
        return fake_data

    panel = yfinance_fetcher.fetch_price_panel(["AAA", "BBB"], downloader=fake_downloader)
    assert list(panel.columns) == ["AAA", "BBB"]
    assert len(panel) == 5
    np.testing.assert_array_equal(panel["AAA"].values, fake_data["Close"]["AAA.NS"].values)


def test_yfinance_fetch_price_panel_ohlc_with_injected_downloader():
    """fetch_price_panel (Close-only) vs fetch_price_panel_ohlc (real OHLC)
    -- the latter exists specifically for indicators like Ichimoku that need
    true high/low, not a close-derived proxy (see backtesting/adaptive_ichimoku.py)."""
    dates = pd.bdate_range("2023-01-02", periods=5)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    tickers = ["AAA.NS", "BBB.NS"]
    columns = pd.MultiIndex.from_product([fields, tickers])
    fake_data = pd.DataFrame(
        np.arange(len(dates) * len(fields) * len(tickers)).reshape(len(dates), len(fields) * len(tickers)),
        index=dates, columns=columns,
    )

    def fake_downloader(tickers, start, end):
        return fake_data

    panel = yfinance_fetcher.fetch_price_panel_ohlc(["AAA", "BBB"], downloader=fake_downloader)
    assert set(panel.keys()) == {"AAA", "BBB"}
    assert list(panel["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    assert len(panel["AAA"]) == 5
    np.testing.assert_array_equal(panel["AAA"]["close"].values, fake_data["Close"]["AAA.NS"].values)
    np.testing.assert_array_equal(panel["AAA"]["high"].values, fake_data["High"]["AAA.NS"].values)


def test_yfinance_fetch_price_panel_ohlc_empty_download_returns_empty_dict():
    def fake_downloader(tickers, start, end):
        return pd.DataFrame()

    panel = yfinance_fetcher.fetch_price_panel_ohlc(["AAA"], downloader=fake_downloader)
    assert panel == {}


def test_yfinance_fetch_benchmark_series_with_injected_downloader():
    dates = pd.bdate_range("2023-01-02", periods=5)
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["^CRSLDX"]])
    fake_data = pd.DataFrame(
        np.arange(len(dates) * 2).reshape(len(dates), 2), index=dates, columns=columns
    )

    def fake_downloader(tickers, start, end):
        return fake_data

    series = yfinance_fetcher.fetch_benchmark_series(downloader=fake_downloader)
    assert len(series) == 5
    assert series.name == "close"


def test_yfinance_fetch_benchmark_ohlcv_with_injected_downloader():
    dates = pd.bdate_range("2023-01-02", periods=5)
    columns = pd.MultiIndex.from_product(
        [["Close", "Open", "High", "Low", "Volume"], ["^CRSLDX"]]
    )
    fake_data = pd.DataFrame(
        np.arange(len(dates) * 5).reshape(len(dates), 5), index=dates, columns=columns
    )

    def fake_downloader(tickers, start, end):
        return fake_data

    ohlcv = yfinance_fetcher.fetch_benchmark_ohlcv(downloader=fake_downloader)
    assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]
    assert len(ohlcv) == 5

    # fetch_benchmark_series must still work as a close-only view when the
    # richer OHLCV downloader is plugged in underneath it.
    series = yfinance_fetcher.fetch_benchmark_series(downloader=fake_downloader)
    assert series.name == "close"
    assert (series.values == ohlcv["close"].values).all()


# --------------------------------------------------------------------------
# Multi-source merge
# --------------------------------------------------------------------------

def test_merge_field_records_priority_and_fallback():
    records = {
        "screener": {"price": 100.0, "revenue": np.nan, "roe": 15.0},
        "yfinance": {"price": 101.0, "revenue": 5000.0, "roe": np.nan},
    }
    merged, provenance = merge_field_records(records, source_priority=["screener", "yfinance"])
    assert merged["price"] == 100.0  # screener wins (present in both, screener has priority)
    assert provenance["price"] == "screener"
    assert merged["revenue"] == 5000.0  # falls back to yfinance since screener had NaN
    assert provenance["revenue"] == "yfinance"
    assert merged["roe"] == 15.0
    assert provenance["roe"] == "screener"


def test_merge_field_records_all_missing():
    records = {"screener": {"foo": np.nan}, "yfinance": {"foo": None}}
    merged, provenance = merge_field_records(records, source_priority=["screener", "yfinance"])
    assert math.isnan(merged["foo"])
    assert provenance["foo"] == "missing"


def test_merge_sources_dataframe_level():
    screener_df = pd.DataFrame(
        {"price": [100.0, np.nan], "revenue": [np.nan, 200.0]}, index=["AAA", "BBB"]
    )
    yfinance_df = pd.DataFrame(
        {"price": [np.nan, 50.0], "revenue": [10.0, 20.0]}, index=["AAA", "BBB"]
    )
    merged, provenance = merge_sources(
        {"screener": screener_df, "yfinance": yfinance_df}, source_priority=["screener", "yfinance"]
    )
    assert merged.loc["AAA", "price"] == 100.0
    assert merged.loc["AAA", "revenue"] == 10.0  # screener was NaN, fell back
    assert merged.loc["BBB", "price"] == 50.0  # screener was NaN, fell back
    assert provenance.loc["BBB", "price"] == "yfinance"


def test_merge_sources_handles_missing_source_entirely():
    screener_df = pd.DataFrame({"price": [100.0]}, index=["AAA"])
    merged, provenance = merge_sources({"screener": screener_df}, source_priority=["screener", "yfinance"])
    assert merged.loc["AAA", "price"] == 100.0


# --------------------------------------------------------------------------
# scraping_utils
# --------------------------------------------------------------------------

def test_disk_cache_round_trip(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path), ttl_days=1)
    assert cache.get("http://example.com") is None
    cache.set("http://example.com", "<html>hello</html>")
    assert cache.get("http://example.com") == "<html>hello</html>"


def test_disk_cache_expires(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path), ttl_days=0)
    cache.set("http://example.com", "hello")
    time.sleep(0.05)
    # ttl_days=0 means anything older than "now" (any elapsed time) is expired.
    assert cache.get("http://example.com") is None


def test_rate_limited_get_enforces_delay(monkeypatch):
    import src.common.scraping_utils as su

    su._last_request_time.clear()
    sleep_calls = []
    monkeypatch.setattr(su.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(su, "is_allowed_by_robots", lambda url, user_agent="*": True)

    class FakeResponse:
        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, url, timeout):
            return FakeResponse()

    session = FakeSession()
    su.rate_limited_get(session, "https://example.com/a", min_delay_seconds=5.0)
    su.rate_limited_get(session, "https://example.com/b", min_delay_seconds=5.0)
    # Second call to the same domain should have triggered a sleep.
    assert len(sleep_calls) == 1
    assert sleep_calls[0] <= 5.0
