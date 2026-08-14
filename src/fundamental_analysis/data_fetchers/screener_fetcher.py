"""Scraper for screener.in company pages — the primary fundamentals source.

Screener.in's company page (``/company/<SYMBOL>/consolidated/``) is a
server-rendered HTML page (no login required for the core financials) built
around a consistent structure: a `#top-ratios` list of headline ratios, and
one `<table>` per financial statement inside a `<section id="...">` —
quarterly results, profit & loss, balance sheet, cash flow, ratios, and
shareholding pattern. This module parses those tables directly.

IMPORTANT — what's actually free here (verified by fetching a live page
during development; see docs/data_sourcing_spec.md for the full breakdown):
  - Reliable: price, market cap, P/E, EPS, book value, ROE/ROCE, revenue,
    net income, borrowings, interest expense, operating cash flow, and the
    full shareholding pattern (promoter/FII/DII/government/public %, and
    shareholder count) — both quarterly and yearly history.
  - Approximate (documented per-field below): EBIT/EBITDA, total equity,
    retained earnings, capex, dividend per share, enterprise value — Screener's
    simplified statements don't break these out explicitly, so they're derived
    from adjacent line items with a clearly-flagged approximation.
  - Not available on the free page at all: a current-assets/current-liabilities
    split (so current ratio, quick ratio, and the Altman Z-score's working-
    capital term can't be computed from Screener), gross profit, promoter
    pledge %, related-party-transaction/auditor-change flags, and analyst
    estimates. These come back NaN from this fetcher — see
    fundamentals_fetcher.py's multi-source merge for where else they might
    come from (they mostly don't, from a free source — this is a genuine data
    gap, not a bug).

FRAGILITY WARNING: this is markup-scraping, not an API. If screener.in changes
its HTML structure, parsing will silently return fewer fields (each parse
step degrades gracefully to NaN/missing rather than raising) — run
scripts/probe_data_source.py after any long gap to sanity-check field
coverage before trusting a full-universe run.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from src.common.logging_utils import get_logger
from src.common.scraping_utils import DiskCache, build_session, cached_get_text

logger = get_logger(__name__)

def _company_url(symbol: str, consolidated: bool = True) -> str:
    suffix = "consolidated/" if consolidated else ""
    return f"https://www.screener.in/company/{symbol}/{suffix}"

# Candidate <section id="..."> values per logical table — first match wins.
# Screener has used more than one id scheme across redesigns; trying a short
# list is cheap insurance against this fetcher breaking on a minor rename.
SECTION_IDS = {
    "quarters": ["quarters", "quarterly-results"],
    "profit_loss": ["profit-loss"],
    "balance_sheet": ["balance-sheet"],
    "cash_flow": ["cash-flow"],
    "ratios": ["ratios"],
    "shareholding_quarterly": ["shareholding", "shareholding-pattern", "quarterly-shp"],
    "shareholding_yearly": ["shareholding", "shareholding-pattern", "yearly-shp"],
}


def _parse_number(text: str) -> float:
    """Parse a Screener-formatted number: strips ₹/%%/commas, handles
    parenthesized negatives and en/em-dash "no data" markers."""
    if text is None:
        return float("nan")
    t = text.strip().replace(",", "").replace("₹", "").replace("%", "").strip()
    if t in ("", "-", "—", "–", "N/A", "NA"):
        return float("nan")
    negative = t.startswith("(") and t.endswith(")")
    if negative:
        t = t[1:-1]
    t = t.strip()
    try:
        value = float(t)
    except ValueError:
        return float("nan")
    return -value if negative else value


def _find_section(soup: BeautifulSoup, candidate_ids: list[str]):
    for cid in candidate_ids:
        section = soup.find(id=cid)
        if section is not None:
            return section
    return None


def _parse_data_table(soup: BeautifulSoup, candidate_ids: list[str]) -> pd.DataFrame:
    """Generic parser for Screener's `<table class="data-table">` blocks.
    Returns a DataFrame indexed by row label, columns = period labels
    (e.g. "Mar 2023", "TTM"). Empty DataFrame if the section isn't found.
    """
    section = _find_section(soup, candidate_ids)
    if section is None:
        logger.warning("Section not found for any of %s", candidate_ids)
        return pd.DataFrame()

    table = section.find("table")
    if table is None:
        return pd.DataFrame()

    thead = table.find("thead")
    periods: list[str] = []
    if thead is not None:
        header_cells = thead.find_all("th")
        periods = [th.get_text(strip=True) for th in header_cells[1:]]

    tbody = table.find("tbody") or table
    rows: dict[str, list[float]] = {}
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        raw_label = cells[0].get_text(strip=True)
        label = re.sub(r"[+ ]+$", "", raw_label).strip()
        values = [_parse_number(td.get_text(strip=True)) for td in cells[1:]]
        if periods and len(values) != len(periods):
            # Row length mismatch (can happen with nested/expandable sub-rows) —
            # pad/truncate defensively rather than dropping the whole table.
            values = (values + [float("nan")] * len(periods))[: len(periods)]
        rows[label] = values

    if not periods:
        return pd.DataFrame(rows).T
    return pd.DataFrame(rows, index=periods).T


def _lookup_row(df: pd.DataFrame, *label_candidates: str) -> pd.Series | None:
    """Case-insensitive substring match against a data table's row labels —
    tolerant of Screener's minor label variation (e.g. 'EPS in Rs' vs
    'Basic EPS (Rs)') without needing an exact-string map for every field."""
    if df.empty:
        return None
    lower_index = {idx.lower(): idx for idx in df.index}
    for candidate in label_candidates:
        c = candidate.lower()
        for lower_label, original_label in lower_index.items():
            if c in lower_label:
                return df.loc[original_label]
    return None


def _parse_top_ratios(soup: BeautifulSoup) -> dict[str, float]:
    container = soup.find(id="top-ratios")
    result: dict[str, float] = {}
    if container is None:
        logger.warning("top-ratios section not found")
        return result
    for li in container.find_all("li"):
        name_el = li.find(class_="name")
        number_el = li.find(class_="number")
        if name_el is None or number_el is None:
            continue
        label = name_el.get_text(strip=True)
        result[label] = _parse_number(number_el.get_text(strip=True))
    return result


def _parse_shareholding(soup: BeautifulSoup) -> pd.DataFrame:
    """Latest-available quarterly shareholding table: rows are category
    (Promoters/FIIs/DIIs/Government/Public/No. of Shareholders), columns are
    quarter labels, most recent last."""
    return _parse_data_table(soup, SECTION_IDS["shareholding_quarterly"])


CR_TO_RUPEES = 1e7  # Screener reports currency figures in ₹ Crore; canonicalize to ₹


_PERIOD_LABEL_RE = re.compile(
    r"^(?P<mon>[A-Za-z]{3})[a-z]*\.?['\u2019]?\s*(?P<year>\d{2}|\d{4})$"
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_period_end(label: str) -> pd.Timestamp | None:
    """Parse a Screener period-column label into the last calendar day of
    that month (the fiscal *quarter-end* date). Handles the annual-table
    style ('Mar 2024') as well as more compact quarterly-table variants
    observed in the wild ('Mar-24', "Mar'24", non-breaking spaces, 2-digit
    years) — see ``fetch_raw_quarters_table`` below if labels still don't
    parse after this; that function's output is the fastest way to see the
    actual label format and fix this regex to match it. Returns None for
    labels that aren't a recognizable period at all (e.g. a trailing
    'TTM'/estimate column).
    """
    cleaned = label.strip().replace("\xa0", " ").replace("-", " ")
    m = _PERIOD_LABEL_RE.match(cleaned)
    if not m:
        return None
    mon = _MONTH_MAP.get(m.group("mon").lower()[:3])
    if mon is None:
        return None
    year_str = m.group("year")
    year = int(year_str)
    if len(year_str) == 2:
        # Screener has no 2-digit-year data pre-2000, so 00-79 -> 2000s is safe.
        year += 2000 if year < 80 else 1900
    return pd.Timestamp(year=year, month=mon, day=1) + pd.offsets.MonthEnd(0)


def fetch_raw_quarters_table(
    symbol: str, session=None, cache: DiskCache | None = None, min_delay_seconds: float = 2.0
) -> pd.DataFrame:
    """Debug helper: fetch and parse Screener's quarterly-results table
    *without* filtering/parsing period labels — returns the raw table exactly
    as ``_parse_data_table`` sees it (row labels x period-column labels).

    Use this first whenever ``fetch_quarterly_history``/``fetch_multiple_quarterly_history``
    come back empty or thin: print ``result.columns.tolist()`` to see the
    actual period-label strings Screener is sending right now, and check them
    against ``_parse_period_end`` — this project's build/test sandbox had no
    live network access, so that regex was written against the documented
    format, not verified against a real page (see docs/data_sourcing_spec.md's
    known gaps). ``scripts/probe_data_source.py SYMBOL --quarterly`` wraps
    this for convenient CLI use.
    """
    session = session or build_session()
    cache = cache or DiskCache()
    from src.common.scraping_utils import cached_get_text

    url = _company_url(symbol, consolidated=True)
    html = cached_get_text(session, url, cache=cache, min_delay_seconds=min_delay_seconds)
    soup = BeautifulSoup(html, "lxml")
    return _parse_data_table(soup, SECTION_IDS["quarters"])


# Which quarterly-results row labels we extract for the point-in-time history,
# and the schema name each maps to. Kept intentionally small (vs. the full
# snapshot schema) — quarterly history is what point_in_time.py replays
# forward through time, and only fields that actually vary quarter-to-quarter
# in a way the composite scorer cares about are worth carrying.
QUARTERLY_FIELD_MAP = {
    "revenue": ("sales", "revenue"),
    "net_income": ("net profit",),
    "eps": ("eps in rs", "eps"),
    "operating_profit_margin_pct": ("opm",),
}


def fetch_quarterly_history(
    symbol: str,
    session=None,
    cache: DiskCache | None = None,
    min_delay_seconds: float = 2.0,
    reporting_lag_days: int = 45,
) -> pd.DataFrame:
    """Fetch Screener's quarterly-results table for one symbol and return it
    in long format: one row per (period_end, field), with a ``known_date``
    column = period_end + ``reporting_lag_days``.

    ``known_date`` is a heuristic, not a scraped filing date — Screener's
    quarterly table only gives the *period* each column covers (e.g.
    "Mar 2024" = the quarter ending 31 Mar 2024), not the date results were
    actually filed/announced. Indian-listed companies must report quarterly
    results within 45 days of quarter-end (SEBI LODR Regulation 33) — using
    period_end + 45 days as ``known_date`` is therefore a conservative
    (never-too-early) upper bound on when a figure could plausibly have been
    public, which is exactly the property point_in_time.py needs to avoid
    look-ahead bias. It is deliberately not tighter than that; see
    docs/backtesting_spec.md's point-in-time section for the exact trade-off.
    """
    session = session or build_session()
    cache = cache or DiskCache()
    from src.common.scraping_utils import cached_get_text

    url = _company_url(symbol, consolidated=True)
    try:
        html = cached_get_text(session, url, cache=cache, min_delay_seconds=min_delay_seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch quarterly-history page for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["symbol", "period_end", "known_date", "field", "value"])

    soup = BeautifulSoup(html, "lxml")
    quarters = _parse_data_table(soup, SECTION_IDS["quarters"])
    if quarters.empty:
        return pd.DataFrame(columns=["symbol", "period_end", "known_date", "field", "value"])

    rows = []
    for period_label in quarters.columns:
        period_end = _parse_period_end(period_label)
        if period_end is None:
            continue  # skip non-period columns (e.g. a trailing "TTM"/estimate column)
        known_date = period_end + pd.Timedelta(days=reporting_lag_days)
        for field_name, candidates in QUARTERLY_FIELD_MAP.items():
            row = _lookup_row(quarters, *candidates)
            if row is None or period_label not in row.index:
                continue
            value = row[period_label]
            scale = CR_TO_RUPEES if field_name in ("revenue", "net_income") else 1.0
            rows.append({
                "symbol": symbol,
                "period_end": period_end,
                "known_date": known_date,
                "field": field_name,
                "value": value * scale if pd.notna(value) else value,
            })
    return pd.DataFrame(rows, columns=["symbol", "period_end", "known_date", "field", "value"])


def fetch_multiple_quarterly_history(
    symbols: list[str],
    session=None,
    cache: DiskCache | None = None,
    min_delay_seconds: float = 2.0,
    reporting_lag_days: int = 45,
) -> pd.DataFrame:
    """Batch wrapper around ``fetch_quarterly_history`` — one long DataFrame
    (symbol, period_end, known_date, field, value) across all symbols, ready
    to feed ``point_in_time.build_pit_panel``.

    Logs a loud warning (not just per-symbol debug noise) if fewer than half
    the requested symbols yielded any quarterly rows at all — a near-total
    failure here silently produces an all-empty/all-NaN point-in-time
    fundamentals panel and a flat 0%-return backtest many minutes later
    (see docs/fundamental_analysis_spec.md's point-in-time section), so it's
    worth catching immediately instead. If you see this warning, run
    ``scripts/probe_data_source.py SYMBOL --quarterly`` on one symbol next —
    it dumps the raw period-column labels Screener actually sent, which is
    almost always either a section-id change or a label-format mismatch in
    ``_parse_period_end``.
    """
    session = session or build_session()
    cache = cache or DiskCache()
    frames = []
    symbols_with_data = 0
    for symbol in symbols:
        frame = fetch_quarterly_history(
            symbol, session=session, cache=cache,
            min_delay_seconds=min_delay_seconds, reporting_lag_days=reporting_lag_days,
        )
        if not frame.empty:
            symbols_with_data += 1
        frames.append(frame)

    if symbols and symbols_with_data < len(symbols) * 0.5:
        logger.warning(
            "fetch_multiple_quarterly_history: only %d/%d symbols yielded any quarterly rows. "
            "This will produce a mostly/entirely empty point-in-time fundamentals panel. "
            "Run `python scripts/probe_data_source.py <SYMBOL> --quarterly` on one symbol to "
            "see why (likely a Screener markup/label-format change — see this function's "
            "docstring).", symbols_with_data, len(symbols),
        )

    if not frames:
        return pd.DataFrame(columns=["symbol", "period_end", "known_date", "field", "value"])
    return pd.concat(frames, ignore_index=True)


def parse_company_page(html: str) -> dict[str, Any]:
    """Parse a single fetched Screener company page into a flat dict of
    SNAPSHOT_SCHEMA-shaped fields (see fundamentals_fetcher.py). Pure
    function — no network — so it's directly unit-testable against a fixture.
    """
    soup = BeautifulSoup(html, "lxml")
    ratios = _parse_top_ratios(soup)
    pl = _parse_data_table(soup, SECTION_IDS["profit_loss"])
    if "TTM" in pl.columns:
        # TTM is a rolling window, not a fiscal-year-end snapshot — drop it so
        # "current"/"prior" below are a clean annual-vs-annual YoY pair (this
        # matters a lot for Piotroski's YoY comparisons in profitability_quality.py;
        # mixing a rolling TTM "current" against an annual "prior" would compare
        # overlapping, inconsistent periods).
        pl = pl.drop(columns="TTM")
    bs = _parse_data_table(soup, SECTION_IDS["balance_sheet"])
    cf = _parse_data_table(soup, SECTION_IDS["cash_flow"])
    rt = _parse_data_table(soup, SECTION_IDS["ratios"])
    shp = _parse_shareholding(soup)

    def latest(series: pd.Series | None, offset: int = 0) -> float:
        if series is None or len(series) <= offset:
            return float("nan")
        return series.iloc[-1 - offset]

    sales = _lookup_row(pl, "sales", "revenue")
    net_profit = _lookup_row(pl, "net profit")
    interest_pl = _lookup_row(pl, "interest")
    depreciation = _lookup_row(pl, "depreciation")
    pbt = _lookup_row(pl, "profit before tax")
    eps_row = _lookup_row(pl, "eps in rs", "eps")

    total_assets = _lookup_row(bs, "total assets")
    equity_capital = _lookup_row(bs, "equity capital", "share capital")
    reserves = _lookup_row(bs, "reserves")
    borrowings = _lookup_row(bs, "borrowings")
    # NOTE: Screener's own "Total Liabilities" row is the balance-sheet total
    # (liabilities + equity, i.e. numerically equal to Total Assets) — not
    # liabilities excluding equity. We don't use it; total_liabilities below
    # is derived as total_assets - total_equity instead, which is what the
    # Altman Z-score (leverage_solvency.py) actually needs.

    cfo = _lookup_row(cf, "cash from operating")
    cfi = _lookup_row(cf, "cash from investing")

    roe = _lookup_row(rt, "roe")
    roce = _lookup_row(rt, "roce")

    total_equity = None
    if equity_capital is not None and reserves is not None:
        total_equity = equity_capital.add(reserves, fill_value=0)
    elif reserves is not None:
        total_equity = reserves

    total_assets_rupees = latest(total_assets) * CR_TO_RUPEES if total_assets is not None else float("nan")
    total_equity_rupees = latest(total_equity) * CR_TO_RUPEES if total_equity is not None else float("nan")
    total_liabilities_rupees = (
        total_assets_rupees - total_equity_rupees
        if pd.notna(total_assets_rupees) and pd.notna(total_equity_rupees)
        else float("nan")
    )  # derived, not scraped directly — see note above on Screener's "Total Liabilities" row

    fields: dict[str, Any] = {
        "price": ratios.get("Current Price", float("nan")),
        "market_cap": ratios.get("Market Cap", float("nan")) * CR_TO_RUPEES if "Market Cap" in ratios else float("nan"),
        "eps_ttm": ratios.get("EPS", latest(eps_row)),
        "book_value_per_share": ratios.get("Book Value", float("nan")),
        "revenue": latest(sales) * CR_TO_RUPEES if sales is not None else float("nan"),
        "revenue_prior": latest(sales, 1) * CR_TO_RUPEES if sales is not None else float("nan"),
        "net_income": latest(net_profit) * CR_TO_RUPEES if net_profit is not None else float("nan"),
        "net_income_prior": latest(net_profit, 1) * CR_TO_RUPEES if net_profit is not None else float("nan"),
        "total_assets": total_assets_rupees,
        "total_assets_prior": latest(total_assets, 1) * CR_TO_RUPEES if total_assets is not None else float("nan"),
        "total_liabilities": total_liabilities_rupees,
        "total_equity": total_equity_rupees,
        "retained_earnings": latest(reserves) * CR_TO_RUPEES if reserves is not None else float("nan"),  # approximation
        "total_debt": latest(borrowings) * CR_TO_RUPEES if borrowings is not None else float("nan"),
        "long_term_debt": latest(borrowings) * CR_TO_RUPEES if borrowings is not None else float("nan"),  # approximation: no current/non-current split available
        "long_term_debt_prior": latest(borrowings, 1) * CR_TO_RUPEES if borrowings is not None else float("nan"),
        "interest_expense": latest(interest_pl) * CR_TO_RUPEES if interest_pl is not None else float("nan"),
        "ebit": (latest(pbt) + latest(interest_pl)) * CR_TO_RUPEES if pbt is not None and interest_pl is not None else float("nan"),  # approximation: PBT + interest
        "ebitda": None,  # filled below once ebit/depreciation are known
        "cfo": latest(cfo) * CR_TO_RUPEES if cfo is not None else float("nan"),
        "capex": -min(latest(cfi), 0) * CR_TO_RUPEES if cfi is not None else float("nan"),  # approximation: proxied by investing cash outflow
        "roe": ratios.get("ROE", latest(roe)),
        "roce": ratios.get("ROCE", latest(roce)),
        # Not available from Screener's free page — left NaN deliberately:
        "current_assets": float("nan"),
        "current_assets_prior": float("nan"),
        "current_liabilities": float("nan"),
        "current_liabilities_prior": float("nan"),
        "gross_profit": float("nan"),
        "gross_profit_prior": float("nan"),
        "promoter_pledge_pct": float("nan"),
        "related_party_transactions_flag": float("nan"),
        "auditor_changed_flag": float("nan"),
    }

    ebit_val = fields["ebit"]
    dep_val = latest(depreciation) * CR_TO_RUPEES if depreciation is not None else float("nan")
    fields["ebitda"] = ebit_val + dep_val if pd.notna(ebit_val) and pd.notna(dep_val) else float("nan")  # approximation

    dividend_yield_pct = ratios.get("Dividend Yield")
    price = fields["price"]
    if dividend_yield_pct is not None and price is not None and pd.notna(price):
        fields["dividend_per_share"] = price * (dividend_yield_pct / 100.0)  # approximation
    else:
        fields["dividend_per_share"] = float("nan")

    if not shp.empty:
        promoter = _lookup_row(shp, "promoter")
        fii = _lookup_row(shp, "fii")
        dii = _lookup_row(shp, "dii")
        fields["promoter_holding_pct"] = latest(promoter) if promoter is not None else float("nan")
        fields["promoter_holding_pct_prior"] = latest(promoter, 1) if promoter is not None else float("nan")
        fields["fii_holding_pct"] = latest(fii) if fii is not None else float("nan")
        fields["fii_holding_pct_prior"] = latest(fii, 1) if fii is not None else float("nan")
        fields["dii_holding_pct"] = latest(dii) if dii is not None else float("nan")
        fields["dii_holding_pct_prior"] = latest(dii, 1) if dii is not None else float("nan")
    else:
        for k in ["promoter_holding_pct", "promoter_holding_pct_prior", "fii_holding_pct",
                  "fii_holding_pct_prior", "dii_holding_pct", "dii_holding_pct_prior"]:
            fields[k] = float("nan")

    return fields


def fetch_company_snapshot(
    symbol: str,
    session=None,
    cache: DiskCache | None = None,
    consolidated: bool = True,
    min_delay_seconds: float = 2.0,
) -> dict[str, Any]:
    """Fetch + parse one symbol's Screener page. Network I/O lives only here;
    ``parse_company_page`` above does the actual parsing and is what tests
    exercise directly against a fixture."""
    session = session or build_session()
    url = _company_url(symbol, consolidated=consolidated)
    try:
        html = cached_get_text(session, url, cache=cache, min_delay_seconds=min_delay_seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Screener page for %s: %s", symbol, exc)
        return {"symbol": symbol}

    fields = parse_company_page(html)
    fields["symbol"] = symbol
    return fields


def fetch_multiple(
    symbols: list[str],
    session=None,
    cache: DiskCache | None = None,
    min_delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Batch fetch — sequential and rate-limited (Screener has no documented
    bulk/API endpoint), so this is slow by design for a full NIFTY500 run.
    Use the DiskCache so interrupted runs can resume without re-fetching."""
    session = session or build_session()
    rows = []
    for i, symbol in enumerate(symbols):
        logger.info("Fetching Screener data for %s (%d/%d)", symbol, i + 1, len(symbols))
        rows.append(fetch_company_snapshot(symbol, session=session, cache=cache, min_delay_seconds=min_delay_seconds))
    return pd.DataFrame(rows).set_index("symbol", drop=False)
