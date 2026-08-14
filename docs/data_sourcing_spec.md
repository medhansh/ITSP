# Data sourcing — Screener, Trendlyne, yfinance

## Summary

Price data comes from **yfinance** (the only one of the three sources with clean,
batchable daily OHLC history). Fundamentals are merged field-by-field from
**Screener.in** (primary), **yfinance** (fallback), and **Trendlyne** (supplementary —
most of its useful data is paywalled, see below), via
`src/fundamental_analysis/data_fetchers/fundamentals_fetcher.py::fetch_fundamentals`.

Read this whole doc before running `scripts/fetch_data.py` at scale — the short
version is: this works, but it's HTML scraping (not an official API) hitting two
sites whose Terms of Service you should read yourself, it's slow by design
(rate-limited), and a meaningful chunk of fields are either approximated or
genuinely unavailable for free. None of that is unique to this project — it's the
actual state of free Indian equity fundamentals data.

## What was actually verified, and how

This scaffold's build sandbox has no outbound network access at all (confirmed with
curl against NSE, Yahoo Finance, and PyPI — see the other spec docs), so none of this
code could be tested end-to-end against live data. What *was* verified, using the
WebFetch tool (a different network path than this sandbox's shell), was the actual
content available on live Screener.in and Trendlyne pages for Reliance Industries —
enough to determine what's genuinely free vs. paywalled on each site, and Screener's
general table structure. Exact CSS selectors could not be verified (WebFetch
converts pages to markdown, stripping HTML markup), so the scrapers are written
against Screener's well-documented, widely-scraped structure (`<section id="...">`
containing a `<table>`, used by many public scraping projects) rather than markup
inspected directly. **Run `scripts/probe_data_source.py` (see below) against a real
symbol before trusting a full-universe run** — if Screener has changed its markup
since this was written, that's how you'll find out.

## Per-field coverage (confirmed during development)

| Field | Screener | Trendlyne | yfinance |
|---|---|---|---|
| Price, market cap, P/E, EPS, book value | ✅ free | price only (free) | ✅ (inconsistent for smaller names) |
| Revenue, net income (current + multi-year) | ✅ free | ❌ | ✅ (weaker) |
| ROE, ROCE | ✅ free (ratios table) | ❌ | partial |
| Operating cash flow | ✅ free | ❌ | partial |
| Borrowings / total debt | ✅ free | ❌ | partial |
| **Current assets / current liabilities split** | ❌ not broken out | ❌ | partial |
| **Gross profit** | ❌ not broken out | ❌ | partial (`grossProfits`) |
| EBIT / EBITDA | approximated (PBT+interest; op. profit+depreciation) | ❌ | approximated |
| Total equity, retained earnings | approximated (equity capital + reserves) | ❌ | ❌ |
| Capex | approximated (proxied by investing cash flow) | ❌ | ✅ (`capitalExpenditures`) |
| Promoter / FII / DII holding % (+ history) | ✅ free (shareholding pattern table) | 🔒 paywalled | ❌ |
| **Promoter pledge %** | ❌ not on free page | 🔒 paywalled | ❌ |
| Durability / Valuation scores | n/a (not a Screener concept) | 🔒 paywalled | n/a |
| Momentum score, SWOT counts | n/a | ✅ free | n/a |
| Analyst target price / estimates | ❌ | 🔒 paywalled | partial (`targetMeanPrice`, `forwardEps`) |
| Related-party-transaction / auditor-change flags | ❌ | ❌ | ❌ |

**❌ = not available for free from this source. 🔒 = confirmed paywalled (requires a
paid Trendlyne "GuruQ"/"StratQ" subscription) — verified by fetching a live page and
seeing the paywall messaging directly, not assumed.** "Approximated" fields are
computed from adjacent line items rather than scraped directly — see the relevant
fetcher's docstring/inline comments for the exact formula and the assumption it rests
on. Promoter pledge %, related-party-transaction flags, and auditor-change flags are
not available from any of these three free sources — they require NSE's
shareholding-pattern XBRL filings (a much bigger parsing project) or a paid vendor.
This is a genuine, currently-unfilled gap in the pipeline, not an oversight.

## Currency units

Screener.in reports currency figures in ₹ Crore; yfinance reports in absolute ₹.
`screener_fetcher.py` converts every Crore-denominated field to absolute rupees
(`CR_TO_RUPEES = 1e7`) before returning it, so every SNAPSHOT_SCHEMA currency field is
in the same units regardless of source — **do not add a second unit conversion
downstream**, and if you extend either fetcher with a new currency field, convert it
at the fetcher boundary, not later.

## Screener's "Total Liabilities" row — a real gotcha

Screener's simplified balance sheet has a row labeled "Total Liabilities" that is
actually the balance-sheet *total* (liabilities + equity — numerically equal to Total
Assets), not liabilities excluding equity. `screener_fetcher.py` does not use that row
directly; `total_liabilities` is instead derived as `total_assets - total_equity`,
which is what `leverage_solvency.py`'s Altman Z-score actually needs (X4 = market cap
/ total liabilities excluding equity). If you're extending this fetcher, don't trust
that row's label at face value.

## Trendlyne ID resolution

Trendlyne URLs use an internal numeric ID + slug
(`trendlyne.com/equity/<id>/<SYMBOL>/<slug>/`), not the NSE symbol directly, and no
reliable public search/resolver endpoint was confirmed during development (a couple
of guessed API endpoints returned 405s). `trendlyne_fetcher.py::resolve_trendlyne_id`
reads a local mapping file (`data/universe/trendlyne_id_map.csv`, one row seeded:
RELIANCE) that you build up by visiting trendlyne.com, searching each constituent,
and copying the id/slug from the URL. Symbols missing from the map are skipped (not
an error) — Trendlyne coverage will start at whatever fraction of NIFTY500 you've
mapped and grow as you add rows.

## Scraping etiquette — read before running at scale

`src/common/scraping_utils.py` makes both scrapers *polite*: a real browser User-Agent,
retry-with-backoff, a minimum delay between requests to the same domain (default 2
seconds — `data_fetchers.fundamentals.min_delay_seconds` in config), a robots.txt
check before each domain's first request, and an on-disk cache so re-runs during
development don't re-fetch unchanged pages. **None of that makes scraping
automatically permitted** — robots.txt and a site's actual Terms of Service are
different things, and this project did not review either site's ToS in depth before
building this. Before running a full NIFTY500 fetch:

- Read Screener.in's and Trendlyne's Terms of Service yourself.
- Keep `min_delay_seconds` at 2+ — a full run is ~500 symbols × 2 sources × 2+ sec =
  well over an hour; don't lower this to "speed things up."
- This is appropriate for personal/academic research (which is the stated context —
  a course project). Prefer an official API or paid data vendor for anything beyond
  that, and don't redistribute scraped data.
- If either site starts returning 403s/CAPTCHAs, that's a signal to stop, not to
  retry harder — see the "Avoid rabbit holes" guidance this project follows generally.

## Caching

`DiskCache` (`src/common/scraping_utils.py`) stores each fetched page's raw HTML
under `data/raw/.cache/`, keyed by URL hash, with a 7-day TTL (config:
`data_fetchers.fundamentals.cache_ttl_days`). This means: a full fetch interrupted
partway through can simply be re-run and will skip everything already cached; and
`fetch_fundamentals_history` (which re-fetches the same Screener page fetched for the
snapshot) hits the cache instead of the network the second time, within the TTL
window. The cache directory is git-ignored (`data/raw/*` is already ignored — see
`.gitignore`).

## Point-in-time data — still an open problem

Everything above fetches the *current* fundamentals snapshot. `fetch_fundamentals_history`
gets multi-year *annual* history for the growth dimension, but that's still "as of
today's page load," not "as of each historical date." The look-ahead-bias caveat in
`docs/backtesting_spec.md` still applies in full: reusing today's snapshot at every
historical backtest rebalance date leaks future information. Building genuinely
point-in-time fundamentals (using filing dates as the availability cutoff, not fiscal
period end dates) remains the top open item before backtest numbers from real data
can be trusted — this session added the data *sourcing*, not a point-in-time *feed*.

## Diagnosing scraper breakage

Run this after any long gap, or if `fetch_fundamentals`'s coverage log (printed by
`scripts/fetch_data.py fundamentals`) shows unexpectedly low coverage for fields that
used to work:

```bash
python3 -c "
from src.fundamental_analysis.data_fetchers import screener_fetcher
snap = screener_fetcher.fetch_company_snapshot('RELIANCE')
for k, v in snap.items():
    print(f'{k:35s} {v}')
"
```

A field that's suddenly all-NaN across many symbols (not just one company's
data-quality issue) usually means Screener renamed a row label or changed a section
id — check `SECTION_IDS` and the `_lookup_row(...)` calls in `screener_fetcher.py`
against the live page's current structure.
