# Fundamental analysis — methodology

## Scope expansion from the original abstract

The abstract scoped this module narrowly: "an earnings surprise predictor analyzing
promoter activity and analyst patterns." Per the brief for this pass, that's been
broadened into a full fundamentals engine with seven dimensions. Earnings surprise +
analyst revisions is retained as one dimension (`metrics/earnings_surprise.py`);
promoter activity now lives under ownership & governance since it's really a distinct
signal family (insider behavior) rather than solely a driver of earnings surprises.

## Dimensions

### 1. Valuation (`metrics/valuation.py`)
P/E, P/B, EV/EBITDA, PEG, dividend yield. Lower multiples score better (PEG and
dividend yield are the exceptions to "lower is better" — see `METRIC_DIRECTION` in
`scoring/composite_score.py`, dividend yield is "higher is better").

### 2. Profitability & quality (`metrics/profitability_quality.py`)
ROE, ROCE, gross/net margin, and the **Piotroski F-score** — a 9-point checklist
(Piotroski, 2000) of YoY fundamental improvements across profitability, leverage,
liquidity, and efficiency. Each of the 9 signals needs a specific current-vs-prior-year
comparison; a signal that can't be computed from available data is *excluded* from
that stock's score, not counted as a failure (see `_gt()` helper and its NaN handling
in `profitability_quality.py` — this was a real bug caught during testing: plain
pandas comparisons treat NaN as False, which would silently and wrongly count missing
data as "did not improve").

### 3. Growth (`metrics/growth.py`)
Revenue/net-income/EPS CAGR over the available history, plus a growth-stability
measure (negative stdev of YoY growth) so a steady 15%/yr compounder outscores a
lumpy +40%/-10%/+40% stock with the same average growth. Needs multi-year history
(a separate `history` DataFrame — see schema below), unlike every other dimension
which works off a single latest snapshot.

### 4. Leverage & solvency (`metrics/leverage_solvency.py`)
Debt/equity, interest coverage, current/quick ratio, and the classic **Altman
Z-score** (1968 formula, for non-financial firms — exclude Financial Services when
screening, since the model wasn't designed for banks/NBFCs' balance sheet structure).

### 5. Cash-flow quality (`metrics/cashflow_quality.py`)
CFO/net-income ratio, free cash flow, FCF yield, and the accruals ratio
`(net_income - cfo) / total_assets` (Sloan, 1996) — high positive accruals are a
classic earnings-quality red flag (profits not backed by cash).

### 6. Ownership & governance (`metrics/ownership_governance.py`)
Promoter holding change, promoter pledge %, FII/DII holding change, and a governance
red-flag count (high pledge + promoter selling + related-party/auditor-change flags).
Called out as deliberately India-specific in the module docstring: concentrated
promoter ownership and pledging are far more informative signals in Indian markets
than in developed markets with dispersed ownership.

### 7. Earnings surprise & analyst revisions (`metrics/earnings_surprise.py`)
The module the original abstract scoped as the entire fundamental-analysis
component: surprise % vs. consensus, and 30-day estimate-revision momentum (a proxy
for post-earnings-announcement drift).

### 8. Pre-earnings options signal (`metrics/options_earnings.py`)
Newest and smallest-weighted dimension (`composite_weights.options_earnings: 0.05`).
ATM implied volatility (expressed as a percentile of the stock's own trailing IV, so
it's comparable across names/vol regimes) and put/call open-interest ratio, both
measured in a window strictly *before* the stock's most recently *already-occurred*
earnings date. A third field, `implied_move_pct` (ATM straddle price / spot), is
carried through for reporting but deliberately NOT sign-scored — there's no
principled "higher/lower is better" direction for a pure magnitude-of-uncertainty
number. See that module's docstring for the full no-look-ahead design (it's the
dimension most exposed to accidentally leaking future information, given it's
explicitly earnings-event-anchored) and `data_fetchers/options_fetcher.py` for the
(currently unimplemented — no free NSE options data source) data-acquisition side.
This dimension degrades to all-NaN (harmless — composite weights renormalize per
row, same as any other missing dimension) unless a pre-built option-summary-history
and earnings-calendar are supplied — see `configs/config.yaml`'s
`data_fetchers.options` section.

## Point-in-time fundamentals (`point_in_time.py`)

Every dimension above, as originally built, worked off a single *current* snapshot —
today's P/E, today's ROE. Reusing that same snapshot at every historical backtest
rebalance date is a look-ahead-bias bug (flagged as the top gap in the "Known gaps"
section below, and in `docs/backtesting_spec.md`). `point_in_time.py` fixes this for
the fields that vary quarter-to-quarter (`revenue`, `net_income`, `eps` — see
`screener_fetcher.QUARTERLY_FIELD_MAP`) by replaying quarterly results forward
through time:

1. `screener_fetcher.fetch_quarterly_history` fetches Screener's quarterly-results
   table and tags each period with a `known_date` = period-end + a configurable
   `reporting_lag_days` (default 45, matching SEBI LODR Regulation 33's filing
   deadline) — a conservative upper bound on when a figure could plausibly have
   become public, since Screener only gives the period a column covers, not an
   actual filing timestamp. **A real bug was found and fixed here against live
   data**: `fetch_quarterly_history` was iterating `quarters.index` (the row
   labels — "Sales", "Net Profit", etc.) instead of `quarters.columns` (the
   actual period labels — "Mar 2024", "Jun 2024", ...) when looking for
   periods to parse, so `_parse_period_end` was being called on row labels
   and correctly returning `None` for all of them — a well-formed table
   parsed by `_parse_data_table` fine, but zero rows ever got extracted. This
   is now covered by a regression test
   (`test_fetch_quarterly_history_extracts_rows_from_real_shaped_table` in
   `tests/test_data_fetchers.py`) built from the exact table shape confirmed
   against a live Screener page. `_parse_period_end` was separately also
   hardened to accept more label variants ("Mar-24", "Mar'24", non-breaking
   spaces) as a precaution, though the live page turned out to already use
   the "Mon YYYY" format the original regex expected — the index/columns
   mix-up was the actual root cause. If `fetch_multiple_quarterly_history`
   ever again comes back mostly/entirely empty (it logs a loud warning
   immediately if so, rather than failing silently), run
   `python scripts/probe_data_source.py SYMBOL --quarterly` first — it prints
   the raw section-parse result and the final extracted rows side by side.
2. `point_in_time.build_pit_snapshot` / `build_pit_panel` take that long-format
   history and, for any query date t (or a whole panel of rebalance dates at once,
   via `pd.merge_asof(..., direction="backward")`), return the most recent value
   with `known_date <= t`, forward-filled across the gap until the next result.
   `merge_asof`'s backward-only search makes it structurally impossible for a later
   result to leak into an earlier date — this is the actual correctness mechanism,
   not just a documented convention.
3. `merge_pit_into_snapshot` overlays those as-of-date fields onto an otherwise-
   current snapshot (sector, industry, shareholding, and every field Screener/
   yfinance/Trendlyne don't expose historically at all still come from the current
   snapshot) — a pragmatic partial fix, not a claim that the whole snapshot is now
   point-in-time. See that function's docstring.
4. `run_pit_fundamental_pipeline` (and `scripts/run_full_pipeline.py`'s
   `step_pit_fundamentals`, which additionally merges in the options-earnings
   metrics per rebalance date) runs `pipeline.run_pipeline` once per rebalance date
   with that date's PIT snapshot, producing the long-format `scores_by_date` table
   `backtesting/strategies.py` expects.

**What this does and doesn't fix**: revenue/net_income/eps-derived metrics (most of
valuation, profitability, growth) are now genuinely point-in-time. Fields with no
historical source at all (shareholding %, promoter pledge, analyst estimates) are
unchanged — still current-snapshot-only, still a look-ahead risk if used naively at
historical rebalance dates. Before a symbol's first scraped quarterly result, its PIT
snapshot is correctly all-NaN (shrinking the effective early-history universe for
symbols with short scraped history) rather than being backfilled from data that
didn't exist yet.

## Scoring (`scoring/composite_score.py`)

1. Every raw metric is **z-scored within its own sector** (`sector_relative_zscore`),
   not across the whole NIFTY500 universe — Indian sectors trade at structurally
   different multiples (IT services vs. capital goods vs. banks), so a universe-wide
   z-score would mostly just rediscover sector membership rather than surface
   genuine over/under-valuation. Sectors with fewer than 5 members fall back to a
   universe-wide z-score to avoid degenerate small-sample statistics.
2. Metrics are sign-adjusted per `METRIC_DIRECTION` (e.g. P/E is negated since lower
   is better) so every z-score, after adjustment, means "higher = better."
3. Sign-adjusted z-scores are averaged within each dimension (`DIMENSION_METRICS`).
4. Dimension scores are combined into `composite_score` using the weights in
   `configs/config.yaml`'s `fundamental_analysis.composite_weights` (must sum to
   1.0, validated at config-load time in `common/io_utils.py`), **renormalized per
   row over whatever dimensions are actually non-NaN** for that stock — so a stock
   with incomplete data still gets a usable score instead of NaN, rather than being
   silently dropped.

## Data schema

`src/fundamental_analysis/data_fetchers/fundamentals_fetcher.py` defines the two
input schemas everything else depends on:

- `SNAPSHOT_SCHEMA` — one row per symbol, latest available fundamentals. Used by
  every dimension except growth.
- `HISTORY_SCHEMA` — one row per (symbol, fiscal_year): `revenue`, `net_income`,
  `eps`. Used only by the growth dimension.

Swapping data vendors means implementing a new fetcher that returns these same
schemas — no changes needed anywhere in `metrics/` or `scoring/`.

## Known gaps / next steps

- **No live data**: the build sandbox for this pass had no outbound access to NSE,
  Yahoo Finance, or any market-data endpoint (verified directly — see
  `data_fetchers/nse_fetcher.py` and `fundamentals_fetcher.py` docstrings). Both
  `fetch_nifty500_list()` and `fetch_fundamentals_yfinance()` are implemented but
  untested end-to-end; run them from a machine with internet access.
- **India-specific fields aren't in yfinance**: promoter holding %, promoter pledge
  %, and FII/DII holding are not available via yfinance for NSE tickers. These need
  NSE shareholding-pattern filings (quarterly XBRL/PDF disclosures) or a paid vendor
  (e.g. an official data provider) — `fetch_fundamentals_yfinance` fills them with
  NaN and logs a warning. This is the biggest real data-acquisition gap to close
  next, since ownership & governance is one of the more India-specific, higher-value
  dimensions in this design.
- **Point-in-time fundamentals is now implemented** (`point_in_time.py`, see above)
  for revenue/net_income/eps, closing the top gap from earlier passes — but only for
  those quarterly-tracked fields; shareholding/pledge/analyst-estimate fields are
  still current-snapshot-only. `known_date` is also a heuristic (period-end + a
  statutory filing-deadline lag), not a scraped actual filing date — tightening that
  (e.g. by scraping Screener's "Results announced on" text where present, if it
  exists) would be the natural next step.
- **Options data has no free bulk source wired in** (`data_fetchers/options_fetcher.py`):
  NSE F&O options chains aren't available via yfinance for NSE names, and Screener/
  Trendlyne don't carry options data at all. The `options_earnings` dimension's
  no-look-ahead logic is implemented and unit-tested against fixtures, but needs a
  real NSE historical-archive or paid-vendor client wired into
  `fetch_option_chain_snapshot`'s `fetch_fn` before it can run on real data — until
  then it degrades gracefully to all-NaN in `run_full_pipeline.py`.
- **Altman Z-score** isn't sector-aware (correctly excludes nothing automatically) —
  the caller is responsible for excluding Financial Services / NBFCs before
  screening on it, since the model assumes an industrial/non-financial balance sheet.
- **Composite weights are static** — see `docs/architecture.md`'s integration note on
  making them regime-conditional once `src/integration/` exists.

## technical_momentum dimension (Ichimoku conviction as selection input)

Added 2026-07-24 as the third and final thing tried on the "does Ichimoku
add value to the fundamentals-selected portfolio" question, after two
post-selection mechanisms (gating and reallocating an already-built
portfolio — see `docs/backtesting_spec.md`'s Ichimoku sections) were both
confirmed negative on real data despite the underlying signal
(`ichimoku_only`) being the single best-performing standalone component
found. This is the one mechanism of the three that can genuinely change
*which* stocks get selected, not just how much of an already-fixed basket
is held or how it's split.

**Data path**: `adaptive_ichimoku.build_ichimoku_conviction_panel` builds a
raw per-symbol daily `[0, 1]` conviction score (NOT the portfolio-
normalized weights `build_ichimoku_weights` produces — see that function's
docstring for why the distinction matters; conflating the two was a real
bug in the tilt mechanism earlier). At each fundamentals rebalance date,
`point_in_time.py`'s `run_pit_fundamental_pipeline` extracts the most
recent conviction value at or before that date (`reindex(...,
method="ffill")`) — PIT-safe by construction, since Ichimoku itself is
already purely causal. That gets fed to `metrics/technical_momentum.py`,
which wraps it into the same `fn(snapshot) -> DataFrame` shape every other
dimension uses, then z-scored and averaged into the composite score exactly
like `valuation`/`growth`/etc.

**Wiring requires two config flags**, not one: `technical_signals.ichimoku.enabled`
(builds the conviction panel at all) AND
`fundamental_analysis.dimensions.technical_momentum` (actually uses it in
the composite score). Either off alone degrades to NaN for this dimension
(harmless, renormalized away), same convention as every other
possibly-missing-data dimension in this pipeline.

**Composite weight**: `0.05`, deliberately small — same "newest,
least-validated dimension" convention as `options_earnings`. Every other
weight was scaled by `0.95` to keep the total at exactly `1.0` (validated
strictly at config load time).

**Orchestration note**: this required reordering `scripts/run_full_pipeline.py`'s
`main()` — OHLC now has to be fetched BEFORE fundamentals scoring runs
(it used to run after), since the conviction panel needs to exist before
`step_pit_fundamentals` can consume it. `step_ichimoku` was split into
three functions (`step_ichimoku_ohlc`, `step_ichimoku_conviction`,
`step_ichimoku_weights`) so the OHLC panel is fetched once and reused for
both the fundamentals dimension and the backtest weight matrix, rather
than fetching it twice.

**Status: implemented, mechanically verified (PIT-safety directly tested
with a value that changes mid-window, confirming the pre-jump date sees
the old value and not the new one; NaN-degradation tested for both "no
conviction panel" and "no OHLC coverage for this symbol"), NOT yet
validated on real data as of writing.** Run
`python scripts/run_full_pipeline.py` with both flags enabled and compare
`fundamentals_only`/`combined`'s numbers against the pre-technical_momentum
baseline before drawing any conclusion — same caveat as every other new
signal in this project.
