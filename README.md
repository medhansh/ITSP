# ITSP — Regime-Aware Algorithmic Trading System (Team DAC)

A machine-learning-driven trading system for Indian equities (NIFTY 500 universe)
that adapts strategy based on the detected market regime. This repository currently
scaffolds three of the four modules needed to answer "does this actually work":

1. **Regime detection** — unsupervised learning over market-wide features to classify
   the prevailing market regime (e.g. trending/mean-reverting, high-vol/low-vol,
   bull/bear/consolidation).
2. **Fundamental analysis** — a broad fundamentals engine for NIFTY 500 constituents
   that goes beyond earnings-report surprises to cover valuation, profitability &
   quality, growth, leverage & solvency, cash-flow quality, ownership & governance
   signals, and analyst/earnings-surprise tracking, rolled up into a composite score.
3. **Backtesting & attribution** — simulates four parallel strategies (benchmark,
   regime-timing-only, fundamentals-selection-only, and both combined), computes
   standard performance metrics for each, and decomposes the combined strategy's
   return into how much came from each signal individually. Produces a Markdown
   report with equity-curve, drawdown, rolling-Sharpe, regime-timeline, and
   attribution charts — see `docs/backtesting_spec.md`.

Sentiment analysis (regional-language NLP on Indian financial news) is **out of scope
for this pass** — a placeholder module exists so the final integration layer has a
stable import path, but it is not implemented yet.

## Quickstart: run everything with one command

```bash
python scripts/run_full_pipeline.py --config configs/config.yaml
```

Downloads/caches the universe list, price panels, fundamentals (current snapshot +
point-in-time quarterly history), fits the regime model, scores the universe
point-in-time at each rebalance date, runs the (vectorbt-backed) backtest, and
writes `reports/backtest_report.md`. Re-running reuses anything already cached and
fresh; pass `--force-refresh` to ignore the cache, or `--offline` to run purely off
whatever's already on disk (useful for testing the orchestration logic itself
without a network connection — see the script's docstring). Requires outbound
network access for the download steps, same as the individual `scripts/fetch_data.py`
subcommands below.

## Why this structure

- `src/<module>/` is organized so each module is independently developable and
  testable (per the June milestone: "development and validation of three ML modules
  independently"), and later composable in a `src/integration/` layer (July milestone)
  that hasn't been created yet since integration comes after the modules are validated.
- `data_fetchers` are isolated from `metrics`/`models` so data source swaps (NSE
  bhavcopy, screener-style fundamentals, a paid vendor, etc.) never touch the analysis
  logic — analysis code only depends on well-defined pandas DataFrame schemas
  documented in `docs/`.
- `configs/` centralizes tunables (universe list location, lookback windows, model
  hyperparameters, scoring weights) so experiments don't require code edits.
- `scripts/` holds thin CLI entry points; all real logic lives in `src/` so it's
  importable and unit-testable.

## Folder layout

```
ITSP/
├── README.md                      # this file
├── requirements.txt                # Python dependencies
├── .gitignore
├── configs/
│   └── config.yaml                 # universe path, lookbacks, model + scoring params
├── data/
│   ├── raw/                        # untouched vendor pulls (gitignored)
│   ├── processed/                  # cleaned/feature-engineered data (gitignored)
│   └── universe/
│       └── nifty500_list.csv       # NIFTY 500 constituent list (symbol, name, sector, industry)
├── src/
│   ├── common/                     # shared io/logging/date/scraping utilities
│   ├── regime_detection/
│   │   ├── features.py             # market-wide feature engineering (returns, vol, breadth,
│   │   │                           #   VIX, + range/volume features if OHLCV is available)
│   │   ├── geometric_signal.py     # optional wedge-product crash-risk signal (off by default)
│   │   ├── models.py               # GMM / KMeans / HMM regime models
│   │   └── pipeline.py             # fit + label pipeline, orchestrates features -> model -> labels
│   ├── fundamental_analysis/
│   │   ├── data_fetchers/          # nse_fetcher (universe list), screener_fetcher,
│   │   │                           #   trendlyne_fetcher, yfinance_fetcher, options_fetcher, merge.py
│   │   ├── metrics/                # one file per fundamental dimension (see below)
│   │   ├── scoring/                # composite scoring across all dimensions
│   │   ├── point_in_time.py        # replay quarterly history forward through time, no look-ahead
│   │   └── pipeline.py             # fetch -> compute metrics -> score, for the NIFTY500 universe
│   ├── backtesting/
│   │   ├── engine.py               # long-only portfolio simulation from a weights matrix (fallback)
│   │   ├── vbt_engine.py           # vectorbt-backed engine (default; falls back to engine.py)
│   │   ├── strategies.py           # regime-exposure / fundamentals-selection / combined weight builders
│   │   ├── technical_signals.py    # multi-scale SMA dispersion trend signal (experimental, price-only)
│   │   ├── adaptive_ichimoku.py    # Ichimoku with dispersion-score-adaptive periods (experimental)
│   │   ├── metrics.py              # CAGR, Sharpe, Sortino, max drawdown, Calmar, alpha/beta, ...
│   │   ├── attribution.py          # runs all 4 components, decomposes combined return by source
│   │   ├── plotting.py             # equity curve / drawdown / regime-timeline / attribution charts
│   │   ├── reporting.py            # assembles the Markdown report
│   │   └── pipeline.py             # orchestrates all of the above end to end
│   └── sentiment_analysis/         # placeholder only — not implemented this pass
├── scripts/
│   ├── run_full_pipeline.py        # CLI: the one-command start-to-finish run (see Quickstart)
│   ├── run_regime_detection.py     # CLI: fit regime model, label history, save output
│   ├── run_fundamental_analysis.py # CLI: score NIFTY500 universe on latest fundamentals
│   ├── run_backtest.py             # CLI: backtest + attribution + Markdown report with charts
│   ├── run_technical_backtest.py   # CLI: standalone backtest for the SMA-dispersion signal (no fundamentals/regime needed)
│   ├── run_adaptive_ichimoku_backtest.py # CLI: static vs adaptive-period Ichimoku, side by side
│   ├── _common_cli.py              # shared price-loading helper for the standalone signal-testing scripts
│   ├── fetch_data.py               # CLI: universe / prices / fundamentals / quarterly-history / sector-prices
│   └── probe_data_source.py        # CLI: diagnose scraper breakage on one symbol
├── reports/                        # generated backtest reports + figures (git-ignored contents; see below)
├── tests/
│   ├── test_regime_detection.py
│   ├── test_fundamental_analysis.py
│   ├── test_backtesting.py
│   └── test_data_fetchers.py
└── docs/
    ├── architecture.md             # system-level design & how modules will integrate
    ├── regime_detection_spec.md    # methodology, feature list, model choices, geometric signal
    ├── fundamental_analysis_spec.md# full breakdown of fundamental dimensions, PIT, & formulas
    ├── backtesting_spec.md         # backtest construction, engines, metrics, and attribution methodology
    └── data_sourcing_spec.md       # per-field source coverage, scraping etiquette, caching
```

## Fundamental analysis: expanded scope

The original abstract scoped fundamental analysis narrowly as an "earnings surprise
predictor analyzing promoter activity and analyst patterns." That's retained as one
submodule (`metrics/earnings_surprise.py`), but the module now covers the full
fundamental picture a discretionary analyst would use to screen NIFTY 500 stocks:

| Dimension | File | Examples |
|---|---|---|
| Valuation | `metrics/valuation.py` | P/E, P/B, EV/EBITDA, PEG, dividend yield |
| Profitability & quality | `metrics/profitability_quality.py` | ROE, ROCE, margins, Piotroski F-score |
| Growth | `metrics/growth.py` | revenue/EPS CAGR, growth stability |
| Leverage & solvency | `metrics/leverage_solvency.py` | debt/equity, interest coverage, Altman Z-score |
| Cash-flow quality | `metrics/cashflow_quality.py` | CFO/NI, FCF yield, accruals ratio |
| Ownership & governance | `metrics/ownership_governance.py` | promoter pledge %, promoter holding change, FII/DII flows |
| Earnings surprise & analyst revisions | `metrics/earnings_surprise.py` | surprise %, estimate revision momentum |
| Pre-earnings options signal | `metrics/options_earnings.py` | pre-earnings IV percentile, put/call OI ratio |

`scoring/composite_score.py` combines per-dimension z-scores (sector-relative,
since Indian sectors trade at structurally different multiples) into a single
fundamental score per stock, with configurable weights in `configs/config.yaml`.

For backtesting, scores are computed **point-in-time** — `src/fundamental_analysis/
point_in_time.py` replays each stock's quarterly results forward through time (via a
strictly backward-only `merge_asof`, so a later result can never leak into an earlier
rebalance date) rather than reusing one current snapshot everywhere. See
`docs/fundamental_analysis_spec.md`'s "Point-in-time fundamentals" section.

See `docs/fundamental_analysis_spec.md` and `docs/regime_detection_spec.md` for
full methodology.

## Backtesting & attribution

**Important, applies to every number below**: `engine.run_backtest`/
`vbt_engine.run_backtest_vbt` apply a 1-trading-day execution lag by default
(`backtesting.lag_days` in `configs/config.yaml`, default 1) — a weight decided
using data through day T only starts earning returns from day T+1. This fixes a
real bug: every weight-builder in this project computes its signal using a given
day's own closing price, so applying that weight directly to that same day's
realized return would mean "knowing" today's close before today's return had
happened — impossible to replicate live. See `docs/backtesting_spec.md`'s
"Execution-timing look-ahead bias" section for the full story and the regression
test that caught it. **This changes every backtest number produced before this fix
existed** — re-run anything you're relying on.

`scripts/run_backtest.py` (or `scripts/run_full_pipeline.py` for the full
start-to-finish run) simulates four strategies side by side (five if the
geometric crash-risk overlay is enabled — see below), using **vectorbt** as
the default simulation engine (`backtesting.engine` in `configs/config.yaml`; falls
back automatically to a dependency-free custom engine if vectorbt isn't installed)
so the combined system's performance can be attributed back to its signals
individually:

| Component | Stock selection | Market timing |
|---|---|---|
| `benchmark` | none (100% index) | none (always fully invested) |
| `regime_only` | none (100% index) | exposure scaled by GMM/KMeans/HMM-detected regime |
| `fundamentals_only` | top-quantile composite score | none (always fully invested) |
| `combined` | top-quantile composite score | exposure scaled by regime, further scaled by the geometric overlay if enabled |
| `geometric_overlay_only` *(opt-in)* | none (100% index) | exposure scaled by the geometric wedge-product crash-risk flag — deliberately **not** the GMM regime label; see `docs/backtesting_spec.md`'s "Geometric overlay" section |

Each run produces `reports/backtest_report.md` — CAGR/Sharpe/Sortino/max
drawdown/Calmar/hit-rate/alpha/beta per component, an additive decomposition of
the combined strategy's excess CAGR into a fundamentals contribution, a regime
contribution, and an interaction term (plus an informational geometric-overlay
contribution line when enabled), and six charts (equity curves, drawdowns,
rolling Sharpe, a regime-shaded price timeline, the attribution breakdown, and the
latest fundamental-score distribution) using a fixed, colorblind-safe palette so
each strategy is the same color across every chart.

See `docs/backtesting_spec.md` for full methodology, including the important
look-ahead-bias caveat on how `scores_by_date` must be point-in-time, and why the
attribution decomposition is an additive approximation rather than an exact split.

## Data sourcing

Price data comes from **yfinance**. Fundamentals are merged field-by-field from
**Screener.in** (primary — scraped, best financial-statement coverage), **yfinance**
(fallback — best analyst-estimate coverage), and **Trendlyne** (supplementary — most
of its useful data is paywalled behind a GuruQ/StratQ subscription, confirmed
directly; only momentum score and SWOT counts are free). Run:

```bash
python scripts/fetch_data.py universe            # NIFTY500 constituent list (NSE)
python scripts/fetch_data.py prices               # stock + benchmark price panels (yfinance)
python scripts/fetch_data.py fundamentals         # merged current snapshot from all 3 sources
python scripts/fetch_data.py history               # multi-year P&L history (Screener), for growth
python scripts/fetch_data.py quarterly-history     # per-quarter results w/ known-date tags, for point-in-time scoring
python scripts/fetch_data.py sector-prices         # sector index prices, for the optional geometric crash signal
```

Or run all of the above (plus regime detection, PIT fundamental scoring, and the
backtest) in one go with `scripts/run_full_pipeline.py` — see Quickstart above.
There is no free bulk source for NSE options-chain data (needed for the
`options_earnings` dimension) — see `src/fundamental_analysis/data_fetchers/options_fetcher.py`.

### Technical signal (standalone, no fundamentals/regime pipeline needed)

`src/backtesting/technical_signals.py`'s multi-scale SMA-dispersion signal can be
backtested on its own — against fresh yfinance data or an existing local price CSV —
without needing Screener scraping, point-in-time fundamentals, or regime detection
set up first:

```bash
# Fresh data straight from yfinance:
python scripts/run_technical_backtest.py \
    --symbols RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK \
    --start 2018-01-01 --mode both --sizing all

# Or against a price panel you already have on disk:
python scripts/run_technical_backtest.py \
    --price-csv data/raw/stock_prices.csv --mode trend --sizing continuous --t 10
```

`--sizing` picks between the original hard-threshold entry/exit gate
(`threshold`) and continuous conviction-weighted position sizing (`continuous`
— weight scales smoothly with signal strength instead of snapping between 0%
and a fixed share); `all` (default) runs both side by side and prints the
`avg_exposure`/beta/CAGR deltas directly. The threshold version's first
real-data run came in with beta ~0.83 against buy-and-hold — continuous
sizing was built as the direct attempted fix, but **whether it actually
raises average exposure is data-dependent, not guaranteed by the mechanism**
— see `docs/backtesting_spec.md`'s "Real-data result and the under-exposure
problem" section for why, and what happened in testing so far.

Prints a performance summary (CAGR/Sharpe/Sortino/max drawdown/alpha/beta) versus a
buy-and-hold benchmark, and saves equity-curve + drawdown charts and a CSV summary to
`--out-dir` (default `reports/technical_signal/`). See `docs/backtesting_spec.md`'s
"Technical signal" section for the construction and the (still unvalidated) status of
this signal — this script is exactly what lets you find out whether it does anything
on real data.

A follow-on signal — the same dispersion score used as a continuous dial on **Ichimoku
Cloud's lookback periods** instead of a hard entry/exit threshold — can be backtested
the same way, running the static (fixed-period) baseline alongside both competing
adaptive directions side by side for direct comparison. This is a full Ichimoku
implementation (true OHLC, a genuinely forward-shifted cloud, Chikou Span
confirmation — see `docs/backtesting_spec.md`), so it needs real high/low, not just
close — `--symbols` fetches full OHLC live; `--price-csv` expects **long format**
(`date,symbol,open,high,low,close[,volume]`), not the wide close-only format the
dispersion script above uses:

```bash
python scripts/run_adaptive_ichimoku_backtest.py \
    --symbols RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK \
    --start 2018-01-01 --variant all
```

See `docs/backtesting_spec.md`'s "Adaptive Ichimoku" section for what motivated this
(an under-exposure problem found in the dispersion-threshold strategy's first real-data
result) and the two open hypotheses about which direction the adaptive periods should
move.

This scaffold's build sandbox had **no outbound network access**, so none of this was
tested against live data end-to-end — the Screener/Trendlyne page structure was
confirmed via a separate tool (not this sandbox's shell), but exact HTML markup could
not be inspected directly (see `docs/data_sourcing_spec.md`). **Run
`scripts/probe_data_source.py <SYMBOL>` first** to sanity-check field coverage before
trusting a full-universe run — and read `docs/data_sourcing_spec.md`'s scraping
etiquette section (rate limits, ToS, what's actually free vs. paywalled per field)
before running it at scale.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/run_regime_detection.py --help
python scripts/run_fundamental_analysis.py --help
python scripts/run_backtest.py --help
python scripts/fetch_data.py --help
```
