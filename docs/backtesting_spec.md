# Backtesting & attribution — methodology

## Goal

Answer two questions: "how did the combined regime + fundamentals system perform,"
and "how much of that came from each component individually." Both require running
more than one strategy, so the module always produces at least four parallel
backtests (five when the geometric crash-risk overlay is enabled), not just one.

## The components (`src/backtesting/attribution.py::run_component_backtests`)

| Component | Stock selection | Market timing |
|---|---|---|
| `benchmark` | none (100% index) | none (always 100% invested) |
| `regime_only` | none (100% index) | exposure scaled by GMM/KMeans/HMM-detected regime |
| `fundamentals_only` | top-quantile composite score | none (always 100% invested) |
| `combined` | top-quantile composite score | exposure scaled by detected regime, further scaled by the geometric overlay if enabled |
| `geometric_overlay_only` *(only if enabled)* | none (100% index) | exposure scaled by the geometric wedge-product crash-risk flag — **not** the GMM regime label |

This design isolates each signal cleanly: `regime_only` vs `benchmark` isolates pure
GMM-based market-timing value; `fundamentals_only` vs `benchmark` isolates pure
stock-selection value; `geometric_overlay_only` vs `benchmark` isolates the geometric
signal's own standalone value, independent of and directly comparable to
`regime_only`; `combined` is what the actual system would have done with every
enabled signal live.

Regime exposure comes from `configs/config.yaml`'s `backtesting.exposure_by_regime`
— a direct mapping from regime label (0 = calmest, per `regime_detection`'s
volatility-ordering) to a 0-1 invested fraction. The shipped defaults
(100% / 85% / 55% / 25% for regimes 0-3) are a reasonable starting de-risking
schedule, not a validated finding — they should be tuned once real backtests exist.

## Geometric overlay (`src/backtesting/strategies.py`, only when `regime_detection.geometric_signal.enabled`)

By explicit design, the geometric wedge-product crash-risk signal
(`regime_detection/geometric_signal.py`) is kept completely separate from the
GMM/KMeans/HMM regime model — see `docs/regime_detection_spec.md`'s section on it.
It reaches the backtest through two mechanisms, both driven by
`strategies.build_geometric_overlay_weights`/`apply_geometric_overlay` and the same
`backtesting.geometric_crash_exposure_multiplier` config value (default 0.5 = cut
exposure in half on flagged days):

1. **`geometric_overlay_only`** — a 5th standalone component, structurally identical
   to `regime_only` (100% benchmark exposure scaled by one signal) but driven by
   `geometric_crash_risk_flag` instead of the regime label. This is the fair,
   apples-to-apples comparison for "does this signal alone do anything" — same
   construction as `regime_only`, different (and independently computed) input.
2. **Applied on top of `combined`** — after `combine_regime_and_fundamentals`
   produces the fundamentals-selection-scaled-by-regime weights, `apply_geometric_overlay`
   multiplies in the same exposure factor, so `combined`'s total invested fraction
   can be cut further on days the geometric signal is flagged, on top of whatever
   fundamentals selection and GMM-based regime timing already decided. This is a
   pure exposure overlay — it never touches which stocks are selected.

Both are no-ops when the signal is disabled (`geometric_crash_flag=None` end to end):
`run_component_backtests` returns exactly the original 4 keys, and `combined` is
computed identically to before this overlay existed —
`tests/test_backtesting.py::test_component_backtests_include_geometric_overlay_only_when_flag_supplied`
pins this down directly, including that `regime_only`'s returns are byte-identical
with and without the flag (since it's never applied there).

`compute_return_decomposition` adds two additional, purely informational fields
when `geometric_overlay_only` is present — `geometric_overlay_cagr` and
`geometric_overlay_contribution` (= geometric CAGR - benchmark CAGR, directly
comparable to `regime_contribution`) — without being folded into the existing
fundamentals/regime/interaction additive identity, so that identity's correctness
doesn't depend on whether the geometric signal happens to be enabled.

## Engine (`src/backtesting/engine.py`, `src/backtesting/vbt_engine.py`)

Both engines apply a 1-trading-day execution lag by default (`lag_days`, see
"Execution-timing look-ahead bias" below) — a weight decided using data through
day T only starts earning returns from day T+1 onward. This is not optional in any
meaningful sense; read that section before ever passing `lag_days=0`.

Two interchangeable engines, selected via `configs/config.yaml`'s `backtesting.engine`
(`"vectorbt"`, the default, or `"custom"`):

- **`vbt_engine.py` (default)** — `vectorbt.Portfolio.from_orders` with
  `size_type="targetpercent"`, given actual daily price levels and the same daily
  weights matrix the custom engine uses. This is now the primary engine (per explicit
  request to move off the hand-rolled loop) — vectorized/numba-backed order
  execution and cash accounting from a well-used library, rather than this project's
  own one-off simulation.
- **`engine.py` (fallback)** — the original, deliberately minimal implementation: a
  daily weights matrix (forward-filled between rebalance dates) dotted with a daily
  returns matrix, long-only, no leverage, transaction costs proportional to
  turnover. Kept specifically as the dependency-free fallback: `attribution.py`'s
  `_run_backtest` dispatcher automatically falls back to this (with a logged
  warning) if vectorbt isn't installed, or if `stock_prices`/`benchmark_prices`
  weren't supplied (the vectorbt path needs actual price levels, not just returns).

Both produce the same output contract (`{"returns", "gross_returns", "turnover",
"equity_curve"}`), so `attribution.py`/`metrics.py`/`reporting.py` don't need to know
which engine ran. **Caveat**: `vectorbt` was not installable in this build sandbox
(no PyPI access — same limitation as `hmmlearn`/`yfinance`/`pytest` elsewhere in this
project), so the vectorbt path is implemented and import-guarded but hasn't actually
been executed here; `tests/test_backtesting.py::test_vbt_engine_falls_back_to_custom_when_unavailable`
covers the fallback behavior, but run a direct numerical comparison against
`engine.run_backtest` on a small panel in an environment with vectorbt installed
before trusting the vectorbt numbers in a real report. Both engines model costs
simply (bps-of-turnover) — no market impact, slippage curves, or liquidity
constraints, which matters more for the smaller/less-liquid tail of NIFTY500 than
the top 100.

## Execution-timing look-ahead bias (`engine.py`'s `lag_days`) — a real bug, found and fixed

Distinct from the point-in-time-*data* issue below: this one is about *when* a
weight decided from a given day's data is allowed to start earning returns, and it
affected **every backtest this project has run**, not just the technical-signal
add-ons — it's in `engine.run_backtest`/`vbt_engine.run_backtest_vbt` themselves.

Every weight-builder in this codebase (the regime label, the fundamentals composite
score, the dispersion score, Ichimoku) computes its signal for day T using data up
to and *including* day T's own close — an SMA at day T necessarily uses T's closing
price. The bug: `run_backtest` was applying that day-T weight directly to day T's
own realized return (`r[T] = close[T]/close[T-1] - 1`). That's impossible to
replicate live — you can't observe today's close, decide to be invested because of
it, and also have captured today's own return; the earliest you could actually act
on a close-derived signal is the next trading session.

Found via a deliberately unambiguous test case: a weight that's 0 the day before a
+50% single-day price spike and flips to 1 *exactly* on the spike day. The
pre-fix engine attributed the entire +50% to that decision — impossible in
practice, since the decision to flip to 1 could only have been made *because of*
seeing that day's close, by which point the day's return had already happened.

**Fix**: `run_backtest`/`run_backtest_vbt` now take a `lag_days` parameter
(**default 1**, applied everywhere in this project — `configs/config.yaml`'s
`backtesting.lag_days`) that shifts the effective weight series by one trading day
before computing returns: `portfolio_return[T] = weights[T-1] * returns[T]`, the
standard no-look-ahead backtesting convention. `lag_days=0` reproduces the old
(unsafe) behavior exactly and is kept only for tests/debugging that need to isolate
the pure weight×return arithmetic — see
`tests/test_backtesting.py::test_engine_does_not_capture_same_day_spike_from_a_same_day_decision`
for the regression test pinning this down, and
`test_engine_lags_weights_by_default` for the general property
(`gross_returns == (returns * weights.shift(1)).sum(axis=1)`).

**This changes every backtest number produced before this fix**, including the
original fundamentals+regime pipeline's very first report generated in this
project, not just the technical-signal scripts — re-run anything you're relying on.
Directionally, results should generally get more conservative (a bit lower returns,
since same-day "lucky" capture is no longer possible), though the exact effect
depends on the specific data and signal.

## Rebalancing and the look-ahead-bias trap

`build_fundamental_portfolio_weights` (`src/backtesting/strategies.py`) expects
`scores_by_date` to already contain **point-in-time** composite scores — i.e. the
score for a stock as of each historical rebalance date must only use data that was
actually available on that date. Running `fundamental_analysis.pipeline.run_pipeline`
once "as of today" and reusing that single snapshot at every historical rebalance
date leaks future information into the backtest and produces an inflated,
unrealistic result.

**This is now addressed** by `src/fundamental_analysis/point_in_time.py` (see
`docs/fundamental_analysis_spec.md`'s "Point-in-time fundamentals" section for the
full mechanism) — `scripts/run_full_pipeline.py`'s `step_pit_fundamentals` builds
`scores_by_date` by calling `fundamental_analysis.pipeline.run_pipeline` once per
rebalance date, each time with that date's PIT snapshot (revenue/net_income/eps
replayed forward from quarterly results via a strictly-backward `merge_asof`, so a
later result cannot leak into an earlier date by construction), rather than a single
current snapshot reused everywhere. What remains current-snapshot-only (and
therefore still a look-ahead risk if trusted too literally at early historical
dates): shareholding/pledge percentages, analyst estimates, and sector/industry
classification — see the "Known gaps" section below and
`docs/fundamental_analysis_spec.md`.

## Metrics (`src/backtesting/metrics.py`)

Standard set: CAGR, annualized volatility, Sharpe ratio, Sortino ratio (downside-only
deviation), max drawdown, Calmar ratio (CAGR / |max drawdown|), hit rate, and — when
a benchmark is supplied — OLS alpha/beta. All computed directly from the daily
returns series, no external dependency (`empyrical`, `quantstats`, etc.) was pulled
in to keep the dependency surface small; swap in one of those later if more exotic
metrics (tail ratio, VaR, Omega ratio, ...) are needed.

## Attribution methodology — and its limits

`compute_return_decomposition` reports:

```
combined_excess_cagr ≈ fundamentals_contribution + regime_contribution + interaction_effect
```

where each `_contribution` is that component's own CAGR minus the benchmark's CAGR,
and `interaction_effect` is whatever's left over. This is an **additive
approximation**, not an exact decomposition — because the combined strategy's
returns are a genuine multiplicative interaction (fundamentals stock returns ×
regime exposure scaling) compounded daily, the CAGRs of the three components don't
sum exactly. The `interaction_effect` line captures that gap and is worth reading:
a small interaction effect means the two signals contribute roughly independently; a
large one (either sign) means they meaningfully amplify or offset each other and the
"contribution" split should be treated as directional, not precise. An exact
decomposition would need period-by-period geometric (Brinson-style) attribution —
worth adding if the interaction term turns out to be large in practice.

## Reports (`src/backtesting/reporting.py`, `plotting.py`)

Each run of `scripts/run_backtest.py` writes to `reports/` (configurable via
`backtesting.report_dir`):

```
reports/
├── backtest_report.md       # the full report: tables + embedded figures
├── figures/
│   ├── equity_curves.png        # all 4 components overlaid
│   ├── drawdowns.png            # all 4 components overlaid
│   ├── rolling_sharpe.png       # 63-day rolling Sharpe, all 4 components
│   ├── regime_timeline.png      # benchmark price with regime-shaded background
│   ├── contribution_bar.png     # the attribution decomposition, visually
│   └── score_distribution.png   # latest composite-score histogram
└── tables/
    ├── attribution_table.csv    # full performance_summary per component
    └── return_decomposition.json
```

Every figure uses a fixed, colorblind-safe categorical palette (Okabe-Ito) with a
strategy-name-to-color mapping that's identical across every chart, so "combined" is
always the same color everywhere in a report — see `plotting.py`'s
`COMPONENT_COLORS`.

## Adaptive Ichimoku (`src/backtesting/adaptive_ichimoku.py`, `scripts/run_adaptive_ichimoku_backtest.py`)

A follow-on to the dispersion signal above, motivated by an initial real-data
result: the threshold-based dispersion strategy came in with beta ~0.83
against buy-and-hold — a signature of an *exposure* problem (time spent
completely flat waiting for a confirmed threshold crossing costs return in
an up-trending market), not necessarily a bad-calls problem. Rather than
gating exposure with a hard threshold, this uses the dispersion score's
magnitude (`|signed_score| ∈ [0,1]`) as a continuous dial on Ichimoku Cloud's
lookback periods — `period(t) = round(base_period * scale(magnitude(t)))`.

Two competing, untested hypotheses about which direction that dial should
turn (`compute_adaptive_periods`'s `direction` parameter):
`shrink_when_high` (periods shrink once a trend is already confirmed —
KAMA-style reasoning) vs `shrink_when_low` (periods shrink during quiet/
ranging periods to catch the next breakout earlier — the one that most
directly targets the specific under-exposure problem that motivated this).
Both are implemented, alongside a `static` (non-adaptive, fixed 9/26/52)
baseline, specifically so they can be run head to head rather than one
being asserted correct — `scripts/run_adaptive_ichimoku_backtest.py --variant all`
does exactly that.

**Full Ichimoku, not a simplified version** — an earlier pass of this module
used a close-derived high/low proxy and skipped the forward-shifted cloud;
both were corrected:

- **True OHLC**: `yfinance_fetcher.fetch_price_panel_ohlc` (multi-symbol) /
  `scripts/_common_cli.load_prices_ohlc` (a long-format
  `date,symbol,open,high,low,close[,volume]` CSV, or live yfinance) supply
  real intraday high/low — `build_ichimoku_weights` takes a
  `{symbol: OHLC DataFrame}` dict, not a close-only wide panel.
- **A genuinely forward-shifted cloud, even with an adaptive (day-varying)
  kijun_period.** Textbook Ichimoku plots Senkou Span A/B `kijun_period`
  days ahead — trivial with a fixed period (`.shift()`), but "shift forward
  by a period that itself changes every day" needs an actual mechanism, not
  a shortcut. `_scatter_forward` projects each day `i`'s causally-computed
  Senkou values to target position `i + round(kijun_period[i])` — the
  offset known at calculation time, the only causal choice. Two
  consequences that are inherent to what a variable-lag forward projection
  *means*, not approximations: if `kijun_period` shrinks over time,
  multiple days' projections can collide on the same future target (the
  most recently computed one wins); if it grows, some future days get no
  direct projection (forward-filled from the last scattered value, same as
  a real continuous cloud). Verified two ways in
  `tests/test_adaptive_ichimoku.py`: no-lookahead under truncation, and —
  the strongest check — the general scatter/variable-window machinery given
  *constant* periods exactly reproduces the independently-implemented
  vectorized `compute_static_ichimoku` bit-for-bit.
- **Chikou Span confirmation**: today's close compared against price from
  `kijun_period` days ago (`_variable_lag_lookup` — purely backward, no
  ambiguity the forward case has). `generate_ichimoku_signal` requires all
  three confirmations (cloud position, Tenkan/Kijun relationship, Chikou vs
  lagged price) to agree — the standard "triple confirmation" reading of
  Ichimoku — before taking a position; no 2-of-3 voting.

Trading rule is a continuous state check, not a same-day cross trigger — an
earlier version used a Tenkan/Kijun-cross trigger and had a real bug caught
by testing: the cloud (longer periods) lags the faster cross, so by the time
price actually confirmed outside the cloud, the triggering cross had already
happened days earlier and didn't re-fire, causing entries to be silently
missed on sustained moves.

## Technical signal: multi-scale SMA dispersion (`src/backtesting/technical_signals.py`, `scripts/run_technical_backtest.py`)

A price-only, per-symbol signal, independent of both `regime_detection` and
`fundamental_analysis` — built to an exact specification the user gave (score
formula, normalization, threshold logic), not derived or validated by this
project. **No backtest of it has been run on real data** — treat it exactly
like the geometric wedge-product signal: mechanically correct and unit-tested,
zero claim about whether it's actually profitable.

**The score**: a ladder of 4 SMAs at t/2t/4t/8t (dyadic scale spread),
`s = |SMA(t)-SMA(2t)| + |SMA(2t)-SMA(4t)| + |SMA(4t)-SMA(8t)|`, each term
normalized by price. This measures trend *strength* across scales, not
*direction* (sum of absolute values). Rolling self-relative z-score (default
252d) + `tanh` squashes it to a magnitude in (0, 1); multiplying by
`sign(SMA(t) - SMA(8t))` turns it into a genuinely signed `signed_score` in
(-1, 1) — sign = direction, magnitude = how historically unusual that
directional dispersion currently is.

**Entry/exit**: a single symmetric band `[-q_entry, +q_entry]` drives two
mirror-image modes (`generate_signal`'s `mode` parameter) —
`trend` enters in the direction `signed_score` already points (momentum);
`mean_reversion` enters opposite it (fade). Both exit when
`abs(signed_score)` falls back below a smaller `q_exit` (hysteresis, default
`0.3 * q_entry`), and both handle a direct reversal (score swings from one
extreme straight past the other without lingering near zero) by flipping
straight to the new side rather than getting stuck — this was a real bug
found during development (a sharp trend reversal left a `trend`-mode
position stuck long the entire way through a subsequent downtrend, since the
original state machine only ever checked the exit condition once already in
a position) — see `tests/test_technical_signals.py`'s
`test_trend_mode_reverses_directly_without_getting_stuck` for the regression
test pinning down the fix.

**On the window size `t`**: no analytically correct answer — this needs
empirical (walk-forward) tuning on real data, same caveat as every other
unvalidated parameter in this project. `t=10` (ladder 10/20/40/80 trading
days) is the implemented default, a reasonable starting point for a daily
swing signal, not a validated optimum.

**Real-data result and the under-exposure problem**: an initial live-data run
of the threshold-gated version (`build_technical_signal_weights`) came in
with beta ~0.83 against buy-and-hold — a signature of spending real time
completely flat waiting for a confirmed threshold crossing, which costs
return in an up-trending market independent of whether the "in" periods were
individually good calls. Two follow-on attempts at this, in order:

1. **Adaptive Ichimoku** (see below) — tried modulating a *different*
   indicator's timeframe with this score instead of gating exposure
   directly. Made things worse in both tested directions (beta 0.27-0.41,
   lower than the original problem) — see that section.
2. **Continuous conviction-weighted sizing**
   (`build_conviction_weighted_signal_weights`) — the direct fix: instead of
   `|signed_score| > q_entry` gating any exposure at all, weight is
   proportional to `signed_score` itself, divided by the fixed universe
   size N (not the count of active symbols) — so AGGREGATE portfolio
   exposure is itself continuous, rather than snapping to "some exposure"
   the moment any one symbol crosses a threshold. Bounded automatically
   (gross exposure ≤ 100%) since each clipped/signed per-symbol weight is
   in [0,1] or [-1,1] and there are N of them each divided by N.

   **Whether this actually raises average exposure/beta is empirically
   ambiguous, not guaranteed** — worth being explicit about, since the
   mechanism could plausibly cut either way. The threshold version gives
   *full* per-symbol weight (`1/n_active`) the instant conviction crosses
   `q_entry` (0.51 treated identically to 0.99); continuous sizing instead
   scales proportionally the whole way, so days with conviction just above
   the old threshold get *less* weight than before, while days with
   conviction below the old threshold now get *some* weight instead of
   zero. Which effect dominates depends on the empirical distribution of
   conviction levels in the data — in one synthetic test it went the
   *opposite* direction from the original hypothesis (lower average
   exposure and lower beta than the threshold version, not higher), while
   CAGR still improved (fewer false-positive periods, much lower turnover:
   roughly a quarter of the threshold version's). This is exactly why
   `scripts/run_technical_backtest.py --sizing all` runs both side by side
   and reports `avg_exposure` and the `continuous - threshold` deltas
   directly, rather than asserting which one wins — that has to be checked
   on the actual data being used, not assumed from the mechanism alone.

**Usage**: `technical_signals.build_technical_signal_weights` takes a price
panel and produces a daily (not sparse) target-weight matrix directly usable
by `engine.run_backtest`/`vbt_engine.run_backtest_vbt` with no separate
alignment step, using the same equal-weight-among-selected convention as
`build_fundamental_portfolio_weights` for direct comparability.
`scripts/run_technical_backtest.py` wraps this into a standalone CLI (fresh
yfinance data or a local price CSV, no fundamentals/regime pipeline needed)
that prints a performance summary against buy-and-hold and saves charts. It
is **not yet wired into `attribution.run_component_backtests`** as a named
component (unlike the geometric overlay) — that's a natural next step if
this signal is worth pursuing further, following the same pattern used for
`geometric_overlay_only`.

## Known gaps / next steps

- **Point-in-time fundamentals is implemented** (see above) for the quarterly-
  tracked fields (revenue/net_income/eps) — the previous top blocker. Shareholding/
  pledge/analyst-estimate fields are still current-snapshot-only; see
  `docs/fundamental_analysis_spec.md`'s known gaps for what that means in practice.
- **vectorbt engine is implemented but unexecuted** in this build sandbox (no PyPI
  access) — see the Engine section above. Validate it numerically against
  `engine.run_backtest` before trusting it in a real report.
- **No real price/regime/fundamentals data** — same root cause as the other two
  modules: the build sandbox had no outbound network access.
  `scripts/run_full_pipeline.py` orchestrates the full download -> regime ->
  PIT-fundamentals -> backtest -> report sequence end-to-end (with per-step disk
  caching/freshness checks), but has only been exercised here in `--offline` mode
  against small hand-built fixtures (see its docstring) — run it for real on a
  machine with network access before trusting the output on the actual NIFTY500.
- **Costs are simplified** (flat bps on turnover) — fine for a first pass on
  large/liquid NIFTY500 names, understates real costs on the smaller/less liquid
  tail of the universe.
- **Rebalance frequency** now has an explicit default (`fundamental_analysis.point_in_time.rebalance_frequency`,
  monthly by default) via `scripts/run_full_pipeline.py`'s PIT scoring loop, rather
  than being implied purely by whatever rows happen to be in `scores_by_date`.
- **No benchmark-relative risk controls** (position caps, sector caps, max
  single-name weight) — the current fundamentals-only portfolio can concentrate
  heavily in one sector if that sector's stocks dominate the top-quantile screen.
- **options_earnings dimension has no live data source** — see
  `docs/fundamental_analysis_spec.md`'s known gaps; it degrades gracefully to NaN
  (harmless to the composite score) without one.
- **geometric overlay (wedge-product crash-risk signal) is unvalidated on real
  data** and off by default — see `docs/regime_detection_spec.md`'s section on it
  before enabling it in a real backtest. It's now a standalone overlay
  (`geometric_overlay_only` component + an exposure cut applied on top of
  `combined`), deliberately decoupled from the GMM/KMeans/HMM regime model — see
  the "Geometric overlay" section above.
- **`geometric_crash_exposure_multiplier` (0.5 default) is an arbitrary starting
  point**, not tuned or validated against real data — same caveat as
  `exposure_by_regime`'s defaults above.

## Consensus-governor components (`governed_regime_only` / `governed_combined`)

Added when `regime_detection.consensus_governor.enabled` is set (see
`docs/regime_detection_spec.md`'s consensus-governor section). Same
construction as `regime_only`/`combined`, but driven by `active_regime`
(entropy-gated, persistence + hysteresis governed) instead of the raw
per-bar `regime` label — purely additive, `regime_only`/`combined`
themselves are never modified. `exposure_by_regime` needs a `"transitional"`
key (default 0.25 in `configs/config.yaml`) since `active_regime` can take
that string value on ambiguous bars; an unmapped `"transitional"` defaults
to full (1.0) exposure with a logged warning rather than failing silently
— see `strategies.build_regime_exposure_weights`.

`compute_return_decomposition` adds two informational pairs:
`governed_regime_cagr`/`governed_regime_contribution` (compare directly
against `regime_contribution`) and `governed_combined_cagr`/
`governed_vs_raw_combined_delta` (governed `combined` minus raw `combined`
— the number that actually answers "did reducing over-switching help once
stock selection is layered on top").

Wired into `scripts/run_full_pipeline.py` automatically: `active_regime` is
pulled from `regime_result` if present and passed straight through to
`run_backtest_pipeline`.

## Ichimoku components (`ichimoku_only` / `combined_with_ichimoku`)

Added when `technical_signals.ichimoku.enabled` is set. This is the first
version of Ichimoku actually wired into the main attribution table — until
now it only existed in the standalone `scripts/run_adaptive_ichimoku_backtest.py`
comparison script (see that script's docstring and `adaptive_ichimoku.py`
for the three variants: `static`, `shrink_when_high`, `shrink_when_low`).

**Default variant is `static`** — not because it's been validated as
good, but because it was the least-bad of the three in every run so far
(real and synthetic; both adaptive directions underperformed it). Change
`technical_signals.ichimoku.variant` in `configs/config.yaml` to try the
others; run `scripts/run_adaptive_ichimoku_backtest.py --variant all`
against real data before trusting any of them.

**How it acts on the same portfolio as everything else** — this was the
specific ask, so worth being explicit: `strategies.apply_ichimoku_gate`
takes the already-built `combined` weights matrix (fundamentals selection
x regime exposure x geometric overlay) and zeroes out any symbol's weight
on any day Ichimoku's triple-confirmation isn't currently bullish for that
symbol. It's a confirmation FILTER on the existing selection, not an
independent sizing signal added on top — a stock still has to clear
fundamentals + regime exposure first; Ichimoku can only cut exposure
further, never add a stock the other signals didn't already select. This
cuts total invested fraction on days confirmation fails rather than
reallocating that capital elsewhere (same convention as
`apply_geometric_overlay`'s exposure cut).

Symbols with no OHLC data (e.g. a fetch failure for one ticker) are passed
through UNGATED, not zeroed — a data gap in the confirmation signal
shouldn't silently exclude a stock the fundamentals/regime signals
otherwise selected. Logged once per call so coverage gaps stay visible
(`strategies.apply_ichimoku_gate`).

`ichimoku_only` is the signal backtested standalone (its own weight
matrix directly, not gating anything) — same "isolate the signal alone"
convention as `fundamentals_only`/`regime_only`/`geometric_overlay_only`.

`compute_return_decomposition` adds `ichimoku_cagr`/`ichimoku_contribution`
(standalone signal vs benchmark) and `combined_with_ichimoku_cagr`/
`ichimoku_vs_raw_combined_delta` (gated `combined` minus raw `combined` —
did requiring technical confirmation on top of fundamentals+regime
selection help or just cost exposure).

**Data plumbing**: `scripts/run_full_pipeline.py`'s new `step_ichimoku`
fetches true OHLC (not close-only) for the universe via
`yfinance_fetcher.fetch_price_panel_ohlc`, caches it as a single
long-format CSV (`data/raw/stock_prices_ohlc_long.csv` — same format
`run_adaptive_ichimoku_backtest.py --price-csv` expects, so the exact same
cached file works with both), then calls `adaptive_ichimoku.build_ichimoku_weights`.
Coverage is typically a SUBSET of the full fundamentals-eligible universe
(whichever symbols yfinance actually returned OHLC for) — logged explicitly
so a silent coverage gap doesn't go unnoticed.

**Status: still experimental/unvalidated on real data**, same caveat as
the standalone script and every other new signal in this project — only
tested end-to-end on synthetic data so far. `technical_signals.ichimoku.enabled`
defaults to `false`.

## Ichimoku conviction tilt (`combined_ichimoku_tilted`) — reallocation instead of gating

Added 2026-07-24 after real-data results showed `ichimoku_only` standalone
was the single best-performing component found (21.8% CAGR, best Sharpe),
while every gate-based `combined_with_ichimoku` construction — `hard_gate`
and `breadth_scalar` alike — lost a large chunk of `combined`'s CAGR (6.6pp)
despite better Sharpe/drawdown. The common thread across both gate modes:
they can only ever CUT total exposure when confirmation is weak, never
redirect it. `strategies.apply_ichimoku_conviction_tilt` is a structurally
different mechanism: it REALLOCATES `combined`'s capital among its
already-held names by relative Ichimoku conviction, with total exposure
held fixed to `combined`'s own to machine precision. It can only move
capital between names the base (fundamentals + regime) selection already
picked — never invest less overall, never add a name the base selection
didn't choose. This directly targets the "gating throws away real
information" hypothesis the real-data numbers pointed at.

Computed as a component genuinely additional to (not a mode switch on)
`combined_with_ichimoku` — both can be produced side by side, controlled
independently via `technical_signals.ichimoku.tilt_strength` (0.0 = off,
the previous behavior; `combined_ichimoku_tilted` only appears when
nonzero) vs. `confirmation_mode` (which only affects `combined_with_ichimoku`).

`compute_return_decomposition` adds `combined_ichimoku_tilted_cagr`/
`ichimoku_tilt_vs_raw_combined_delta`, directly comparable to
`ichimoku_vs_raw_combined_delta` — both measure "combined plus Ichimoku"
against plain `combined`, via two different mechanisms.

**Status: implemented, verified mechanically (exposure-preservation to
1e-16, correct tilt direction, correct missing-coverage pass-through),
CONFIRMED NEGATIVE on real data as of 2026-07-24.**

After fixing a real bug (the tilt was initially fed `build_ichimoku_weights`'s
already-normalized output — ~1/n_active scale, e.g. ~0.002 for 500 symbols —
instead of raw `[0,1]` conviction, making it a silent no-op; fixed via
cross-sectional z-scoring, see `strategies.apply_ichimoku_conviction_tilt`'s
docstring), a real run showed `combined_ichimoku_tilted` underperforming
plain `combined` on CAGR (-0.57pp), Sharpe (-0.097), AND max drawdown
(4.7pp worse) simultaneously. A diagnostic ruled out "compressed conviction
spread within the selected basket" as the explanation (0.0036 vs 0.0041
std, full universe vs. within-basket — not meaningfully different).

Combined with `confirmation_mode`'s gate ALSO being confirmed negative
(large CAGR loss, though better Sharpe/drawdown), the current read: two
structurally different post-selection mechanisms (cut exposure, reallocate
exposure) have both failed to convert `ichimoku_only`'s standalone edge
(21.7% CAGR, best of any component) into value for `combined`. Working
hypothesis: that edge is a stock-picking/rotation signal across the FULL
500-stock universe, not a within-basket conviction/timing enhancer —
neither gating nor reallocating an already-fixed ~100-name selection can
recover value that depends on being free to choose which names to hold in
the first place. `tilt_strength` now defaults to `0.0` (off), matching how
`consensus_governor` was handled after its own confirmed-negative result.

### Two further integration ideas — the first is now the clear next step

Both came out of the same real-data finding. With both post-selection
mechanisms (gate, tilt) now confirmed negative, #1 is the recommended next
thing to build if Ichimoku is worth pursuing further at all:

1. **Ichimoku conviction as an eighth fundamentals composite dimension.**
   Rather than acting on the portfolio post-selection at all (gating or
   tilting), feed the conviction score into stock SELECTION itself — a
   cross-sectionally z-scored `technical_momentum` dimension alongside
   `valuation`/`growth`/etc. in `fundamental_analysis/pipeline.py`'s
   composite score, using the conviction value as of each rebalance date
   (no look-ahead, since it only uses OHLC data up to that date, same
   discipline as everything else PIT-scored). This is the only one of the
   three mechanisms that can genuinely change WHICH stocks get selected,
   not just how much of the already-selected basket is held or how it's
   split — and given both mechanisms that only act after selection has
   already happened turned out negative, that distinction now looks load-
   bearing, not academic.

2. **Regime-conditional trust of Ichimoku.** The original research doc's
   own strategy-family table ties trend-following approaches to
   bullish/trending regimes specifically, and flags them as failure-prone
   in choppy/crisis conditions — exactly the environment where a
   trend-following signal like Ichimoku should be least reliable. Rather
   than applying the tilt/gate unconditionally every day, scale
   `tilt_strength` (or gate/no-gate) by the current regime — e.g. full
   tilt strength in calm/trending regimes, zero in high-vol/transitional
   ones. Lower priority than #1 now, since it's still a post-selection
   mechanism and both of those have underperformed so far regardless of
   when they're applied.
