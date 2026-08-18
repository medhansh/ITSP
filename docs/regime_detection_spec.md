# Regime detection — methodology

## Goal

Classify each trading day into one of N market regimes using only unsupervised
learning over market-wide (not single-stock) features, so downstream strategy logic
can condition on "what kind of market are we in" rather than assuming one static
approach works everywhere in time.

## VIX-bucket regime (production default, replacing GMM for exposure scaling / beta rotation / blend ladder)

`regime_detection.production_regime_source` controls which series feeds
`exposure_by_regime`, `beta_rotation.stress_by_regime`, and (when
`technical_signals.blend.stress_mode: "rigid"`) the `technical_momentum`
blend ladder — i.e. every consumer of "the" regime label, not just one
signal-weighting input. As of this build the default is
**`"vix_bucket_contemporaneous"`**, replacing the GMM(4) price-feature
regime described below, which remains available as an explicit fallback
(`production_regime_source: "gmm"`).

**What it is.** Same-day India VIX, bucketed via a 1D GMM fit directly on
VIX level (`src/regime_detection/vix_regime.py`, `n_buckets=4` fixed, not
swept per run — matching what was actually validated), then passed through
an asymmetric hysteresis filter (`apply_bucket_hysteresis`): a bucket
UPGRADE (more stressed) is always accepted instantly; a bucket DOWNGRADE
(calmer) only takes effect once VIX has stayed at or below the calmer
level for `min_days_to_downgrade` consecutive days (currently **10**, see
below). Fit on all available VIX history up to each date — no train/test
split in production, since "production" means using everything known up
to today.

**Why this replaced GMM.** A walk-forward comparison (fit each regime
source only on each fold's training window, score on the held-out test
window) found the price-feature GMM regime was prone to a specific
failure mode: flagging elevated stress during calendar years where India
VIX's own level was among the *lowest* in the sample (e.g. one fold spent
98% of its days in a non-calmest GMM state despite that year's mean VIX
being the second-lowest of any fold tested) — plausibly choppy/range-bound
price action producing elevated short-window realized-volatility features
without corresponding genuine options-implied fear. The VIX-bucket regime
does not make this mistake by construction, since it only ever reacts to
VIX itself. Across a 6-fold non-overlapping walk-forward, the VIX-bucket
arm's mean per-fold Sharpe delta versus GMM was positive but modest and
not conclusively different from zero at that sample size (bootstrap 90%
CI approximately [-0.10, +0.23] before hysteresis tuning) — this is a
real, evidenced, but not overwhelming edge, not a landslide.

**What was tried and NOT chosen** (documented so nobody re-tries these
blind):
- **A VIX-bucket forecasting arm** (`vix_bucket_supervised` in earlier
  development): an N-days-ahead gradient-boosted classifier forecasting
  the VIX bucket, rather than using today's contemporaneous bucket
  directly. Consistently underperformed the simple contemporaneous
  version — worse aggregate Sharpe, roughly double the turnover, and a
  specific, repeatable failure during the COVID fold (the classifier,
  trained on pre-2020 data, mistimed entry/exit around the fastest,
  largest single shock in the sample). The added forecasting complexity
  did not earn its keep; this project's "always test a cheap control
  before the complex version" discipline paid off here.
- **A GMM+VIX hybrid gate**: GMM's regime label, vetoed to calmest unless
  VIX's contemporaneous bucket confirmed elevation. Motivated by wanting
  "the best of both" (GMM's apparent edge in some crisis folds, VIX's
  edge in false-positive-stress folds), and deliberately designed to avoid
  the specific failure mode that made `consensus_governor` (below) a
  confirmed negative — this gate is asymmetric (instant entry, only the
  exit is ever delayed) rather than symmetric hysteresis, and its
  confirming signal is externally observed (VIX) rather than a second
  view of the same price data GMM already used. Despite that, on real
  walk-forward data it came out WEAKER than the plain VIX-bucket-only
  arm (mean fold delta +0.002 vs the plain arm's +0.070), because a pure
  veto mechanism can only ever suppress GMM, never correct a genuinely
  bad GMM call, and gave back most of its crisis-fold protection anyway.
  Not pursued further.

**Why `min_days_to_downgrade: 10`, specifically — and the risk of pushing
it higher.** A sweep across `min_days_to_downgrade ∈ {0..72}` showed most
INDIVIDUAL folds' own improvement plateauing or reversing somewhere around
10-20: e.g. the COVID fold's delta improved from a loss to a solid win by
`min_days≈10`, but had reversed to a LOSS again by `min_days=72`, despite
the coarse "was the regime ever calm" diagnostic reporting the exact same
(fully saturated) reading at both settings — the specific bucket LEVEL the
hysteresis locks onto, not just whether it's "elevated at all," continued
changing well past the point where that diagnostic looked stable. The
aggregate mean kept climbing well past `min_days=20`, but that climb was
increasingly the product of one or two folds reaching values 2-3x larger
than anything else in the whole sweep — the signature of overfitting a
handful of historical years' exact shape, not a genuinely better general
setting. `min_days_to_downgrade: 10` sits at the edge of the range where
improvement was broad-based across folds, deliberately short of the
aggregate-maximizing value. **Do not raise this casually** — re-run the
full walk-forward sweep and inspect the per-fold breakdown (not just the
aggregate mean) before changing it, and treat any setting above roughly
15-20 as unvalidated regardless of what the aggregate number shows.

**Known gaps in this evidence, honestly stated:**
- The walk-forward comparison used 6 non-overlapping folds (2020-2025) --
  a small sample for any of the above to be statistically decisive. An
  overlapping-fold re-run (more folds, but no longer independent
  observations) showed the same broad direction but did not tighten the
  uncertainty in any way that should be over-interpreted.
- The comparison held point-in-time fundamental scoring, beta-panel
  construction, and every other pipeline stage fixed across regime
  sources -- it isolates the regime source specifically, not the whole
  strategy end-to-end.
- Not yet re-validated after this session's default-config change (i.e.
  this documentation reflects the comparison's findings; a full
  `run_full_pipeline.py` run under the new default has not itself been
  re-run and re-reported here).

## Feature set (`src/regime_detection/features.py`)

| Feature | Rationale |
|---|---|
| Rolling returns (5d, 21d, 63d) | Captures trend direction/strength at multiple horizons |
| Rolling realized volatility (21d, 63d, annualized) | The single strongest regime discriminator empirically — vol clusters |
| Drawdown from running high | Distinguishes "healthy uptrend" from "recovering from stress" even at similar vol |
| Advance/decline ratio (breadth) | A market can be up on a few large caps while breadth deteriorates — an early regime-change signal; NIFTY500-specific since it needs full-universe advance/decline counts |
| India VIX level, 5d change, 1y z-score | Direct, forward-looking risk-pricing signal, independent of realized price action |
| Parkinson / Garman-Klass range volatility (21d, 63d) | Range-based vol estimators using intraday high/low (+ open for Garman-Klass) instead of just close-to-close returns — statistically more efficient than `realized_vol` over the same window, and captures intraday range that a close-only series simply doesn't have |
| Volume z-score (21d, 63d) | Flags volume spikes (capitulation, breakout, panic) independent of price direction — a close-only feature set can't distinguish a move on collapsing volume from one on expanding volume, even though those are meaningfully different regimes |
| OBV trend (21d, 63d) | Net directional volume pressure over the window, normalized to [-1, 1] — participation/conviction behind a price move |

Breadth and VIX are optional (`build_feature_matrix` degrades gracefully without
them) so the pipeline still runs price-only if those feeds aren't wired up yet. The
same is true of the range-volatility and volume features: they're only added if the
price data has high/low (+ open, + volume) columns — see "OHLCV features" below.

## OHLCV features: why close-only was the default, and what changed

Earlier passes of this module used **close only** — `data_loader`/`yfinance_fetcher`
only ever pulled the `Close` column from yfinance's OHLCV download, discarding
Open/High/Low/Volume entirely. That was a real gap, not a considered trade-off: it
meant the regime model was blind to intraday range and to volume/participation, both
of which are standard regime-detection inputs.

This is now fixed at the data layer, not just the feature layer, so it applies
automatically end-to-end:

- `data_fetchers/yfinance_fetcher.fetch_benchmark_ohlcv` (new) pulls the full
  open/high/low/close/volume panel for the benchmark index, not just close.
  `fetch_benchmark_series` (used wherever only a close Series is needed, e.g. the
  backtest engine) is now a thin wrapper around it. `scripts/fetch_data.py prices`
  and `scripts/run_full_pipeline.py` both write the full OHLCV to
  `data/raw/benchmark_prices.csv`.
- `data_loader.load_from_csv` / `load_from_yfinance` pass open/high/low/volume
  through if present — all still optional and independent of each other (a file
  with just high/low but no volume, or vice versa, is fine).
- `features.build_feature_matrix` picks up whichever of `open_`/`high`/`low`/
  `volume` it's given and adds the corresponding feature block(s); `pipeline.run_pipeline`
  wires this through automatically from whatever columns are in `price_csv` — no
  separate flag to enable it, it's driven purely by data availability, same
  convention as breadth/VIX.

**Caveat carried over from the original `^CRSLDX` ticker note**: verify the
benchmark ticker's `Volume` field is actually populated on Yahoo Finance before
relying on the volume-derived features — some India index/total-return-proxy
tickers report zero or missing volume. If that happens, the volume features come
back all-NaN and are silently dropped by `dropna()`, same graceful-degradation
behavior as any other missing optional input — not an error, just fewer features.

## Model choices (`src/regime_detection/models.py`)

All three share one `RegimeModel` interface (`fit` / `predict` / `predict_proba`),
selected via `configs/config.yaml`'s `regime_detection.model.type`:

- **GMM (default)** — soft clustering with full covariance, gives regime
  *probabilities* (useful for regime-confidence-weighted strategy sizing), and
  doesn't assume regimes are equal-sized or spherical the way KMeans does.
- **KMeans** — simple, fast, hard-assignment baseline to sanity-check GMM against.
  If GMM and KMeans disagree substantially, that's a sign the feature set or
  `n_regimes` needs revisiting before trusting either.
- **HMM (GaussianHMM)** — adds a transition-probability matrix, so today's regime is
  informed by yesterday's, not just today's features in isolation. This directly
  addresses regime "flicker" (single-day noisy reclassification) that i.i.d.
  clustering (GMM/KMeans) is prone to. Costs extra complexity and a `hmmlearn`
  dependency — worth adopting once GMM/KMeans are validated and flicker is
  empirically shown to be a problem, not before.

Regime labels are always re-ordered by mean volatility after fitting (see
`RegimeModel.fit`'s use of `regime_order_`), so label 0 is the calmest regime and the
highest label is the most volatile — independent of each backend's arbitrary internal
cluster ordering. This makes labels comparable across re-fits and across backends.

## Choosing `n_regimes`

The config defaults to 4 (`configs/config.yaml`), roughly: low-vol uptrend,
high-vol uptrend, low-vol downtrend/consolidation, high-vol downtrend/stress. This is
a starting point, not a finding — validate with silhouette score / BIC (GMM exposes
`.bic()`) across a range of `n_regimes` before treating 4 as settled.

## Known gaps / next steps

- `data_loader.load_from_yfinance` needs a live internet connection to pull NIFTY500
  and India VIX history — untested end-to-end in this pass since the build sandbox
  has no outbound access to Yahoo Finance. Verify the index ticker (`^CRSLDX`) is
  still correct before relying on it.
- No walk-forward validation yet — the model is fit once on the full history. For any
  backtest, regimes must be relabeled on an expanding/rolling window, not fit once
  on the full sample and looked up historically (that leaks future information into
  the vol-based label ordering).
- HMM branch is implemented but not covered by `tests/test_regime_detection.py`
  (`hmmlearn` wasn't installable in the build sandbox — no PyPI access). Install it
  and add an HMM test case before relying on that path.

## Geometric wedge-product crash-risk signal (`geometric_signal.py`)

An optional supplementary signal, off by default (`regime_detection.geometric_signal.enabled: false`
in `configs/config.yaml`). Source and full write-up: `src/regime_detection/geometric_signal.py`'s
module docstring — read it before turning this on, it covers both the math and an
important caveat.

**The idea, briefly**: pairwise correlation matrices only capture two-asset
relationships and miss a market-wide "everything sells off together" collapse.
Geometric algebra's wedge product gives the n-dimensional oriented "volume" spanned
by n (direction-normalized) sector return vectors — large when sectors move
independently (healthy rotation), collapsing toward zero when they move together
(panic/liquidation). It's a model-free structural signal, not a fitted distribution,
so (per the source article) it doesn't exhibit the same day-to-day "flickering" GMM/HMM
can around single-day outliers.

**Where the idea came from, and why it's gated off by default**: this was pointed at us
by the user (an article + a YouTube video), not derived by this team. The article
claims a 16.7% out-of-sample Sharpe improvement and a 98% transaction-cost reduction
versus an HMM, citing "a recent analysis of regime detection methods (2015-2023)"
without a traceable source. Those numbers are unverified and NOT reproduced on our
data — treat the signal as experimental until `geometric_signal.validate_against_known_crises`
has been run against real (not synthetic) NIFTY sector history and shown a genuine
lift over the flag's base rate around known Indian market stress periods (e.g. the
Mar 2020 COVID crash, the 2018 NBFC/IL&FS liquidity crunch). Turn `enabled: true` on
only after that validation, and don't repeat the source's specific performance
figures in any report this project produces.

**Inputs required**: a multi-column sector-price panel (`data_loader.load_sector_prices_from_csv`
/ `load_sector_prices_from_yfinance`; NSE sector index tickers are configured under
`regime_detection.geometric_signal.sector_tickers` — verify them before relying on them,
same caveat as the existing `^CRSLDX` note above). A single index price series (what the
rest of this module otherwise runs on) is NOT sufficient — the wedge product is
undefined for one asset. `load_sector_prices_from_yfinance`/`load_from_yfinance` are
hardened against a real bug found on live yfinance: some versions return `(field,
ticker)` MultiIndex columns even for a single-ticker download, which silently broke
per-ticker Series extraction and crashed with a cryptic pandas "must pass an index"
error deep in `pd.DataFrame(frames)`. Both flat and MultiIndex column shapes are now
handled, individual failed tickers are skipped with a warning instead of crashing the
whole batch, and a `ValueError` with the specific failed ticker names is raised if
fewer than 2 tickers succeed (see `tests/test_regime_detection.py`'s
`test_sector_loader_*` / `test_load_from_yfinance_handles_multiindex_index_and_vix`
tests).

**By explicit design, this signal is NOT fed into the GMM/KMeans/HMM clustering** —
this was a deliberate change from an earlier version of this module, where it was one
of several inputs `features.build_feature_matrix` handed to the model. It's now
computed entirely separately, in `regime_detection/pipeline.py::run_pipeline`, strictly
*after* the model has already been fit and predicted (`_attach_geometric_overlay` runs
as the very last step and left-joins the geometric columns onto the finished regime
history) — so it structurally cannot influence `regime`/`regime_name`, not just by
convention. `RegimeModel.feature_names_` (what the model actually saw) never contains
`wedge_volume_*`/`geometric_crash_risk_flag`; see
`tests/test_regime_detection.py::test_geometric_signal_never_reaches_the_clustering_model`
for the regression test pinning this down.

The rationale: the source article's own argument is that thresholding this signal
*directly* — bypassing a probabilistic classifier entirely — is where the (still
unverified on our data) edge is claimed to come from, not from folding it into GMM/HMM
as just another feature the model may or may not weight sensibly.

**What gets produced**: `wedge_volume_{window}d`, its smoothed version, a rolling
self-relative percentile rank (so it's comparable across market eras without a fixed
absolute threshold — a walk-forward-safe departure from the source article's
fixed-sample 15th percentile), and a binary `geometric_crash_risk_flag`, all joined
onto `regime_history.csv` as informational/overlay columns alongside (not part of)
the regime label.

**How it reaches the backtest**: see `docs/backtesting_spec.md`'s "Geometric overlay"
section — `backtesting/strategies.py`'s `build_geometric_overlay_weights` /
`apply_geometric_overlay` turn `geometric_crash_risk_flag` into an exposure multiplier
used two ways: (1) a standalone `geometric_overlay_only` backtest component, isolating
the signal's own effect for direct comparison against `regime_only`, and (2) an
additional exposure cut applied "on top of" the `combined` (fundamentals x regime)
strategy. Both are no-ops (and `geometric_overlay_only` doesn't appear at all) when
the signal is disabled — `combined` behaves identically to before this overlay
existed.

## Hybrid HMM + Wasserstein consensus governor (`consensus_governor`)

Sourced from an external research review of alternative regime-detection
architectures (`Regime_Detection_Module_Research.pdf`, five options compared).
Four of the five (MSAR-TVP/Gibbs, BOCPD+BCT, Kalman+ALS, VAE+SAC-Transformer)
were rejected as out of scope — either built for a different asset class
(pairs trading, multi-asset macro cycles), requiring infrastructure this
project doesn't have (GPU training, MCMC sampling per rebalance), or solving
a different problem than K-state classification (permanent-break detection).
Only **Option 1 (Hybrid Gaussian HMM + Wasserstein clustering)** was judged a
good fit, and only its cheap parts: reweighting the existing `model.type`
posterior by Wasserstein-1 proximity to training-time regime templates, plus
the paper's three-rule "state governor" (entropy gating, persistence,
hysteresis) for deciding when to actually flip the *active* regime.

**Modules**: `wasserstein_proximity.py` (template building + rolling
proximity scoring) and `state_governor.py` (the sequential consensus/gating
state machine). Wired into `pipeline.py::run_pipeline` via
`_attach_consensus_governor`, gated behind `config["consensus_governor"]["enabled"]`
(default `false`).

**By explicit design, this is purely additive, same pattern as the geometric
overlay**: it adds `wasserstein_proximity_*`, `consensus_entropy`,
`proposed_regime`, `active_regime`, `is_transitional`, `candidate_count`
columns to the output of `run_pipeline`, and does not touch `regime`,
`regime_name`, or `p_regime_*` — those remain exactly what `model_type`
(gmm/kmeans/hmm) produced on its own. Downstream strategy code that already
keys off `regime` is completely unaffected until it's deliberately switched
to key off `active_regime` instead.

**Motivation tie-in**: the project's observed under-exposure problem (beta
well below 1.0 in up-trending markets) has one plausible contributing cause
that this doesn't fix by itself but is related to: if the *raw* regime label
flickers near a decision boundary, every flicker can trigger a position-size
cut even when the market never really left a permissive regime. The
persistence + hysteresis rules are aimed at that specific failure mode for
whichever downstream code opts into `active_regime`. This is a hypothesis,
not yet validated on real (non-synthetic) NIFTY500 data — same caveat as
everything else in this module that hasn't been backtested for real yet.

**Look-ahead-bias note**: regime templates are built via
`build_regime_templates(features, labels, ...)` using the exact in-sample
`features`/`labels` pair `run_pipeline` already used to fit `model` — no
extra data is used. If/when `run_pipeline` is called per walk-forward window
(as it should be for real backtest numbers, same caveat as the "Known gaps"
section on point-in-time fundamentals), this in-sample discipline carries
through automatically, since the templates are just whatever was already
passed in for that window.

**Config** (`regime_detection.consensus_governor` in `configs/config.yaml`):
`enabled` (default `false`), `wasserstein_columns` (default: `return_5d`,
`return_21d`, `realized_vol_21d`, `realized_vol_63d` — a distributional
channel, not the full clustering feature set), `wasserstein_window` (21d
rolling window for the empirical distribution), `entropy_limit` (0.85 bits,
per the paper), `persistence_window` (5 bars), `hysteresis_epsilon` (0.05).

**Tests**: `tests/test_hybrid_consensus_governor.py` — template building,
proximity-score correctness (rows sum to 1, own-regime windows score
highest), and each governor rule in isolation (entropy gating forces
"transitional" on ambiguous days, persistence blocks premature switches,
hysteresis blocks narrow-margin switches, candidate counter resets on
flip-flops), plus an end-to-end run against synthetic three-regime data.
Verified manually against a lightweight pytest-compatible shim since pytest
isn't installable in this build sandbox (same constraint as the rest of the
project — see "Known gaps").

**Not yet done**: real-data validation (only synthetic-data tested, same
caveat as the rest of the module), and no downstream code (`backtesting/`)
has been switched over to consume `active_regime` yet — it exists in the
output but nothing reads it. That's a deliberate, separate next step so this
change stays isolated and easy to review/revert on its own.
