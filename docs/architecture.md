# System architecture

## Milestone alignment

- **May — data infrastructure & literature review**: `data_fetchers/` interfaces,
  `data/universe/nifty500_list.csv` schema, and this scaffold itself.
- **June — develop & validate three ML modules independently**: `src/regime_detection/`
  and `src/fundamental_analysis/` are built this way now — each has its own data
  loading, its own tests, and no cross-module imports. `src/sentiment_analysis/` is
  stubbed for a later pass.
- **July — integration, backtesting, paper trading, dashboard**: backtesting
  (`src/backtesting/`) is now implemented, ahead of the June modules being fully
  validated on real data — it's the piece that actually combines regime detection and
  fundamental analysis (see below), so building it early was necessary to get any
  read on whether the combined system does anything useful, even on synthetic data.
  Paper trading and a live dashboard are still not started.

## Module boundaries

```
data_fetchers  →  features/metrics  →  models/scoring  →  pipeline  →  scripts (CLI)
```

Each of the first two modules follows this same shape so they're structurally consistent:

- **regime_detection**: `data_loader.py` → `features.py` → `models.py` → `pipeline.py`
- **fundamental_analysis**: `data_fetchers/` → `metrics/*.py` → `scoring/composite_score.py` → `pipeline.py`

Nothing in `features.py`/`metrics/` or `models.py`/`scoring/` imports from
`data_fetchers`/`data_loader` directly — they only consume pandas DataFrames with a
documented schema. That's deliberate: it means data sources can change (NSE archives
today, a paid vendor tomorrow) without touching analysis code, and it's what makes the
synthetic-data tests in `tests/` possible without any live data connection.

**backtesting** sits downstream of both and is the actual integration point, shaped
slightly differently since it *consumes* two modules rather than one:
`engine.py` (generic portfolio simulation) + `strategies.py` (turns regime labels and
fundamental scores into weight matrices) → `metrics.py` + `attribution.py`
(performance + per-signal decomposition) → `plotting.py` + `reporting.py` → `pipeline.py`.

## How the modules combine today

Regime detection outputs a per-day regime label/probabilities for the *market as a
whole*. Fundamental analysis outputs a per-stock composite score *at a point in time*.
`src/backtesting/strategies.py::combine_regime_and_fundamentals` is the current
integration logic: it takes the fundamentals-selected stock portfolio and scales its
total exposure (not its composition) by the regime-implied exposure fraction from
`configs/config.yaml`'s `backtesting.exposure_by_regime`. This is intentionally the
simplest possible combination — multiplicative exposure scaling, not
regime-conditional re-weighting of the fundamental dimensions themselves (e.g.
up-weighting leverage/solvency in stress regimes and growth in calm ones). That
richer version is a natural next step once the simple version's attribution results
(see `docs/backtesting_spec.md`) show whether it's worth the added complexity.

## Why unsupervised learning for regime detection

The abstract specifies unsupervised learning deliberately: market regimes aren't
labeled in nature, and hand-labeling risks look-ahead bias (you tend to "see" a
regime shift only once you know how the period ended). GMM/KMeans/HMM all discover
regimes purely from the statistical structure of market features, avoiding that trap.
HMM additionally models transition persistence, which matters because regimes are
sticky — see `docs/regime_detection_spec.md` for the tradeoffs between the three.

## Testing philosophy

Every module ships with synthetic-data tests (`tests/`) that don't require network
access or a real data vendor. This was a hard requirement in the sandbox this scaffold
was built in (no outbound access to NSE/exchange/data-vendor endpoints — verified
directly), but it's good practice regardless: it means CI can validate the modeling
logic itself, independent of data-pipeline flakiness.
