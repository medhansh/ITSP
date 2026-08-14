# ITSP — Indian Trading Strategy Platform

Systematic long-only equity strategy for the NIFTY500, with paper trading.

## Strategy

Monthly rebalance. Stocks ranked by an 9-dimension composite score, top
quantile held equal-weighted, then two overlays:

| Component | Setting | Evidence |
|---|---|---|
| Conviction signal | 12-1 momentum blended with mean reversion, stress-weighted | Sharpe +0.156 in 6/6 walk-forward folds vs the previous Ichimoku signal; blending adds +0.053 in 5/6 |
| Beta rotation | `rotation_strength: 3.0` | Drawdown better in 5/6 folds; first mechanism to give the selection a better drawdown than the index |
| Regime exposure | GMM, 4 states | Better Sharpe, under half the index drawdown |
| Selection | `top_quantile: 0.2` | U-shaped curve; 0.2 retained pending further work |

Walk-forward (6 rolling folds, 17bps one-way): **CAGR 30.3%, Sharpe 1.96**.
Those are the numbers to quote. `run_full_pipeline.py` reports a single
full-history in-sample backtest, which will read higher.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/fetch_data.py            # prices + fundamentals (needs network)
python scripts/run_full_pipeline.py     # backtest, charts, report -> reports/
python scripts/run_paper_trading.py --init --start 2025-01-01
python scripts/run_paper_trading.py     # daily: appends the latest session
open reports/paper_trading.html
```

## Paper trading

`SimulatedBroker` fills at the **next** day's price, never the price the
decision used, and charges slippage, STT, exchange, SEBI, stamp duty and
GST. State persists to `data/processed/paper_ledger.json`, so runs resume.

It validates: forward-data performance, whether point-in-time fundamentals
are published when the backtest assumes, and realised turnover.

It does not validate: real fill prices, market impact, or liquidity in
smaller names. Those need a live broker.

`KiteBroker` is a stub. Zerodha Kite Connect costs ₹2,000/month and has no
free sandbox. Implementing that one class and setting
`paper_trading.broker: kite` is the only change needed — no strategy code
moves.

## Layout

```
configs/config.yaml     every tunable, with the evidence for each in comments
src/fundamental_analysis 9 scoring dimensions, point-in-time replay
src/regime_detection     GMM/KMeans/HMM clustering on price features
src/backtesting          engine, attribution, momentum-reversal blend, charts
src/paper_trading        broker, ledger, engine, dashboard
scripts/                 fetch_data, run_full_pipeline, run_paper_trading
```

## Notes carried forward

- Regime states are quantile slices of a volatility **continuum**, not
  separable states: DBSCAN finds one cluster at every eps, silhouette
  prefers n=2 at every backend. `n_regimes: 4` is the best-performing
  discretisation, not a discovered state count.
- The composite score carries return information but essentially no risk
  information. Four attempts to extract risk management from it failed;
  beta rotation worked by using trailing beta instead.
- Confirmed negatives are documented inline in `configs/config.yaml` next
  to the flags that control them.

## Deployment

See **SETUP.md** for step-by-step GitHub Desktop, Pages, Actions and
external-trigger setup.

Two workflows ship in `.github/workflows/`:

- `paper-trading.yml` — triggered by an external HTTP POST to the
  `repository_dispatch` API (preferred), with GitHub cron as a fallback.
  Fetches prices server-side, advances the ledger, commits it back, and
  publishes the dashboard to Pages.
- `keepalive.yml` — weekly no-op commit. GitHub disables scheduled workflows
  after 60 days of repository inactivity, silently. Redundant once the
  external trigger is running, since each run commits the ledger.

Trigger request:

```
POST https://api.github.com/repos/<owner>/<repo>/dispatches
Authorization: Bearer <fine-grained PAT with Contents: read and write>
Accept: application/vnd.github+json
X-GitHub-Api-Version: 2022-11-28

{"event_type": "run-paper-trading"}
```

Returns HTTP 204 with an empty body on success.
