"""Run paper trading and write the dashboard.

  --init      seed a new ledger by replaying a historical window
  (default)   append the latest trading day to an existing ledger

Both modes share the same engine, so the daily path is the one that was
exercised during seeding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.io_utils import load_config
from src.common.logging_utils import get_logger
from src.fundamental_analysis.orthogonalization import compute_rolling_beta_panel
from src.fundamental_analysis.point_in_time import run_pit_fundamental_pipeline
from src.paper_trading.broker import Ledger, SimulatedBroker
from src.paper_trading.dashboard import render_dashboard
from src.paper_trading.engine import build_conviction_panel, step_day

logger = get_logger(__name__)


def load_inputs(cfg):
    sp = pd.read_csv("data/raw/stock_prices.csv", index_col=0, parse_dates=True)
    bm = pd.read_csv("data/raw/benchmark_prices.csv", index_col=0, parse_dates=True)["close"]
    snap = pd.read_csv("data/raw/fundamentals_snapshot.csv", index_col=0)
    qh = pd.read_csv("data/raw/fundamentals_quarterly_history.csv")
    # known_date/period_end must be datetimes: point_in_time replays with
    # merge_asof, which refuses to merge a datetime key against strings.
    for col in ("known_date", "period_end"):
        if col in qh.columns:
            qh[col] = pd.to_datetime(qh[col], errors="coerce")
    # data/processed/regime_history.csv is written by
    # scripts/run_full_pipeline.py's step_production_regime -- whichever
    # source regime_detection.production_regime_source selects there (VIX-
    # bucket by default, GMM as an explicit fallback) is what shows up
    # here. Paper trading intentionally has no regime-construction logic of
    # its own; it always mirrors whatever the last full pipeline run
    # validated, by reading this one shared file. That means: re-run
    # `python scripts/run_full_pipeline.py` (which re-fetches VIX and
    # rebuilds this file) before a live paper-trading session that needs
    # today's regime reflected -- an existing but easy-to-miss requirement,
    # not something --init/daily append handles on its own.
    rh = Path("data/processed/regime_history.csv")
    regime = (pd.read_csv(rh, index_col=0, parse_dates=True)["regime"] if rh.exists() else None)
    return sp, bm, snap, qh, regime


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--ledger", default="data/processed/paper_ledger.json")
    p.add_argument("--init", action="store_true", help="seed a new ledger (overwrites)")
    p.add_argument("--start", default=None, help="replay start date for --init, e.g. 2024-01-01")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--out", default="reports/paper_trading.html")
    args = p.parse_args()

    cfg = load_config(args.config)
    sp, bm, snap, qh, regime = load_inputs(cfg)
    broker = SimulatedBroker(slippage_bps=args.slippage_bps)
    freq = cfg["fundamental_analysis"].get("point_in_time", {}).get("rebalance_frequency", "MS")

    if args.init:
        start = pd.Timestamp(args.start) if args.start else sp.index[-252]
        ledger = Ledger.new(args.capital)
    else:
        if not Path(args.ledger).exists():
            print(f"No ledger at {args.ledger}. Run with --init first.")
            sys.exit(1)
        ledger = Ledger.load(args.ledger)
        last = pd.Timestamp(ledger.nav_history[-1]["date"]) if ledger.nav_history else sp.index[0]
        start = last + pd.Timedelta(days=1)

    days = sp.index[(sp.index >= start)]
    if len(days) == 0:
        print("No new trading days.")
        render_dashboard(ledger, sp, bm, args.out)
        return

    conviction = build_conviction_panel(cfg, sp, bm, regime)
    beta_panel = None
    if cfg["backtesting"].get("beta_rotation", {}).get("rotation_strength", 0.0):
        bw = cfg["fundamental_analysis"].get(
            "technical_momentum_beta_orthogonalization", {}).get("beta_window", 252)
        beta_panel = compute_rolling_beta_panel(
            {s: sp[s].dropna() for s in sp.columns}, bm, window=bw)

    rebal = pd.date_range(days[0], days[-1], freq=freq)
    rebal_days = {sp.index[sp.index.searchsorted(d)] for d in rebal
                  if sp.index.searchsorted(d) < len(sp.index)}

    # A window that does not span a period boundary yields NO rebalance dates
    # -- e.g. any run starting mid-month under a monthly schedule. Without
    # this, the portfolio never buys anything and sits in cash reporting a
    # flat 0.00%, which looks like a working run rather than a broken one.
    # If the book is empty there is nothing to preserve, so establish it on
    # the first available day.
    if not ledger.positions and days[0] not in rebal_days:
        rebal_days.add(days[0])
        logger.info(
            "[paper] no scheduled rebalance in this window and the portfolio is empty -- "
            "opening positions on %s", days[0].date(),
        )
    scores = run_pit_fundamental_pipeline(
        cfg["fundamental_analysis"], snap, qh,
        pd.DatetimeIndex(sorted(rebal_days)), conviction_panel=conviction,
    )

    for i, day in enumerate(days):
        # Execute at the NEXT day's price where one exists, never at the price
        # the decision was made on.
        pos = sp.index.get_loc(day)
        exec_row = sp.iloc[pos + 1] if pos + 1 < len(sp.index) else sp.loc[day]
        step_day(ledger, day, sp.loc[day], day in rebal_days, broker,
                 cfg=cfg, scores=scores, beta_panel=beta_panel, regime=regime,
                 exec_prices_row=exec_row)

    ledger.save(args.ledger)
    out = render_dashboard(ledger, sp, bm, args.out)
    nav = ledger.nav_history[-1]["nav"]
    print(f"{len(days)} day(s) | NAV {nav:,.0f} | {len(ledger.positions)} positions | {out}")
    if not ledger.positions:
        print("WARNING: no open positions. Check that scores were produced for this window "
              "(fundamentals snapshot present, and the window reaches a rebalance date).")


if __name__ == "__main__":
    main()
