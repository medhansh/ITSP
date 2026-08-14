#!/usr/bin/env python
"""Standalone backtest for the multi-scale SMA-dispersion technical signal
(``src/backtesting/technical_signals.py``) — deliberately independent of the
regime-detection / fundamental-analysis / point-in-time machinery the rest
of this project's backtests need, so you can test this signal on real price
data without first standing up Screener scraping, PIT history, etc.

**Status reminder**: this signal is an exact implementation of a
user-specified construction, not something this project has validated —
see technical_signals.py's module docstring. This script lets you actually
find out whether it does anything, on your own data; it doesn't come with
that answer built in.

Two position-sizing schemes, both available via ``--sizing``:
  - ``threshold``  — hard entry/exit band (the original design). Came in at
    beta ~0.83 against buy-and-hold in initial real-data testing: spending
    real time completely flat waiting for a confirmed threshold crossing
    costs return in an up-trending market.
  - ``continuous`` — conviction-weighted position sizing, no hard gate:
    weight is directly proportional to the signal's magnitude. The direct
    fix attempt for the exposure problem above -- but whether it actually
    raises average exposure/beta is empirically ambiguous, not guaranteed
    (see build_conviction_weighted_signal_weights's docstring). ``--sizing
    all`` (default) runs both side by side specifically so you can see
    which one actually happens on your data, rather than trusting either
    claim.

Usage — against fresh data pulled live from yfinance:
    python scripts/run_technical_backtest.py \\
        --symbols RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK \\
        --start 2018-01-01 \\
        --mode both --sizing all

Usage — against an already-fetched local price CSV (e.g. from
``scripts/fetch_data.py prices``, at data/raw/stock_prices.csv):
    python scripts/run_technical_backtest.py \\
        --price-csv data/raw/stock_prices.csv \\
        --mode trend --sizing continuous --t 10

Requires network access (only if using --symbols, not --price-csv) and
yfinance — same requirement/caveat as the rest of this project's live-data
paths (see docs/data_sourcing_spec.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts._common_cli import add_price_data_args, load_prices_safe
from src.backtesting.engine import compute_returns_panel, run_backtest as run_backtest_custom
from src.backtesting.metrics import performance_summary
from src.backtesting.plotting import plot_drawdowns, plot_equity_curves
from src.backtesting.technical_signals import (
    build_conviction_weighted_signal_weights,
    build_technical_signal_weights,
)
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

MIN_HISTORY_DAYS_NOTE = (
    "Needs roughly (8 * t) + zscore_window trading days of history before the "
    "signal is even defined (default t=10, zscore_window=252 -> ~330 days "
    "~1.3yr) — shorter price histories will show mostly/all-flat positions "
    "and aren't a meaningful test of this signal."
)


def _build_weights(sizing: str, mode: str, price_panel: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if sizing == "threshold":
        return build_technical_signal_weights(
            price_panel, t=args.t, q_entry=args.q_entry, q_exit=args.q_exit,
            mode=mode, zscore_window=args.zscore_window, long_only=not args.allow_short,
        )
    return build_conviction_weighted_signal_weights(
        price_panel, t=args.t, zscore_window=args.zscore_window, mode=mode, long_only=not args.allow_short,
    )


def _run_one_variant(
    sizing: str,
    mode: str,
    price_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.Series]:
    weights = _build_weights(sizing, mode, price_panel, args)
    returns_panel = compute_returns_panel(price_panel)

    if args.engine == "vectorbt":
        from src.backtesting.vbt_engine import run_backtest_with_fallback
        result = run_backtest_with_fallback(
            price_panel, returns_panel, weights, args.transaction_cost_bps, engine="vectorbt", lag_days=args.lag_days
        )
    else:
        result = run_backtest_custom(returns_panel, weights, args.transaction_cost_bps, lag_days=args.lag_days)

    daily_exposure = weights.abs().sum(axis=1)
    n_days_active = (daily_exposure > 0).sum()
    logger.info(
        "[%s/%s] active on %d/%d days (%.1f%%), avg exposure %.3f, mean turnover %.4f",
        sizing, mode, n_days_active, len(weights), 100 * n_days_active / max(len(weights), 1),
        daily_exposure.mean(), result["turnover"].mean(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_price_data_args(parser)

    sig_group = parser.add_argument_group("signal parameters")
    sig_group.add_argument("--t", type=int, default=10, help="Base SMA window; ladder is t/2t/4t/8t. Default 10.")
    sig_group.add_argument("--sizing", choices=["threshold", "continuous", "all"], default="all",
                            help="Position-sizing scheme. 'all' (default) runs both side by side.")
    sig_group.add_argument("--q-entry", type=float, default=0.5, help="Entry threshold in (0,1] -- only used by 'threshold' sizing. Default 0.5.")
    sig_group.add_argument("--q-exit", type=float, default=None, help="Exit threshold, 0<=q_exit<q_entry -- only used by 'threshold' sizing. Default 0.3*q_entry.")
    sig_group.add_argument("--zscore-window", type=int, default=252, help="Rolling z-score lookback in days. Default 252 (~1yr).")
    sig_group.add_argument("--mode", choices=["trend", "mean_reversion", "both"], default="trend")
    sig_group.add_argument("--allow-short", action="store_true", help="Allow negative (short) weights -- NOTE: neither backtest engine models margin/borrow costs, so short P&L here is optimistic. Off by default.")

    bt_group = parser.add_argument_group("backtest parameters")
    bt_group.add_argument("--transaction-cost-bps", type=float, default=10.0)
    bt_group.add_argument("--engine", choices=["vectorbt", "custom"], default="vectorbt")
    bt_group.add_argument("--lag-days", type=int, default=1, help="Trading days between a weight being decided and earning returns. Default 1 (no-look-ahead). See engine.run_backtest's docstring before changing.")
    bt_group.add_argument("--out-dir", default="reports/technical_signal")

    args = parser.parse_args()

    print(MIN_HISTORY_DAYS_NOTE)
    price_panel, benchmark = load_prices_safe(args)
    price_panel = price_panel.dropna(axis=1, how="all")
    logger.info("Price panel: %d symbols x %d days (%s to %s)",
                price_panel.shape[1], price_panel.shape[0],
                price_panel.index.min().date(), price_panel.index.max().date())

    min_required = 8 * args.t + args.zscore_window
    if len(price_panel) < min_required:
        logger.warning(
            "Only %d days of price history -- need ~%d for the signal to be fully defined "
            "(see the note above). Results will mostly show a flat/inactive signal.",
            len(price_panel), min_required,
        )

    sizings = ["threshold", "continuous"] if args.sizing == "all" else [args.sizing]
    modes = ["trend", "mean_reversion"] if args.mode == "both" else [args.mode]

    results: dict[str, dict[str, pd.Series]] = {}
    exposure_by_variant: dict[str, float] = {}
    for sizing in sizings:
        for mode in modes:
            name = f"{sizing}_{mode}"
            weights = _build_weights(sizing, mode, price_panel, args)
            exposure_by_variant[name] = weights.abs().sum(axis=1).mean()
            results[name] = _run_one_variant(sizing, mode, price_panel, args)

    if benchmark is not None:
        bench_returns = benchmark.pct_change(fill_method=None).reindex(price_panel.index).dropna()
        results["benchmark_buy_hold"] = {"returns": bench_returns}
    equal_weight_bh = compute_returns_panel(price_panel).mean(axis=1)
    results["equal_weight_buy_hold"] = {"returns": equal_weight_bh}

    print("\nPerformance summary:")
    summary_rows = {}
    bench_for_alpha = results.get("benchmark_buy_hold", results["equal_weight_buy_hold"])["returns"]
    for name, result in results.items():
        summary_rows[name] = performance_summary(result["returns"], benchmark_returns=bench_for_alpha)
    summary_df = pd.DataFrame(summary_rows).T
    for name, avg_exposure in exposure_by_variant.items():
        summary_df.loc[name, "avg_exposure"] = avg_exposure
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 180):
        print(summary_df)

    if len(sizings) == 2:
        print("\nContinuous vs threshold sizing, same mode:")
        for mode in modes:
            t_name, c_name = f"threshold_{mode}", f"continuous_{mode}"
            if t_name in summary_df.index and c_name in summary_df.index:
                d_cagr = summary_df.loc[c_name, "cagr"] - summary_df.loc[t_name, "cagr"]
                d_beta = summary_df.loc[c_name, "beta"] - summary_df.loc[t_name, "beta"]
                d_exp = exposure_by_variant[c_name] - exposure_by_variant[t_name]
                print(f"  [{mode}] continuous - threshold: CAGR {d_cagr:+.4f}, beta {d_beta:+.4f}, avg_exposure {d_exp:+.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "technical_signal_summary.csv")

    returns_dict = {name: result["returns"] for name, result in results.items()}
    plot_equity_curves(returns_dict, str(out_dir / "equity_curves.png"), title="Technical Signal — Equity Curves")
    plot_drawdowns(returns_dict, str(out_dir / "drawdowns.png"), title="Technical Signal — Drawdowns")

    print(f"\nSaved summary + charts to {out_dir}/")


if __name__ == "__main__":
    main()
