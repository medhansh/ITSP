#!/usr/bin/env python
"""Standalone backtest for the adaptive-period Ichimoku signal
(``src/backtesting/adaptive_ichimoku.py``) — full Ichimoku (true OHLC,
forward-shifted cloud, Chikou Span confirmation), runs the static
(non-adaptive) baseline alongside both competing adaptive-direction
hypotheses (``shrink_when_high`` / ``shrink_when_low``) side by side, AND a
``no_ichimoku`` control (statically 100% invested, equal-weighted, run
through the exact same backtest engine/transaction-costs/lag as every
Ichimoku variant — not a shortcut raw-mean-of-returns baseline), so you can
see directly whether using Ichimoku at all helps, and if so, which variant.
``equal_weight_buy_hold``/``benchmark_buy_hold`` are also included as
secondary reference points, but ``no_ichimoku`` is the one that's an
apples-to-apples comparison against the Ichimoku variants specifically.

**Status reminder**: experimental and unvalidated — see
adaptive_ichimoku.py's module docstring. This script is what actually
answers the "does this help" question; it doesn't come with the answer
built in.

Usage — against fresh OHLC data pulled live from yfinance:
    python scripts/run_adaptive_ichimoku_backtest.py \\
        --symbols RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK \\
        --start 2018-01-01

Usage — against a local LONG-format OHLC CSV (columns:
date,symbol,open,high,low,close[,volume] — NOT the wide close-only format
run_technical_backtest.py uses, since OHLC for many symbols doesn't fit one
wide table the way a single close value per cell does):
    python scripts/run_adaptive_ichimoku_backtest.py \\
        --price-csv data/raw/stock_prices_ohlc.csv --variant all

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
import numpy as np

from scripts._common_cli import add_ohlc_price_data_args, load_prices_ohlc_safe
from src.backtesting.adaptive_ichimoku import DEFAULT_BASE_KIJUN, DEFAULT_BASE_SENKOU_B, build_ichimoku_weights
from src.backtesting.engine import compute_returns_panel, run_backtest as run_backtest_custom
from src.backtesting.metrics import performance_summary
from src.backtesting.plotting import plot_drawdowns, plot_equity_curves
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

VARIANT_CHOICES = ["static", "shrink_when_high", "shrink_when_low", "all"]

MIN_HISTORY_DAYS_NOTE = (
    "The adaptive variants also need the dispersion score's z-score lookback "
    "(default 252 days) warmed up on top of the base Ichimoku periods "
    "(default senkou_b=52) -- roughly 8*t + zscore_window + senkou_b days "
    "before results are meaningful (default t=10 -> ~380 days ~1.5yr)."
)


def _close_panel_from_ohlc(price_panel_ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide close-only panel (for the buy-and-hold comparison / returns
    computation), built from the per-symbol OHLC dict — aligned on the
    union of all symbols' dates."""
    closes = {symbol: ohlc["close"] for symbol, ohlc in price_panel_ohlc.items() if "close" in ohlc.columns}
    return pd.DataFrame(closes)


def _run_one_variant(
    variant: str,
    price_panel_ohlc: dict[str, pd.DataFrame],
    close_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.Series]:
    weights = build_ichimoku_weights(
        price_panel_ohlc, t=args.t, zscore_window=args.zscore_window, variant=variant,
        scale_min=args.scale_min, scale_max=args.scale_max,
        base_tenkan=args.base_tenkan, base_kijun=args.base_kijun, base_senkou_b=args.base_senkou_b,
        long_only=not args.allow_short, signal_mode=args.signal_mode,
    )
    weights = weights.reindex(columns=close_panel.columns, fill_value=0.0)

    if args.engine == "vectorbt":
        from src.backtesting.vbt_engine import run_backtest_with_fallback
        result = run_backtest_with_fallback(
            close_panel, returns_panel, weights, args.transaction_cost_bps, engine="vectorbt", lag_days=args.lag_days
        )
    else:
        result = run_backtest_custom(returns_panel, weights, args.transaction_cost_bps, lag_days=args.lag_days)

    n_days_active = (weights.abs().sum(axis=1) > 0).sum()
    logger.info(
        "[%s] active on %d/%d days (%.1f%%), mean turnover %.4f",
        variant, n_days_active, len(weights), 100 * n_days_active / max(len(weights), 1),
        result["turnover"].mean(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_ohlc_price_data_args(parser)

    sig_group = parser.add_argument_group("signal parameters")
    sig_group.add_argument("--variant", choices=VARIANT_CHOICES, default="all",
                            help="Which variant(s) to run. 'all' (default) runs static + both adaptive "
                                 "directions side by side -- the actual comparison this script exists for.")
    sig_group.add_argument("--t", type=int, default=10, help="Base SMA window for the dispersion score driving adaptivity. Default 10.")
    sig_group.add_argument("--zscore-window", type=int, default=252, help="Dispersion score z-score lookback. Default 252 (~1yr).")
    sig_group.add_argument("--scale-min", type=float, default=0.5, help="Minimum period scale factor. Default 0.5.")
    sig_group.add_argument("--scale-max", type=float, default=1.5, help="Maximum period scale factor. Default 1.5.")
    sig_group.add_argument("--base-tenkan", type=int, default=9)
    sig_group.add_argument("--base-kijun", type=int, default=DEFAULT_BASE_KIJUN)
    sig_group.add_argument("--base-senkou-b", type=int, default=DEFAULT_BASE_SENKOU_B)
    sig_group.add_argument("--allow-short", action="store_true", help="Allow negative (short) weights -- NOTE: neither backtest engine models margin/borrow costs, so short P&L here is optimistic. Off by default.")
    sig_group.add_argument("--signal-mode", choices=["triple_confirmation", "conviction_score"], default="conviction_score",
                            help="triple_confirmation = traditional binary AND-of-3 gate (equal-weighted among confirmed names). "
                                 "conviction_score (default) = continuous tanh-bounded score, conviction-weighted, added specifically "
                                 "to raise the low day-to-day hit rate the strict AND produces -- see adaptive_ichimoku.py.")

    bt_group = parser.add_argument_group("backtest parameters")
    bt_group.add_argument("--transaction-cost-bps", type=float, default=10.0)
    bt_group.add_argument("--engine", choices=["vectorbt", "custom"], default="vectorbt")
    bt_group.add_argument("--lag-days", type=int, default=1, help="Trading days between a weight being decided and earning returns. Default 1 (no-look-ahead). See engine.run_backtest's docstring before changing.")
    bt_group.add_argument("--out-dir", default="reports/adaptive_ichimoku")

    args = parser.parse_args()

    print(MIN_HISTORY_DAYS_NOTE)
    price_panel_ohlc, benchmark = load_prices_ohlc_safe(args)
    logger.info(
        "OHLC panel: %d symbols (%s to %s)",
        len(price_panel_ohlc),
        min(ohlc.index.min() for ohlc in price_panel_ohlc.values()).date(),
        max(ohlc.index.max() for ohlc in price_panel_ohlc.values()).date(),
    )

    close_panel = _close_panel_from_ohlc(price_panel_ohlc)
    returns_panel = compute_returns_panel(close_panel)

    min_required = 8 * args.t + args.zscore_window + args.base_senkou_b
    if len(close_panel) < min_required:
        logger.warning(
            "Only %d days of price history -- need ~%d for the adaptive signal to be fully "
            "defined (see the note above). Results will mostly show a flat/inactive signal.",
            len(close_panel), min_required,
        )

    variants = ["static", "shrink_when_high", "shrink_when_low"] if args.variant == "all" else [args.variant]
    results: dict[str, dict[str, pd.Series]] = {}
    for variant in variants:
        results[variant] = _run_one_variant(variant, price_panel_ohlc, close_panel, returns_panel, args)

    # --- no_ichimoku: the actual "what if we don't use Ichimoku at all"
    # control. Equal-weighted across whichever symbols have valid data on
    # a given day, run through the EXACT SAME _run_backtest dispatch (same
    # transaction-cost-bps, same engine, same lag_days) as every Ichimoku
    # variant above -- NOT just a raw mean-of-returns shortcut. This
    # matters because "equal_weight_buy_hold" below skips transaction
    # costs/lag entirely, which isn't a fair comparison against variants
    # that pay costs on every entry/exit; a strategy that trades in and out
    # should be judged against a baseline priced through the same engine,
    # not one that gets a free pass on costs it never has the chance to
    # pay.
    #
    # IMPORTANT: weights are a FIXED 1/N_total only in appearance -- if
    # built that way literally, any symbol with a data gap that day (early
    # listing, missing OHLC fetch, etc.) gets zero-filled by the engine's
    # NaN handling (engine.run_backtest: `r.fillna(0.0)`), which silently
    # DILUTES this control's realized return/volatility relative to
    # equal_weight_buy_hold's `returns_panel.mean(axis=1)` (pandas' default
    # skipna=True excludes missing symbols from both sum and count, i.e.
    # naturally re-normalizes across only the live names). A first version
    # of this fix used a truly-fixed 1/N matrix and this diluted no_ichimoku
    # to a materially lower beta/vol than equal_weight_buy_hold even though
    # they're supposed to be nearly the same portfolio -- confirmed via a
    # small synthetic reproduction showing the exact same ~0.79 vol ratio
    # this bug produced on the real run. Fixed by re-deriving equal weight
    # PER DAY only among symbols with a valid (non-NaN) return that day, so
    # a missing symbol's "share" is redistributed across the live ones
    # instead of silently becoming dead weight -- this is what
    # equal_weight_buy_hold already does implicitly via skipna.
    valid_mask = returns_panel.notna()
    n_valid_per_day = valid_mask.sum(axis=1).replace(0, np.nan)
    static_full_weights = valid_mask.div(n_valid_per_day, axis=0).fillna(0.0)
    if args.engine == "vectorbt":
        from src.backtesting.vbt_engine import run_backtest_with_fallback
        results["no_ichimoku"] = run_backtest_with_fallback(
            close_panel, returns_panel, static_full_weights, args.transaction_cost_bps,
            engine="vectorbt", lag_days=args.lag_days,
        )
    else:
        results["no_ichimoku"] = run_backtest_custom(
            returns_panel, static_full_weights, args.transaction_cost_bps, lag_days=args.lag_days
        )

    if benchmark is not None:
        bench_returns = benchmark.pct_change(fill_method=None).reindex(close_panel.index).dropna()
        results["benchmark_buy_hold"] = {"returns": bench_returns}
    equal_weight_bh = returns_panel.mean(axis=1)
    results["equal_weight_buy_hold"] = {"returns": equal_weight_bh}

    print("\nPerformance summary:")
    summary_rows = {}
    bench_for_alpha = results.get("benchmark_buy_hold", results["equal_weight_buy_hold"])["returns"]
    for name, result in results.items():
        summary_rows[name] = performance_summary(result["returns"], benchmark_returns=bench_for_alpha)
    summary_df = pd.DataFrame(summary_rows).T
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(summary_df)

    if args.variant == "all":
        static_cagr = summary_df.loc["static", "cagr"] if "static" in summary_df.index else None
        if static_cagr is not None:
            for adaptive_name in ("shrink_when_high", "shrink_when_low"):
                if adaptive_name in summary_df.index:
                    delta = summary_df.loc[adaptive_name, "cagr"] - static_cagr
                    print(f"  {adaptive_name} vs static baseline: {delta:+.4f} CAGR")

    no_ichimoku_cagr = summary_df.loc["no_ichimoku", "cagr"]
    print(f"\nNo-Ichimoku-at-all baseline (static full exposure, same engine/costs/lag): {no_ichimoku_cagr:+.4f} CAGR")
    for variant in variants:
        if variant in summary_df.index:
            delta = summary_df.loc[variant, "cagr"] - no_ichimoku_cagr
            print(f"  {variant} vs no_ichimoku: {delta:+.4f} CAGR")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "adaptive_ichimoku_summary.csv")

    returns_dict = {name: result["returns"] for name, result in results.items()}
    plot_equity_curves(returns_dict, str(out_dir / "equity_curves.png"), title="Adaptive Ichimoku — Equity Curves")
    plot_drawdowns(returns_dict, str(out_dir / "drawdowns.png"), title="Adaptive Ichimoku — Drawdowns")

    print(f"\nSaved summary + charts to {out_dir}/")


if __name__ == "__main__":
    main()
