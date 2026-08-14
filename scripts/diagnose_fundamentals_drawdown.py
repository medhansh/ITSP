"""Diagnose WHY fundamentals_only's max drawdown has been worse than the
benchmark's, rather than continuing to guess. Checks, in order:

1. Is 'sector' actually populated in your fundamentals snapshot? (the
   sector cap is a silent no-op otherwise, and this is checkable directly
   rather than inferred from log-grepping)
2. Is the max drawdown a SYSTEMIC event (benchmark/regime_only/
   fundamentals_only/combined all trough on the same date) -- if so,
   sector diversification structurally cannot help, since everything
   drops together regardless of sector.
3. What was the ACTUAL sector concentration of the selected portfolio
   at each historical rebalance date -- if it was never very concentrated
   to begin with, the sector cap had nothing to fix.

Reuses cached data -- no re-fetching.

Usage:
    python scripts/diagnose_fundamentals_drawdown.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", default="data/raw/fundamentals_snapshot.csv")
    parser.add_argument("--scores", default=None,
                         help="Optional: a saved scores_by_date CSV (date,symbol,sector,composite_score,...). "
                              "If not given, only checks #1 (sector coverage in the snapshot) and skips #3.")
    parser.add_argument("--stock-prices", default="data/raw/stock_prices.csv")
    parser.add_argument("--benchmark-prices", default="data/raw/benchmark_prices.csv")
    parser.add_argument("--top-quantile", type=float, default=0.2)
    parser.add_argument("--min-positions", type=int, default=5)
    args = parser.parse_args()

    print("=" * 78)
    print("CHECK 1: is 'sector' actually populated in your fundamentals snapshot?")
    print("=" * 78)
    if not Path(args.snapshot).exists():
        print(f"  Snapshot not found at {args.snapshot} -- skipping this check.")
    else:
        snapshot = pd.read_csv(args.snapshot, index_col="symbol")
        if "sector" not in snapshot.columns:
            print("  NO 'sector' COLUMN AT ALL in the snapshot -- the cap has been a total no-op. "
                  "This is your answer: fix sector data ingestion before anything else here matters.")
        else:
            n_total = len(snapshot)
            n_populated = snapshot["sector"].notna().sum()
            n_unique = snapshot["sector"].nunique()
            pct = 100 * n_populated / max(n_total, 1)
            print(f"  {n_populated}/{n_total} symbols ({pct:.1f}%) have a non-null sector, across {n_unique} distinct sectors.")
            if pct < 70:
                print("  -> LOW COVERAGE. A large fraction of symbols have no sector label, so they were treated "
                      "as an '__unknown__' bucket by the cap (or excluded from cap logic entirely, depending on "
                      "how many). This alone could explain why the cap had little effect -- it can't diversify "
                      "away concentration risk in names it has no sector label for.")
            else:
                print("  -> Coverage looks reasonable. Sector data isn't the bottleneck.")
            if n_unique <= 3:
                print(f"  -> Only {n_unique} distinct sector VALUES total -- if your sector taxonomy is this "
                      "coarse, a 30% cap might not be a meaningful constraint (e.g. if it's really just "
                      "'Financials' vs 'Non-Financials', capping at 30% still allows huge within-bucket "
                      "concentration in, say, one specific industry).")

    print("\n" + "=" * 78)
    print("CHECK 2: is the max drawdown a SYSTEMIC (market-wide) event?")
    print("=" * 78)
    if not (Path(args.stock_prices).exists() and Path(args.benchmark_prices).exists()):
        print("  Price data not found -- skipping this check.")
    else:
        benchmark_prices = pd.read_csv(args.benchmark_prices, index_col=0, parse_dates=True)["close"]
        bench_returns = benchmark_prices.pct_change(fill_method=None).dropna()
        bench_dd = compute_drawdown_series(bench_returns)
        trough_date = bench_dd.idxmin()
        print(f"  Benchmark's own worst drawdown trough: {trough_date.date()} (drawdown {bench_dd.min():.1%})")
        print(f"  If your reports/figures/drawdowns.png shows fundamentals_only ALSO troughing at or near")
        print(f"  {trough_date.date()}, this is very likely a systemic/market-wide event (e.g. a broad")
        print(f"  crash) that sector diversification cannot fix -- everything drops together regardless of")
        print(f"  sector composition. If fundamentals_only's trough is at a DIFFERENT date than the")
        print(f"  benchmark's, or is deeper/longer-lasting specifically during periods the benchmark")
        print(f"  recovers faster, that's the concentration/selection-quality signature the cap targets.")

    if args.scores is None:
        print("\n(Skipping CHECK 3 -- pass --scores <path to a saved scores_by_date CSV> to check actual "
              "historical sector concentration of the selected portfolio directly.)")
        return

    print("\n" + "=" * 78)
    print("CHECK 3: how concentrated was the selected portfolio, historically?")
    print("=" * 78)
    if not Path(args.scores).exists():
        print(f"  {args.scores} not found -- skipping.")
        return
    scores = pd.read_csv(args.scores, parse_dates=["date"])
    if "sector" not in scores.columns:
        print("  scores_by_date has no 'sector' column either -- same root cause as CHECK 1.")
        return

    rows = []
    for date, g in scores.groupby("date"):
        g = g.dropna(subset=["composite_score"])
        n_select = max(args.min_positions, int(len(g) * args.top_quantile))
        n_select = min(n_select, len(g))
        if n_select == 0:
            continue
        selected = g.nlargest(n_select, "composite_score")
        sector_counts = selected["sector"].value_counts(normalize=True)
        rows.append({
            "date": date,
            "n_selected": n_select,
            "max_sector_fraction": sector_counts.max() if not sector_counts.empty else np.nan,
            "top_sector": sector_counts.idxmax() if not sector_counts.empty else None,
            "herfindahl": (sector_counts ** 2).sum() if not sector_counts.empty else np.nan,
        })
    concentration = pd.DataFrame(rows).set_index("date")

    print(concentration.describe().to_string())
    print()
    worst = concentration.nlargest(5, "max_sector_fraction")
    print("5 most sector-concentrated rebalance dates (pre-cap selection):")
    print(worst.to_string())

    mean_max_frac = concentration["max_sector_fraction"].mean()
    if mean_max_frac < 0.35:
        print(f"\n  -> Mean max-sector-fraction across all rebalance dates was only {mean_max_frac:.1%} -- "
              "the PRE-CAP selection was rarely very concentrated to begin with, which would explain why "
              "a 30% cap barely moved the drawdown: there wasn't much concentration for it to fix. The "
              "drawdown driver is probably something else -- systemic risk (see CHECK 2), a market-cap/"
              "liquidity tilt, or genuine stock-specific risk in the composite score's picks.")
    else:
        print(f"\n  -> Mean max-sector-fraction was {mean_max_frac:.1%} -- real concentration was present "
              "pre-cap. If the cap still isn't showing up in the backtest's drawdown number, check that "
              "max_sector_weight is actually being read from your config at runtime (CHECK 1's config grep).")

    if not Path(args.stock_prices).exists():
        return

    print("\n" + "=" * 78)
    print("CHECK 4: does fundamentals_only ACTUALLY trough on the same date as the benchmark,")
    print("and does the selected portfolio carry higher realized beta?")
    print("=" * 78)
    from src.backtesting.engine import align_weights_to_returns, compute_returns_panel, run_backtest
    from src.backtesting.strategies import build_fundamental_portfolio_weights

    stock_prices = pd.read_csv(args.stock_prices, index_col=0, parse_dates=True)
    benchmark_prices = pd.read_csv(args.benchmark_prices, index_col=0, parse_dates=True)["close"]
    stock_returns = compute_returns_panel(stock_prices)
    bench_returns = benchmark_prices.pct_change(fill_method=None).dropna()

    weights_sparse = build_fundamental_portfolio_weights(scores, top_quantile=args.top_quantile, min_positions=args.min_positions)
    common_index = stock_returns.index.intersection(bench_returns.index)
    weights_daily = align_weights_to_returns(weights_sparse, common_index, stock_returns.columns)
    result = run_backtest(stock_returns.loc[common_index], weights_daily, transaction_cost_bps=0.0, lag_days=1)

    fund_dd = compute_drawdown_series(result["returns"])
    fund_trough = fund_dd.idxmin()
    bench_dd_aligned = compute_drawdown_series(bench_returns.loc[common_index])
    bench_trough = bench_dd_aligned.idxmin()

    print(f"  benchmark trough:          {bench_trough.date()}  (drawdown {bench_dd_aligned.min():.1%})")
    print(f"  fundamentals_only trough:  {fund_trough.date()}  (drawdown {fund_dd.min():.1%})")
    same_event = abs((fund_trough - bench_trough).days) <= 10
    print(f"  same crash event (within 10 days)? {'YES' if same_event else 'NO'}")

    # realized beta: portfolio returns vs benchmark returns, over the full
    # period AND specifically during a window around the benchmark's own
    # worst drawdown -- a portfolio with a genuine size/volatility tilt
    # will show beta > 1 especially in the crash window, sector
    # composition aside entirely.
    aligned = pd.DataFrame({"portfolio": result["returns"], "benchmark": bench_returns.loc[common_index]}).dropna()
    full_beta = aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var()
    crash_window = aligned.loc[bench_trough - pd.Timedelta(days=30): bench_trough + pd.Timedelta(days=30)]
    crash_beta = crash_window["portfolio"].cov(crash_window["benchmark"]) / crash_window["benchmark"].var() if len(crash_window) > 5 else float("nan")

    print(f"\n  realized beta, full period:              {full_beta:.2f}")
    print(f"  realized beta, +/-30 days around the crash: {crash_beta:.2f}")
    if same_event and (crash_beta > 1.15 or full_beta > 1.1):
        print(
            "\n  -> This IS a systemic event (same trough date as the benchmark), and the selected portfolio "
            "carries meaningfully higher beta than the benchmark -- especially if the crash-window beta is "
            "well above 1. Sector diversification cannot fix this: it's a SIZE/VOLATILITY TILT in the "
            "composite score's picks (e.g. favoring smaller, more cyclical, or more volatile names), not a "
            "concentration problem. Worth checking whether the composite score should penalize high beta/"
            "volatility directly, or whether a market-cap floor is needed in the selection universe."
        )
    elif same_event:
        print(
            "\n  -> Same trough date as the benchmark, but beta isn't dramatically elevated -- the crash hit "
            "the selected portfolio about as hard as the market itself, just with the disadvantage of no "
            "diversification benefit during a systemic event (expected -- there's rarely anywhere to hide "
            "in a true systemic crash regardless of what you hold)."
        )
    else:
        print(
            "\n  -> Different trough dates -- NOT simply a shared systemic event. Worth checking what "
            "happened specifically around the fundamentals_only trough date that didn't also hit the "
            "benchmark as hard."
        )


if __name__ == "__main__":
    main()
