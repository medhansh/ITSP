"""Pull each walk-forward fold's TEST-period regime composition and
benchmark return alongside its selected-vs-baseline edge, to check whether
the ``technical_momentum`` weight's out-of-sample benefit is a stable,
regime-independent effect or concentrated in specific market conditions
(e.g. strongly trending years) -- the follow-up to
``walk_forward_technical_momentum.py``'s per-fold edge sizes being wildly
uneven (a couple of huge wins, a couple of small wins, one loss).

Reuses ``data/processed/regime_history.csv`` (from a prior
``run_full_pipeline.py`` run) and ``data/raw/benchmark_prices.csv`` --
no re-fetching, no re-running any model.

Usage:
    python scripts/analyze_walk_forward_regimes.py
    python scripts/analyze_walk_forward_regimes.py --results reports/technical_momentum_walk_forward/walk_forward_results.csv
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


def regime_composition_for_window(
    regime_history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, regime_col: str,
) -> dict[str, float]:
    """Fraction of trading days in each regime during ``[start, end]``.
    Returns e.g. ``{"regime_frac_Bullish_LowVol": 0.62, ...}``."""
    window = regime_history.loc[start:end, regime_col]
    if window.empty:
        return {}
    fractions = window.value_counts(normalize=True)
    return {f"regime_frac_{name}": frac for name, frac in fractions.items()}


def benchmark_return_for_window(benchmark_prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Simple total return of the benchmark over the window -- the most
    direct available proxy for 'was this a trending-up, trending-down, or
    flat/choppy period', independent of whatever the regime model itself
    concluded."""
    window = benchmark_prices.loc[start:end]
    if len(window) < 2:
        return None
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="reports/technical_momentum_walk_forward/walk_forward_results.csv",
                         help="Path to walk_forward_technical_momentum.py's output CSV.")
    parser.add_argument("--regime-history", default="data/processed/regime_history.csv")
    parser.add_argument("--benchmark-prices", default="data/raw/benchmark_prices.csv")
    parser.add_argument("--regime-col", default="regime_name",
                         help="Which regime_history.csv column to break down by. Default regime_name "
                              "(falls back to 'regime' if regime_name isn't present).")
    parser.add_argument("--out", default="reports/technical_momentum_walk_forward/regime_analysis.csv")
    args = parser.parse_args()

    for label, path in (("walk-forward results", args.results), ("regime history", args.regime_history),
                         ("benchmark prices", args.benchmark_prices)):
        if not Path(path).exists():
            print(f"ERROR: {label} file not found at {path}")
            print("Run walk_forward_technical_momentum.py first (for the results CSV) "
                  "and run_full_pipeline.py at least once (for regime history / benchmark prices).")
            sys.exit(1)

    results = pd.read_csv(args.results, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    regime_history = pd.read_csv(args.regime_history, index_col=0, parse_dates=True)
    benchmark_prices = pd.read_csv(args.benchmark_prices, index_col=0, parse_dates=True)["close"]

    regime_col = args.regime_col
    if regime_col not in regime_history.columns:
        logger.warning("%s not found in %s -- falling back to 'regime'", regime_col, args.regime_history)
        regime_col = "regime"
        if regime_col not in regime_history.columns:
            print(f"ERROR: neither '{args.regime_col}' nor 'regime' found in {args.regime_history}. "
                  f"Available columns: {list(regime_history.columns)}")
            sys.exit(1)

    # Figure out which "selected minus baseline" edge column(s) are present
    # -- walk_forward_technical_momentum.py names these per target component
    # (fundamentals_only and/or combined), so support either/both.
    edge_cols = {}
    for component in ("fundamentals_only", "combined"):
        sel_col, base_col = f"selected_test_{component}_cagr", f"baseline_test_{component}_cagr"
        if sel_col in results.columns and base_col in results.columns:
            edge_col = f"{component}_edge"
            results[edge_col] = results[sel_col] - results[base_col]
            edge_cols[component] = edge_col

    if not edge_cols:
        print(f"ERROR: no selected/baseline CAGR columns found in {args.results}. Columns present: {list(results.columns)}")
        sys.exit(1)

    rows = []
    for _, fold in results.iterrows():
        row = {
            "fold": int(fold["fold"]),
            "test_start": fold["test_start"].date(),
            "test_end": fold["test_end"].date(),
            "selected_weight": fold["selected_weight"],
        }
        for component, edge_col in edge_cols.items():
            row[edge_col] = fold[edge_col]
        row["benchmark_return"] = benchmark_return_for_window(benchmark_prices, fold["test_start"], fold["test_end"])
        row.update(regime_composition_for_window(regime_history, fold["test_start"], fold["test_end"], regime_col))
        rows.append(row)

    table = pd.DataFrame(rows).set_index("fold")
    # Sort regime fraction columns by how much they vary across folds (most
    # informative first) rather than alphabetically, so a quick glance at
    # the leftmost columns already shows what's moving.
    regime_frac_cols = sorted(
        [c for c in table.columns if c.startswith("regime_frac_")],
        key=lambda c: -table[c].std(skipna=True),
    )
    ordered_cols = ["test_start", "test_end", "selected_weight", *edge_cols.values(), "benchmark_return", *regime_frac_cols]
    table = table[ordered_cols].fillna(0.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)

    print("=" * 100)
    print(f"WALK-FORWARD EDGE vs. TEST-PERIOD MARKET CONDITIONS  (regime column: {regime_col})")
    print("=" * 100)
    pd.set_option("display.width", 160)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 100)
    print("CORRELATION CHECK  (n={} folds -- treat as directional, not statistically conclusive)".format(len(table)))
    print("-" * 100)
    for component, edge_col in edge_cols.items():
        edge = table[edge_col]
        print(f"\n{component} edge vs.:")
        bench_corr = edge.corr(table["benchmark_return"])
        print(f"  benchmark return in test period:  corr = {bench_corr:+.3f}"
              f"{'  (edge tends to be BIGGER in stronger up years)' if bench_corr > 0.3 else ''}"
              f"{'  (edge tends to be BIGGER in weaker/down years -- NOT simply riding a rising benchmark)' if bench_corr < -0.3 else ''}")
        for col in regime_frac_cols:
            c = edge.corr(table[col])
            if abs(c) > 0.3:
                direction = "more time in this regime -> BIGGER edge" if c > 0 else "more time in this regime -> SMALLER edge"
                print(f"  {col}: corr = {c:+.3f}  ({direction})")

    print("\n" + "-" * 100)
    print("READ THIS AS: a strong positive correlation with benchmark_return means the edge is largely riding a")
    print("rising market (a trend-following signal doing what trend-following signals do) rather than adding a")
    print("stable, condition-independent improvement -- worth knowing before committing to this weight for periods")
    print("that might look nothing like the big-win folds above. With only a handful of folds, don't over-read a")
    print("single strong-looking correlation, but a CONSISTENT pattern (e.g. every big win in an up year, the one")
    print("loss in a down/choppy year) is worth taking seriously even at small n.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
