#!/usr/bin/env python
"""CLI: run the full backtest + attribution + report pipeline.

Usage:
    python scripts/run_backtest.py \\
        --stock-prices-csv data/raw/stock_prices.csv \\
        --benchmark-prices-csv data/raw/benchmark_prices.csv \\
        --regime-csv data/processed/regime_detection/regime_history.csv \\
        --scores-csv data/processed/fundamental_analysis/scores_by_date.csv \\
        --out-dir reports

Input formats:
    --stock-prices-csv: date, <symbol1>, <symbol2>, ... (wide format, close prices)
    --benchmark-prices-csv: date, close
    --regime-csv: date, regime  (as saved by scripts/run_regime_detection.py)
    --scores-csv: date, symbol, composite_score  (long format, one row per
        rebalance date x symbol — see docs/backtesting_spec.md for how to
        build this from repeated fundamental_analysis pipeline runs)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.io_utils import load_config
from src.backtesting.pipeline import run_backtest_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stock-prices-csv", required=True)
    parser.add_argument("--benchmark-prices-csv", required=True)
    parser.add_argument("--regime-csv", required=True)
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default=None, help="Defaults to backtesting.report_dir in config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    stock_prices = pd.read_csv(args.stock_prices_csv, parse_dates=["date"]).set_index("date")
    benchmark_prices = (
        pd.read_csv(args.benchmark_prices_csv, parse_dates=["date"]).set_index("date")["close"]
    )
    regime_df = pd.read_csv(args.regime_csv, parse_dates=["date"]).set_index("date")
    regime = regime_df["regime"]
    # geometric_crash_risk_flag is present only if regime_detection.geometric_signal.enabled
    # was set when regime_history.csv was generated — see regime_detection/pipeline.py.
    geometric_crash_flag = regime_df.get("geometric_crash_risk_flag")
    scores_by_date = pd.read_csv(args.scores_csv, parse_dates=["date"])

    out_dir = args.out_dir or cfg["backtesting"].get("report_dir", "reports")
    result = run_backtest_pipeline(
        cfg["backtesting"], stock_prices, benchmark_prices, regime, scores_by_date, out_dir=out_dir,
        geometric_crash_flag=geometric_crash_flag,
    )

    print(result["attribution_table"][["cagr", "sharpe_ratio", "max_drawdown", "excess_cagr_vs_benchmark"]])
    print("\nReturn decomposition:")
    for k, v in result["decomposition"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nFull report: {result['report_path']}")


if __name__ == "__main__":
    main()
