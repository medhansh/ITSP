#!/usr/bin/env python
"""CLI: score the NIFTY500 universe on fundamentals and save the ranked output.

Usage:
    python scripts/run_fundamental_analysis.py \\
        --snapshot-csv data/raw/fundamentals_snapshot.csv \\
        --history-csv data/raw/fundamentals_history.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.io_utils import load_config
from src.fundamental_analysis.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-csv", required=True,
        help="Per-symbol latest fundamentals; see SNAPSHOT_SCHEMA in "
             "src/fundamental_analysis/data_fetchers/fundamentals_fetcher.py",
    )
    parser.add_argument(
        "--history-csv", default=None,
        help="Multi-year fundamentals for the growth dimension (symbol, fiscal_year, "
             "revenue, net_income, eps). Omit to skip growth scoring.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-csv", default="data/processed/fundamental_analysis/scores.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    snapshot = pd.read_csv(args.snapshot_csv).set_index("symbol", drop=False)
    history = pd.read_csv(args.history_csv) if args.history_csv else None

    result = run_pipeline(cfg["fundamental_analysis"], snapshot, history)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path)
    print(result[["composite_score"]].head(20))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
