#!/usr/bin/env python
"""CLI: fit the regime-detection model on a price history CSV and save labels.

Usage:
    python scripts/run_regime_detection.py --price-csv data/raw/nifty500_history.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io_utils import load_config
from src.regime_detection.pipeline import run_pipeline, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-csv",
        required=True,
        help="CSV with columns: date, close[, advances, declines, vix]",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="data/processed/regime_detection")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result, model = run_pipeline(cfg["regime_detection"], args.price_csv)
    save_outputs(result, model, args.out_dir)
    print(result[["regime", "regime_name"]].tail(10))


if __name__ == "__main__":
    main()
