#!/usr/bin/env python
"""Quick diagnostic: fetch one symbol from Screener (and optionally Trendlyne)
and print every field that was successfully parsed vs. came back NaN.

Run this after any long gap before trusting a full-universe fetch — if a
field that used to populate is suddenly NaN across the board, it usually
means the source site changed its markup and screener_fetcher.py /
trendlyne_fetcher.py need updating (see docs/data_sourcing_spec.md).

Usage:
    python scripts/probe_data_source.py RELIANCE
    python scripts/probe_data_source.py RELIANCE --trendlyne
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.fundamental_analysis.data_fetchers import screener_fetcher, trendlyne_fetcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol")
    parser.add_argument("--trendlyne", action="store_true", help="Also probe Trendlyne (requires a row in trendlyne_id_map.csv)")
    parser.add_argument(
        "--quarterly", action="store_true",
        help="Also probe the quarterly-results table used for point-in-time fundamentals — "
             "shows the raw period-column labels Screener sends and whether each one parsed "
             "into a date. Run this first if fetch_multiple_quarterly_history is coming back "
             "empty/thin (see docs/fundamental_analysis_spec.md's point-in-time section).",
    )
    args = parser.parse_args()

    print(f"--- Screener.in: {args.symbol} ---")
    snapshot = screener_fetcher.fetch_company_snapshot(args.symbol)
    n_populated = sum(1 for v in snapshot.values() if not (v is None or (isinstance(v, float) and pd.isna(v))))
    for k, v in snapshot.items():
        print(f"  {k:35s} {v}")
    print(f"\n{n_populated}/{len(snapshot)} fields populated (non-NaN).")

    if args.trendlyne:
        print(f"\n--- Trendlyne: {args.symbol} ---")
        tl = trendlyne_fetcher.fetch_equity_snapshot(args.symbol)
        for k, v in tl.items():
            print(f"  {k:35s} {v}")

    if args.quarterly:
        print(f"\n--- Quarterly results table (raw): {args.symbol} ---")
        raw = screener_fetcher.fetch_raw_quarters_table(args.symbol)
        if raw.empty:
            print(
                "  EMPTY — the 'quarters'/'quarterly-results' section wasn't found at all on "
                "the page (see screener_fetcher.SECTION_IDS). This is a bigger markup change "
                "than a label-format mismatch — the section id itself may have changed."
            )
        else:
            print(f"  Row labels found: {raw.index.tolist()}")
            print(f"  Raw period-column labels: {raw.columns.tolist()}")
            print("\n  Parse result per column (via _parse_period_end):")
            parsed_any = False
            for col in raw.columns:
                parsed = screener_fetcher._parse_period_end(col)
                print(f"    {col!r:20s} -> {parsed}")
                parsed_any = parsed_any or parsed is not None
            if not parsed_any:
                print(
                    "\n  NONE of the columns parsed. Copy the raw labels above and update "
                    "_PERIOD_LABEL_RE / _parse_period_end in screener_fetcher.py to match "
                    "the actual format — that's the fix."
                )

        print(f"\n--- fetch_quarterly_history() output: {args.symbol} ---")
        history = screener_fetcher.fetch_quarterly_history(args.symbol)
        print(history if not history.empty else "  EMPTY")


if __name__ == "__main__":
    main()
