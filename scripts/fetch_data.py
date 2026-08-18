#!/usr/bin/env python
"""CLI: pull the NIFTY500 universe list, price data, and fundamentals into
data/raw/, ready to feed the regime_detection / fundamental_analysis /
backtesting pipelines.

Requires outbound internet access (this scaffold's build sandbox had none —
see docs/data_sourcing_spec.md). Fundamentals fetching hits Screener.in and
Trendlyne HTML pages directly (no bulk API), so a full NIFTY500 run is slow
by design (rate-limited, ~2 sec/request/source) and will take well over an
hour — run it once, then rely on the disk cache for iteration.

Usage:
    # Universe list (from NSE archives)
    python scripts/fetch_data.py universe --out data/universe/nifty500_list.csv

    # Price panel (yfinance) for the whole universe + benchmark
    python scripts/fetch_data.py prices --universe-csv data/universe/nifty500_list.csv \\
        --out-stocks data/raw/stock_prices.csv --out-benchmark data/raw/benchmark_prices.csv

    # Fundamentals snapshot (merged Screener + yfinance + Trendlyne)
    python scripts/fetch_data.py fundamentals --universe-csv data/universe/nifty500_list.csv \\
        --out-snapshot data/raw/fundamentals_snapshot.csv --out-provenance data/raw/fundamentals_provenance.csv

    # Multi-year history (Screener P&L, for the growth dimension)
    python scripts/fetch_data.py history --universe-csv data/universe/nifty500_list.csv \\
        --out data/raw/fundamentals_history.csv

    # India VIX (yfinance) -- production regime-detection input (see
    # regime_detection.production_regime_source in configs/config.yaml)
    python scripts/fetch_data.py vix --out data/raw/vix.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.io_utils import load_config
from src.fundamental_analysis.data_fetchers import fundamentals_fetcher, nse_fetcher, yfinance_fetcher
from src.fundamental_analysis.data_fetchers import screener_fetcher
from src.regime_detection import data_loader as regime_data_loader


def cmd_universe(args, cfg) -> None:
    df = nse_fetcher.fetch_nifty500_list()
    nse_fetcher.save_universe_list(df, out_path=args.out)
    print(f"Saved {len(df)} symbols to {args.out}")


def cmd_prices(args, cfg) -> None:
    universe = pd.read_csv(args.universe_csv, comment="#")
    symbols = universe["symbol"].tolist()

    dcfg = cfg["data_fetchers"]
    panel = yfinance_fetcher.fetch_price_panel(symbols, start=dcfg["price_start_date"])
    Path(args.out_stocks).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_stocks)
    print(f"Saved price panel for {panel.shape[1]} symbols x {panel.shape[0]} days to {args.out_stocks}")

    benchmark = yfinance_fetcher.fetch_benchmark_ohlcv(
        benchmark_ticker=dcfg["benchmark_ticker"], start=dcfg["price_start_date"]
    )
    benchmark.to_csv(args.out_benchmark)
    print(f"Saved benchmark OHLCV series ({len(benchmark)} days) to {args.out_benchmark}")


def cmd_fundamentals(args, cfg) -> None:
    universe = pd.read_csv(args.universe_csv, comment="#")
    symbols = universe["symbol"].tolist()

    fcfg = cfg["data_fetchers"]["fundamentals"]
    snapshot, provenance = fundamentals_fetcher.fetch_fundamentals(
        symbols,
        sources=fcfg["sources"],
        source_priority=fcfg["source_priority"],
        min_delay_seconds=fcfg["min_delay_seconds"],
        cache_dir=fcfg["cache_dir"],
        cache_ttl_days=fcfg["cache_ttl_days"],
    )
    Path(args.out_snapshot).parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(args.out_snapshot)
    provenance.to_csv(args.out_provenance)
    print(f"Saved fundamentals snapshot ({snapshot.shape}) to {args.out_snapshot}")
    print(f"Saved field provenance to {args.out_provenance}")
    print("\nField coverage (fraction of symbols with a value):")
    print(snapshot.notna().mean().sort_values(ascending=False))


def cmd_history(args, cfg) -> None:
    universe = pd.read_csv(args.universe_csv, comment="#")
    symbols = universe["symbol"].tolist()

    fcfg = cfg["data_fetchers"]["fundamentals"]
    history = fundamentals_fetcher.fetch_fundamentals_history(
        symbols, min_delay_seconds=fcfg["min_delay_seconds"]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.out, index=False)
    print(f"Saved history ({history.shape}) to {args.out}")


def cmd_quarterly_history(args, cfg) -> None:
    """Fetch per-quarter results (with a known-date-of-availability tag) for
    the point-in-time fundamentals pipeline — see src/fundamental_analysis/
    point_in_time.py. Distinct from `history` (annual P&L for growth.py)."""
    universe = pd.read_csv(args.universe_csv, comment="#")
    symbols = universe["symbol"].tolist()

    fcfg = cfg["data_fetchers"]["fundamentals"]
    pit_cfg = cfg["fundamental_analysis"].get("point_in_time", {})
    quarterly = screener_fetcher.fetch_multiple_quarterly_history(
        symbols,
        min_delay_seconds=fcfg["min_delay_seconds"],
        reporting_lag_days=pit_cfg.get("reporting_lag_days", 45),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    quarterly.to_csv(args.out, index=False)
    print(f"Saved quarterly PIT history ({quarterly.shape}) to {args.out}")


def cmd_vix(args, cfg) -> None:
    """Fetch India VIX as a standalone series -- the production regime
    source (regime_detection.production_regime_source in
    configs/config.yaml, default 'vix_bucket_contemporaneous') is built
    directly from it. Separate from the bundled VIX fetch inside
    regime_detection.data_loader.load_from_yfinance (that one feeds the
    GMM price-feature clustering matrix specifically)."""
    dcfg = cfg["data_fetchers"]
    vix_ticker = dcfg.get("vix_ticker", "^INDIAVIX")
    vix = yfinance_fetcher.fetch_india_vix_series(vix_ticker=vix_ticker, start=dcfg["price_start_date"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vix.to_csv(args.out)
    if vix.empty:
        print(f"WARNING: yfinance returned no data for {vix_ticker} -- {args.out} is empty.")
    else:
        print(f"Saved India VIX series ({len(vix)} days, {vix.index.min().date()}..{vix.index.max().date()}) to {args.out}")


def cmd_sector_prices(args, cfg) -> None:
    """Fetch sector index price history for the geometric wedge-product
    crash-risk signal — see src/regime_detection/geometric_signal.py."""
    geo_cfg = cfg["regime_detection"].get("geometric_signal", {})
    sector_tickers = geo_cfg.get("sector_tickers") or regime_data_loader.DEFAULT_SECTOR_TICKERS
    dcfg = cfg["data_fetchers"]
    prices = regime_data_loader.load_sector_prices_from_yfinance(
        sector_tickers=sector_tickers, start=dcfg["price_start_date"]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out)
    print(f"Saved sector price panel ({prices.shape}) to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_universe = subparsers.add_parser("universe", help="Fetch the NIFTY500 constituent list")
    p_universe.add_argument("--out", default="data/universe/nifty500_list.csv")
    p_universe.set_defaults(func=cmd_universe)

    p_prices = subparsers.add_parser("prices", help="Fetch stock + benchmark price panels via yfinance")
    p_prices.add_argument("--universe-csv", default="data/universe/nifty500_list.csv")
    p_prices.add_argument("--out-stocks", default="data/raw/stock_prices.csv")
    p_prices.add_argument("--out-benchmark", default="data/raw/benchmark_prices.csv")
    p_prices.set_defaults(func=cmd_prices)

    p_fund = subparsers.add_parser("fundamentals", help="Fetch + merge fundamentals snapshot from all sources")
    p_fund.add_argument("--universe-csv", default="data/universe/nifty500_list.csv")
    p_fund.add_argument("--out-snapshot", default="data/raw/fundamentals_snapshot.csv")
    p_fund.add_argument("--out-provenance", default="data/raw/fundamentals_provenance.csv")
    p_fund.set_defaults(func=cmd_fundamentals)

    p_hist = subparsers.add_parser("history", help="Fetch multi-year financial history (Screener) for the growth dimension")
    p_hist.add_argument("--universe-csv", default="data/universe/nifty500_list.csv")
    p_hist.add_argument("--out", default="data/raw/fundamentals_history.csv")
    p_hist.set_defaults(func=cmd_history)

    p_qhist = subparsers.add_parser(
        "quarterly-history", help="Fetch per-quarter results w/ known-date tags for point-in-time fundamentals"
    )
    p_qhist.add_argument("--universe-csv", default="data/universe/nifty500_list.csv")
    p_qhist.add_argument("--out", default="data/raw/fundamentals_quarterly_history.csv")
    p_qhist.set_defaults(func=cmd_quarterly_history)

    p_sector = subparsers.add_parser(
        "sector-prices", help="Fetch sector index prices for the geometric wedge-product crash signal"
    )
    p_sector.add_argument("--out", default="data/raw/sector_prices.csv")
    p_sector.set_defaults(func=cmd_sector_prices)

    p_vix = subparsers.add_parser(
        "vix", help="Fetch India VIX for the production regime source (regime_detection.production_regime_source)"
    )
    p_vix.add_argument("--out", default="data/raw/vix.csv")
    p_vix.set_defaults(func=cmd_vix)

    args = parser.parse_args()
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
