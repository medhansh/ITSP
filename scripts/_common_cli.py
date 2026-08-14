"""Shared price-data loading for standalone backtest CLI scripts
(``run_technical_backtest.py``, ``run_adaptive_ichimoku_backtest.py``).
Factored out so both scripts load data identically instead of drifting.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def add_price_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("price data (pick one)")
    group.add_argument("--symbols", help="Comma-separated NSE symbols to fetch live via yfinance, e.g. RELIANCE,TCS,INFY")
    group.add_argument("--price-csv", help="Path to an existing wide-format price CSV (date index, one column per symbol)")
    group.add_argument("--benchmark-ticker", default="^CRSLDX", help="Benchmark ticker for comparison (live-fetch mode); '' to skip")
    group.add_argument("--benchmark-csv", help="Benchmark CSV path (local-file mode); optional")
    group.add_argument("--start", default="2015-01-01")
    group.add_argument("--end", default=None)


def add_ohlc_price_data_args(parser: argparse.ArgumentParser) -> None:
    """Same as ``add_price_data_args`` but for indicators (Ichimoku) that
    need true OHLC, not just close — see
    ``backtesting/adaptive_ichimoku.py``'s docstring on why a close-derived
    high/low proxy isn't good enough. ``--price-csv`` here expects
    LONG format: columns ``date, symbol, open, high, low, close[, volume]``
    (not the wide close-only format ``add_price_data_args`` uses) — OHLC
    for many symbols doesn't naturally fit one wide table the way a single
    close value per (date, symbol) cell does.
    """
    group = parser.add_argument_group("OHLC price data (pick one)")
    group.add_argument("--symbols", help="Comma-separated NSE symbols to fetch live via yfinance, e.g. RELIANCE,TCS,INFY")
    group.add_argument("--price-csv", help="Path to a LONG-format OHLC CSV: columns date,symbol,open,high,low,close[,volume]")
    group.add_argument("--benchmark-ticker", default="^CRSLDX", help="Benchmark ticker for comparison (live-fetch mode); '' to skip")
    group.add_argument("--benchmark-csv", help="Benchmark CSV path (close-only is fine; local-file mode); optional")
    group.add_argument("--start", default="2015-01-01")
    group.add_argument("--end", default=None)


def load_prices_ohlc(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.Series | None]:
    """Returns ({symbol: OHLC DataFrame}, benchmark_series_or_None). Each
    OHLC DataFrame has lowercase columns open/high/low/close[/volume],
    date index.
    """
    if getattr(args, "benchmark_ticker", None) == "":
        args.benchmark_ticker = None

    if args.price_csv:
        logger.info("Loading long-format OHLC CSV from %s", args.price_csv)
        long_df = pd.read_csv(args.price_csv, parse_dates=["date"])
        required = {"date", "symbol", "open", "high", "low", "close"}
        missing = required - set(long_df.columns)
        if missing:
            raise SystemExit(
                f"{args.price_csv} is missing required OHLC columns: {missing}. Expected long "
                "format: date, symbol, open, high, low, close[, volume]."
            )
        panel: dict[str, pd.DataFrame] = {}
        for symbol, group_df in long_df.groupby("symbol"):
            df = group_df.set_index("date").sort_index()
            cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[cols]
            if args.start:
                df = df.loc[args.start:]
            if args.end:
                df = df.loc[:args.end]
            panel[symbol] = df

        benchmark = None
        if args.benchmark_csv:
            logger.info("Loading benchmark series from %s", args.benchmark_csv)
            bdf = pd.read_csv(args.benchmark_csv, index_col=0, parse_dates=True)
            benchmark = bdf["close"] if "close" in bdf.columns else bdf.iloc[:, 0]
        return panel, benchmark

    if not args.symbols:
        raise SystemExit("Pass either --price-csv (long-format OHLC) or --symbols (comma-separated NSE symbols).")

    from src.fundamental_analysis.data_fetchers import yfinance_fetcher

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Fetching live OHLC panel for %d symbols via yfinance (start=%s)", len(symbols), args.start)
    panel = yfinance_fetcher.fetch_price_panel_ohlc(symbols, start=args.start, end=args.end)
    if not panel:
        raise SystemExit(
            "yfinance returned no OHLC data for any of the requested symbols. Check the symbol "
            "names (NSE symbols, no .NS suffix needed) and your network connection."
        )

    benchmark = None
    if getattr(args, "benchmark_ticker", None):
        logger.info("Fetching benchmark %s via yfinance", args.benchmark_ticker)
        benchmark = yfinance_fetcher.fetch_benchmark_series(
            benchmark_ticker=args.benchmark_ticker, start=args.start, end=args.end
        )
        if benchmark.empty:
            logger.warning("No benchmark data for %s -- proceeding without a benchmark comparison", args.benchmark_ticker)
            benchmark = None

    return panel, benchmark


def load_prices_ohlc_safe(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.Series | None]:
    """``load_prices_ohlc`` wrapped with a clean error message instead of a
    raw traceback on common failures."""
    try:
        return load_prices_ohlc(args)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from None
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to load OHLC price data ({type(exc).__name__}: {exc}). If using --symbols, "
            "check your network connection and that the symbol names are valid NSE tickers; "
            "if using --price-csv, check the file exists and is long-format "
            "(date, symbol, open, high, low, close[, volume])."
        ) from None


def load_prices(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series | None]:
    """Returns (stock_price_panel, benchmark_series_or_None). Close prices
    only, wide format (date index, one column per symbol) — this project's
    price fetchers/CSVs don't carry per-symbol OHLC (see
    ``adaptive_ichimoku.py``'s docstring for what that means for Tenkan/
    Kijun/Senkou-B, which are traditionally computed from intraday high/low).
    """
    if getattr(args, "benchmark_ticker", None) == "":
        args.benchmark_ticker = None

    if args.price_csv:
        logger.info("Loading price panel from %s", args.price_csv)
        panel = pd.read_csv(args.price_csv, index_col=0, parse_dates=True)
        if args.start:
            panel = panel.loc[args.start:]
        if args.end:
            panel = panel.loc[:args.end]
        benchmark = None
        if args.benchmark_csv:
            logger.info("Loading benchmark series from %s", args.benchmark_csv)
            bdf = pd.read_csv(args.benchmark_csv, index_col=0, parse_dates=True)
            benchmark = bdf["close"] if "close" in bdf.columns else bdf.iloc[:, 0]
        return panel, benchmark

    if not args.symbols:
        raise SystemExit("Pass either --price-csv or --symbols (comma-separated NSE symbols).")

    from src.fundamental_analysis.data_fetchers import yfinance_fetcher

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Fetching live price panel for %d symbols via yfinance (start=%s)", len(symbols), args.start)
    panel = yfinance_fetcher.fetch_price_panel(symbols, start=args.start, end=args.end)
    if panel.empty:
        raise SystemExit(
            "yfinance returned no data for any of the requested symbols. Check the symbol "
            "names (NSE symbols, no .NS suffix needed -- fetch_price_panel appends it) and "
            "your network connection."
        )

    benchmark = None
    if getattr(args, "benchmark_ticker", None):
        logger.info("Fetching benchmark %s via yfinance", args.benchmark_ticker)
        benchmark = yfinance_fetcher.fetch_benchmark_series(
            benchmark_ticker=args.benchmark_ticker, start=args.start, end=args.end
        )
        if benchmark.empty:
            logger.warning("No benchmark data for %s -- proceeding without a benchmark comparison", args.benchmark_ticker)
            benchmark = None

    return panel, benchmark


def load_prices_safe(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series | None]:
    """``load_prices`` wrapped with a clean error message instead of a raw
    traceback on common failures (missing yfinance, bad file path, network
    issues) — suitable to call directly from a script's ``main()``."""
    try:
        return load_prices(args)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from None
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to load price data ({type(exc).__name__}: {exc}). If using --symbols, "
            "check your network connection and that the symbol names are valid NSE tickers; "
            "if using --price-csv, check the file exists and has a parseable date index."
        ) from None
