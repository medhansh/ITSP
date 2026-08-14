#!/usr/bin/env python
"""One-command, start-to-finish run: download/cache every input this system
needs, then run regime detection -> point-in-time fundamental scoring ->
backtest -> report.

This is the orchestration layer requested on top of the individual
`scripts/fetch_data.py` subcommands and the three pipeline modules
(`regime_detection.pipeline`, `fundamental_analysis.pipeline` /
`point_in_time`, `backtesting.pipeline`) — it does not duplicate their logic,
it just sequences them and adds "skip if already cached and fresh" so
re-running during development doesn't re-download or re-scrape everything
from scratch (`--force-refresh` bypasses that).

Requires outbound network access for the download steps — same requirement
(and the same caveat that this scaffold's own build/dev sandbox has none) as
`scripts/fetch_data.py`. Run this from your own machine or a server with
internet access; see docs/data_sourcing_spec.md.

Usage:
    python scripts/run_full_pipeline.py --config configs/config.yaml

    # Skip the (slow) full-universe fundamentals scrape and reuse whatever's
    # already cached on disk, even if stale:
    python scripts/run_full_pipeline.py --skip-fundamentals-refresh

    # Force a full re-download of everything, ignoring existing cache files:
    python scripts/run_full_pipeline.py --force-refresh

    # Smoke-test the orchestration logic itself against a small local
    # fixture universe instead of the real NIFTY500 (no network needed):
    python scripts/run_full_pipeline.py --universe-csv tests/fixtures/mini_universe.csv --offline
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.backtesting.pipeline import run_backtest_pipeline
from src.fundamental_analysis.orthogonalization import compute_rolling_beta_panel
from src.common.io_utils import load_config
from src.common.logging_utils import get_logger
from src.fundamental_analysis.data_fetchers import fundamentals_fetcher, nse_fetcher, screener_fetcher
from src.fundamental_analysis.point_in_time import run_pit_fundamental_pipeline
from src.regime_detection import data_loader as regime_data_loader
from src.regime_detection.pipeline import run_pipeline as run_regime_pipeline

logger = get_logger(__name__)


def _stale(path: str, ttl_days: float, force_refresh: bool) -> bool:
    """True if ``path`` doesn't exist, is older than ``ttl_days``, or
    ``force_refresh`` was requested — i.e. "this needs to be (re)downloaded".
    """
    if force_refresh:
        return True
    p = Path(path)
    if not p.exists():
        return True
    age_days = (time.time() - p.stat().st_mtime) / 86400.0
    return age_days > ttl_days


def step_universe(cfg: dict, args: argparse.Namespace) -> str:
    out = args.universe_csv or cfg["universe"]["list_path"]
    if args.offline:
        logger.info("[universe] --offline: using existing file at %s as-is", out)
        return out
    if not _stale(out, ttl_days=30, force_refresh=args.force_refresh):
        logger.info("[universe] %s is fresh (< 30 days old) — skipping re-fetch", out)
        return out
    logger.info("[universe] fetching NIFTY500 constituent list -> %s", out)
    df = nse_fetcher.fetch_nifty500_list()
    nse_fetcher.save_universe_list(df, out_path=out)
    return out


def step_prices(cfg: dict, args: argparse.Namespace, universe_csv: str) -> tuple[str, str]:
    dcfg = cfg["data_fetchers"]
    stocks_out = "data/raw/stock_prices.csv"
    bench_out = "data/raw/benchmark_prices.csv"
    if args.offline:
        logger.info("[prices] --offline: using existing price files as-is")
        return stocks_out, bench_out
    if not (_stale(stocks_out, 1, args.force_refresh) or _stale(bench_out, 1, args.force_refresh)):
        logger.info("[prices] price panel is fresh (< 1 day old) — skipping re-fetch")
        return stocks_out, bench_out

    from src.fundamental_analysis.data_fetchers import yfinance_fetcher

    universe = pd.read_csv(universe_csv, comment="#")
    symbols = universe["symbol"].tolist()
    logger.info("[prices] fetching price panel for %d symbols via yfinance", len(symbols))
    panel = yfinance_fetcher.fetch_price_panel(symbols, start=dcfg["price_start_date"])
    Path(stocks_out).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(stocks_out)

    # Full OHLCV (not just close) so regime_detection picks up range-based
    # volatility (Parkinson/Garman-Klass) and volume features automatically
    # — see src/regime_detection/features.py. Downstream consumers that only
    # need close (the backtest engine) just read that one column back out.
    benchmark = yfinance_fetcher.fetch_benchmark_ohlcv(
        benchmark_ticker=dcfg["benchmark_ticker"], start=dcfg["price_start_date"]
    )
    benchmark.to_csv(bench_out)
    return stocks_out, bench_out


def step_sector_prices(cfg: dict, args: argparse.Namespace) -> str | None:
    geo_cfg = cfg["regime_detection"].get("geometric_signal", {})
    if not geo_cfg.get("enabled"):
        return None
    out = geo_cfg.get("sector_price_csv", "data/raw/sector_prices.csv")
    if args.offline:
        logger.info("[sector-prices] --offline: using existing file as-is")
        return out
    if not _stale(out, 1, args.force_refresh):
        logger.info("[sector-prices] fresh — skipping re-fetch")
        return out
    logger.info("[sector-prices] fetching sector index prices for the geometric signal")
    tickers = geo_cfg.get("sector_tickers") or regime_data_loader.DEFAULT_SECTOR_TICKERS
    prices = regime_data_loader.load_sector_prices_from_yfinance(
        sector_tickers=tickers, start=cfg["data_fetchers"]["price_start_date"]
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out)
    return out


def step_fundamentals(cfg: dict, args: argparse.Namespace, universe_csv: str) -> tuple[str, str]:
    """Current snapshot (non-PIT fields like sector/shareholding) + quarterly
    PIT history (revenue/net_income/eps, replayed forward through time)."""
    fcfg = cfg["data_fetchers"]["fundamentals"]
    pit_cfg = cfg["fundamental_analysis"].get("point_in_time", {})
    snapshot_out = "data/raw/fundamentals_snapshot.csv"
    quarterly_out = pit_cfg.get("quarterly_history_path", "data/raw/fundamentals_quarterly_history.csv")

    if args.offline or args.skip_fundamentals_refresh:
        logger.info("[fundamentals] skipping refresh — using existing files as-is")
        return snapshot_out, quarterly_out

    universe = pd.read_csv(universe_csv, comment="#")
    symbols = universe["symbol"].tolist()

    if _stale(snapshot_out, fcfg["cache_ttl_days"], args.force_refresh):
        logger.info("[fundamentals] fetching current snapshot for %d symbols (slow — rate-limited)", len(symbols))
        snapshot, provenance = fundamentals_fetcher.fetch_fundamentals(
            symbols,
            sources=fcfg["sources"], source_priority=fcfg["source_priority"],
            min_delay_seconds=fcfg["min_delay_seconds"], cache_dir=fcfg["cache_dir"],
            cache_ttl_days=fcfg["cache_ttl_days"], trendlyne_mapping_path=fcfg["trendlyne_mapping_path"],
        )
        Path(snapshot_out).parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(snapshot_out)
        provenance.to_csv("data/raw/fundamentals_provenance.csv")
    else:
        logger.info("[fundamentals] snapshot is fresh — skipping re-fetch")

    if pit_cfg.get("enabled", True) and _stale(quarterly_out, fcfg["cache_ttl_days"], args.force_refresh):
        logger.info("[fundamentals] fetching quarterly PIT history for %d symbols", len(symbols))
        quarterly = screener_fetcher.fetch_multiple_quarterly_history(
            symbols, min_delay_seconds=fcfg["min_delay_seconds"],
            reporting_lag_days=pit_cfg.get("reporting_lag_days", 45),
        )
        Path(quarterly_out).parent.mkdir(parents=True, exist_ok=True)
        quarterly.to_csv(quarterly_out, index=False)
    else:
        logger.info("[fundamentals] quarterly PIT history is fresh or PIT disabled — skipping")

    return snapshot_out, quarterly_out


def step_options(cfg: dict, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a pre-built option_summary_history + earnings_calendar from
    disk. There is no free bulk NSE options source wired in (see
    data_fetchers/options_fetcher.py's docstring) — this step does NOT
    scrape anything; it just loads whatever CSVs are configured, and
    degrades to empty (all-NaN options_earnings dimension downstream,
    harmless — weights get renormalized) if they don't exist.
    """
    ocfg = cfg["data_fetchers"].get("options", {})
    if not ocfg.get("enabled"):
        return pd.DataFrame(), pd.DataFrame()
    hist_path = ocfg.get("option_summary_history_path")
    cal_path = ocfg.get("earnings_calendar_path")
    history = pd.DataFrame()
    calendar = pd.DataFrame()
    if hist_path and Path(hist_path).exists():
        history = pd.read_csv(hist_path, parse_dates=["date"])
    else:
        logger.warning("[options] option_summary_history_path not found (%s) — options_earnings dimension will be NaN", hist_path)
    if cal_path and Path(cal_path).exists():
        calendar = pd.read_csv(cal_path, parse_dates=["earnings_date"])
    else:
        logger.warning("[options] earnings_calendar_path not found (%s) — options_earnings dimension will be NaN", cal_path)
    return history, calendar


def step_conviction_panel(cfg, stock_prices, benchmark_prices, regime):
    """Build the conviction panel that feeds the ``technical_momentum``
    dimension, honouring ``technical_signals.conviction_source``.

    Sources:
      ``mom121``   -- plain 12-1 momentum: trailing 12-month return skipping
                      the most recent month.
      ``blend``    -- 12-1 momentum blended with a mean-reversion signal,
                      weighted by market stress.

    **Why the default changed.** A direct information-coefficient
    measurement (scripts/diagnose_signal_ic_by_regime.py) found the
    Ichimoku conviction score has essentially NO cross-sectional
    predictive power on this universe: IC -0.0010, t = -0.10, correct sign
    on 49.6% of rebalance dates. Plain 12-1 momentum measured IC +0.0383,
    t = 2.88. Walk-forward backtesting confirmed it: swapping Ichimoku for
    12-1 momentum improved out-of-sample Sharpe in 6/6 folds (+0.156) and
    CAGR in 6/6 folds, at LOWER turnover (4.25x vs 4.80x annualized).
    Blending on top added a further +0.053 Sharpe in 5/6 folds.

    Note this does not mean the earlier walk-forward validation of the
    Ichimoku dimension was wrong -- that test compared it against a
    ZERO-weight baseline ("something beats nothing"), never against a
    competing signal. It answered a weaker question than it appeared to.
    """
    ts_cfg = cfg.get("technical_signals", {})
    source = ts_cfg.get("conviction_source", "blend")
    if not cfg.get("fundamental_analysis", {}).get("dimensions", {}).get("technical_momentum", False):
        return None

    from src.backtesting.momentum_reversal_blend import (
        build_blend, build_momentum_panel, build_reversal_panel,
        stress_from_regime, stress_from_volatility,
    )

    momentum = build_momentum_panel(stock_prices, kind="12_1")
    if source == "mom121":
        logger.info("[conviction] source=mom121 (12-1 momentum, no blending)")
        panel = momentum
    elif source == "blend":
        b_cfg = ts_cfg.get("blend", {})
        mode = b_cfg.get("stress_mode", "continuous")
        reversal = build_reversal_panel(
            stock_prices, kind=b_cfg.get("reversal_kind", "dist_from_ma"),
            ma_window=b_cfg.get("ma_window", 63),
        )
        if mode == "continuous":
            stress = stress_from_volatility(
                benchmark_prices, stock_prices.index,
                vol_window=b_cfg.get("vol_window", 21),
            )
        else:
            stress = stress_from_regime(regime, stock_prices.index)
        logger.info("[conviction] source=blend (12-1 momentum x %s reversal, stress_mode=%s)",
                    b_cfg.get("reversal_kind", "dist_from_ma"), mode)
        panel = build_blend(momentum, reversal, stress, mode=mode,
                            max_reversal_weight=b_cfg.get("max_reversal_weight", 1.0))
    else:
        raise ValueError(
            f"technical_signals.conviction_source must be 'mom121' or 'blend', got {source!r}. "
            f"The 'ichimoku' source was retired from this build: it measured IC -0.0010 "
            f"(t = -0.10) against 12-1 momentum's +0.0383 (t = 2.88), and lost on 6/6 "
            f"walk-forward folds. Its OHLC fetch and modules are gone."
        )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    panel.to_csv("data/processed/conviction_panel.csv")
    return panel


def step_beta_panel(cfg: dict, stock_prices: pd.DataFrame, benchmark_prices: pd.Series):
    """Build the trailing rolling-beta panel that COMPOSITIONAL de-risking
    (``strategies.apply_beta_rotation``) needs. Returns ``None`` when
    ``backtesting.beta_rotation.rotation_strength`` is 0 or unset, in which
    case the backtest behaves exactly as it did before rotation existed.

    **Why this step exists at all**: ``rotation_strength`` in config does
    NOTHING on its own -- ``run_backtest_pipeline`` no-ops the rotation
    unless it is actually handed a ``beta_panel``. Before this step was
    added, setting the config value produced an attribution table with the
    usual 4 components and no ``fundamentals_beta_rotated`` row at all, and
    the silence looked exactly like a normal run. Anyone changing that
    config value and seeing 4 components instead of 5 is looking at a
    rotation that never ran.

    Built from the daily close panel already loaded for the backtest rather
    than from the Ichimoku OHLC file, so this works whether or not
    the retired Ichimoku signal is present -- beta rotation has no
    dependency on the Ichimoku signal and should not inherit one.
    """
    rot_cfg = cfg["backtesting"].get("beta_rotation", {})
    strength = rot_cfg.get("rotation_strength", 0.0)
    if not strength:
        logger.info("[beta_rotation] rotation_strength=%s -> disabled, no beta panel built", strength)
        return None

    window = (cfg["fundamental_analysis"]
              .get("technical_momentum_beta_orthogonalization", {})
              .get("beta_window", 252))
    logger.info("[beta_rotation] building trailing %dd beta panel for %d symbols (strength=%.2f)",
                window, stock_prices.shape[1], strength)
    beta_panel = compute_rolling_beta_panel(
        {s: stock_prices[s].dropna() for s in stock_prices.columns},
        benchmark_prices, window=window,
    )
    n_usable = int(beta_panel.iloc[-1].notna().sum())
    logger.info("[beta_rotation] beta panel: %d symbols, %d with a usable beta at the latest date",
                beta_panel.shape[1], n_usable)
    if n_usable == 0:
        logger.warning(
            "[beta_rotation] NO symbol has a usable beta -- rotation would be a silent no-op. "
            "Check that stock_prices has at least %d rows of history.", window,
        )
    return beta_panel


def step_regime(cfg: dict, price_csv: str, sector_price_csv: str | None) -> pd.DataFrame:
    logger.info("[regime] fitting regime model (sector signal %s)", "enabled" if sector_price_csv else "disabled")
    result, model = run_regime_pipeline(cfg["regime_detection"], price_csv, sector_price_csv=sector_price_csv)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    result.to_csv("data/processed/regime_history.csv")
    model.save("data/processed/regime_model.joblib")
    return result


def step_pit_fundamentals(
    cfg: dict, snapshot_csv: str, quarterly_csv: str,
    options_history: pd.DataFrame, earnings_calendar: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    conviction_panel: pd.DataFrame | None = None,
    regime: pd.Series | None = None,
) -> pd.DataFrame:
    fa_cfg = cfg["fundamental_analysis"]
    pit_cfg = fa_cfg.get("point_in_time", {})
    snapshot = pd.read_csv(snapshot_csv, index_col="symbol")
    snapshot = snapshot.reindex(columns=fundamentals_fetcher.SNAPSHOT_SCHEMA).combine_first(snapshot)
    quarterly = pd.read_csv(quarterly_csv, parse_dates=["period_end", "known_date"]) if Path(quarterly_csv).exists() else pd.DataFrame(
        columns=["symbol", "period_end", "known_date", "field", "value"]
    )

    rebalance_dates = pd.date_range(
        price_index.min(), price_index.max(), freq=pit_cfg.get("rebalance_frequency", "MS")
    )
    logger.info("[fundamentals-PIT] scoring %d rebalance dates x %d symbols", len(rebalance_dates), len(snapshot))

    if not options_history.empty and not earnings_calendar.empty:
        ocfg = cfg["data_fetchers"].get("options", {})
        conviction_at_rebalance = None
        if conviction_panel is not None:
            conviction_at_rebalance = conviction_panel.sort_index().reindex(rebalance_dates, method="ffill")
        results = []
        for date in rebalance_dates:
            from src.fundamental_analysis.point_in_time import (
                build_annual_growth_history_pit,
                build_pit_panel,
                merge_pit_into_snapshot,
            )
            from src.fundamental_analysis.metrics.options_earnings import compute_options_earnings_metrics
            from src.fundamental_analysis.pipeline import run_pipeline as run_fundamental_pipeline

            pit_fields = build_pit_panel(quarterly, [date])
            pit_wide = pit_fields.pivot(index="symbol", columns="field", values="value") if not pit_fields.empty else pd.DataFrame(index=snapshot.index)
            snap_as_of = merge_pit_into_snapshot(snapshot, pit_wide)
            options_metrics = compute_options_earnings_metrics(
                snap_as_of, options_history, earnings_calendar, as_of_date=date,
                pre_earnings_window_days=ocfg.get("pre_earnings_window_days", 5),
                max_lookback_days=ocfg.get("max_lookback_days", 10),
                iv_percentile_lookback_days=ocfg.get("iv_percentile_lookback_days", 252),
            )
            snap_as_of = snap_as_of.join(options_metrics)
            # Same PIT-safe TTM-rollup growth history as the non-options branch
            # below (run_pit_fundamental_pipeline) -- see
            # build_annual_growth_history_pit's docstring for why this was a
            # real gap (growth was previously silently skipped here too, via
            # a hardcoded history=None).
            growth_history_as_of = build_annual_growth_history_pit(quarterly, date)
            technical_conviction = None
            if conviction_at_rebalance is not None and date in conviction_at_rebalance.index:
                row = conviction_at_rebalance.loc[date].dropna()
                if not row.empty:
                    technical_conviction = row
            scored = run_fundamental_pipeline(
                fa_cfg, snap_as_of,
                history=growth_history_as_of if not growth_history_as_of.empty else None,
                technical_conviction=technical_conviction,
            )
            scored = scored.reset_index().rename(columns={"index": "symbol"})
            scored["date"] = date
            results.append(scored)
        scores_by_date = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["date", "symbol", "composite_score"])
    else:
        tm_cfg = fa_cfg.get("technical_momentum_regime_conditioning", {})
        regime_weight_multipliers = tm_cfg.get("multipliers") if tm_cfg.get("enabled", False) else None
        scores_by_date = run_pit_fundamental_pipeline(
            fa_cfg, snapshot, quarterly, rebalance_dates, conviction_panel=conviction_panel,
            regime=regime, regime_weight_multipliers=regime_weight_multipliers,
        )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    scores_by_date.to_csv("data/processed/pit_fundamental_scores.csv", index=False)

    n_rows = len(scores_by_date)
    n_scored = int(scores_by_date["composite_score"].notna().sum()) if n_rows else 0
    coverage = n_scored / n_rows if n_rows else 0.0
    logger.info(
        "[fundamentals-PIT] %d/%d (rebalance_date, symbol) rows have a composite_score (%.1f%% coverage)",
        n_scored, n_rows, coverage * 100,
    )
    if n_rows == 0:
        logger.error(
            "[fundamentals-PIT] scores_by_date is EMPTY (0 rows) — every strategy that needs "
            "stock selection (fundamentals_only, combined) will trade nothing and report a "
            "flat 0%% return for the whole backtest. This almost always means the quarterly "
            "PIT history (%s) itself came back empty — check the "
            "fetch_multiple_quarterly_history warning earlier in this log, and run "
            "`python scripts/probe_data_source.py <SYMBOL> --quarterly` to diagnose.",
            quarterly_csv,
        )
    elif coverage < 0.05:
        logger.warning(
            "[fundamentals-PIT] composite_score coverage is only %.1f%% — the fundamentals_only/"
            "combined backtests will be selecting from a near-empty universe most/all of the "
            "time. Check data/raw/fundamentals_snapshot.csv and %s coverage per column.",
            coverage * 100, quarterly_csv,
        )
    return scores_by_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--universe-csv", default=None, help="Override the universe CSV path")
    parser.add_argument("--force-refresh", action="store_true", help="Re-download everything, ignoring cache freshness")
    parser.add_argument("--skip-fundamentals-refresh", action="store_true", help="Always reuse whatever fundamentals data is already on disk")
    parser.add_argument("--offline", action="store_true", help="Never touch the network — use only what's already on disk (for testing the orchestration logic itself)")
    parser.add_argument("--out-dir", default=None, help="Override backtesting.report_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()

    universe_csv = step_universe(cfg, args)
    stock_price_csv, benchmark_price_csv = step_prices(cfg, args, universe_csv)
    sector_price_csv = step_sector_prices(cfg, args)
    snapshot_csv, quarterly_csv = step_fundamentals(cfg, args, universe_csv)
    options_history, earnings_calendar = step_options(cfg, args)

    stock_prices = pd.read_csv(stock_price_csv, index_col=0, parse_dates=True)
    benchmark_prices = pd.read_csv(benchmark_price_csv, index_col=0, parse_dates=True)["close"]

    regime_result = step_regime(cfg, benchmark_price_csv, sector_price_csv)
    regime = regime_result["regime"]
    active_regime = regime_result.get("active_regime")
    if active_regime is not None:
        n_transitional = int((active_regime == "transitional").sum())
        n_switches = int((active_regime != active_regime.shift()).sum())
        raw_switches = int((regime != regime.shift()).sum())
        logger.info(
            "[regime] consensus governor active: %d/%d days transitional, %d active_regime "
            "switches vs %d raw regime switches",
            n_transitional, len(active_regime), n_switches, raw_switches,
        )
    geometric_crash_flag = regime_result.get("geometric_crash_risk_flag")
    if geometric_crash_flag is not None:
        logger.info(
            "[regime] geometric overlay active: %d/%d days flagged as elevated crash risk "
            "(computed independently of the GMM regime — see regime_detection/pipeline.py)",
            int(geometric_crash_flag.fillna(0).sum()), geometric_crash_flag.notna().sum(),
        )

    # OHLC fetch happens BEFORE fundamentals scoring now (it didn't used to)
    # so the technical_momentum dimension's conviction panel can be threaded
    # why this reordering was necessary.
    conviction_panel = step_conviction_panel(
        cfg, stock_prices, benchmark_prices, regime_result["regime"],
    )
    if conviction_panel is not None:
        logger.info(
            "[conviction] technical_momentum dimension: conviction panel covers %d symbols",
            conviction_panel.shape[1],
        )

    scores_by_date = step_pit_fundamentals(
        cfg, snapshot_csv, quarterly_csv, options_history, earnings_calendar, stock_prices.index,
        conviction_panel=conviction_panel,
        regime=regime_result.get("regime_name", regime_result.get("regime")),
    )

    beta_panel = step_beta_panel(cfg, stock_prices, benchmark_prices)

    out_dir = args.out_dir or cfg["backtesting"].get("report_dir", "reports")
    logger.info("[backtest] running component backtests + attribution + report -> %s", out_dir)
    result = run_backtest_pipeline(
        cfg["backtesting"], stock_prices, benchmark_prices, regime, scores_by_date, out_dir=out_dir,
        geometric_crash_flag=geometric_crash_flag,
        active_regime=active_regime,
        beta_panel=beta_panel,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")
    print(f"Report: {result['report_path']}")
    print("\nPerformance by component:")
    print(result["attribution_table"][["cagr", "sharpe_ratio", "max_drawdown"]])

    # Surface whether compositional de-risking actually ran. Its absence is
    # otherwise indistinguishable from a normal run: you just get the usual
    # 4 components and no error.
    rot_strength = cfg["backtesting"].get("beta_rotation", {}).get("rotation_strength", 0.0)
    if "fundamentals_beta_rotated" in result["attribution_table"].index:
        print(f"\n[beta_rotation] ACTIVE at rotation_strength={rot_strength:g} "
              f"-- see the fundamentals_beta_rotated row above.")
    elif rot_strength:
        print(f"\n[beta_rotation] WARNING: rotation_strength={rot_strength:g} is set but the "
              f"fundamentals_beta_rotated component is MISSING -- the rotation did not run. "
              f"Check the [beta_rotation] log lines above for why.")


if __name__ == "__main__":
    main()
