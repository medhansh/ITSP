"""Sweep a fundamentals composite-score dimension's weight (default:
technical_momentum) across a range and measure the effect on
fundamentals_only/combined performance.

Reuses already-cached data (prices, fundamentals snapshot/quarterly,
regime, Ichimoku conviction panel) from a prior ``run_full_pipeline.py``
run -- does NOT re-fetch anything over the network or recompute regime/
Ichimoku, only the composite-score weighting and the resulting backtest.
Run ``run_full_pipeline.py`` at least once first (with
``technical_signals.ichimoku.enabled`` and
``fundamental_analysis.dimensions.technical_momentum`` both true, if
you're sweeping that dimension) so those cache files exist.

Usage:
    python scripts/sweep_technical_momentum_weight.py --min 0.0 --max 0.25 --step 0.025
    python scripts/sweep_technical_momentum_weight.py --weights 0,0.05,0.10,0.15,0.20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtesting.pipeline import run_backtest_pipeline
from src.common.io_utils import load_config
from src.common.logging_utils import get_logger
from src.fundamental_analysis.data_fetchers import fundamentals_fetcher
from src.fundamental_analysis.point_in_time import run_pit_fundamental_pipeline
from src.fundamental_analysis.scoring.composite_score import rebalanced_weights

logger = get_logger(__name__)


def load_cached_inputs(cfg: dict) -> dict:
    """Load everything a prior ``run_full_pipeline.py`` run already cached
    to disk -- raises a clear error naming the missing file(s) rather than
    a confusing downstream KeyError/FileNotFoundError if the pipeline
    hasn't been run yet.
    """
    fa_cfg = cfg["fundamental_analysis"]
    required = {
        "stock_prices": "data/raw/stock_prices.csv",
        "benchmark_prices": "data/raw/benchmark_prices.csv",
        "snapshot": "data/raw/fundamentals_snapshot.csv",
        "regime_history": "data/processed/regime_history.csv",
    }
    missing = [path for path in required.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cached input(s), run scripts/run_full_pipeline.py at least once first:\n  "
            + "\n  ".join(missing)
        )

    stock_prices = pd.read_csv(required["stock_prices"], index_col=0, parse_dates=True)
    benchmark_prices = pd.read_csv(required["benchmark_prices"], index_col=0, parse_dates=True)["close"]
    regime_history = pd.read_csv(required["regime_history"], index_col=0, parse_dates=True)
    regime = regime_history["regime"]
    # regime-conditional composite weighting (apply_regime_conditional_weight)
    # needs STRING labels matching the multipliers config (e.g. "low_vol_calm"),
    # NOT the raw int regime id the backtest's exposure_by_regime uses -- these
    # are genuinely different things and conflating them silently degrades
    # every regime-conditional lookup to a no-op (int never matches a string
    # key, so it always falls back to the default multiplier). Prefer
    # regime_name if present; fall back to stringified ints with a warning
    # otherwise, since that's at least self-consistent even if it won't match
    # a human-written multipliers config out of the box.
    if "regime_name" in regime_history.columns:
        regime_name = regime_history["regime_name"]
    else:
        logger.warning(
            "regime_name column not found in %s -- falling back to stringified integer regime ids "
            "for regime-conditional weighting, which likely won't match a multipliers config written "
            "against label names like 'low_vol_calm'. Re-run run_full_pipeline.py to regenerate "
            "regime_history.csv with regime_name included.",
            required["regime_history"],
        )
        regime_name = regime.astype(str)

    snapshot = pd.read_csv(required["snapshot"], index_col="symbol")
    snapshot = snapshot.reindex(columns=fundamentals_fetcher.SNAPSHOT_SCHEMA).combine_first(snapshot)

    quarterly_path = fa_cfg.get("point_in_time", {}).get(
        "quarterly_history_path", "data/raw/fundamentals_quarterly_history.csv"
    )
    quarterly = (
        pd.read_csv(quarterly_path, parse_dates=["period_end", "known_date"])
        if Path(quarterly_path).exists()
        else pd.DataFrame(columns=["symbol", "period_end", "known_date", "field", "value"])
    )

    conviction_panel = None
    conviction_path = "data/processed/ichimoku_conviction_panel.csv"
    if Path(conviction_path).exists():
        conviction_panel = pd.read_csv(conviction_path, index_col=0, parse_dates=True)
    else:
        logger.warning(
            "%s not found -- sweeping technical_momentum will score it as all-NaN "
            "(harmless, but the sweep won't show any real effect). Run run_full_pipeline.py "
            "with technical_signals.ichimoku.enabled: true first if that's not intended.",
            conviction_path,
        )

    return dict(
        stock_prices=stock_prices, benchmark_prices=benchmark_prices, regime=regime, regime_name=regime_name,
        snapshot=snapshot, quarterly=quarterly, conviction_panel=conviction_panel,
    )


def run_one_weight(
    cfg: dict, dim: str, weight: float, inputs: dict, out_dir: str, engine: str,
) -> dict:
    """Re-score fundamentals with ``dim`` set to ``weight`` (rebalancing
    every other weight to compensate) and run the backtest -- everything
    else (prices, regime, Ichimoku panels) is reused unchanged from
    ``inputs``.
    """
    fa_cfg = cfg["fundamental_analysis"]
    weights_this_run = rebalanced_weights(fa_cfg["composite_weights"], dim, weight)
    total = sum(weights_this_run.values())
    assert abs(total - 1.0) < 1e-6, f"rebalanced weights don't sum to 1.0: {total}"

    run_fa_cfg = dict(fa_cfg)
    run_fa_cfg["composite_weights"] = weights_this_run

    pit_cfg = fa_cfg.get("point_in_time", {})
    rebalance_dates = pd.date_range(
        inputs["stock_prices"].index.min(), inputs["stock_prices"].index.max(),
        freq=pit_cfg.get("rebalance_frequency", "MS"),
    )

    scores_by_date = run_pit_fundamental_pipeline(
        run_fa_cfg, inputs["snapshot"], inputs["quarterly"], rebalance_dates,
        conviction_panel=inputs["conviction_panel"] if dim == "technical_momentum" else None,
    )

    backtest_cfg = dict(cfg["backtesting"])
    backtest_cfg["engine"] = engine
    bt_result = run_backtest_pipeline(
        backtest_cfg, inputs["stock_prices"], inputs["benchmark_prices"], inputs["regime"],
        scores_by_date, out_dir=out_dir,
    )

    table = bt_result["attribution_table"]
    row = {"weight": weight}
    for component in ("fundamentals_only", "combined"):
        if component in table.index:
            row[f"{component}_cagr"] = table.loc[component, "cagr"]
            row[f"{component}_sharpe"] = table.loc[component, "sharpe_ratio"]
            row[f"{component}_max_drawdown"] = table.loc[component, "max_drawdown"]
    return row


def analyze(summary: pd.DataFrame) -> str:
    """Plain-text analysis of the sweep: best weight per metric, spread,
    and a monotonicity read so a peak-in-the-middle result (a real
    optimum) can be told apart from a monotonic edge result (the range
    tested wasn't wide enough) or a noisy non-trend (treat with caution).
    """
    lines = []
    for component in ("fundamentals_only", "combined"):
        cagr_col = f"{component}_cagr"
        sharpe_col = f"{component}_sharpe"
        dd_col = f"{component}_max_drawdown"
        if cagr_col not in summary.columns:
            continue

        best_cagr_w = summary[cagr_col].idxmax()
        best_sharpe_w = summary[sharpe_col].idxmax()
        best_dd_w = summary[dd_col].idxmax()  # max_drawdown is negative; idxmax = least-negative = best

        lines.append(f"\n{component}:")
        lines.append(f"  best CAGR    at weight={best_cagr_w:.4f}  ({summary.loc[best_cagr_w, cagr_col]:+.4f})")
        lines.append(f"  best Sharpe  at weight={best_sharpe_w:.4f}  ({summary.loc[best_sharpe_w, sharpe_col]:.4f})")
        lines.append(f"  best drawdown at weight={best_dd_w:.4f}  ({summary.loc[best_dd_w, dd_col]:.4f})")
        lines.append(
            f"  CAGR range: {summary[cagr_col].min():.4f} to {summary[cagr_col].max():.4f} "
            f"(spread {summary[cagr_col].max() - summary[cagr_col].min():.4f})"
        )

        diffs = np.diff(summary[cagr_col].values)
        n_up, n_down = (diffs > 1e-9).sum(), (diffs < -1e-9).sum()
        n_steps = len(diffs)
        w_min, w_max = summary.index.min(), summary.index.max()
        # A handful of tiny reversals in an otherwise-consistent direction is
        # noise, not "no trend" -- require the MINORITY direction to be at
        # least 20% of steps before calling it genuinely non-monotonic.
        # (An earlier stricter version required literally zero reversals in
        # either direction, which misclassified a real 18-of-20-steps-up
        # trend as "no clean trend" purely because of two ~0.1pp wiggles.)
        minority_frac = min(n_up, n_down) / n_steps if n_steps else 0
        if n_down == 0:
            trend = f"MONOTONICALLY INCREASING across the tested range -- consider testing weights above {w_max:.4f}"
        elif n_up == 0:
            trend = f"MONOTONICALLY DECREASING across the tested range -- the lowest tested weight ({w_min:.4f}) is best so far; consider testing lower/zero"
        elif minority_frac <= 0.20 and best_cagr_w == w_max:
            trend = (
                f"MOSTLY INCREASING ({n_up}/{n_steps} steps up, {n_down} minor reversal(s)) all the way to the "
                f"edge of the tested range ({w_max:.4f}) with no interior peak -- this is the signature of the "
                f"range simply not being wide enough, OR of overfitting to a single fixed backtest period rather "
                f"than a genuine optimum. Do not trust this weight without out-of-sample validation."
            )
        elif minority_frac <= 0.20 and best_cagr_w == w_min:
            trend = (
                f"MOSTLY DECREASING ({n_down}/{n_steps} steps down, {n_up} minor reversal(s)); lowest tested "
                f"weight ({w_min:.4f}) is still best -- consider testing lower/zero."
            )
        elif best_cagr_w not in (w_min, w_max):
            trend = f"PEAKS IN THE INTERIOR at weight={best_cagr_w:.4f} -- could be a genuine optimum, but verify with out-of-sample validation before trusting it, especially if the peak is at an extreme weight (e.g. >0.5) for a single dimension."
        else:
            trend = "NO CLEAN TREND -- CAGR moves non-monotonically with substantial reversals; treat any single 'best' weight with real caution, likely noise rather than a real effect"
        lines.append(f"  CAGR trend: {trend}")
    return "\n".join(lines)


def plot_sweep(summary: pd.DataFrame, dim: str, out_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- skipping chart.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(
        axes, ("cagr", "sharpe", "max_drawdown"), ("CAGR", "Sharpe ratio", "Max drawdown")
    ):
        for component in ("fundamentals_only", "combined"):
            col = f"{component}_{metric}"
            if col in summary.columns:
                ax.plot(summary.index, summary[col], marker="o", label=component)
        ax.set_xlabel(f"{dim} composite weight")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = f"{out_dir}/sweep_chart.png"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dimension", default="technical_momentum",
                         help="Which composite-score dimension to sweep. Default technical_momentum.")
    parser.add_argument("--weights", default=None,
                         help="Comma-separated explicit weights to test, e.g. '0,0.05,0.10,0.15'. Overrides --min/--max/--step.")
    parser.add_argument("--min", type=float, default=0.0, help="Lower bound (inclusive). Default 0.0.")
    parser.add_argument("--max", type=float, default=0.25, help="Upper bound (inclusive). Default 0.25.")
    parser.add_argument("--step", type=float, default=0.025, help="Step size. Default 0.025.")
    parser.add_argument("--engine", choices=["vectorbt", "custom"], default="vectorbt")
    parser.add_argument("--out-dir", default="reports/technical_momentum_sweep")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fa_cfg = cfg["fundamental_analysis"]
    dim = args.dimension

    if not fa_cfg["dimensions"].get(dim, False):
        print(f"ERROR: fundamental_analysis.dimensions.{dim} is not enabled in {args.config} -- enable it first.")
        sys.exit(1)
    if dim not in fa_cfg["composite_weights"]:
        print(f"ERROR: {dim!r} not found in fundamental_analysis.composite_weights.")
        sys.exit(1)

    if args.weights:
        weights_to_test = [float(w) for w in args.weights.split(",")]
    else:
        if args.step <= 0:
            print("ERROR: --step must be positive.")
            sys.exit(1)
        weights_to_test = [round(w, 6) for w in np.arange(args.min, args.max + args.step / 2, args.step)]

    print(f"Sweeping {dim!r} weight across: {weights_to_test}")

    try:
        inputs = load_cached_inputs(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    results = []
    for w in weights_to_test:
        print(f"\n=== {dim} weight = {w:.4f} ===")
        row = run_one_weight(cfg, dim, w, inputs, out_dir=f"{args.out_dir}/w_{w:.4f}", engine=args.engine)
        results.append(row)
        print(pd.DataFrame([row]).set_index("weight").to_string())

    summary = pd.DataFrame(results).set_index("weight")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary_path = f"{args.out_dir}/sweep_summary.csv"
    summary.to_csv(summary_path)

    print("\n" + "=" * 78)
    print("SWEEP SUMMARY")
    print("=" * 78)
    print(summary.to_string())

    print("\n" + "-" * 78)
    print("ANALYSIS")
    print("-" * 78)
    print(analyze(summary))

    chart_path = plot_sweep(summary, dim, args.out_dir)

    print(f"\nSaved summary CSV to {summary_path}")
    if chart_path:
        print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
