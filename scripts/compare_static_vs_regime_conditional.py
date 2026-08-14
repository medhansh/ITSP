"""Compare STATIC vs REGIME-CONDITIONAL technical_momentum weighting on the
full backtest -- the direct verification step for
``point_in_time.apply_regime_conditional_weight``, built after walk-forward
validation found the dimension's edge concentrated in trending conditions.

Reuses already-cached data (prices, fundamentals, regime, Ichimoku
conviction panel) -- no re-fetching.

Usage:
    python scripts/compare_static_vs_regime_conditional.py
    python scripts/compare_static_vs_regime_conditional.py --static-weight 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.sweep_technical_momentum_weight import load_cached_inputs, rebalanced_weights
from src.backtesting.pipeline import run_backtest_pipeline
from src.common.io_utils import load_config
from src.common.logging_utils import get_logger
from src.fundamental_analysis.point_in_time import run_pit_fundamental_pipeline

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--static-weight", type=float, default=None,
                         help="technical_momentum weight to use for the STATIC comparison arm. "
                              "Default: whatever's currently in configs/config.yaml.")
    parser.add_argument("--engine", choices=["vectorbt", "custom"], default="vectorbt")
    parser.add_argument("--out-dir", default="reports/static_vs_regime_conditional")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fa_cfg = cfg["fundamental_analysis"]
    dim = "technical_momentum"

    if not fa_cfg["dimensions"].get(dim, False):
        print(f"ERROR: fundamental_analysis.dimensions.{dim} is not enabled -- enable it first.")
        sys.exit(1)

    tm_cfg = fa_cfg.get("technical_momentum_regime_conditioning", {})
    multipliers = tm_cfg.get("multipliers")
    if not multipliers:
        print("ERROR: fundamental_analysis.technical_momentum_regime_conditioning.multipliers not "
              "found in config -- add that section first (see configs/config.yaml for the default).")
        sys.exit(1)

    static_weight = args.static_weight if args.static_weight is not None else fa_cfg["composite_weights"].get(dim, 0.05)
    print(f"Static arm weight: {static_weight}")
    print(f"Regime-conditional multipliers: {multipliers}")

    try:
        inputs = load_cached_inputs(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    pit_cfg = fa_cfg.get("point_in_time", {})
    rebalance_dates = pd.date_range(
        inputs["stock_prices"].index.min(), inputs["stock_prices"].index.max(),
        freq=pit_cfg.get("rebalance_frequency", "MS"),
    )

    # --- STATIC arm ---
    print("\nScoring STATIC arm...")
    static_weights = rebalanced_weights(fa_cfg["composite_weights"], dim, static_weight)
    static_fa_cfg = dict(fa_cfg)
    static_fa_cfg["composite_weights"] = static_weights
    static_scores = run_pit_fundamental_pipeline(
        static_fa_cfg, inputs["snapshot"], inputs["quarterly"], rebalance_dates,
        conviction_panel=inputs["conviction_panel"],
    )

    # --- REGIME-CONDITIONAL arm ---
    print("Scoring REGIME-CONDITIONAL arm...")
    conditional_fa_cfg = dict(fa_cfg)
    conditional_fa_cfg["composite_weights"] = rebalanced_weights(fa_cfg["composite_weights"], dim, static_weight)
    conditional_scores = run_pit_fundamental_pipeline(
        conditional_fa_cfg, inputs["snapshot"], inputs["quarterly"], rebalance_dates,
        conviction_panel=inputs["conviction_panel"],
        regime=inputs["regime_name"], regime_weight_multipliers=multipliers,
    )

    backtest_cfg = dict(cfg["backtesting"])
    backtest_cfg["engine"] = args.engine

    print("\nBacktesting STATIC arm...")
    static_bt = run_backtest_pipeline(
        backtest_cfg, inputs["stock_prices"], inputs["benchmark_prices"], inputs["regime"], static_scores,
        out_dir=f"{args.out_dir}/static",
    )
    print("Backtesting REGIME-CONDITIONAL arm...")
    conditional_bt = run_backtest_pipeline(
        backtest_cfg, inputs["stock_prices"], inputs["benchmark_prices"], inputs["regime"], conditional_scores,
        out_dir=f"{args.out_dir}/regime_conditional",
    )

    rows = []
    for label, bt in (("static", static_bt), ("regime_conditional", conditional_bt)):
        table = bt["attribution_table"]
        for component in ("fundamentals_only", "combined"):
            if component in table.index:
                rows.append({
                    "arm": label, "component": component,
                    "cagr": table.loc[component, "cagr"],
                    "sharpe": table.loc[component, "sharpe_ratio"],
                    "max_drawdown": table.loc[component, "max_drawdown"],
                })
    summary = pd.DataFrame(rows).set_index(["component", "arm"]).sort_index()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary_path = f"{args.out_dir}/comparison_summary.csv"
    summary.to_csv(summary_path)

    print("\n" + "=" * 78)
    print("STATIC vs REGIME-CONDITIONAL COMPARISON")
    print("=" * 78)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 78)
    print("DELTA (regime_conditional minus static)")
    print("-" * 78)
    for component in ("fundamentals_only", "combined"):
        if (component, "static") in summary.index and (component, "regime_conditional") in summary.index:
            static_row = summary.loc[(component, "static")]
            cond_row = summary.loc[(component, "regime_conditional")]
            print(f"\n{component}:")
            for metric in ("cagr", "sharpe", "max_drawdown"):
                delta = cond_row[metric] - static_row[metric]
                print(f"  {metric}: {static_row[metric]:+.4f} -> {cond_row[metric]:+.4f}  (delta {delta:+.4f})")

    print(f"\nSaved to {summary_path}")
    print(
        "\nNOTE: this compares two SPECIFIC configurations on the SAME historical data used throughout "
        "this whole investigation -- a win here is encouraging but is still in-sample. If you want the "
        "same rigor as the static-weight validation, run scripts/walk_forward_technical_momentum.py's "
        "underlying logic against the regime-conditional arm too before fully trusting a specific result."
    )


if __name__ == "__main__":
    main()
