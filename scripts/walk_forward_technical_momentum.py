"""Walk-forward validation for a fundamentals composite-score dimension's
weight (default: technical_momentum) -- the rigorous follow-up to
``sweep_technical_momentum_weight.py``'s in-sample sweep.

**Why this exists**: an in-sample sweep answers "which weight performed
best over this one fixed historical period" -- which is exactly the
question that gets contaminated by overfitting the more weight values you
try against the same backtest window. A monotonic-looking improvement all
the way to an extreme weight (e.g. >0.5 on a single dimension out of nine)
is the classic signature of that happening, not evidence of a genuine
factor blend. See ``docs/backtesting_spec.md``'s technical_momentum
section for the specific real-data sweep that motivated this.

**What this does instead**: splits history into successive (TRAIN, TEST)
folds. For each fold, the weight that performed best on TRAIN is selected
-- using ONLY train-period returns, never touching test -- then that fixed
choice is evaluated on TEST, a period it had zero influence over. If the
walk-forward-selected weight reliably beats a zero-weight baseline on TEST
performance, averaged across folds, that's real evidence of value. If it
doesn't, the in-sample sweep's apparent "optimum" was overfitting.

**Why this is fast despite re-running many backtests**: fundamentals
composite scoring at any given rebalance date only uses data available AS
OF that date (already point-in-time by construction throughout this
project) -- so ``scores_by_date`` for a given weight can be computed ONCE
over the full history and safely reused for every fold's train/test
slice; there's no leakage risk from doing so, since the composite weight
itself is a fixed a-priori choice per candidate value, not something
fit via regression against subsequent returns. Only the BACKTEST
(converting scores into weights into realized P&L over a specific date
range) needs to be re-run per fold.

Usage:
    python scripts/walk_forward_technical_momentum.py --weights 0,0.05,0.1,0.2,0.3,0.5,0.7,0.775,0.9
    python scripts/walk_forward_technical_momentum.py --min-train-years 5 --test-years 1 --step-years 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.sweep_technical_momentum_weight import load_cached_inputs, rebalanced_weights
from src.backtesting.pipeline import run_backtest_pipeline
from src.common.io_utils import load_config
from src.common.logging_utils import get_logger
from src.fundamental_analysis.point_in_time import run_pit_fundamental_pipeline

logger = get_logger(__name__)


def generate_walk_forward_folds(
    start: pd.Timestamp, end: pd.Timestamp,
    min_train_years: float = 5.0, test_years: float = 1.0, step_years: float = 1.0,
    fold_mode: str = "expanding",
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Two fold modes:

    ``"expanding"`` (default): TRAIN always starts at ``start`` and grows;
    TEST is a fixed-length window immediately following TRAIN, rolling
    forward by ``step_years`` each fold. Uses all available data each
    fold, matching how you'd actually deploy this in practice -- but a
    real limitation worth being explicit about: consecutive folds' TRAIN
    windows overlap heavily (fold N's train is a near-superset of fold
    N-1's), so folds picking the same "best" weight is much weaker
    independent confirmation than it looks -- it can just mean the same,
    mostly-unchanged data produced a similar (possibly flat/noisy)
    train-selection result each time, not that the weight is robustly
    validated across genuinely different conditions.

    ``"rolling"``: TRAIN is a FIXED-length ``min_train_years`` window that
    slides forward by ``step_years`` each fold (not expanding), so
    consecutive folds' train windows overlap far less -- e.g. with
    ``min_train_years=5, step_years=1``, fold N and fold N+1 share only 4
    of their 5 train years, and folds more than ``min_train_years`` apart
    share none at all. This is a genuine (if still imperfect, given
    limited total history) independence check: if the selected weight
    still clusters together across ROLLING folds, especially ones that
    barely overlap, that's real evidence, not an artifact of shared data.
    Costs some efficiency (less data per fold, more train-set turnover).
    """
    if fold_mode not in ("expanding", "rolling"):
        raise ValueError(f"fold_mode must be 'expanding' or 'rolling', got {fold_mode!r}")
    folds = []
    test_start = start + pd.DateOffset(years=min_train_years)
    while True:
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if test_end > end:
            break
        train_end = test_start - pd.Timedelta(days=1)
        train_start = start if fold_mode == "expanding" else max(start, train_end - pd.DateOffset(years=min_train_years) + pd.Timedelta(days=1))
        folds.append((train_start, train_end, test_start, min(test_end, end)))
        test_start = test_start + pd.DateOffset(years=step_years)
    return folds


def compute_scores_for_weights(
    cfg: dict, dim: str, weights: list[float], inputs: dict,
) -> dict[float, pd.DataFrame]:
    """Compute ``scores_by_date`` once per candidate weight over the FULL
    history -- safe to reuse across every fold (see module docstring for
    why there's no leakage from doing this)."""
    fa_cfg = cfg["fundamental_analysis"]
    pit_cfg = fa_cfg.get("point_in_time", {})
    rebalance_dates = pd.date_range(
        inputs["stock_prices"].index.min(), inputs["stock_prices"].index.max(),
        freq=pit_cfg.get("rebalance_frequency", "MS"),
    )
    scores_by_weight = {}
    for w in weights:
        logger.info("Scoring full history for %s weight=%.4f ...", dim, w)
        weights_this_run = rebalanced_weights(fa_cfg["composite_weights"], dim, w)
        assert abs(sum(weights_this_run.values()) - 1.0) < 1e-6
        run_fa_cfg = dict(fa_cfg)
        run_fa_cfg["composite_weights"] = weights_this_run
        scores_by_weight[w] = run_pit_fundamental_pipeline(
            run_fa_cfg, inputs["snapshot"], inputs["quarterly"], rebalance_dates,
            conviction_panel=inputs["conviction_panel"] if dim == "technical_momentum" else None,
        )
    return scores_by_weight


def backtest_window(
    cfg: dict, inputs: dict, scores_by_date: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp, out_dir: str, engine: str,
) -> dict | None:
    """Run the backtest restricted to ``[start, end]`` only, reusing
    already-computed prices/regime/scores (just sliced) -- no re-fetching,
    no re-scoring. Returns None if the window has too little data to
    produce a meaningful backtest (e.g. a test fold cut short at the end
    of available history)."""
    stock_prices = inputs["stock_prices"].loc[start:end]
    benchmark_prices = inputs["benchmark_prices"].loc[start:end]
    regime = inputs["regime"].loc[start:end]
    window_scores = scores_by_date[(scores_by_date["date"] >= start) & (scores_by_date["date"] <= end)]

    if len(stock_prices) < 20 or window_scores["date"].nunique() < 2:
        logger.warning("Window %s to %s has too little data (%d price rows, %d rebalance dates) -- skipping",
                        start.date(), end.date(), len(stock_prices), window_scores["date"].nunique())
        return None

    backtest_cfg = dict(cfg["backtesting"])
    backtest_cfg["engine"] = engine
    bt_result = run_backtest_pipeline(
        backtest_cfg, stock_prices, benchmark_prices, regime, window_scores, out_dir=out_dir,
    )
    table = bt_result["attribution_table"]
    row = {}
    for component in ("fundamentals_only", "combined"):
        if component in table.index:
            row[f"{component}_cagr"] = table.loc[component, "cagr"]
            row[f"{component}_sharpe"] = table.loc[component, "sharpe_ratio"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dimension", default="technical_momentum")
    parser.add_argument("--weights", default="0,0.025,0.05,0.1,0.2,0.3,0.5,0.7,0.775,0.9",
                         help="Comma-separated candidate weights to select among on each fold's TRAIN window.")
    parser.add_argument("--min-train-years", type=float, default=5.0)
    parser.add_argument("--test-years", type=float, default=1.0)
    parser.add_argument("--step-years", type=float, default=1.0)
    parser.add_argument("--fold-mode", choices=["expanding", "rolling"], default="expanding",
                         help="'expanding' (default) reuses all history each fold but folds overlap heavily -- "
                              "weight agreement across folds is weak evidence. 'rolling' uses fixed-length, "
                              "far-less-overlapping train windows -- a genuine independence check. Run BOTH "
                              "and compare if you want real confidence, not just one or the other.")
    parser.add_argument("--selection-metric", choices=["sharpe", "cagr"], default="sharpe",
                         help="Which TRAIN metric picks the weight for each fold. Default sharpe (less noisy / more standard for model selection than raw CAGR).")
    parser.add_argument("--target-component", choices=["fundamentals_only", "combined"], default="fundamentals_only")
    parser.add_argument("--engine", choices=["vectorbt", "custom"], default="vectorbt")
    parser.add_argument("--out-dir", default="reports/technical_momentum_walk_forward")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fa_cfg = cfg["fundamental_analysis"]
    dim = args.dimension
    target = args.target_component
    metric_col = f"{target}_{args.selection_metric}"

    if not fa_cfg["dimensions"].get(dim, False):
        print(f"ERROR: fundamental_analysis.dimensions.{dim} is not enabled in {args.config} -- enable it first.")
        sys.exit(1)

    weights = sorted(set(float(w) for w in args.weights.split(",")))
    print(f"Candidate weights for walk-forward selection: {weights}")

    try:
        inputs = load_cached_inputs(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    full_start, full_end = inputs["stock_prices"].index.min(), inputs["stock_prices"].index.max()
    folds = generate_walk_forward_folds(
        full_start, full_end, args.min_train_years, args.test_years, args.step_years, fold_mode=args.fold_mode
    )
    if not folds:
        print(
            f"ERROR: no folds fit -- history spans {full_start.date()} to {full_end.date()} "
            f"({(full_end - full_start).days / 365.25:.1f} years), need at least "
            f"{args.min_train_years + args.test_years:.1f} years. Reduce --min-train-years/--test-years."
        )
        sys.exit(1)
    print(f"\nGenerated {len(folds)} walk-forward fold(s):")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"  fold {i}: train {tr_s.date()} to {tr_e.date()}  |  test {te_s.date()} to {te_e.date()}")

    scores_by_weight = compute_scores_for_weights(cfg, dim, weights, inputs)

    fold_results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n{'=' * 78}\nFOLD {i}: train {train_start.date()}-{train_end.date()}, test {test_start.date()}-{test_end.date()}\n{'=' * 78}")

        train_rows = {}
        for w in weights:
            row = backtest_window(
                cfg, inputs, scores_by_weight[w], train_start, train_end,
                out_dir=f"{args.out_dir}/fold{i}_train_w{w:.4f}", engine=args.engine,
            )
            if row is not None:
                train_rows[w] = row
        if not train_rows:
            print(f"  fold {i}: no usable train results, skipping fold")
            continue

        train_df = pd.DataFrame(train_rows).T
        train_df.index.name = "weight"
        best_w = train_df[metric_col].idxmax()
        print(f"  TRAIN: best weight by {args.selection_metric} = {best_w:.4f} "
              f"({target} {args.selection_metric}={train_df.loc[best_w, metric_col]:.4f})")

        # Evaluate the walk-forward-selected weight, the zero-weight
        # baseline, AND (for illustration only, never for selection) the
        # single overall best-on-full-in-sample-history weight, all on
        # this fold's TEST window -- the baseline/reference comparisons
        # never influenced weight selection, only the walk-forward pick did.
        test_selected = backtest_window(
            cfg, inputs, scores_by_weight[best_w], test_start, test_end,
            out_dir=f"{args.out_dir}/fold{i}_test_selected", engine=args.engine,
        )
        test_baseline = backtest_window(
            cfg, inputs, scores_by_weight[0.0], test_start, test_end,
            out_dir=f"{args.out_dir}/fold{i}_test_baseline", engine=args.engine,
        ) if 0.0 in scores_by_weight else None

        if test_selected is None:
            print(f"  fold {i}: test window too short/unusable, skipping")
            continue

        fold_results.append({
            "fold": i, "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "selected_weight": best_w,
            f"selected_test_{target}_cagr": test_selected.get(f"{target}_cagr"),
            f"selected_test_{target}_sharpe": test_selected.get(f"{target}_sharpe"),
            f"baseline_test_{target}_cagr": test_baseline.get(f"{target}_cagr") if test_baseline else np.nan,
            f"baseline_test_{target}_sharpe": test_baseline.get(f"{target}_sharpe") if test_baseline else np.nan,
        })
        print(f"  TEST (out-of-sample): selected weight {best_w:.4f} -> "
              f"{target} cagr={test_selected.get(f'{target}_cagr'):.4f}, sharpe={test_selected.get(f'{target}_sharpe'):.4f}")
        if test_baseline:
            print(f"  TEST (out-of-sample): baseline weight 0.0    -> "
                  f"{target} cagr={test_baseline.get(f'{target}_cagr'):.4f}, sharpe={test_baseline.get(f'{target}_sharpe'):.4f}")

    if not fold_results:
        print("\nNo usable folds produced results -- try a shorter --min-train-years/--test-years given your available history.")
        sys.exit(1)

    results_df = pd.DataFrame(fold_results)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results_path = f"{args.out_dir}/walk_forward_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'=' * 78}\nWALK-FORWARD SUMMARY ({len(fold_results)} fold(s))\n{'=' * 78}")
    print(results_df.to_string(index=False))

    cagr_col_sel = f"selected_test_{target}_cagr"
    cagr_col_base = f"baseline_test_{target}_cagr"
    sharpe_col_sel = f"selected_test_{target}_sharpe"
    sharpe_col_base = f"baseline_test_{target}_sharpe"

    print(f"\n{'-' * 78}\nVERDICT\n{'-' * 78}")
    mean_sel_cagr, mean_base_cagr = results_df[cagr_col_sel].mean(), results_df[cagr_col_base].mean()
    mean_sel_sharpe, mean_base_sharpe = results_df[sharpe_col_sel].mean(), results_df[sharpe_col_base].mean()
    n_folds_sel_wins_cagr = (results_df[cagr_col_sel] > results_df[cagr_col_base]).sum()
    n_folds_sel_wins_sharpe = (results_df[sharpe_col_sel] > results_df[sharpe_col_base]).sum()
    n_folds = len(results_df)

    print(f"Mean out-of-sample CAGR:   walk-forward-selected={mean_sel_cagr:.4f}  vs  baseline(weight=0)={mean_base_cagr:.4f}  (delta {mean_sel_cagr - mean_base_cagr:+.4f})")
    print(f"Mean out-of-sample Sharpe: walk-forward-selected={mean_sel_sharpe:.4f}  vs  baseline(weight=0)={mean_base_sharpe:.4f}  (delta {mean_sel_sharpe - mean_base_sharpe:+.4f})")
    print(f"Selected weight beat baseline on CAGR in {n_folds_sel_wins_cagr}/{n_folds} folds, Sharpe in {n_folds_sel_wins_sharpe}/{n_folds} folds")
    selected_weights = results_df["selected_weight"].tolist()
    n_unique_selected = results_df["selected_weight"].nunique()
    print(f"Selected weights across folds: {selected_weights}")

    beats_baseline = mean_sel_cagr > mean_base_cagr and mean_sel_sharpe > mean_base_sharpe and n_folds_sel_wins_cagr >= (n_folds + 1) // 2

    if args.fold_mode == "expanding" and n_unique_selected == 1:
        print(
            f"\n-> CAUTION: every fold selected the EXACT SAME weight ({selected_weights[0]:.4f}). With "
            f"--fold-mode expanding, consecutive folds' TRAIN windows overlap heavily (fold N's train is a "
            f"near-superset of fold N-1's) -- so this agreement is much weaker evidence than it looks. It can "
            f"mean the same, mostly-unchanged data just produced a similar train-selection result each time, "
            f"NOT that the weight is robustly validated across genuinely different conditions. Re-run with "
            f"--fold-mode rolling (fixed-length, far-less-overlapping train windows) before trusting this "
            f"specific weight -- if it STILL lands on the same value across genuinely non-overlapping folds, "
            f"that's real confirmation; if it scatters, the expanding-fold agreement above was an artifact of "
            f"overlap, not a stable result."
        )
    elif n_unique_selected == 1:
        print(
            f"\n-> Every fold selected the same weight ({selected_weights[0]:.4f}) even under --fold-mode "
            f"rolling (far-less-overlapping train windows) -- this is a meaningfully stronger signal than the "
            f"same result would be under expanding folds, since these folds share much less training data."
        )

    if beats_baseline:
        print(
            "\n-> The walk-forward-selected weight beat the zero-weight baseline OUT OF SAMPLE, on average and "
            "in most folds, on both CAGR and Sharpe. This IS real (if still limited-sample) evidence that SOME "
            "nonzero weight on this dimension adds value out of sample. It is NOT strong evidence for the "
            "SPECIFIC selected weight being correct, especially if that weight sits in a region your earlier "
            "in-sample sweep showed to be a flat/noisy plateau rather than a sharp peak -- check whether nearby "
            "weights scored similarly in-sample before treating the exact selected value as meaningful, and "
            "consider using a more moderate weight from within the plateau rather than its edge."
        )
    else:
        print(
            "\n-> The walk-forward-selected weight did NOT consistently beat the zero-weight baseline out of "
            "sample. This is consistent with the in-sample sweep's apparent optimum being overfitting rather "
            "than a real, generalizable effect -- the in-sample numbers looked good specifically because the "
            "weight was chosen to look good on that exact historical period, and that advantage did not "
            "survive being tested on data it never saw."
        )

    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
