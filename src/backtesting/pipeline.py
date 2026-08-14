"""End-to-end backtest pipeline: prices + regime labels + fundamental scores
-> per-component backtests -> attribution -> figures -> Markdown report.

This is the module that ties regime_detection and fundamental_analysis
together and answers "how did the combined system do, and how much did each
piece contribute" — see docs/backtesting_spec.md for full methodology.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtesting.attribution import (
    compute_attribution_table,
    compute_return_decomposition,
    run_component_backtests,
)
from src.backtesting.engine import compute_returns_panel
from src.backtesting.plotting import (
    plot_contribution_bar,
    plot_drawdowns,
    plot_equity_curves,
    plot_fundamental_score_distribution,
    plot_regime_timeline,
    plot_rolling_sharpe,
)
from src.backtesting.reporting import generate_markdown_report
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def run_backtest_pipeline(
    config: dict[str, Any],
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    regime: pd.Series,
    scores_by_date: pd.DataFrame,
    out_dir: str = "reports",
    geometric_crash_flag: pd.Series | None = None,
    active_regime: pd.Series | None = None,
    ichimoku_weights: pd.DataFrame | None = None,
    ichimoku_mode: str = "breadth_scalar",
    ichimoku_confirmation_floor: float = 0.0,
    ichimoku_tilt_strength: float = 0.0,
    beta_panel: pd.DataFrame | None = None,
    risk_panel: pd.DataFrame | None = None,
    vol_target_exposure: pd.Series | None = None,
) -> dict[str, Any]:
    """Run the full backtest + attribution + reporting pipeline.

    Args:
        config: the ``backtesting`` section of configs/config.yaml.
        stock_prices: daily close prices, index=date, columns=symbols (the
            fundamentals-eligible universe).
        benchmark_prices: daily close level of the benchmark (e.g. NIFTY500),
            a Series indexed by the same dates.
        regime: daily regime label Series from regime_detection.pipeline
            (already volatility-ordered: 0 = calmest).
        scores_by_date: long-format (date, symbol, composite_score) fundamental
            scores at each historical rebalance date — see
            src.backtesting.strategies.build_fundamental_portfolio_weights.
        out_dir: directory to write the report and its figures/ and tables/
            subdirectories into.
        geometric_crash_flag: optional ``geometric_crash_risk_flag`` column
            from ``regime_detection.pipeline.run_pipeline``'s output (present
            when ``regime_detection.geometric_signal.enabled`` is set). When
            given, adds the standalone ``geometric_overlay_only`` component
            and applies the same overlay on top of ``combined`` — see
            ``attribution.run_component_backtests``. ``None`` (default)
            reproduces the exact pre-overlay behavior.
        active_regime: optional ``active_regime`` column from
            ``regime_detection.pipeline.run_pipeline``'s output (present
            when ``regime_detection.consensus_governor.enabled`` is set) —
            entropy-gated, persistence + hysteresis governed regime label,
            a mix of int regime ids and the string ``"transitional"``. When
            given, adds ``governed_regime_only``/``governed_combined``
            components alongside (not replacing) ``regime_only``/
            ``combined`` — see ``attribution.run_component_backtests``.
            ``None`` (default) reproduces the exact pre-governor behavior.
        ichimoku_weights: optional daily weight matrix from
            ``adaptive_ichimoku.build_ichimoku_weights`` (present when
            ``technical_signals.ichimoku.enabled`` is set). When given,
            adds ``ichimoku_only``/``combined_with_ichimoku`` components —
            see ``attribution.run_component_backtests``. ``None`` (default)
            reproduces the exact pre-Ichimoku behavior.
        ichimoku_mode: ``"breadth_scalar"`` (default, recommended) or
            ``"hard_gate"`` — which construction builds
            ``combined_with_ichimoku``. See
            ``attribution.run_component_backtests``'s docstring and
            ``strategies.apply_ichimoku_gate``'s warning for why
            ``"hard_gate"`` produced severe cash drag (beta ~0.19) in the
            first real backtest.
        ichimoku_confirmation_floor: forwarded to
            ``strategies.apply_ichimoku_breadth_scalar`` when
            ``ichimoku_mode="breadth_scalar"``. Default 0.0.
        ichimoku_tilt_strength: default 0.0 (off). When nonzero, adds
            ``combined_ichimoku_tilted`` — ``combined`` REALLOCATED among
            its held names by relative Ichimoku conviction with total
            exposure held fixed, via ``strategies.apply_ichimoku_conviction_tilt``.
            Independent of/additional to ``ichimoku_mode`` — both
            ``combined_with_ichimoku`` and ``combined_ichimoku_tilted`` can
            be produced side by side. See ``attribution.run_component_backtests``.

    Returns:
        dict with keys: attribution_table (DataFrame), decomposition (dict),
        report_path (str), component_results (dict of engine outputs).
    """
    out_path = Path(out_dir)
    fig_dir = out_path / "figures"
    table_dir = out_path / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    stock_returns = compute_returns_panel(stock_prices)
    benchmark_returns = compute_returns_panel(benchmark_prices.to_frame("benchmark"))["benchmark"]

    engine = config.get("engine", "vectorbt")
    components_desc = "benchmark, regime_only, fundamentals_only, combined" + (
        ", geometric_overlay_only" if geometric_crash_flag is not None else ""
    ) + (
        ", governed_regime_only, governed_combined" if active_regime is not None else ""
    ) + (
        ", ichimoku_only, combined_with_ichimoku" if ichimoku_weights is not None else ""
    ) + (
        ", combined_ichimoku_tilted" if ichimoku_weights is not None and ichimoku_tilt_strength != 0.0 else ""
    )
    logger.info("Running component backtests: %s (engine=%s)", components_desc, engine)
    exposure_by_regime_cfg = config["exposure_by_regime"]
    exposure_by_regime = {
        (int(k) if str(k) != "transitional" else "transitional"): v
        for k, v in exposure_by_regime_cfg.items()
    }
    component_results = run_component_backtests(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores_by_date=scores_by_date,
        regime=regime,
        exposure_by_regime=exposure_by_regime,
        top_quantile=config.get("top_quantile", 0.2),
        exclude_bottom_quantile=config.get("exclude_bottom_quantile"),
        weighting=config.get("weighting", "equal"),
        risk_panel=risk_panel,
        exclude_riskiest_quantile=config.get("exclude_riskiest_quantile"),
        vol_target_exposure=vol_target_exposure,
        min_positions=config.get("min_positions", 5),
        max_sector_weight=config.get("max_sector_weight"),
        max_position_weight=config.get("max_position_weight"),
        beta_panel=beta_panel,
        stress_by_regime=config.get("beta_rotation", {}).get("stress_by_regime"),
        rotation_strength=config.get("beta_rotation", {}).get("rotation_strength", 1.0),
        transaction_cost_bps=config.get("transaction_cost_bps", 0.0),
        stock_prices=stock_prices,
        benchmark_prices=benchmark_prices,
        engine=engine,
        geometric_crash_flag=geometric_crash_flag,
        lag_days=config.get("lag_days", 1),
        crash_exposure_multiplier=config.get("geometric_crash_exposure_multiplier", 0.5),
        active_regime=active_regime,
        ichimoku_weights=ichimoku_weights,
        ichimoku_mode=ichimoku_mode,
        ichimoku_confirmation_floor=ichimoku_confirmation_floor,
        ichimoku_tilt_strength=ichimoku_tilt_strength,
    )

    attribution_table = compute_attribution_table(component_results, benchmark_returns=benchmark_returns)

    # The full table is the ATTRIBUTION decomposition: regime_only and
    # fundamentals_only exist to show WHERE return comes from, and are what
    # exposed the negative interaction effect between exposure timing and
    # stock selection. They are diagnostics, not deliverables.
    # `backtesting.report_components` narrows what is reported without
    # changing what is computed, so the headline shows the strategy while the
    # decomposition stays available in the CSV for when something needs
    # explaining. Unknown names are ignored rather than raising, and an empty
    # intersection falls back to the full table rather than reporting nothing.
    wanted = config.get("report_components")
    full_attribution_table = attribution_table
    if wanted:
        keep = [c for c in wanted if c in attribution_table.index]
        if keep:
            attribution_table = attribution_table.loc[keep]
        else:
            logger.warning(
                "report_components %s matched no computed component; reporting all of %s",
                wanted, list(full_attribution_table.index),
            )
    decomposition = compute_return_decomposition(component_results)
    logger.info("Return decomposition: %s", decomposition)

    returns_dict = {name: result["returns"] for name, result in component_results.items()}

    figure_paths_abs = {
        "equity_curves": plot_equity_curves(returns_dict, str(fig_dir / "equity_curves.png")),
        "drawdowns": plot_drawdowns(returns_dict, str(fig_dir / "drawdowns.png")),
        "rolling_sharpe": plot_rolling_sharpe(
            returns_dict, str(fig_dir / "rolling_sharpe.png"), window=config.get("rolling_sharpe_window", 63)
        ),
        "regime_timeline": plot_regime_timeline(
            benchmark_prices, regime, str(fig_dir / "regime_timeline.png")
        ),
        "contribution_bar": plot_contribution_bar(decomposition, str(fig_dir / "contribution_bar.png")),
    }

    latest_date = scores_by_date["date"].max()
    latest_snapshot = scores_by_date[scores_by_date["date"] == latest_date]
    figure_paths_abs["score_distribution"] = plot_fundamental_score_distribution(
        latest_snapshot, str(fig_dir / "score_distribution.png")
    )

    # Report links should be relative to the report file (which sits in out_dir).
    figure_paths_rel = {k: f"figures/{Path(v).name}" for k, v in figure_paths_abs.items()}

    # Narrowed table for the report; full decomposition always written beside it.
    attribution_table.to_csv(table_dir / "attribution_table.csv")
    full_attribution_table.to_csv(table_dir / "attribution_table_full.csv")
    with open(table_dir / "return_decomposition.json", "w") as f:
        json.dump(decomposition, f, indent=2, default=float)

    report_path = generate_markdown_report(
        attribution=attribution_table,
        decomposition=decomposition,
        figure_paths=figure_paths_rel,
        out_path=str(out_path / "backtest_report.md"),
        period_start=str(stock_returns.index.min().date()) if len(stock_returns) else None,
        period_end=str(stock_returns.index.max().date()) if len(stock_returns) else None,
    )
    logger.info("Backtest report written to %s", report_path)

    return {
        "attribution_table": attribution_table,
        "attribution_table_full": full_attribution_table,
        "decomposition": decomposition,
        "report_path": report_path,
        "component_results": component_results,
    }
