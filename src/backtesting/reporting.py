"""Assemble a single Markdown backtest report from an attribution table,
a return decomposition, and a set of already-generated figure files.

Markdown (not HTML/PDF) is used so the report renders natively on GitHub,
in most editors, and can be converted to PDF/HTML later with pandoc if
needed — it's the lowest-friction format for a report that lives in a repo.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

METRIC_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "annualized_volatility": "Annualized volatility",
    "sharpe_ratio": "Sharpe ratio",
    "sortino_ratio": "Sortino ratio",
    "max_drawdown": "Max drawdown",
    "calmar_ratio": "Calmar ratio",
    "hit_rate": "Hit rate (% positive days)",
    "alpha_annualized": "Alpha (annualized)",
    "beta": "Beta",
    "excess_cagr_vs_benchmark": "Excess CAGR vs. benchmark",
}

PCT_METRICS = {
    "total_return", "cagr", "annualized_volatility", "max_drawdown",
    "hit_rate", "alpha_annualized", "excess_cagr_vs_benchmark",
}


def _fmt(metric: str, value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if metric in PCT_METRICS:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def _attribution_table_markdown(attribution: pd.DataFrame) -> str:
    column_labels = [METRIC_LABELS.get(col, col) for col in attribution.columns]
    row_labels = [i.replace("_", " ").title() for i in attribution.index]
    header = "| Component | " + " | ".join(column_labels) + " |"
    sep = "|---" * (len(column_labels) + 1) + "|"
    rows = []
    for row_label, (_, row) in zip(row_labels, attribution.iterrows()):
        formatted = [_fmt(col, row[col]) for col in attribution.columns]
        rows.append(f"| {row_label} | " + " | ".join(formatted) + " |")
    return "\n".join([header, sep] + rows)


def _decomposition_markdown(decomposition: dict[str, float]) -> str:
    rows = [
        ("Benchmark CAGR", decomposition["benchmark_cagr"], "cagr"),
        ("Fundamentals-only CAGR", decomposition["fundamentals_only_cagr"], "cagr"),
        ("Regime-only CAGR", decomposition["regime_only_cagr"], "cagr"),
    ]
    if "geometric_overlay_cagr" in decomposition:
        rows.append(("Geometric-overlay-only CAGR", decomposition["geometric_overlay_cagr"], "cagr"))
    rows += [
        ("Combined CAGR", decomposition["combined_cagr"], "cagr"),
        ("— Combined excess over benchmark", decomposition["combined_excess_cagr"], "excess_cagr_vs_benchmark"),
        ("— — Fundamentals contribution", decomposition["fundamentals_contribution"], "excess_cagr_vs_benchmark"),
        ("— — Regime contribution", decomposition["regime_contribution"], "excess_cagr_vs_benchmark"),
        ("— — Interaction effect", decomposition["interaction_effect"], "excess_cagr_vs_benchmark"),
    ]
    if "geometric_overlay_contribution" in decomposition:
        rows.append((
            "— Geometric overlay contribution (standalone, informational)",
            decomposition["geometric_overlay_contribution"], "excess_cagr_vs_benchmark",
        ))
    header = "| Line item | Value |\n|---|---|"
    lines = [f"| {label} | {_fmt(metric_key, value)} |" for label, value, metric_key in rows]
    return "\n".join([header] + lines)


def generate_markdown_report(
    attribution: pd.DataFrame,
    decomposition: dict[str, float],
    figure_paths: dict[str, str],
    out_path: str,
    universe_name: str = "NIFTY500",
    period_start: str | None = None,
    period_end: str | None = None,
) -> str:
    """Write the full Markdown report to ``out_path``. Figure paths should be
    relative to the report file's own directory so links work when the
    ``reports/`` folder is moved or shared as a whole.
    """
    out_path = Path(out_path)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    period = f"{period_start} to {period_end}" if period_start and period_end else "full backtest sample"
    has_geometric = "geometric_overlay_only" in attribution.index

    summary_text = (
        "Four strategies are compared: a fully-invested **benchmark**, a "
        "**regime-only** market-timing overlay (no stock selection), a "
        "**fundamentals-only** stock-selection strategy (always fully invested), "
        "and the **combined** strategy (fundamentals selection, exposure scaled "
        "by regime). See `docs/backtesting_spec.md` for exact construction and "
        "the caveats on the attribution methodology below."
    )
    if has_geometric:
        summary_text += (
            " A fifth, **geometric-overlay-only** strategy is also shown: "
            "exposure scaled purely by the geometric wedge-product crash-risk "
            "flag (`regime_detection/geometric_signal.py`), computed and applied "
            "*completely independently* of the GMM/KMeans/HMM regime label used "
            "everywhere above — it never influenced the regime model's fit or "
            "prediction. It is also applied as an additional exposure cut on top "
            "of `combined` (see the decomposition table's \"Geometric overlay "
            "contribution\" line, which is informational/additive-only and not "
            "folded into the interaction-effect math)."
        )

    sections = [
        f"# Backtest Report — {universe_name}",
        f"*Generated {generated_at} · period: {period}*",
        "",
        "## Summary",
        "",
        summary_text,
        "",
        "## Performance by component",
        "",
        _attribution_table_markdown(attribution),
        "",
        f"![Equity curves]({figure_paths['equity_curves']})",
        "",
        f"![Drawdowns]({figure_paths['drawdowns']})",
        "",
        f"![Rolling Sharpe]({figure_paths['rolling_sharpe']})",
        "",
        "## Regime detection over the backtest period",
        "",
        f"![Regime timeline]({figure_paths['regime_timeline']})",
        "",
        "## Fundamental score distribution (latest snapshot)",
        "",
        f"![Fundamental score distribution]({figure_paths['score_distribution']})",
        "",
        "## Return attribution: individual contribution of each component",
        "",
        "Additive decomposition of the combined strategy's CAGR spread over the "
        "benchmark. This is an approximation (see `docs/backtesting_spec.md` — "
        "compounding means contributions aren't perfectly additive), not an "
        "exact Brinson-style attribution, but is directionally reliable for "
        "\"roughly how much did each signal add.\"",
        "",
        _decomposition_markdown(decomposition),
        "",
        f"![Contribution breakdown]({figure_paths['contribution_bar']})",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections))
    return str(out_path)
