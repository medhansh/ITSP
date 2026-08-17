"""Two poster-ready charts: benchmark vs. the strategy only.

Deliberately separate from ``plotting.py``'s report charts rather than
reusing them directly — a poster needs large fonts, thick lines, and
exactly two series with nothing else competing for attention, none of
which are the right defaults for a multi-panel report figure. Both
functions here call into ``plotting.py``'s existing math (equity curve,
rolling Sharpe) so the numbers are identical to the report; only the
presentation differs.

Usage from a backtest result:

    from src.backtesting.poster_charts import plot_poster_charts
    plot_poster_charts(result["component_results"], out_dir="reports/poster")

``result`` is whatever ``run_backtest_pipeline`` returned — this reads
``component_results[name]["returns"]`` directly, so it works even when
``report_components`` has trimmed the attribution table down to just the
two series shown on the poster.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCHMARK_KEY = "benchmark"
STRATEGY_KEY = "fundamentals_beta_rotated"

BENCHMARK_COLOR = "#000000"
STRATEGY_COLOR = "#009E73"  # same green as the report charts, kept consistent

POSTER_RC = {
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 20,
    "axes.linewidth": 1.4,
}


def _get_returns(component_results: dict, key: str, label: str) -> pd.Series:
    if key not in component_results:
        raise KeyError(
            f"'{key}' not found in component_results (have: {list(component_results)}). "
            f"Run the backtest with beta_panel supplied so {label} is computed — "
            f"report_components only affects which rows are PRINTED, not which are computed."
        )
    return component_results[key]["returns"]


def plot_poster_equity_curve(
    component_results: dict,
    out_path: str = "reports/poster/equity_curve.png",
    title: str = "Strategy vs. Benchmark",
) -> str:
    """Growth of ₹1, benchmark vs. the strategy, log scale.

    Log scale because the comparison spans over a decade — on a linear
    axis a strategy that started compounding faster early on visually
    dwarfs everything else even where the more recent, more relevant gap
    has narrowed.
    """
    bench = _get_returns(component_results, BENCHMARK_KEY, "the benchmark")
    strat = _get_returns(component_results, STRATEGY_KEY, "the strategy")

    with plt.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
        for returns, color, label in (
            (bench, BENCHMARK_COLOR, "Benchmark"),
            (strat, STRATEGY_COLOR, "Strategy"),
        ):
            curve = (1.0 + returns.fillna(0.0)).cumprod()
            ax.plot(curve.index, curve.values, color=color, linewidth=3.2, label=label)

        ax.set_yscale("log")
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(title, fontweight="bold", pad=16)
        ax.set_ylabel("Growth of ₹1 (log scale)")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_poster_rolling_sharpe(
    component_results: dict,
    out_path: str = "reports/poster/rolling_sharpe.png",
    window: int = 126,
    periods_per_year: int = 252,
    title: str | None = None,
) -> str:
    """Rolling Sharpe, benchmark vs. the strategy.

    Window defaults to 126 trading days (~6 months) rather than the
    report's 63-day window: a poster is read from a distance in a few
    seconds, so the line needs to read as one clear trend rather than
    short-window noise.
    """
    bench = _get_returns(component_results, BENCHMARK_KEY, "the benchmark")
    strat = _get_returns(component_results, STRATEGY_KEY, "the strategy")

    with plt.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
        for returns, color, label in (
            (bench, BENCHMARK_COLOR, "Benchmark"),
            (strat, STRATEGY_COLOR, "Strategy"),
        ):
            roll_mean = returns.rolling(window).mean()
            roll_std = returns.rolling(window).std()
            roll_sharpe = (roll_mean / roll_std) * np.sqrt(periods_per_year)
            ax.plot(roll_sharpe.index, roll_sharpe.values, color=color, linewidth=3.0, label=label)

        ax.axhline(0, color="#888888", linewidth=1.4, linestyle="--")
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        months = max(round(window / 21), 1)
        ax.set_title(title or f"Rolling {months}-Month Sharpe Ratio", fontweight="bold", pad=16)
        ax.set_ylabel("Sharpe ratio")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_poster_charts(component_results: dict, out_dir: str = "reports/poster") -> dict[str, str]:
    """Both charts in one call. Returns the two file paths."""
    out_dir = str(out_dir).rstrip("/")
    return {
        "equity_curve": plot_poster_equity_curve(component_results, f"{out_dir}/equity_curve.png"),
        "rolling_sharpe": plot_poster_rolling_sharpe(component_results, f"{out_dir}/rolling_sharpe.png"),
    }
