"""Matplotlib chart generation for backtest reports.

Design choices (kept simple and explicit rather than pulling in a plotting
framework, since this only needs to run headless and save PNGs):
  - A fixed, colorblind-safe categorical palette (Okabe-Ito) is assigned to
    each strategy component *by name*, in the same order everywhere, so
    "combined" is always the same color across every figure in a report.
  - One y-axis per chart — never a dual-axis overlay of differently-scaled
    series (see docs/backtesting_spec.md's plotting notes).
  - Regime shading uses a single sequential ramp (light -> dark) since regime
    labels are ordered by volatility (an actual magnitude), not by identity.
  - Recessive gridlines, direct end-of-line labels instead of a legend where
    there's only 1-2 series worth naming inline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.metrics import drawdown_series

# Okabe-Ito: colorblind-safe categorical palette. Fixed assignment, never cycled.
COMPONENT_COLORS = {
    "benchmark": "#000000",          # black — the reference line
    "regime_only": "#E69F00",        # orange
    "fundamentals_only": "#0072B2",  # blue
    "combined": "#009E73",           # green
    "fundamentals_beta_rotated": "#009E73",  # green — THE strategy in the lean build
    "geometric_overlay_only": "#CC79A7",  # reddish-purple — only appears when the signal is enabled
    "trend": "#D55E00",              # vermillion — scripts/run_technical_backtest.py
    "mean_reversion": "#56B4E9",     # sky blue — scripts/run_technical_backtest.py
    "benchmark_buy_hold": "#000000",
    "equal_weight_buy_hold": "#999999",
}
GRID_COLOR = "#DDDDDD"
REGIME_CMAP = "YlOrRd"  # sequential: light (calm) -> dark (stressed)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_equity_curves(returns_dict: dict[str, pd.Series], out_path: str, title: str = "Equity Curve") -> str:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for name, returns in returns_dict.items():
        curve = (1.0 + returns.fillna(0.0)).cumprod()
        color = COMPONENT_COLORS.get(name, "#999999")
        ax.plot(curve.index, curve.values, label=name.replace("_", " ").title(), color=color, linewidth=2)
    _style_axes(ax)
    ax.set_title(title)
    ax.set_ylabel("Growth of ₹1")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_drawdowns(returns_dict: dict[str, pd.Series], out_path: str, title: str = "Drawdown") -> str:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    for name, returns in returns_dict.items():
        dd = drawdown_series(returns) * 100
        color = COMPONENT_COLORS.get(name, "#999999")
        ax.plot(dd.index, dd.values, color=color, linewidth=1.5, label=name.replace("_", " ").title())
        ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.12)
    _style_axes(ax)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series], out_path: str, window: int = 63, periods_per_year: int = 252
) -> str:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    for name, returns in returns_dict.items():
        roll_mean = returns.rolling(window).mean()
        roll_std = returns.rolling(window).std()
        roll_sharpe = (roll_mean / roll_std) * np.sqrt(periods_per_year)
        color = COMPONENT_COLORS.get(name, "#999999")
        ax.plot(roll_sharpe.index, roll_sharpe.values, color=color, linewidth=1.5, label=name.replace("_", " ").title())
    ax.axhline(0, color="#888888", linewidth=1, linestyle="--")
    _style_axes(ax)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_regime_timeline(prices: pd.Series, regime: pd.Series, out_path: str, title: str = "Price & Detected Regime") -> str:
    """Price line with the background shaded by detected regime (light = calm,
    dark = stressed), so regime shifts are visible against actual price action."""
    common = prices.index.intersection(regime.index)
    prices, regime = prices.loc[common], regime.loc[common]

    n_regimes = int(regime.max()) + 1
    cmap = plt.get_cmap(REGIME_CMAP)
    regime_colors = [cmap(0.15 + 0.65 * i / max(n_regimes - 1, 1)) for i in range(n_regimes)]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    # Shade contiguous regime blocks.
    block_start = regime.index[0]
    current = regime.iloc[0]
    for i in range(1, len(regime)):
        if regime.iloc[i] != current:
            ax.axvspan(block_start, regime.index[i], color=regime_colors[int(current)], alpha=0.35, linewidth=0)
            block_start = regime.index[i]
            current = regime.iloc[i]
    ax.axvspan(block_start, regime.index[-1], color=regime_colors[int(current)], alpha=0.35, linewidth=0)

    ax.plot(prices.index, prices.values, color="#000000", linewidth=1.3)
    _style_axes(ax)
    ax.set_title(title)
    ax.set_ylabel("Price")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=regime_colors[i], alpha=0.6, label=f"Regime {i}")
        for i in range(n_regimes)
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=n_regimes)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_contribution_bar(decomposition: dict[str, float], out_path: str, title: str = "Return Attribution (CAGR)") -> str:
    labels = ["Fundamentals\ncontribution", "Regime\ncontribution", "Interaction\neffect", "Combined\nexcess (total)"]
    values = [
        decomposition["fundamentals_contribution"],
        decomposition["regime_contribution"],
        decomposition["interaction_effect"],
        decomposition["combined_excess_cagr"],
    ]
    colors = [COMPONENT_COLORS["fundamentals_only"], COMPONENT_COLORS["regime_only"], "#999999", COMPONENT_COLORS["combined"]]

    if "geometric_overlay_contribution" in decomposition:
        labels.insert(2, "Geometric overlay\ncontribution")
        values.insert(2, decomposition["geometric_overlay_contribution"])
        colors.insert(2, COMPONENT_COLORS.get("geometric_overlay_only", "#CC79A7"))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(labels, [v * 100 for v in values], color=colors, width=0.6, zorder=3)
    ax.axhline(0, color="#444444", linewidth=1)
    _style_axes(ax)
    ax.set_title(title)
    ax.set_ylabel("Contribution to CAGR (percentage points)")
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v * 100:+.2f}pp",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4 if v >= 0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fundamental_score_distribution(scores_snapshot: pd.DataFrame, out_path: str, title: str = "Composite Fundamental Score Distribution") -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(
        scores_snapshot["composite_score"].dropna(),
        bins=30,
        color=COMPONENT_COLORS["fundamentals_only"],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    _style_axes(ax)
    ax.set_title(title)
    ax.set_xlabel("Composite score (sector-relative z, higher = better)")
    ax.set_ylabel("Number of stocks")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
