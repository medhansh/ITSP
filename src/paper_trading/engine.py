"""Paper-trading engine: advances the portfolio one trading day at a time,
rebalancing on schedule and marking to market daily.

Runs in two modes from the same code path:

  ``replay``  -- step through a historical window day by day, used to seed
                 the ledger and to check the machinery against known data.
  ``live``    -- append the latest trading day to an existing ledger, which
                 is what a scheduled daily job calls.

Both share ``step_day``, so a bug cannot appear in one mode and not the
other.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.momentum_reversal_blend import (
    build_blend, build_momentum_panel, build_reversal_panel,
    stress_from_regime, stress_from_volatility,
)
from src.backtesting.strategies import apply_beta_rotation, build_fundamental_portfolio_weights
from src.common.logging_utils import get_logger
from src.fundamental_analysis.orthogonalization import compute_rolling_beta_panel
from src.fundamental_analysis.point_in_time import run_pit_fundamental_pipeline
from src.paper_trading.broker import Ledger, SimulatedBroker

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def build_conviction_panel(cfg, stock_prices, benchmark_prices, regime):
    """Conviction panel for the technical_momentum dimension, honouring
    ``technical_signals.conviction_source``. Mirrors the pipeline's own
    dispatcher so paper trading scores identically to the backtest."""
    ts = cfg.get("technical_signals", {})
    src = ts.get("conviction_source", "blend")
    momentum = build_momentum_panel(stock_prices, kind="12_1")
    if src == "mom121":
        return momentum
    if src != "blend":
        raise ValueError(
            f"paper trading supports conviction_source 'mom121' or 'blend', got {src!r}. "
            "The 'ichimoku' source needs the OHLC panel and was retired from this build."
        )
    b = ts.get("blend", {})
    reversal = build_reversal_panel(stock_prices, kind=b.get("reversal_kind", "dist_from_ma"),
                                     ma_window=b.get("ma_window", 63))
    stress = (stress_from_volatility(benchmark_prices, stock_prices.index,
                                     vol_window=b.get("vol_window", 21))
              if b.get("stress_mode", "continuous") == "continuous"
              else stress_from_regime(regime, stock_prices.index))
    return build_blend(momentum, reversal, stress,
                       mode=b.get("stress_mode", "continuous"),
                       max_reversal_weight=b.get("max_reversal_weight", 1.0))


def target_weights_for(cfg, scores_asof, beta_row, regime_value) -> pd.Series:
    """Target weights for one rebalance: score-selected, then beta-rotated."""
    bt = cfg["backtesting"]
    sparse = build_fundamental_portfolio_weights(
        scores_asof,
        top_quantile=bt.get("top_quantile", 0.2),
        min_positions=bt.get("min_positions", 5),
        max_sector_weight=bt.get("max_sector_weight"),
        max_position_weight=bt.get("max_position_weight"),
    )
    if sparse.empty:
        return pd.Series(dtype=float)
    w = sparse.iloc[-1].dropna()
    w = w[w > 0]

    rot = bt.get("beta_rotation", {}).get("rotation_strength", 0.0)
    if rot and beta_row is not None and regime_value is not None:
        frame = pd.DataFrame([w.values], columns=w.index, index=[0])
        betas = pd.DataFrame([beta_row.reindex(w.index).values], columns=w.index, index=[0])
        rotated = apply_beta_rotation(
            frame, betas, pd.Series([regime_value], index=[0]),
            stress_by_regime=bt.get("beta_rotation", {}).get("stress_by_regime"),
            rotation_strength=rot,
        )
        w = rotated.iloc[0]
        w = w[w > 0]
    return w


def step_day(ledger: Ledger, day, prices_row, is_rebalance, broker,
             cfg=None, scores=None, beta_panel=None, regime=None,
             exec_prices_row=None) -> float:
    """Advance one trading day. Rebalances first (at ``exec_prices_row``,
    which is the NEXT day's price when available), then marks to market at
    the day's own close.

    Executing at a price the decision could not have used is the single
    easiest way to reintroduce look-ahead, so the caller passes execution
    prices explicitly rather than the engine reaching for them.
    """
    dstr = str(pd.Timestamp(day).date())
    if is_rebalance and scores is not None:
        asof = scores[scores["date"] <= pd.Timestamp(day)]
        if not asof.empty:
            last = asof["date"].max()
            asof = asof[asof["date"] == last]
            nav = ledger.cash + sum(p.market_value for p in ledger.positions.values())
            if nav <= 0:
                nav = ledger.starting_capital
            beta_row = None
            if beta_panel is not None:
                prior = beta_panel.index[beta_panel.index <= pd.Timestamp(day)]
                if len(prior):
                    beta_row = beta_panel.loc[prior[-1]]
            reg_val = None
            if regime is not None:
                prior = regime.index[regime.index <= pd.Timestamp(day)]
                if len(prior):
                    reg_val = regime.loc[prior[-1]]
            targets = target_weights_for(cfg, asof, beta_row, reg_val)
            if not targets.empty:
                fill_px = exec_prices_row if exec_prices_row is not None else prices_row
                for f in broker.execute(targets, fill_px, nav, ledger.positions, dstr):
                    ledger.apply_fill(f)
                ledger.rebalance_dates.append(dstr)
    return ledger.mark_to_market(prices_row, dstr)


def compute_statistics(ledger: Ledger, benchmark: pd.Series | None = None) -> dict:
    """Headline statistics from the NAV history."""
    if len(ledger.nav_history) < 2:
        return {}
    nav = pd.Series({pd.Timestamp(r["date"]): r["nav"] for r in ledger.nav_history}).sort_index()
    ret = nav.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)

    # Annualising a handful of days produces numbers that are arithmetically
    # correct and completely meaningless -- a 4-day window extrapolates to
    # CAGR figures in the tens of percent from noise. Below a quarter of data
    # the annualised measures are reported as None so the dashboard shows a
    # dash instead of inventing precision.
    MIN_DAYS_FOR_ANNUALISED = 63
    annualisable = len(nav) >= MIN_DAYS_FOR_ANNUALISED
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0
    vol = ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (ret.mean() * TRADING_DAYS_PER_YEAR) / vol if vol else 0.0
    dd = (nav / nav.cummax() - 1.0)
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    stats = {
        "start_date": str(nav.index[0].date()), "end_date": str(nav.index[-1].date()),
        "trading_days": int(len(nav)), "starting_capital": ledger.starting_capital,
        "annualised": bool(annualisable),
        "current_nav": float(nav.iloc[-1]), "total_return": float(total),
        "cagr": float(cagr) if annualisable else None,
        "volatility": float(vol) if annualisable else None,
        "sharpe": float(sharpe) if annualisable else None,
        "sortino": (float((ret.mean() * TRADING_DAYS_PER_YEAR) / downside)
                    if (annualisable and downside and pd.notna(downside)) else None),
        "max_drawdown": float(dd.min()), "current_drawdown": float(dd.iloc[-1]),
        "hit_rate": float((ret > 0).mean()),
        "n_rebalances": len(ledger.rebalance_dates),
        "n_fills": len(ledger.fills),
        "total_costs": float(sum(f.costs for f in ledger.fills)),
        "cash": float(ledger.cash), "n_positions": len(ledger.positions),
    }
    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.reindex(nav.index).ffill().dropna()
        if len(b) > 1:
            b_total = b.iloc[-1] / b.iloc[0] - 1.0
            b_ret = b.pct_change().dropna()
            b_vol = b_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            stats["benchmark_total_return"] = float(b_total)
            stats["benchmark_cagr"] = (float((b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1.0)
                                       if annualisable else None)
            stats["benchmark_sharpe"] = (float((b_ret.mean() * TRADING_DAYS_PER_YEAR) / b_vol)
                                         if (annualisable and b_vol) else None)
            stats["benchmark_max_drawdown"] = float((b / b.cummax() - 1.0).min())
            stats["excess_return"] = float(total - b_total)
    return stats


def interval_returns(ledger: Ledger, windows=(1, 3, 5, 10, 21, 63, 126, 252)) -> dict:
    """Portfolio return over the last N TRADING days for each window.

    Trading days, not calendar days: the NAV series only contains days the
    market was open, so counting rows is the correct unit and avoids
    weekends and holidays silently shortening a window.
    """
    if len(ledger.nav_history) < 2:
        return {}
    nav = pd.Series({pd.Timestamp(r["date"]): r["nav"] for r in ledger.nav_history}).sort_index()
    out = {}
    for w in windows:
        if len(nav) > w:
            out[f"{w}D"] = float(nav.iloc[-1] / nav.iloc[-(w + 1)] - 1.0)
    if len(nav) > 1:
        out["ALL"] = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    return out


def holdings_detail(ledger: Ledger, stock_prices: pd.DataFrame) -> list[dict]:
    """Per-holding rows for the dashboard: return since the position was
    opened, yesterday's move, and the price path since entry."""
    if not ledger.positions:
        return []
    idx = stock_prices.index
    rows = []
    for sym, pos in sorted(ledger.positions.items()):
        entry = pd.Timestamp(pos.entry_date)
        series = (stock_prices[sym].loc[stock_prices.index >= entry].dropna()
                  if sym in stock_prices.columns else pd.Series(dtype=float))
        day_change = np.nan
        if sym in stock_prices.columns:
            s = stock_prices[sym].dropna()
            if len(s) > 1:
                day_change = float(s.iloc[-1] / s.iloc[-2] - 1.0)
        held_days = int((idx >= entry).sum())
        rows.append({
            "symbol": sym, "quantity": pos.quantity, "avg_price": pos.avg_price,
            "last_price": pos.last_price, "market_value": pos.market_value,
            "weight": 0.0,  # filled by caller once NAV is known
            "unrealized_pnl": pos.unrealized_pnl,
            "return_since_entry": pos.return_since_entry,
            "entry_date": pos.entry_date, "held_trading_days": held_days,
            "day_change": None if pd.isna(day_change) else day_change,
            "path": [round(float(v), 2) for v in series.tolist()[-260:]],
        })
    nav = ledger.cash + sum(p.market_value for p in ledger.positions.values())
    if nav > 0:
        for r in rows:
            r["weight"] = r["market_value"] / nav
    return rows
