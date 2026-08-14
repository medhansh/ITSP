"""Standard performance metrics computed from a daily returns series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    running_max = curve.cummax()
    return curve / running_max - 1.0


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    curve = equity_curve(returns)
    if len(curve) < 2 or curve.iloc[-1] <= 0:
        return np.nan
    n_years = len(curve) / periods_per_year
    if n_years <= 0:
        return np.nan
    return curve.iloc[-1] ** (1.0 / n_years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_dev = downside.std()
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return (excess.mean() / downside_dev) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return dd.min() if len(dd) else np.nan


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr(returns, periods_per_year) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    nonzero = returns.dropna()
    if len(nonzero) == 0:
        return np.nan
    return (nonzero > 0).mean()


def alpha_beta(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> tuple[float, float]:
    """OLS alpha (annualized) and beta of ``returns`` vs. ``benchmark_returns``."""
    df = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(df) < 2:
        return np.nan, np.nan
    y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
    var_x = np.var(x)
    if var_x == 0:
        return np.nan, np.nan
    beta = np.cov(x, y, ddof=1)[0, 1] / var_x
    alpha_daily = y.mean() - beta * x.mean()
    alpha_annualized = alpha_daily * periods_per_year
    return alpha_annualized, beta


def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """One-stop dict of every metric above for a single returns series."""
    curve = equity_curve(returns)
    summary = {
        "total_return": curve.iloc[-1] - 1.0 if len(curve) else np.nan,
        "cagr": cagr(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
    }
    if benchmark_returns is not None:
        alpha, beta = alpha_beta(returns, benchmark_returns, periods_per_year)
        summary["alpha_annualized"] = alpha
        summary["beta"] = beta
    return summary
