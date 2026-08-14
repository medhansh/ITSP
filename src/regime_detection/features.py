"""Feature engineering for market-regime detection.

Core functions operate on a DataFrame indexed by trading date with a ``close``
column (the index/benchmark level, e.g. NIFTY 500 or NIFTY 50), and optionally
join in breadth, volatility-index, and — new in this pass — open/high/low/
volume data for range-based volatility (Parkinson/Garman-Klass) and
volume-derived features (volume z-score, OBV trend). Every function returns
columns that can be concatenated into a single feature matrix by
``build_feature_matrix``, and every non-``close`` input is independently
optional: the pipeline degrades gracefully to whatever's actually available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_returns(prices: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Rolling (non-overlapping-lookback) log returns over each window in days."""
    log_price = np.log(prices)
    out = {}
    for w in windows:
        out[f"return_{w}d"] = log_price.diff(w)
    return pd.DataFrame(out, index=prices.index)


def compute_realized_vol(prices: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Annualized rolling realized volatility of daily log returns."""
    daily_log_ret = np.log(prices).diff()
    out = {}
    for w in windows:
        out[f"realized_vol_{w}d"] = daily_log_ret.rolling(w).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    return pd.DataFrame(out, index=prices.index)


def compute_drawdown(prices: pd.Series) -> pd.DataFrame:
    """Distance from the trailing all-time (or since-start) high."""
    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    return pd.DataFrame({"drawdown": drawdown}, index=prices.index)


def compute_breadth(
    advances: pd.Series, declines: pd.Series, windows: list[int]
) -> pd.DataFrame:
    """Market-breadth features from NSE advance/decline counts.

    advances / declines: daily count of NIFTY500 constituents that advanced /
    declined. Smoothed advance-decline ratio is a classic regime signal —
    persistently >1 suggests broad participation (healthy trend), <1 suggests
    a narrow or deteriorating market even if the index itself is flat/up.
    """
    ad_ratio = advances / declines.replace(0, np.nan)
    out = {}
    for w in windows:
        out[f"ad_ratio_{w}d_avg"] = ad_ratio.rolling(w).mean()
    return pd.DataFrame(out, index=advances.index)


def compute_vix_features(vix: pd.Series) -> pd.DataFrame:
    """India VIX level and short-term change — a direct, forward-looking fear gauge."""
    return pd.DataFrame(
        {
            "vix_level": vix,
            "vix_change_5d": vix.diff(5),
            "vix_zscore_1y": (vix - vix.rolling(TRADING_DAYS_PER_YEAR).mean())
            / vix.rolling(TRADING_DAYS_PER_YEAR).std(),
        },
        index=vix.index,
    )


def compute_parkinson_volatility(high: pd.Series, low: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Parkinson (1980) range-based volatility estimator: uses the daily
    high-low range instead of just close-to-close returns. More statistically
    efficient than ``compute_realized_vol`` for the same window (close-only
    realized vol throws away the intraday range entirely, and Parkinson's
    variance has ~5x lower theoretical variance than the close-to-close
    estimator under a GBM assumption) — but it only captures range, not
    close-to-close drift/gaps, so it's a complement to ``compute_realized_vol``,
    not a replacement. Needs only high/low (no open), so it degrades
    gracefully when open isn't available (see ``compute_garman_klass_volatility``
    below for the richer OHLC estimator).
    """
    log_hl = np.log(high / low)
    daily_var = (log_hl ** 2) / (4.0 * np.log(2.0))
    out = {}
    for w in windows:
        out[f"parkinson_vol_{w}d"] = np.sqrt(daily_var.rolling(w).mean() * TRADING_DAYS_PER_YEAR)
    return pd.DataFrame(out, index=high.index)


def compute_garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, windows: list[int]
) -> pd.DataFrame:
    """Garman-Klass (1980) range-based volatility estimator: extends Parkinson
    with the open-close term, capturing overnight gaps as well as intraday
    range. Needs the full OHLC — see ``compute_parkinson_volatility`` for the
    high/low-only fallback when open isn't available.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    daily_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    out = {}
    for w in windows:
        # Clip negative rolling-mean variance (can happen with small/degenerate
        # windows or noisy data — the per-day GK term isn't itself guaranteed
        # non-negative) rather than propagating NaN from sqrt of a negative.
        rolling_var = daily_var.rolling(w).mean().clip(lower=0.0)
        out[f"garman_klass_vol_{w}d"] = np.sqrt(rolling_var * TRADING_DAYS_PER_YEAR)
    return pd.DataFrame(out, index=close.index)


def compute_volume_features(
    close: pd.Series, volume: pd.Series, zscore_windows: list[int], obv_trend_windows: list[int]
) -> pd.DataFrame:
    """Volume-derived features — participation/conviction behind a price move,
    which close-only features can't see at all (the same % move on rising vs.
    collapsing volume is a meaningfully different regime).

    - ``volume_zscore_{w}d``: today's volume vs. its own trailing distribution
      (rolling mean/std over window w) — flags volume spikes (capitulation,
      breakout, panic) independent of price direction.
    - ``obv_trend_{w}d``: On-Balance Volume (cumulative signed daily volume —
      Granville, 1963) change over window w, normalized by that window's total
      volume so it's scale-free and bounded in [-1, 1]: +1 means every unit of
      volume in the window occurred on an up day (maximal buying-volume
      pressure), -1 the reverse, 0 means no net directional volume pressure.
      Normalizing this way (rather than using OBV's raw level, which is an
      arbitrary-scale cumulative sum) keeps it comparable across symbols/time.
    """
    daily_sign = np.sign(close.diff()).fillna(0.0)
    obv = (daily_sign * volume).cumsum()

    out = {}
    for w in zscore_windows:
        rolling_mean = volume.rolling(w).mean()
        rolling_std = volume.rolling(w).std()
        out[f"volume_zscore_{w}d"] = (volume - rolling_mean) / rolling_std.replace(0, np.nan)
    for w in obv_trend_windows:
        window_volume_sum = volume.rolling(w).sum().replace(0, np.nan)
        out[f"obv_trend_{w}d"] = obv.diff(w) / window_volume_sum
    return pd.DataFrame(out, index=close.index)


def build_feature_matrix(
    prices: pd.Series,
    return_windows: list[int],
    vol_windows: list[int],
    advances: pd.Series | None = None,
    declines: pd.Series | None = None,
    breadth_windows: list[int] | None = None,
    vix: pd.Series | None = None,
    open_: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
    range_vol_windows: list[int] | None = None,
    volume_zscore_windows: list[int] | None = None,
    obv_trend_windows: list[int] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Assemble the regime-detection **clustering** feature matrix — i.e.
    everything that actually gets fit/predicted by the GMM/KMeans/HMM model
    in ``models.py``. This is deliberately the only place that feature set
    is assembled, so anything NOT plumbed through here structurally cannot
    influence the regime label.

    ``prices`` should be the broad-market index level (NIFTY500, falling back
    to NIFTY50) so regimes reflect market-wide conditions rather than a single
    stock. Breadth and VIX inputs are optional — the pipeline degrades
    gracefully to price-derived features only, which is useful in the sandbox
    or for quick experiments before breadth/VIX feeds are wired up.

    ``open_``/``high``/``low``/``volume`` (all optional, independently):
    everything above this point in the docstring is derived from ``close``
    alone. If ``high``+``low`` are supplied, Parkinson range volatility is
    added; if ``open_`` is *also* supplied, the richer Garman-Klass estimator
    is added too (see ``compute_parkinson_volatility`` /
    ``compute_garman_klass_volatility``). If ``volume`` is supplied, volume
    z-score and OBV-trend features are added (see ``compute_volume_features``).
    Each is independently optional and skipped (not an error) if its inputs
    aren't available — same graceful-degradation convention as breadth/VIX.
    ``range_vol_windows``/``volume_zscore_windows``/``obv_trend_windows``
    default to ``vol_windows`` if not given.

    NOTE on the geometric wedge-product crash-risk signal
    (``geometric_signal.py``): it is intentionally NOT assembled here. By
    explicit design decision, it is computed separately in
    ``pipeline.run_pipeline`` (after the model has already been fit/predicted)
    and used only as a standalone post-hoc exposure overlay in
    ``backtesting/strategies.py`` — see that pipeline's docstring and
    ``docs/regime_detection_spec.md``'s "Geometric wedge-product crash-risk
    signal" section for why: it never gets a chance to influence the GMM/
    KMeans/HMM regime label, by construction, not just by convention.
    """
    parts = [
        compute_returns(prices, return_windows),
        compute_realized_vol(prices, vol_windows),
        compute_drawdown(prices),
    ]
    if advances is not None and declines is not None:
        parts.append(compute_breadth(advances, declines, breadth_windows or [21]))
    if vix is not None:
        parts.append(compute_vix_features(vix))
    if high is not None and low is not None:
        rv_windows = range_vol_windows or vol_windows
        parts.append(compute_parkinson_volatility(high, low, rv_windows))
        if open_ is not None:
            parts.append(compute_garman_klass_volatility(open_, high, low, prices, rv_windows))
    if volume is not None:
        parts.append(
            compute_volume_features(
                prices, volume,
                zscore_windows=volume_zscore_windows or vol_windows,
                obv_trend_windows=obv_trend_windows or vol_windows,
            )
        )

    features = pd.concat(parts, axis=1)
    if dropna:
        features = features.dropna()
    return features
