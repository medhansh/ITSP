"""Adaptive-period Ichimoku Cloud, full implementation — true OHLC (not a
close-derived high/low proxy), a genuinely forward-shifted cloud (even with
a day-varying kijun_period), and Chikou Span confirmation. The dispersion-
score magnitude from ``technical_signals.py`` modulates Ichimoku's lookback
periods day by day, instead of driving a hard entry/exit threshold directly.

**Why this exists**: the threshold-based dispersion strategy in
``technical_signals.py`` under-exposed the portfolio relative to a
buy-and-hold benchmark (beta ~0.83 in initial testing) — spending real time
completely flat while waiting for a confirmed threshold crossing costs
return in an up-trending market, independent of whether the "in" periods
were individually good calls. This module tries a different way of using
the same underlying signal: as a continuous *dial* on an existing
trend-following indicator's timeframe, rather than a binary in/out gate.

**Status: experimental, unvalidated.** Same caveat as everywhere else new
in this project — implemented and unit-tested for mechanical correctness,
zero claim about real-data profitability. In particular, there are two
competing, genuinely untested hypotheses about which *direction* the
adaptive mapping should go (see ``compute_adaptive_periods``'s
``direction`` parameter) — this module implements both, plus a static
(non-adaptive) baseline, specifically so they can be compared head to head
rather than one being asserted as correct. See
``scripts/run_adaptive_ichimoku_backtest.py``.

**The three components of "proper" Ichimoku, all implemented here:**

1. **True high/low.** Tenkan-sen/Kijun-sen/Senkou-Span-B are
   ``(period-high + period-low) / 2`` using actual intraday high/low (via
   ``yfinance_fetcher.fetch_price_panel_ohlc`` / a long-format OHLC CSV —
   see ``scripts/_common_cli.py``'s ``load_prices_ohlc``), not a
   close-only proxy.

2. **A genuinely forward-shifted cloud.** Textbook Ichimoku plots Senkou
   Span A/B ``kijun_period`` days *ahead* — the cloud visible on day D was
   calculated using data up to day ``D - kijun_period``. With a FIXED
   kijun_period this is a trivial ``.shift()``. With an ADAPTIVE (day-
   varying) kijun_period, "shift forward by kijun_period" needs an actual
   definition, not a shortcut: each day i's Senkou values are *scattered*
   forward to target position ``i + round(kijun_period[i])`` (the offset
   known and fixed at the moment of calculation — the only causal choice).
   Two consequences that are inherent to what an adaptive forward
   projection means, not approximations chosen to save effort:
     - If kijun_period shrinks over time, multiple days' projections can
       land on the same future target day; the most recently computed one
       (larger source index i) wins, since it reflects newer information.
     - If kijun_period grows over time, some future days receive no direct
       projection (a temporal "gap" between two scattered values); these
       are forward-filled from the most recent scattered value, exactly
       like a real Ichimoku cloud is a continuous line/area, not a set of
       isolated points.
   See ``_scatter_forward``.

3. **Chikou Span confirmation.** Textbook Ichimoku plots today's close
   ``kijun_period`` days *behind*, as a third confirmation line — compared
   against price action from that earlier point. This is a backward-only
   lookup (``close[i] vs close[i - kijun_period[i]]``), well-defined even
   with an adaptive period (no ambiguity the way the forward case has,
   since it's indexing into the already-known past). ``generate_ichimoku_signal``
   requires all three confirmations (cloud position, Tenkan/Kijun
   relationship, Chikou vs lagged price) to agree before taking a position —
   the standard "triple confirmation" reading of Ichimoku, not a
   single-filter simplification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.technical_signals import compute_signed_normalized_score
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_TENKAN = 9
DEFAULT_BASE_KIJUN = 26
DEFAULT_BASE_SENKOU_B = 52


def compute_adaptive_periods(
    magnitude: pd.Series,
    base_tenkan: int = DEFAULT_BASE_TENKAN,
    base_kijun: int = DEFAULT_BASE_KIJUN,
    base_senkou_b: int = DEFAULT_BASE_SENKOU_B,
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    direction: str = "shrink_when_low",
) -> pd.DataFrame:
    """Day-by-day Tenkan/Kijun/Senkou-B periods, scaled by the dispersion
    score's magnitude (``technical_signals.compute_signed_normalized_score``'s
    ``magnitude`` column, in [0, 1] — 0 = historically quiet dispersion,
    1 = historically extreme).

        scale(magnitude) = scale_min + (scale_max - scale_min) * g(magnitude)
        period(t) = round(base_period * scale(magnitude(t)))

    ``direction`` picks ``g`` — two competing, untested hypotheses:

    - ``"shrink_when_high"``: g = 1 - magnitude. Periods shrink (become more
      reactive) once a trend is already confirmed/historically strong, on
      the theory that you want to ride a confirmed trend closely rather
      than lag it with a slow-moving indicator. This is the logic behind
      Kaufman's Efficiency-Ratio-driven adaptive moving averages (KAMA).
    - ``"shrink_when_low"`` (default): g = magnitude. Periods shrink during
      QUIET/ranging periods (more reactive, catches the next breakout
      earlier) and lengthen once a trend is already confirmed (smoother,
      less prone to whipsaw once you're already riding a strong move). This
      directly targets the specific failure mode that motivated this
      module: a fixed-period indicator's confirmation lag costs the early
      part of every move.

    Neither has been validated against real data as of writing — run both
    (``scripts/run_adaptive_ichimoku_backtest.py --variant all``) and
    compare against the ``static`` (non-adaptive) baseline before trusting
    either. Relative ordering (tenkan < kijun < senkou_b) is preserved
    automatically since all three scale by the same factor from bases that
    already satisfy it (9 < 26 < 52).

    Returns a DataFrame with integer-valued (float dtype, NaN before
    ``magnitude`` is itself defined) columns ``tenkan_period``,
    ``kijun_period``, ``senkou_b_period``, floored at 1.
    """
    if direction not in ("shrink_when_high", "shrink_when_low"):
        raise ValueError(f"direction must be 'shrink_when_high' or 'shrink_when_low', got {direction!r}")
    if scale_max <= scale_min:
        raise ValueError(f"scale_max ({scale_max}) must be > scale_min ({scale_min})")

    g = (1.0 - magnitude) if direction == "shrink_when_high" else magnitude
    scale = scale_min + (scale_max - scale_min) * g

    def _period(base: int) -> pd.Series:
        return (base * scale).round().clip(lower=1)

    return pd.DataFrame(
        {
            "tenkan_period": _period(base_tenkan),
            "kijun_period": _period(base_kijun),
            "senkou_b_period": _period(base_senkou_b),
        },
        index=magnitude.index,
    )


def _variable_window_extreme(series: pd.Series, windows: pd.Series, agg: str) -> pd.Series:
    """Rolling max/min of ``series`` with a window length that varies by
    day (``windows``, integer-valued). Sequential (not vectorized): pandas'
    ``.rolling()`` requires a fixed window, and an adaptive indicator by
    definition doesn't have one — same "explicit loop for auditability over
    cleverness" convention as ``geometric_signal.calculate_wedge_volume``
    and ``technical_signals.generate_signal`` elsewhere in this project.
    """
    values = series.values
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        w = windows.iloc[i]
        if pd.isna(w):
            continue
        w = int(w)
        start = i - w + 1
        if start < 0:
            continue
        window_slice = values[start : i + 1]
        if np.isnan(window_slice).any():
            continue
        out[i] = np.nanmax(window_slice) if agg == "max" else np.nanmin(window_slice)
    return pd.Series(out, index=series.index)


def _variable_window_hl_midpoint(high: pd.Series, low: pd.Series, windows: pd.Series) -> pd.Series:
    """(rolling_max(high, w) + rolling_min(low, w)) / 2, w varying by day —
    the actual Tenkan-sen/Kijun-sen/Senkou-Span-B formula using true
    high/low (not a same-series close-derived proxy)."""
    period_high = _variable_window_extreme(high, windows, "max")
    period_low = _variable_window_extreme(low, windows, "min")
    return (period_high + period_low) / 2.0


def _scatter_forward(raw_values: pd.Series, offsets: pd.Series) -> pd.Series:
    """Project ``raw_values[i]`` (computed causally as of day i) forward to
    target position ``i + round(offsets[i])`` — the mechanism behind
    Ichimoku's "leading" Senkou spans, generalized to a day-varying offset.
    See module docstring point 2 for exactly what happens on
    collision (multiple i's mapping to the same target — most recent wins,
    since the loop runs i in increasing order) and gaps (forward-filled).
    Every output value at position T is built only from ``raw_values[i]``
    with i <= T, so this is causal/no-lookahead by construction regardless
    of how ``offsets`` behaves.
    """
    n = len(raw_values)
    values = raw_values.values
    scattered = np.full(n, np.nan)
    for i in range(n):
        v = values[i]
        off = offsets.iloc[i]
        if np.isnan(v) or pd.isna(off):
            continue
        target = i + int(round(off))
        if 0 <= target < n:
            scattered[target] = v  # larger i overwrites -> "most recent wins" on collision
    return pd.Series(scattered, index=raw_values.index).ffill()


def _variable_lag_lookup(series: pd.Series, lags: pd.Series) -> pd.Series:
    """``series[i - round(lags[i])]`` for each i — Chikou Span's reference
    point (price from ``kijun_period`` days ago). Purely backward-looking,
    so (unlike the forward Senkou projection) this has no collision/gap
    ambiguity: it's a direct lookup into already-known history.
    """
    n = len(series)
    values = series.values
    out = np.full(n, np.nan)
    for i in range(n):
        lag = lags.iloc[i]
        if pd.isna(lag):
            continue
        source = i - int(round(lag))
        if source >= 0:
            out[i] = values[source]
    return pd.Series(out, index=series.index)


def compute_adaptive_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, periods: pd.DataFrame) -> pd.DataFrame:
    """Full Ichimoku, adaptive periods: Tenkan-sen, Kijun-sen, the
    forward-shifted Senkou Span A/B cloud, and the Chikou Span reference —
    each day using that day's adaptive period lengths from ``periods`` (see
    ``compute_adaptive_periods``). Uses TRUE high/low, not a close-derived
    proxy — see module docstring.
    """
    tenkan = _variable_window_hl_midpoint(high, low, periods["tenkan_period"])
    kijun = _variable_window_hl_midpoint(high, low, periods["kijun_period"])
    senkou_a_raw = (tenkan + kijun) / 2.0
    senkou_b_raw = _variable_window_hl_midpoint(high, low, periods["senkou_b_period"])

    senkou_a_cloud = _scatter_forward(senkou_a_raw, periods["kijun_period"])
    senkou_b_cloud = _scatter_forward(senkou_b_raw, periods["kijun_period"])
    cloud_top = pd.concat([senkou_a_cloud, senkou_b_cloud], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a_cloud, senkou_b_cloud], axis=1).min(axis=1)

    chikou_reference = _variable_lag_lookup(close, periods["kijun_period"])

    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a_raw": senkou_a_raw,
            "senkou_b_raw": senkou_b_raw,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "chikou_reference": chikou_reference,
        },
        index=close.index,
    )


def compute_static_ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = DEFAULT_BASE_TENKAN,
    kijun_period: int = DEFAULT_BASE_KIJUN,
    senkou_b_period: int = DEFAULT_BASE_SENKOU_B,
) -> pd.DataFrame:
    """Fixed-period Ichimoku — the non-adaptive baseline
    ``scripts/run_adaptive_ichimoku_backtest.py`` compares the adaptive
    versions against. Vectorized (fixed window/offset = ``.rolling()``/
    ``.shift()`` work fine directly), unlike ``compute_adaptive_ichimoku``.
    True high/low and a genuinely forward-shifted cloud + Chikou reference,
    same as the adaptive version — a constant kijun_period is just the
    special case where the scatter-forward/variable-lag machinery collapses
    to a plain ``.shift(kijun_period)``.
    """
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2.0
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2.0
    senkou_a_raw = (tenkan + kijun) / 2.0
    senkou_b_raw = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2.0

    senkou_a_cloud = senkou_a_raw.shift(kijun_period)
    senkou_b_cloud = senkou_b_raw.shift(kijun_period)
    cloud_top = pd.concat([senkou_a_cloud, senkou_b_cloud], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a_cloud, senkou_b_cloud], axis=1).min(axis=1)

    chikou_reference = close.shift(kijun_period)

    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a_raw": senkou_a_raw,
            "senkou_b_raw": senkou_b_raw,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "chikou_reference": chikou_reference,
        },
        index=close.index,
    )


def generate_ichimoku_signal(ichimoku: pd.DataFrame, close: pd.Series, long_only: bool = True) -> pd.Series:
    """Full "triple confirmation" Ichimoku trading rule, a continuous
    trend-STATE read checked every day — all three must agree:

        bullish: close > cloud_top   AND  tenkan > kijun   AND  close > chikou_reference
        bearish: close < cloud_bottom AND  tenkan < kijun   AND  close < chikou_reference
        otherwise: flat (no 2-of-3 or majority rule — Ichimoku's cloud,
                   TK relationship, and Chikou confirmation are traditionally
                   read as three independent checks that all need to line up,
                   not a voting system)

    Position = whatever state is currently true (1 / -1 / 0). Holds the
    last valid state through any NaN gap (warmup, missing data) rather than
    force-flattening — same convention as
    ``technical_signals.generate_signal``.

    Uses a continuous state check rather than a same-day cross-triggered
    entry — an earlier cross-triggered version had a real bug (the lagging
    cloud meant the triggering cross had often already happened and didn't
    re-fire by the time price actually confirmed outside the cloud); see
    ``tests/test_adaptive_ichimoku.py``.
    """
    close_aligned = close.reindex(ichimoku.index)
    required_cols = ["tenkan", "kijun", "cloud_top", "cloud_bottom", "chikou_reference"]
    inputs_valid = ichimoku[required_cols].notna().all(axis=1) & close_aligned.notna()

    bullish = (
        inputs_valid
        & (close_aligned > ichimoku["cloud_top"])
        & (ichimoku["tenkan"] > ichimoku["kijun"])
        & (close_aligned > ichimoku["chikou_reference"])
    )
    bearish = (
        inputs_valid
        & (close_aligned < ichimoku["cloud_bottom"])
        & (ichimoku["tenkan"] < ichimoku["kijun"])
        & (close_aligned < ichimoku["chikou_reference"])
    )

    position = pd.Series(np.nan, index=ichimoku.index)
    position[inputs_valid] = 0.0
    position[bullish] = 1.0
    position[bearish] = -1.0
    position = position.ffill().fillna(0.0)

    if long_only:
        position = position.clip(lower=0.0)
    return position.rename("position")


def compute_ichimoku_conviction_score(
    ichimoku: pd.DataFrame, close: pd.Series, long_only: bool = True
) -> pd.Series:
    """Continuous, ADDITIVE alternative to ``generate_ichimoku_signal``'s
    strict triple-confirmation gate — sums four normalized distance scores
    and squashes with ``tanh`` into ``[-1, 1]`` (or ``[0, 1]`` if
    ``long_only``):

        score1 = (tenkan_t   - kijun_t)   / kijun_t
        score2 = (tenkan_t-1 - kijun_t-1) / kijun_t-1   (yesterday's same ratio
                 -- a 1-day-lagged read on the Tenkan/Kijun relationship,
                 giving the score a small persistence/momentum component
                 rather than reacting to a single day's cross alone)
        score3 = (close - cloud_top) / cloud_top   (signed distance of price
                 above/below the cloud's UPPER edge -- positive means
                 fully above the cloud, more negative the deeper price sits
                 inside or below it)
        score4 = (close - chikou_reference) / chikou_reference

        conviction = tanh(score1 + score2 + score3 + score4)

    **Why this exists**: ``generate_ichimoku_signal`` requires cloud
    position AND the Tenkan/Kijun relationship AND Chikou confirmation to
    ALL independently agree before registering anything other than flat —
    a strict logical AND across three binary checks. That's traditional
    Ichimoku reading, but it's also mechanically why the triple-confirmed
    "hit rate" on any given day, across any stock universe, tends to be
    low (see ``docs/backtesting_spec.md``'s Ichimoku section and this
    project's own earlier finding of beta ~0.27-0.41 even standalone) —
    one weakly-disagreeing sub-condition zeroes out an otherwise strong
    signal from the other two. Summing continuous, normalized versions of
    the same four underlying relationships instead means a symbol can
    register real positive conviction even when not every single
    sub-condition is individually positive, as long as the balance leans
    bullish overall — same Tenkan/Kijun/cloud/Chikou primitives, additive
    rather than multiplicative-AND combination. This does not touch
    ``generate_ichimoku_signal`` itself (kept as-is for comparison via
    ``build_ichimoku_weights``'s ``signal_mode`` parameter).

    ``tanh`` bounds the otherwise-unbounded sum into a stable, comparable
    ``[-1, 1]`` range regardless of how large the individual percentage
    distances get (e.g. a stock far above its cloud during a strong
    trend), rather than letting outlier days dominate downstream
    conviction-weighted sizing.

    Same NaN-during-warmup / hold-last-valid-value-through-gaps convention
    as ``generate_ichimoku_signal`` (via natural NaN propagation through
    the arithmetic above, then forward-filled).

    ``long_only=True`` (default, matching ``generate_ichimoku_signal``'s
    default and this project's long-only-everywhere backtest engines):
    clips to ``[0, 1]`` — negative conviction is just "no position", not a
    short.
    """
    close_aligned = close.reindex(ichimoku.index)
    required_cols = ["tenkan", "kijun", "cloud_top", "chikou_reference"]
    inputs_valid_today = ichimoku[required_cols].notna().all(axis=1) & close_aligned.notna()

    tenkan, kijun = ichimoku["tenkan"], ichimoku["kijun"]
    score1 = (tenkan - kijun) / kijun
    score2 = score1.shift(1)  # naturally NaN if yesterday's tenkan/kijun weren't valid yet
    score3 = (close_aligned - ichimoku["cloud_top"]) / ichimoku["cloud_top"]
    score4 = (close_aligned - ichimoku["chikou_reference"]) / ichimoku["chikou_reference"]

    total = score1 + score2 + score3 + score4
    conviction = np.tanh(total)
    conviction[~inputs_valid_today] = np.nan

    conviction = conviction.ffill().fillna(0.0)
    if long_only:
        conviction = conviction.clip(lower=0.0)
    return conviction.rename("conviction")


def build_ichimoku_conviction_panel(
    price_panel_ohlc: dict[str, pd.DataFrame],
    t: int = 10,
    zscore_window: int = 252,
    variant: str = "static",
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    base_tenkan: int = DEFAULT_BASE_TENKAN,
    base_kijun: int = DEFAULT_BASE_KIJUN,
    base_senkou_b: int = DEFAULT_BASE_SENKOU_B,
) -> pd.DataFrame:
    """Per-symbol daily Ichimoku conviction scores (``compute_ichimoku_conviction_score``,
    always ``long_only=True``) — RAW ``[0, 1]`` values, deliberately NOT
    portfolio-normalized the way ``build_ichimoku_weights`` normalizes its
    output (that normalization divides by the day's sum of active
    conviction, so with ~500 symbols the typical value becomes ~1/500 —
    fine for building portfolio weights directly, but the wrong shape to
    feed as a per-symbol SCORE into something else, like a fundamentals
    composite score).

    **Why this exists, separately from ``build_ichimoku_weights``**: both
    post-selection uses of Ichimoku tried so far — gating an existing
    selection (``strategies.apply_ichimoku_gate``/``apply_ichimoku_breadth_scalar``)
    and reallocating within one (``strategies.apply_ichimoku_conviction_tilt``)
    — were confirmed negative on real data (see
    ``docs/backtesting_spec.md``'s Ichimoku sections), despite
    ``ichimoku_only`` (the signal on its own, free to pick from the full
    universe) being the single best-performing component found. The
    working hypothesis: the edge is a stock-picking/rotation signal, not a
    within-basket timing enhancer — which means the right place to use it
    is at SELECTION time, not after. This panel is the input to
    ``fundamental_analysis/point_in_time.py``'s per-rebalance-date
    conviction extraction, which feeds
    ``fundamental_analysis/metrics/technical_momentum.py`` as an eighth
    composite-score dimension — see that module and
    ``docs/fundamental_analysis_spec.md``'s "technical_momentum dimension"
    section for the full integration.

    Same variant/adaptive-period machinery as ``build_ichimoku_weights``
    (see that function's docstring for ``variant``'s meaning) — this
    literally shares the per-symbol Ichimoku computation, just stops
    before the portfolio-normalization step and always uses the
    continuous conviction score (never the binary triple-confirmation
    signal — a raw 0/1 value would make a poor continuous composite-score
    input).

    Returns a DataFrame indexed by date, columns = symbols with usable
    OHLC data (a subset of ``price_panel_ohlc``'s keys if any were
    skipped for missing columns).
    """
    conviction_scores = {}
    for symbol, ohlc in price_panel_ohlc.items():
        missing = {"high", "low", "close"} - set(ohlc.columns)
        if missing:
            logger.warning("Symbol %s is missing OHLC columns %s -- skipping", symbol, missing)
            continue
        high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
        if variant == "static":
            ichimoku = compute_static_ichimoku(high, low, close, base_tenkan, base_kijun, base_senkou_b)
        else:
            scores = compute_signed_normalized_score(close, t=t, zscore_window=zscore_window)
            periods = compute_adaptive_periods(
                scores["magnitude"], base_tenkan, base_kijun, base_senkou_b,
                scale_min=scale_min, scale_max=scale_max, direction=variant,
            )
            ichimoku = compute_adaptive_ichimoku(high, low, close, periods)
        conviction_scores[symbol] = compute_ichimoku_conviction_score(ichimoku, close, long_only=True)

    if not conviction_scores:
        raise ValueError("No symbols had usable OHLC data -- nothing to build a conviction panel from.")

    panel = pd.DataFrame(conviction_scores)
    panel.index.name = "date"
    return panel


def build_ichimoku_weights(
    price_panel_ohlc: dict[str, pd.DataFrame],
    t: int = 10,
    zscore_window: int = 252,
    variant: str = "static",
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    base_tenkan: int = DEFAULT_BASE_TENKAN,
    base_kijun: int = DEFAULT_BASE_KIJUN,
    base_senkou_b: int = DEFAULT_BASE_SENKOU_B,
    long_only: bool = True,
    signal_mode: str = "triple_confirmation",
) -> pd.DataFrame:
    """Apply the Ichimoku signal independently to every symbol in
    ``price_panel_ohlc`` (``{symbol: DataFrame}``, each with columns
    open/high/low/close[/volume] — see
    ``fundamental_analysis.data_fetchers.yfinance_fetcher.fetch_price_panel_ohlc`` /
    ``scripts._common_cli.load_prices_ohlc``) and build a daily
    target-weight matrix across whichever symbols are currently active —
    same convention as ``technical_signals.build_technical_signal_weights``,
    for direct comparability.

    ``variant``: ``"static"`` (fixed base periods, no adaptivity — the
    ablation baseline), ``"shrink_when_high"``, or ``"shrink_when_low"``
    (adaptive, driven by the dispersion-score magnitude with ``t``/
    ``zscore_window`` controlling that score — see
    ``technical_signals.compute_signed_normalized_score``, which uses
    ``close`` only, same as the rest of that module).

    ``signal_mode`` (default ``"triple_confirmation"``): which per-symbol
    signal function drives the weights.
    - ``"triple_confirmation"``: ``generate_ichimoku_signal`` — the
      traditional binary AND-of-three-checks reading. Every active symbol
      gets EQUAL weight (``1/n_active``), since the signal itself is
      already binary (in/out), not graded.
    - ``"conviction_score"``: ``compute_ichimoku_conviction_score`` — a
      continuous, additive tanh-bounded score in ``[0, 1]`` (long_only)
      combining the same four underlying Tenkan/Kijun/cloud/Chikou
      relationships without requiring all of them to individually agree.
      Weights are then CONVICTION-weighted rather than equal-weighted (see
      below) — a stock the score is more confident about gets
      proportionally more capital, not just an equal split among whichever
      names cleared a hard threshold. Added specifically because the
      strict AND in ``"triple_confirmation"`` mode produces a low
      day-to-day hit rate (see that function's docstring), which is a
      structural driver of low realized exposure when this signal gates or
      combines with another already-narrow selection (e.g.
      ``strategies.apply_ichimoku_gate``/``apply_ichimoku_breadth_scalar``).

    The weight-normalization formula (divide each day's raw per-symbol
    signal by that day's sum of absolute active signals) is IDENTICAL for
    both modes — it was already generic enough to handle a continuous
    ``[0, 1]`` conviction value exactly the same way it handles a binary
    ``{0, 1}`` value; a binary signal is simply the special case where
    every active name has equal conviction.

    Symbols' OHLC frames may have different date ranges/calendars (e.g. IPO
    dates, live-fetch quirks) — positions are aligned onto the union of all
    symbols' dates, with 0 (no position) wherever a symbol has no data for
    that day, rather than requiring a fully-rectangular panel upfront.
    """
    if variant not in ("static", "shrink_when_high", "shrink_when_low"):
        raise ValueError(f"variant must be 'static', 'shrink_when_high', or 'shrink_when_low', got {variant!r}")
    if signal_mode not in ("triple_confirmation", "conviction_score"):
        raise ValueError(f"signal_mode must be 'triple_confirmation' or 'conviction_score', got {signal_mode!r}")

    positions = {}
    for symbol, ohlc in price_panel_ohlc.items():
        missing = {"high", "low", "close"} - set(ohlc.columns)
        if missing:
            logger.warning("Symbol %s is missing OHLC columns %s -- skipping", symbol, missing)
            continue
        high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
        if variant == "static":
            ichimoku = compute_static_ichimoku(high, low, close, base_tenkan, base_kijun, base_senkou_b)
        else:
            scores = compute_signed_normalized_score(close, t=t, zscore_window=zscore_window)
            periods = compute_adaptive_periods(
                scores["magnitude"], base_tenkan, base_kijun, base_senkou_b,
                scale_min=scale_min, scale_max=scale_max, direction=variant,
            )
            ichimoku = compute_adaptive_ichimoku(high, low, close, periods)

        if signal_mode == "triple_confirmation":
            positions[symbol] = generate_ichimoku_signal(ichimoku, close, long_only=long_only)
        else:
            positions[symbol] = compute_ichimoku_conviction_score(ichimoku, close, long_only=long_only)

    if not positions:
        raise ValueError("No symbols had usable OHLC data -- nothing to build weights from.")

    position_panel = pd.DataFrame(positions).fillna(0.0)
    n_active = position_panel.abs().sum(axis=1).replace(0, np.nan)
    weights = position_panel.div(n_active, axis=0).fillna(0.0)
    weights.index.name = "date"
    return weights
