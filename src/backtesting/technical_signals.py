"""Multi-scale SMA-dispersion trend-strength signal — a price-only,
per-symbol technical signal, independent of both the regime-detection and
fundamental-analysis modules (see ``backtesting/strategies.py`` for how its
output gets turned into portfolio weights).

**Status: experimental / user-specified test construction.** This was built
to an exact specification the user gave (score formula, normalization,
threshold logic), not derived or empirically validated by this project — no
backtest of it has been run yet, on synthetic or real data, at the time this
module was written. Treat it exactly like the geometric wedge-product signal
elsewhere in this codebase: implemented and unit-tested for correctness of
the *mechanics*, but with zero claim about whether it actually makes money.
Validate with a real backtest (``strategies.build_technical_signal_weights``
plugs into the same ``backtesting`` pipeline as everything else) before
trusting it.

The construction, in order:

1. **Dispersion score** — a ladder of 4 SMAs at t, 2t, 4t, 8t (a dyadic/
   octave scale spread), summed pairwise-adjacent absolute differences:

       s = |SMA(t) - SMA(2t)| + |SMA(2t) - SMA(4t)| + |SMA(4t) - SMA(8t)|

   By construction (sum of absolute values) this measures trend *strength*
   across scales, not *direction* — a strong uptrend and a strong downtrend
   both produce a large s; a flat/range-bound market produces a small one
   (all four SMAs converge). Each term is normalized by price before
   summing (``normalize_by_price=True``, the default) so the score is a
   dimensionless percentage, comparable across symbols/price levels/time —
   without this, a ₹5000 stock and a ₹50 stock (or the same stock's own
   history a decade apart) aren't comparable even after z-scoring.

2. **Self-relative normalization** — rolling z-score of s against its own
   trailing history (default 252d ≈ 1yr, matching the z-score lookback
   convention used elsewhere in this project, e.g.
   ``regime_detection/features.py``'s ``vix_zscore_1y``), then
   ``tanh`` to squash into (-1, 1). This keeps the score comparable across
   volatility regimes and market eras without a fixed absolute threshold.

3. **Direction** — added on top of the (direction-blind) magnitude score:
   ``sign(SMA(t) - SMA(8t))``. Multiplying this onto the tanh'd z-score
   gives a genuinely *signed* quantity in (-1, +1): sign = which way the
   short-vs-long-scale trend points, magnitude = how historically unusual
   that directional dispersion currently is. This is what makes a single
   symmetric [-q, +q] band able to drive both trend-following AND
   mean-reversion logic — see ``generate_signal``'s docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def compute_sma_ladder(prices: pd.Series, t: int) -> pd.DataFrame:
    """Simple moving averages at t, 2t, 4t, 8t."""
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}")
    return pd.DataFrame(
        {
            f"sma_{t}": prices.rolling(t).mean(),
            f"sma_{2 * t}": prices.rolling(2 * t).mean(),
            f"sma_{4 * t}": prices.rolling(4 * t).mean(),
            f"sma_{8 * t}": prices.rolling(8 * t).mean(),
        },
        index=prices.index,
    )


def compute_dispersion_score(prices: pd.Series, t: int, normalize_by_price: bool = True) -> pd.Series:
    """Raw (unnormalized-across-time, but optionally price-normalized)
    dispersion score s = |SMA(t)-SMA(2t)| + |SMA(2t)-SMA(4t)| + |SMA(4t)-SMA(8t)|.

    ``normalize_by_price=True`` (default, recommended) divides each term by
    the current price, so s is a sum of percentage gaps rather than raw
    price-unit gaps — makes it comparable across symbols/price levels/time.
    Set False only if you specifically want raw-unit dispersion (e.g. for a
    single symbol where you've already reasoned about the scale).
    """
    ladder = compute_sma_ladder(prices, t)
    sma_t, sma_2t, sma_4t, sma_8t = (ladder[c] for c in ladder.columns)
    gaps = [(sma_t - sma_2t).abs(), (sma_2t - sma_4t).abs(), (sma_4t - sma_8t).abs()]
    if normalize_by_price:
        gaps = [g / prices for g in gaps]
    s = gaps[0] + gaps[1] + gaps[2]
    s.name = f"dispersion_score_t{t}"
    return s


def compute_signed_normalized_score(
    prices: pd.Series, t: int, zscore_window: int = 252, normalize_by_price: bool = True
) -> pd.DataFrame:
    """Full pipeline: raw score -> rolling self-relative z-score -> tanh ->
    signed by direction. Returns a DataFrame with columns:
    ``raw_score``, ``zscore``, ``magnitude`` (tanh'd, always >= 0),
    ``direction`` (+1/-1/0), ``signed_score`` (direction * magnitude, in
    (-1, 1) — this is what ``generate_signal`` thresholds against).

    ``zscore_window`` needs at least this many trailing days of ``raw_score``
    history before it produces a non-NaN value (plus the 8t warmup for the
    longest SMA itself) — with the default t=10/zscore_window=252, that's
    roughly 80 (8*10) + 252 ≈ 330 trading days of price history needed
    before the signal is defined at all. Shorter zscore_window trades
    stability for faster adaptation to genuine regime changes in the
    dispersion baseline; there's no single right answer, same caveat as
    choosing t (see module docstring).
    """
    raw_score = compute_dispersion_score(prices, t, normalize_by_price=normalize_by_price)

    rolling_mean = raw_score.rolling(zscore_window).mean()
    rolling_std = raw_score.rolling(zscore_window).std()
    zscore = (raw_score - rolling_mean) / rolling_std.replace(0, np.nan)
    magnitude = np.tanh(zscore)

    ladder = compute_sma_ladder(prices, t)
    direction = np.sign(ladder[f"sma_{t}"] - ladder[f"sma_{8 * t}"])

    signed_score = direction * magnitude

    return pd.DataFrame(
        {
            "raw_score": raw_score,
            "zscore": zscore,
            "magnitude": magnitude,
            "direction": direction,
            "signed_score": signed_score,
        },
        index=prices.index,
    )


def _target_position(v: float, q_entry: float, mode: str) -> float | None:
    """What position an extreme value of ``v`` implies under ``mode``, or
    ``None`` if ``v`` isn't past the entry threshold either way."""
    if v > q_entry:
        return 1.0 if mode == "trend" else -1.0
    if v < -q_entry:
        return -1.0 if mode == "trend" else 1.0
    return None


def generate_signal(
    signed_score: pd.Series,
    q_entry: float,
    q_exit: float | None = None,
    mode: str = "trend",
) -> pd.Series:
    """Turn a signed_score series into a daily position in {-1, 0, +1}
    (short/flat/long) using entry/exit thresholds at +/-q with hysteresis.

    Args:
        q_entry: entry threshold, 0 < q_entry <= 1 (a value outside (0,1]
            makes no sense since signed_score is bounded in (-1,1)).
        q_exit: exit threshold, 0 <= q_exit < q_entry (defaults to
            0.3 * q_entry if not given — enough hysteresis to avoid
            whipsawing right at the entry boundary, without being so wide
            the position never exits). Position flattens whenever
            ``abs(signed_score) < q_exit`` REGARDLESS of mode — "the
            dispersion has normalized" means exit, whether you entered on
            a trend or a mean-reversion thesis.
        mode: "trend" (momentum — enter in the direction the signed_score
            already points) or "mean_reversion" (fade — enter opposite the
            signed_score's direction, betting on reversion). Same score,
            same thresholds, entries mirrored:

              trend:          signed_score > +q_entry -> long
                               signed_score < -q_entry -> short
              mean_reversion: signed_score < -q_entry -> long  (fade the dip)
                               signed_score > +q_entry -> short (fade the spike)

    Returns:
        Position series, same index as ``signed_score``, values in
        {-1, 0, +1}. This is a STATEFUL signal (today's position can depend
        on yesterday's, because of the exit hysteresis band) — computed with
        an explicit sequential pass, not vectorized, for correctness/
        auditability over cleverness on typical single-symbol daily-data
        lengths (thousands of rows, not a bottleneck).

        Handles direct reversals (score swings from one extreme straight
        past the opposite extreme without lingering in the exit band in
        between — a sharp trend reversal, not a gradual fade) by flipping
        straight to the new side that same day, not just exiting to flat and
        waiting for a fresh entry: without this, a position that was long
        into a sharp reversal would ride the entire opposite move stuck in
        the wrong direction, since ``abs(v)`` never dips below ``q_exit``
        when the move is simply strong-but-flipped-sign, not weakening.
    """
    if mode not in ("trend", "mean_reversion"):
        raise ValueError(f"mode must be 'trend' or 'mean_reversion', got {mode!r}")
    if not (0 < q_entry <= 1):
        raise ValueError(f"q_entry must be in (0, 1], got {q_entry}")
    q_exit = q_exit if q_exit is not None else 0.3 * q_entry
    if not (0 <= q_exit < q_entry):
        raise ValueError(f"q_exit must satisfy 0 <= q_exit < q_entry, got q_exit={q_exit}, q_entry={q_entry}")

    values = signed_score.values
    positions = np.zeros(len(values))
    position = 0.0

    for i, v in enumerate(values):
        if np.isnan(v):
            positions[i] = position  # hold through NaN gaps rather than force-flatten
            continue
        target = _target_position(v, q_entry, mode)
        if position == 0.0:
            if target is not None:
                position = target
        else:
            if abs(v) < q_exit:
                position = 0.0
            elif target is not None and target != position:
                position = target  # direct reversal, no need to pass through flat first
        positions[i] = position

    return pd.Series(positions, index=signed_score.index, name="position")


def build_technical_signal_weights(
    price_panel: pd.DataFrame,
    t: int = 10,
    q_entry: float = 0.5,
    q_exit: float | None = None,
    mode: str = "trend",
    zscore_window: int = 252,
    normalize_by_price: bool = True,
    long_only: bool = True,
) -> pd.DataFrame:
    """Apply the dispersion-score signal independently to every symbol in
    ``price_panel`` and build a daily equal-weighted target-weight matrix
    across whichever symbols are currently "in" (long, or long+short if
    ``long_only=False``) — same equal-weight-among-selected convention as
    ``build_fundamental_portfolio_weights``, for direct comparability in
    ``attribution.run_component_backtests``.

    **Real-data result, worth reading before using this over
    ``build_conviction_weighted_signal_weights`` below**: this
    threshold-gated version came in at beta ~0.83 against buy-and-hold in
    initial testing — spending real time completely flat waiting for a
    confirmed threshold crossing costs return in an up-trending market,
    independent of whether the "in" periods were individually good calls.
    Kept here (not removed) as the ablation baseline the continuous version
    is compared against — see ``scripts/run_technical_backtest.py --sizing all``.

    ``long_only=True`` (default): short signals are treated as flat (0),
    not negative weight — this project's backtest engines and the rest of
    the strategies in this codebase are long-only/no-leverage; actually
    shorting would need margin/borrow-cost modeling this system doesn't
    have. Set False only if you're prepared to extend the engine
    accordingly first.

    Returns a DAILY (not sparse/rebalance-date) weights DataFrame — unlike
    the fundamentals weights, this signal is naturally computed every day
    (it's a rolling technical indicator, not a periodic snapshot), so no
    separate ``align_weights_to_returns`` forward-fill step is needed before
    handing this to ``run_backtest``/``run_backtest_vbt``.
    """
    positions = {}
    for symbol in price_panel.columns:
        scores = compute_signed_normalized_score(
            price_panel[symbol], t=t, zscore_window=zscore_window, normalize_by_price=normalize_by_price
        )
        positions[symbol] = generate_signal(scores["signed_score"], q_entry, q_exit, mode)

    position_panel = pd.DataFrame(positions)
    if long_only:
        position_panel = position_panel.clip(lower=0.0)

    n_active = position_panel.abs().sum(axis=1).replace(0, np.nan)
    weights = position_panel.div(n_active, axis=0).fillna(0.0)
    weights.index.name = "date"
    return weights


def build_conviction_weighted_signal_weights(
    price_panel: pd.DataFrame,
    t: int = 10,
    zscore_window: int = 252,
    mode: str = "trend",
    normalize_by_price: bool = True,
    long_only: bool = True,
) -> pd.DataFrame:
    """Continuous, conviction-weighted position sizing — the direct fix for
    the under-exposure problem found in ``build_technical_signal_weights``
    (threshold-gated: beta ~0.83 against buy-and-hold in initial real-data
    testing). Instead of requiring ``|signed_score| > q_entry`` just to have
    ANY exposure, and then equal-weighting whichever symbols cleared that
    bar, weight is directly proportional to each symbol's own
    ``signed_score`` — so exposure scales smoothly with conviction rather
    than snapping between 0% and a fixed per-symbol share.

        weight_i(t) = signed_score_i(t) / N      (mode="trend")
        weight_i(t) = -signed_score_i(t) / N     (mode="mean_reversion")
        [long_only: clipped to >= 0 before dividing]

    where ``N = len(price_panel.columns)`` is the FIXED universe size, NOT
    the count of symbols with nonzero conviction that day. This is the key
    structural difference from ``build_technical_signal_weights``, and the
    actual mechanism of the fix: dividing by a fixed N means AGGREGATE
    portfolio exposure is itself continuous — if every symbol has weak
    conviction, total invested fraction is small and the rest sits in cash;
    if every symbol has strong conviction, exposure approaches fully
    invested. ``build_technical_signal_weights``'s equal-weight-among-active
    scheme is effectively still binary at the portfolio level even though
    individual entries are threshold-gated: SOME exposure appears the
    moment any one symbol crosses the threshold, and total exposure doesn't
    otherwise reflect how strong or weak conviction is across the universe.

    Bounded automatically: since each clipped/signed per-symbol weight is
    in [0, 1] (long-only) or [-1, 1] (short-allowed) and there are N of
    them each divided by N, total gross exposure never exceeds 100% without
    needing a separate cap.

    ``mode`` mirrors ``generate_signal``'s trend vs. mean-reversion framing
    exactly (same score, sign-flipped conviction) — see that function's
    docstring. Same DAILY (not sparse) output convention as
    ``build_technical_signal_weights``.
    """
    if mode not in ("trend", "mean_reversion"):
        raise ValueError(f"mode must be 'trend' or 'mean_reversion', got {mode!r}")

    scores_by_symbol = {}
    for symbol in price_panel.columns:
        scores = compute_signed_normalized_score(
            price_panel[symbol], t=t, zscore_window=zscore_window, normalize_by_price=normalize_by_price
        )
        scores_by_symbol[symbol] = scores["signed_score"]

    signed_score_panel = pd.DataFrame(scores_by_symbol).fillna(0.0)
    conviction = signed_score_panel if mode == "trend" else -signed_score_panel
    if long_only:
        conviction = conviction.clip(lower=0.0)

    n = len(price_panel.columns)
    weights = conviction / n
    weights.index.name = "date"
    return weights
