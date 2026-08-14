"""Turn regime-detection and fundamental-analysis outputs into portfolio weights.

Weight schemes, each isolating one signal so their individual contribution
can be measured in ``attribution.py``:

- ``build_benchmark_weights``          — 100% in the benchmark, always. The baseline.
- ``build_regime_exposure_weights``    — benchmark exposure scaled 0-100% by the
                                          GMM/KMeans/HMM-detected market regime (a
                                          market-timing overlay; no stock selection).
- ``build_fundamental_portfolio_weights`` — equal-weighted top-quantile stocks by
                                          composite fundamental score, fully invested
                                          regardless of regime (a stock-selection
                                          strategy; no market timing).
- ``combine_regime_and_fundamentals``  — the fundamentals stock selection, with
                                          total exposure scaled by the regime-timing
                                          factor. This is "both components together."
- ``build_geometric_overlay_weights``  — benchmark exposure scaled by the geometric
                                          wedge-product crash-risk flag
                                          (``regime_detection/geometric_signal.py``)
                                          ALONE — not the GMM regime label, by
                                          explicit design (see that module's and
                                          ``regime_detection/pipeline.py``'s
                                          docstrings). Isolates the geometric
                                          signal's own standalone effect.
- ``apply_geometric_overlay``          — multiplies an already-built weights matrix
                                          (e.g. the combined regime+fundamentals
                                          weights) by that same geometric exposure
                                          factor — the "on top of everything" hook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def build_benchmark_weights(dates: pd.DatetimeIndex, benchmark_col: str = "benchmark") -> pd.DataFrame:
    return pd.DataFrame({benchmark_col: 1.0}, index=dates)


def build_regime_exposure_weights(
    regime: pd.Series,
    exposure_by_regime: dict,
    benchmark_col: str = "benchmark",
) -> pd.DataFrame:
    """Daily benchmark exposure fraction implied by the detected regime.

    ``regime`` is typically the integer regime label series from
    ``regime_detection.pipeline.run_pipeline`` (already volatility-ordered,
    so ``exposure_by_regime`` keys are stable/comparable across re-fits:
    0 = calmest) — but this also accepts the governed ``active_regime``
    series from ``regime_detection.state_governor``, whose values are a
    MIX of those same int regime ids and the string ``"transitional"``
    (see ``docs/regime_detection_spec.md``'s consensus-governor section).
    ``exposure_by_regime`` therefore accepts both int and ``"transitional"``
    keys in the same dict; a value missing a ``"transitional"`` mapping
    when governed regimes are actually present defaults to full (1.0)
    exposure with a logged warning (same "unmapped -> 1.0" convention as
    unmapped int regimes below), since silently trading full exposure
    through a bar the governor explicitly flagged as ambiguous is a
    meaningfully different (and probably unwanted) choice worth surfacing,
    not a safe default to apply quietly.
    """
    unmapped = set(regime.unique()) - set(exposure_by_regime.keys())
    if "transitional" in unmapped:
        logger.warning(
            "build_regime_exposure_weights: 'transitional' regime label present but not in "
            "exposure_by_regime -- defaulting to 1.0 (full exposure) on transitional bars. "
            "Add a 'transitional' key to exposure_by_regime (e.g. 0.25, per the consensus "
            "governor research doc's 'Transitional' target) if that's not what you want."
        )
    exposure = regime.map(exposure_by_regime).fillna(1.0)
    return pd.DataFrame({benchmark_col: exposure}, index=regime.index)


def build_fundamental_portfolio_weights(
    scores_by_date: pd.DataFrame,
    top_quantile: float = 0.2,
    min_positions: int = 5,
    max_sector_weight: float | None = None,
    max_position_weight: float | None = None,
) -> pd.DataFrame:
    """Equal-weighted top-quantile-by-composite-score portfolio, one row per
    rebalance date.

    ``scores_by_date`` must have columns: date, symbol, composite_score — one
    row per (rebalance date, symbol). Typically produced by re-running
    ``fundamental_analysis.pipeline.run_pipeline`` at each historical rebalance
    date on point-in-time data (see module docstring caveat on look-ahead bias
    in docs/backtesting_spec.md).

    ``max_sector_weight`` (optional, e.g. ``0.30``): caps any single sector's
    total weight in the portfolio at this fraction. Requires a ``sector``
    column in ``scores_by_date`` (present by default — ``fundamental_analysis.pipeline.run_pipeline``
    always carries the snapshot's ``sector`` column through). ``None``
    (default) applies no sector cap — exact pre-cap behavior.

    **Why this exists**: the fundamentals selection's max drawdown has
    consistently been WORSE than the raw benchmark's across every real
    backtest run so far (e.g. -45% vs benchmark's -38%), despite picking
    supposedly higher-quality names — flagged early in this project as a
    likely concentration problem, confirmed later when regime-conditional
    weighting's fallback (the 8 non-technical_momentum dimensions) turned
    out to inherit this same worse-than-benchmark drawdown on its own. See
    ``docs/fundamental_analysis_spec.md``'s "Position and sector caps"
    section.

    ``max_position_weight`` (optional, e.g. ``0.05``): caps any single
    symbol's weight. NOTE: with the default equal-weighting scheme here,
    every position already gets exactly ``1/n_select`` — this only bites
    if ``n_select`` is small enough that ``1/n_select`` alone exceeds the
    cap (e.g. ``min_positions=5`` with a 10% cap), OR if this function is
    later extended to non-equal weighting (e.g. conviction-weighted). Kept
    as a first-class option now for that reason, not because equal
    weighting needs it today.

    **How the sector cap works**: greedy, rank-preserving. Selection walks
    down the composite-score-sorted candidate list in order and admits a
    symbol UNLESS admitting it would push that symbol's sector over the
    cap, in which case it's skipped (not just capped in place) and the
    next-best candidate is considered instead — so a capped sector doesn't
    quietly shrink the portfolio, it makes room for the next-best
    idea from elsewhere. If sector data is entirely missing for a date,
    behaves as if no cap were set for that date only (logged once), rather
    than silently producing a smaller/skewed portfolio from missing data
    it can't actually evaluate against the cap.

    Returns a sparse weights DataFrame indexed by rebalance date, columns=symbols,
    NaN/0 for non-selected stocks. Feed through
    ``engine.align_weights_to_returns`` to forward-fill onto a daily index.
    """
    required = {"date", "symbol", "composite_score"}
    missing = required - set(scores_by_date.columns)
    if missing:
        raise ValueError(f"scores_by_date is missing columns: {missing}")

    has_sector = "sector" in scores_by_date.columns
    if max_sector_weight is not None and not has_sector:
        logger.warning(
            "build_fundamental_portfolio_weights: max_sector_weight=%.2f requested but no 'sector' "
            "column in scores_by_date -- sector cap will be a no-op for every date.", max_sector_weight,
        )

    rows = {}
    for date, g in scores_by_date.groupby("date"):
        g = g.dropna(subset=["composite_score"])
        n_select = max(min_positions, int(len(g) * top_quantile))
        n_select = min(n_select, len(g))
        if n_select == 0:
            rows[date] = pd.Series(dtype=float)
            continue

        candidates = g.sort_values("composite_score", ascending=False)

        if max_sector_weight is not None and has_sector and candidates["sector"].notna().any():
            selected_symbols: list[str] = []
            sector_counts: dict[str, int] = {}
            for _, row in candidates.iterrows():
                if len(selected_symbols) >= n_select:
                    break
                sector = row["sector"] if pd.notna(row["sector"]) else "__unknown__"
                # Check the cap using the FINAL portfolio size (n_select), not
                # the running count -- e.g. a 30% cap with n_select=10 means
                # at most 3 names from one sector, checked against the target
                # 10, not against however many are filled so far.
                max_from_this_sector = max(1, int(n_select * max_sector_weight))
                if sector_counts.get(sector, 0) >= max_from_this_sector:
                    continue
                selected_symbols.append(row["symbol"])
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            # If the sector cap left the portfolio under-filled (e.g. very
            # few sectors represented among candidates), fill remaining
            # slots from the next-best candidates regardless of sector,
            # rather than silently running a smaller-than-requested
            # portfolio -- a cap is meant to enforce diversification among
            # otherwise-viable choices, not to shrink the portfolio when
            # there simply isn't enough diversity to fill it any other way.
            if len(selected_symbols) < n_select:
                remaining = [s for s in candidates["symbol"] if s not in selected_symbols]
                selected_symbols.extend(remaining[: n_select - len(selected_symbols)])
            selected = candidates[candidates["symbol"].isin(selected_symbols)]
        else:
            selected = candidates.head(n_select)

        weight = 1.0 / len(selected)
        if max_position_weight is not None and weight > max_position_weight:
            # Cap and let gross exposure shrink rather than silently
            # renormalizing back up to 100% invested -- renormalizing would
            # defeat the point of a position cap (it would just redistribute
            # the capped amount right back among the same few names).
            weight = max_position_weight
        rows[date] = pd.Series(weight, index=selected["symbol"])

    weights = pd.DataFrame(rows).T.fillna(0.0)
    weights.index = pd.to_datetime(weights.index)
    weights.index.name = "date"
    return weights.sort_index()


def combine_regime_and_fundamentals(
    fundamental_weights_daily: pd.DataFrame, regime_exposure_daily: pd.Series
) -> pd.DataFrame:
    """Scale the fundamentals stock-selection weights by the regime exposure
    factor, so total invested fraction on any day = regime_exposure while the
    *composition* of that invested fraction still comes from fundamentals.

    Both inputs must already be aligned to the same daily DatetimeIndex (via
    ``engine.align_weights_to_returns``).
    """
    return fundamental_weights_daily.multiply(regime_exposure_daily, axis=0)


def apply_ichimoku_gate(
    weights_daily: pd.DataFrame,
    ichimoku_weights_daily: pd.DataFrame | None,
) -> pd.DataFrame:
    """Gate an already daily-aligned weights matrix (e.g. ``combined``) by
    Ichimoku confirmation: zero out any symbol's weight on any day the
    Ichimoku triple-confirmation signal (``adaptive_ichimoku.py``) isn't
    bullish for that symbol, leaving everything else untouched.

    **WARNING — real-data result: this construction produces severe cash
    drag, not a useful confirmation filter, when gating an already-narrow
    selection.** ``combined`` typically holds ~20% of the universe
    (``top_quantile``); Ichimoku independently confirms some other subset
    of the full universe on any given day. A hard per-symbol AND of two
    independently-selective filters intersects down to a tiny (often near-
    empty) overlap almost by construction — in the first real backtest this
    produced beta ~0.19 and annualized volatility ~5% (vs. ~12-17% for
    every other component), i.e. the strategy sat mostly in cash rather
    than expressing any real view. That's a structural property of AND-
    gating two independently-narrow selections, not evidence Ichimoku lacks
    edge. **Prefer ``apply_ichimoku_breadth_scalar`` below** — this
    function is kept for comparison purposes (e.g. an explicit
    ``ichimoku_confirmation_mode: "hard_gate"`` config choice) and as a
    cautionary example, not as the default.

    ``ichimoku_weights_daily`` is the direct output of
    ``adaptive_ichimoku.build_ichimoku_weights`` — an equal-weighted daily
    matrix that's already exactly 0 for non-confirmed symbols and
    ``1/n_active`` for confirmed ones (long_only), so ``> 0`` IS the
    confirmation mask; no separate raw position panel needs to be plumbed
    through.

    Symbols present in ``weights_daily`` but missing from
    ``ichimoku_weights_daily`` (e.g. no OHLC data was fetchable for that
    symbol) are treated as "no Ichimoku opinion" and passed through
    ungated (mask=1), NOT zeroed out — a data gap in the confirmation
    signal shouldn't silently exclude a stock the fundamentals/regime
    signals otherwise selected. Logged once per call so silent OHLC
    coverage gaps are visible rather than just quietly shrinking the
    effective universe.

    No-op (returns ``weights_daily`` unchanged) if
    ``ichimoku_weights_daily`` is ``None`` — i.e. the Ichimoku signal isn't
    configured/enabled, so ``combined`` behaves exactly as it did before
    this gate existed.
    """
    if ichimoku_weights_daily is None:
        return weights_daily

    ichimoku_aligned = ichimoku_weights_daily.reindex(
        index=weights_daily.index, columns=weights_daily.columns
    )
    missing_symbols = [c for c in weights_daily.columns if c not in ichimoku_weights_daily.columns]
    if missing_symbols:
        logger.warning(
            "apply_ichimoku_gate: %d/%d symbols have no Ichimoku data (no OHLC fetched) -- "
            "passing them through ungated rather than excluding them: %s%s",
            len(missing_symbols), weights_daily.shape[1], missing_symbols[:10],
            " ..." if len(missing_symbols) > 10 else "",
        )
    confirmed_mask = ichimoku_aligned > 0
    # columns that were entirely missing from ichimoku_weights_daily -> all-NaN after
    # reindex -> confirmed_mask is False everywhere for them; force those back to True
    # (pass-through) per the "no opinion, don't exclude" rule above.
    confirmed_mask[missing_symbols] = True

    return weights_daily.where(confirmed_mask, 0.0)


def apply_ichimoku_breadth_scalar(
    weights_daily: pd.DataFrame,
    ichimoku_weights_daily: pd.DataFrame | None,
    floor: float = 0.0,
) -> pd.DataFrame:
    """Scale an already daily-aligned weights matrix's TOTAL exposure by
    what fraction of its currently-held names Ichimoku confirms bullish —
    the recommended replacement for ``apply_ichimoku_gate``'s hard AND-gate
    (see that function's warning for why the hard version produces severe
    cash drag: intersecting two independently-narrow selections collapses
    to a near-empty overlap almost by construction).

    Per day t:
        held(t)      = symbols with weights_daily[t] > 0
        covered(t)   = held(t) that ALSO have Ichimoku data at all (i.e.
                       OHLC was fetchable -- symbols missing Ichimoku data
                       entirely are excluded from both numerator and
                       denominator, not counted against the fraction)
        confirmed(t) = covered(t) where Ichimoku is currently bullish
        fraction(t)  = |confirmed(t)| / |covered(t)|   (1.0 if covered(t) is empty
                       -- "no opinion available today" defaults to full
                       pass-through, same convention as the missing-symbol
                       handling in apply_ichimoku_gate)

    Every held symbol's weight that day is multiplied by fraction(t) — ALL
    selected names stay in the portfolio (unlike the hard gate, no single
    name is ever individually zeroed out for lack of confirmation); only
    the portfolio's AGGREGATE exposure moves with how broadly confirmed
    today's selection is. This means exposure degrades gracefully with
    partial confirmation instead of collapsing to near-zero the moment the
    (typically small) exact-overlap set is empty.

    ``floor`` (default 0.0): a minimum fraction below which exposure is not
    cut further -- e.g. ``floor=0.3`` means even on a day with zero
    confirmed names, exposure is only cut to 30%, not 0%. 0.0 (no floor,
    can go fully to cash) is the conservative default; raise it if fully
    flat days feel too aggressive once you've looked at real results.

    No-op if ``ichimoku_weights_daily`` is ``None``.
    """
    if ichimoku_weights_daily is None:
        return weights_daily

    held = weights_daily > 0
    ichimoku_aligned = ichimoku_weights_daily.reindex(index=weights_daily.index, columns=weights_daily.columns)
    has_coverage = ichimoku_aligned.notna()
    if ichimoku_weights_daily.shape[1] < weights_daily.shape[1]:
        missing_symbols = [c for c in weights_daily.columns if c not in ichimoku_weights_daily.columns]
        logger.warning(
            "apply_ichimoku_breadth_scalar: %d/%d symbols have no Ichimoku data (no OHLC "
            "fetched) -- excluded from the confirmation-fraction calculation entirely "
            "(neither helps nor hurts it): %s%s",
            len(missing_symbols), weights_daily.shape[1], missing_symbols[:10],
            " ..." if len(missing_symbols) > 10 else "",
        )

    covered_and_held = held & has_coverage
    confirmed_and_held = covered_and_held & (ichimoku_aligned > 0)

    covered_count = covered_and_held.sum(axis=1)
    confirmed_count = confirmed_and_held.sum(axis=1)
    fraction = (confirmed_count / covered_count.replace(0, np.nan)).fillna(1.0).clip(lower=floor)

    scaled = weights_daily.multiply(fraction, axis=0)
    # Only OVERWRITE cells that actually have Ichimoku coverage; symbols with
    # no coverage at all keep their original (unscaled) weight -- the
    # fraction is computed from covered names only (see above), so it must
    # also only be APPLIED to covered names, or a name with no Ichimoku
    # opinion would still get penalized by some other symbol's confirmation
    # rate, which contradicts the "no opinion -> don't affect it" rule this
    # function is supposed to follow (and that apply_ichimoku_gate follows
    # for its own missing-coverage symbols).
    result = weights_daily.copy()
    result[has_coverage] = scaled[has_coverage]
    return result


def apply_ichimoku_conviction_tilt(
    weights_daily: pd.DataFrame,
    ichimoku_weights_daily: pd.DataFrame | None,
    tilt_strength: float = 0.5,
) -> pd.DataFrame:
    """REALLOCATE an already-selected weights matrix's capital among its
    currently-held names based on relative Ichimoku conviction, WITHOUT
    ever changing total exposure — a structurally different mechanism from
    ``apply_ichimoku_gate``/``apply_ichimoku_breadth_scalar``, both of which
    can only ever CUT exposure (confirmation failing means less capital
    deployed, full stop). This function can only move capital BETWEEN
    names ``combined`` already selected; it never invests less overall
    than ``weights_daily`` did on that day, and never adds a name the base
    (fundamentals + regime) selection didn't already pick. That's the
    literal difference between "removing value" (gating) and "adding value"
    (tilting) — real-data results back this distinction up: gating
    `combined_with_ichimoku` down to a CAGR of 6.6% vs. `combined`'s 13.2%
    despite `ichimoku_only` standalone hitting 21.8% (the best of any
    component) shows the conviction signal has real information, and that
    information was being thrown away by exposure-cutting rather than
    reallocation.

    Per day t, for the set of currently-held names H (``weights_daily[t] >
    0``):

        conviction_i = ichimoku_weights_daily[t, i] if i has Ichimoku
                       coverage, else the mean conviction across H's
                       covered names that day (i.e. "no opinion" -> "don't
                       tilt away from this name", same convention as
                       ``apply_ichimoku_gate``'s missing-coverage handling)
        z_i           = (conviction_i - mean(conviction over H)) / std(conviction over H)
        tilt_i        = max(0, 1 + tilt_strength * z_i)
        weight_i(new) = weights_daily[t, i] * tilt_i, then RENORMALIZED so
                        sum(weight(new)) == sum(weights_daily[t]) exactly
                        — total exposure that day is unchanged to machine
                        precision; only the split across held names moves.

    **Why z-scored, not just demeaned (fixed 2026-07-24)**: an earlier
    version used a raw demeaned term (``conviction_i - mean``) with
    ``tilt_strength`` calibrated assuming conviction values live on
    ``[0, 1]``. That assumption broke when this function was fed
    ``adaptive_ichimoku.build_ichimoku_weights``'s actual OUTPUT — which is
    NOT raw conviction, it's already a normalized portfolio-weight matrix
    (each day's active symbols sum to 1, so with e.g. 500 active names the
    typical value is ~1/500 = 0.002, not ~0.5). Demeaned values at that
    scale are two orders of magnitude smaller than what ``tilt_strength``
    was calibrated for, so the tilt was silently ~1.0 (a no-op) for every
    name on real data — confirmed on a real run where
    ``combined_ichimoku_tilted`` came back numerically indistinguishable
    from plain ``combined`` (CAGR delta of -0.00004). Z-scoring makes the
    result invariant to whatever absolute scale the input happens to be
    in — only each day's RELATIVE spread across held names matters, which
    is the only thing that should matter for a reallocation decision
    anyway.

    ``tilt_strength`` (default 0.5, re-calibrated for the z-scored
    formulation): interpretable as "how much relative weight moves per
    standard deviation of conviction spread". At 0.5, a name 1 standard
    deviation above that day's mean gets 1.5x its base weight before
    renormalization; a name 1 std below gets 0.5x. 0.0 is a no-op (returns
    ``weights_daily`` unchanged). Tilt is floored at 0 (never flips a
    position negative/short) rather than allowed to go arbitrarily
    negative for a strongly below-average-conviction name in a long-only
    engine — meaning ``tilt_strength`` values above ~1.0 will start
    zeroing out any name more than ~1 std below the mean entirely, which
    is a legitimate thing to want but worth being deliberate about, not
    a side effect of picking a large number.

    Days where all covered&held names have identical conviction (std=0,
    e.g. very early in history before any real dispersion has developed)
    fall back to no tilt (z=0 for everyone) rather than dividing by zero.

    No-op if ``ichimoku_weights_daily`` is ``None`` or ``tilt_strength == 0.0``.
    """
    if ichimoku_weights_daily is None or tilt_strength == 0.0:
        return weights_daily

    held = weights_daily > 0
    ichimoku_aligned = ichimoku_weights_daily.reindex(index=weights_daily.index, columns=weights_daily.columns)
    has_coverage = ichimoku_aligned.notna()
    covered_and_held = held & has_coverage

    masked_conviction = ichimoku_aligned.where(covered_and_held)
    row_mean_conviction = masked_conviction.mean(axis=1).fillna(0.0)
    row_std_conviction = masked_conviction.std(axis=1).replace(0.0, np.nan)  # avoid div-by-zero

    # Uncovered names (or an entirely-uncovered day) get filled with that
    # day's own mean conviction across covered&held names -> their
    # z-scored term is exactly 0 -> tilt exactly 1 -> unchanged relative
    # weight.
    conviction_filled = ichimoku_aligned.mask(~has_coverage, row_mean_conviction, axis=0)

    demeaned = conviction_filled.sub(row_mean_conviction, axis=0)
    z_scored = demeaned.div(row_std_conviction, axis=0).fillna(0.0)  # std=NaN (was 0) -> z=0, no tilt
    tilt = (1.0 + tilt_strength * z_scored).clip(lower=0.0)

    tilted = weights_daily * tilt
    original_row_exposure = weights_daily.sum(axis=1)
    tilted_row_sum = tilted.sum(axis=1).replace(0, np.nan)
    renormalized = tilted.div(tilted_row_sum, axis=0).mul(original_row_exposure, axis=0).fillna(0.0)
    return renormalized


def build_geometric_overlay_weights(
    crash_flag: pd.Series,
    crash_exposure_multiplier: float = 0.5,
    benchmark_col: str = "benchmark",
) -> pd.DataFrame:
    """Standalone benchmark exposure driven purely by the geometric
    wedge-product crash-risk flag (``regime_detection/geometric_signal.py``'s
    ``geometric_crash_risk_flag``) — deliberately NOT the GMM/KMeans/HMM
    regime label (see ``regime_detection/pipeline.py``'s docstring: the two
    are computed completely independently). 100% exposure on every day the
    flag isn't active; cut to ``crash_exposure_multiplier`` on every day it
    is (``flag >= 0.5``, so it works whether the flag arrives as a strict
    0.0/1.0 or has picked up any float noise from an upstream join).

    This isolates the geometric signal's *own* effect end-to-end — run it
    through ``attribution.run_component_backtests`` alongside
    ``regime_only`` for a direct, apples-to-apples comparison of "does this
    signal alone do anything useful," independent of whether it happens to
    correlate with what the GMM already does.

    NaN in ``crash_flag`` (e.g. before the signal's rolling windows warm up,
    or before a full year of sector history exists — see
    ``geometric_signal.compute_geometric_crash_features``) is treated as "no
    flag" (full exposure), not as missing/excluded — same convention as
    ``build_regime_exposure_weights``'s "unmapped regime defaults to 1.0".
    """
    flag = crash_flag.fillna(0.0)
    exposure = np.where(flag >= 0.5, crash_exposure_multiplier, 1.0)
    return pd.DataFrame({benchmark_col: exposure}, index=crash_flag.index)


def apply_geometric_overlay(
    weights_daily: pd.DataFrame,
    crash_flag: pd.Series | None,
    crash_exposure_multiplier: float = 0.5,
) -> pd.DataFrame:
    """Multiply an already daily-aligned weights matrix (e.g. the combined
    fundamentals x regime weights from ``combine_regime_and_fundamentals``)
    by the same geometric exposure factor used in
    ``build_geometric_overlay_weights`` — the "apply it on top of
    everything" hook: this can independently cut total exposure further on
    top of whatever fundamentals selection and regime timing already
    decided, without the geometric signal ever having influenced either of
    those two components (it was computed and joined onto the regime
    history entirely separately — see ``regime_detection/pipeline.py``).

    No-op (returns ``weights_daily`` unchanged) if ``crash_flag`` is
    ``None`` — i.e. the geometric signal isn't configured/enabled, so
    ``combined`` behaves exactly as it did before this overlay existed.
    """
    if crash_flag is None:
        return weights_daily
    flag = crash_flag.reindex(weights_daily.index).fillna(0.0)
    exposure_multiplier = pd.Series(
        np.where(flag >= 0.5, crash_exposure_multiplier, 1.0), index=weights_daily.index
    )
    return weights_daily.multiply(exposure_multiplier, axis=0)
