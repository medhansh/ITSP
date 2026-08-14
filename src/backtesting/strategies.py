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



def _assign_weights(
    symbols, date, weighting: str, risk_panel: pd.DataFrame | None,
    max_position_weight: float | None,
) -> pd.Series:
    """Split 1.0 across ``symbols``, either equally or inverse to risk."""
    symbols = list(symbols)
    n = len(symbols)
    if n == 0:
        return pd.Series(dtype=float)

    if weighting == "equal" or risk_panel is None:
        w = pd.Series(1.0 / n, index=symbols)
    else:
        if date in risk_panel.index:
            risk = risk_panel.loc[date].reindex(symbols)
        else:
            # Use the most recent risk row at or before this date; the panel is
            # daily and rebalance dates can fall on non-trading days.
            prior = risk_panel.index[risk_panel.index <= date]
            risk = risk_panel.loc[prior[-1]].reindex(symbols) if len(prior) else pd.Series(np.nan, index=symbols)

        risk = pd.to_numeric(risk, errors="coerce")
        risk[risk <= 0] = np.nan
        if risk.notna().sum() == 0:
            logger.warning(
                "_assign_weights: no usable risk values on %s -- falling back to equal weighting "
                "for this date.", date,
            )
            w = pd.Series(1.0 / n, index=symbols)
        else:
            # Missing/invalid risk -> that date's median, so a data gap neither
            # drops the name nor hands it an outsized inverse-risk weight.
            risk = risk.fillna(risk.median())
            inv = 1.0 / risk
            w = inv / inv.sum()

    if max_position_weight is not None:
        w = w.clip(upper=max_position_weight)
    return w



def build_volatility_target_exposure(
    benchmark_prices: pd.Series,
    target_vol: float = 0.15,
    vol_window: int = 21,
    max_exposure: float = 1.0,
    min_exposure: float = 0.0,
    benchmark_col: str = "benchmark",
) -> pd.DataFrame:
    """Continuous volatility-targeted exposure: ``target_vol / realized_vol``,
    clipped to ``[min_exposure, max_exposure]``.

    **Why this exists.** It is the direct alternative to
    ``build_regime_exposure_weights``, and a test of whether discretizing
    the market into four clustered states was ever adding anything over the
    raw volatility measurement underneath it. This uses NO regime model at
    all: no clustering, no labels, no states, just a smooth function of
    trailing realized volatility. If it matches or beats regime-based
    exposure scaling, the entire regime subsystem can be removed rather
    than maintained -- and ``regime_only`` currently returns less than the
    benchmark, so that possibility is live rather than hypothetical.

    Strictly backward-looking: the volatility window ends at t inclusive,
    same PIT convention as every other rolling feature here, so reading
    this at any date only uses prices from on or before that date.

    ``max_exposure`` defaults to 1.0 (never leveraged), matching the
    long-only, cash-only constraint of the rest of this project. Raising it
    above 1.0 implies borrowing, which the backtest engine does not model
    financing costs for -- so a levered result would be optimistic in a way
    the engine cannot see.

    Early dates without a full ``vol_window`` of history get NaN, which the
    caller should treat as "not yet tradeable" rather than backfilling.
    """
    returns = np.log(benchmark_prices).diff()
    realized_vol = returns.rolling(vol_window, min_periods=max(vol_window // 2, 5)).std() * np.sqrt(252)
    exposure = (target_vol / realized_vol.replace(0, np.nan)).clip(lower=min_exposure, upper=max_exposure)
    n_capped = int((target_vol / realized_vol.replace(0, np.nan) > max_exposure).sum())
    if n_capped:
        logger.info(
            "build_volatility_target_exposure: %d/%d days wanted exposure above the %.2f cap "
            "(realized vol below the %.2f target) and were clipped.",
            n_capped, int(realized_vol.notna().sum()), max_exposure, target_vol,
        )
    return pd.DataFrame({benchmark_col: exposure}, index=benchmark_prices.index)


def build_fundamental_portfolio_weights(
    scores_by_date: pd.DataFrame,
    top_quantile: float = 0.2,
    min_positions: int = 5,
    max_sector_weight: float | None = None,
    max_position_weight: float | None = None,
    exclude_bottom_quantile: float | None = None,
    weighting: str = "equal",
    risk_panel: pd.DataFrame | None = None,
    exclude_riskiest_quantile: float | None = None,
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

    ``exclude_bottom_quantile`` (optional, e.g. ``0.2``): switches this
    function from SELECTION mode to EXCLUSION mode. Instead of holding the
    top ``top_quantile``, hold everything EXCEPT the worst
    ``exclude_bottom_quantile`` by composite score, equal-weighted.
    ``top_quantile`` is ignored entirely when this is set. ``None``
    (default) is exact pre-existing behavior.

    **Why exclusion mode exists.** The top-quantile sweep
    (``scripts/sweep_top_quantile.py``) came back U-shaped, with the
    project's long-standing 0.2 setting sitting almost exactly at the
    BOTTOM of the curve -- the worst risk-adjusted point in the whole
    range, with better results in both directions. The right-hand side is
    the informative part: holding 400 of 500 names still produced ~17.4%
    CAGR against the benchmark's ~11.6%. Excess return of that size cannot
    come from stock-picking skill when you hold 80% of the universe, so it
    must be coming from what is being LEFT OUT. That points at the
    composite score being substantially better at identifying names to
    AVOID than names to buy -- which, if true, means the score has been
    used the wrong way round for the whole project.

    Exclusion mode tests that hypothesis directly and much more cheaply
    than a wide quantile does: it targets the bottom tail explicitly
    rather than approaching it by holding almost everything. It is also
    conceptually the same idea as ``fundamental_analysis.forensic_gates``
    (drop bad names before ranking), just expressed as a score cut rather
    than as hard rule-based gates.

    Sector and position caps are deliberately NOT applied in exclusion
    mode: with most of the universe held, a sector cap would force
    arbitrary further exclusions unrelated to the score, contaminating
    exactly the effect being measured. Caps still apply normally in
    selection mode.

    ``weighting`` (default ``"equal"``): how capital is split across the
    SELECTED names. ``"equal"`` is the historical behavior (1/n each).
    ``"inverse_risk"`` sizes each position at ``(1/risk_i) / sum(1/risk_j)``,
    so a name with twice the risk gets half the capital. Requires
    ``risk_panel``.

    **Why inverse-risk weighting is worth testing.** Every portfolio in this
    project has been equal-weighted, which is a deliberate non-decision: it
    asserts that every selected name contributes equal risk. That is plainly
    false -- a 0.4-beta staples name and a 1.8-beta smallcap do not. More
    importantly, the accumulated evidence says the composite score carries
    RETURN information but essentially no RISK information: concentrating
    into the top 10 names gave the best CAGR in the quantile sweep with a
    -55% drawdown; excluding the bottom half by score gave more CAGR with a
    WORSE drawdown in 6 of 6 walk-forward folds; and the original tail-risk
    diagnostic found the selected portfolio's worst-decile crash return was
    statistically indistinguishable from random selection.

    Inverse-risk weighting responds to that directly: keep using the score
    for what it demonstrably does (rank expected return) and take position
    SIZING from a measure that actually contains risk information. This is
    the always-on generalization of ``apply_beta_rotation``, which was the
    one mechanism on this project to survive walk-forward and which worked
    for exactly this reason -- it stopped asking the score for risk
    information and used trailing beta instead.

    ``risk_panel``: daily (date x symbol) risk measure, higher = riskier.
    Trailing realized volatility or trailing beta both work; beta reuses
    ``fundamental_analysis.orthogonalization.compute_rolling_beta_panel``.
    Values are read at each REBALANCE date only. Non-positive or missing
    values fall back to that date's median risk across selected names
    rather than being dropped or given infinite weight -- same "absent
    evidence is not evidence" convention used by ``forensic_gates`` and
    ``orthogonalization.residualize_against_beta``. If the panel is missing
    entirely for a date, that date falls back to equal weighting and is
    logged.

    ``exclude_riskiest_quantile`` (optional, e.g. ``0.2``): a RISK SCREEN
    applied to the candidate pool BEFORE score-based selection. Drops the
    riskiest X% of the universe by ``risk_panel``, then selects normally
    from what remains. Requires ``risk_panel``. ``None`` (default) is exact
    pre-existing behavior.

    **Why this is different from ``exclude_bottom_quantile``.** That one
    drops the worst names by COMPOSITE SCORE and was tested and rejected:
    walk-forward showed worse drawdown in 6 of 6 folds and a Sharpe delta
    of +0.006 against holding everything. The reason it failed is the
    reason this exists -- the score carries return information but
    essentially no risk information, so screening on it cannot remove
    risk. This screens on realized risk instead, which by construction
    does contain that information.

    Ordering matters and is deliberate: the risk screen runs FIRST, on the
    full universe, and score-based selection then picks from the survivors.
    Screening after selection would just shrink an already-small portfolio;
    screening first lets the score reach further down its ranking to
    replace what the screen removed, keeping the position count intact.

    Returns a sparse weights DataFrame indexed by rebalance date, columns=symbols,
    NaN/0 for non-selected stocks. Feed through
    ``engine.align_weights_to_returns`` to forward-fill onto a daily index.
    """
    if exclude_riskiest_quantile is not None:
        if risk_panel is None:
            raise ValueError(
                "exclude_riskiest_quantile requires risk_panel. Passing None would silently "
                "apply no screen at all, which looks identical to a normal run."
            )
        if not 0.0 <= exclude_riskiest_quantile < 1.0:
            raise ValueError(
                f"exclude_riskiest_quantile must be in [0, 1), got {exclude_riskiest_quantile}"
            )
    if weighting not in ("equal", "inverse_risk"):
        raise ValueError(f"weighting must be 'equal' or 'inverse_risk', got {weighting!r}")
    if weighting == "inverse_risk" and risk_panel is None:
        raise ValueError(
            "weighting='inverse_risk' requires risk_panel. Passing None would silently fall back "
            "to equal weighting for every date, which looks identical to a normal run."
        )
    required = {"date", "symbol", "composite_score"}
    missing = required - set(scores_by_date.columns)
    if missing:
        raise ValueError(f"scores_by_date is missing columns: {missing}")

    # Pre-screen universe size per date. n_select must be computed from THIS,
    # not from the post-screen pool: otherwise the risk screen silently shrinks
    # the portfolio (a 20% screen + 20% quantile would hold 16 names, not 20),
    # confounding "screened out risk" with "held fewer names".
    pre_screen_universe = scores_by_date.dropna(subset=["composite_score"]).groupby("date").size()

    if exclude_riskiest_quantile:
        kept_frames, n_before, n_after = [], 0, 0
        for date, g in scores_by_date.groupby("date"):
            n_before += len(g)
            if date in risk_panel.index:
                risk_row = risk_panel.loc[date]
            else:
                prior = risk_panel.index[risk_panel.index <= date]
                risk_row = risk_panel.loc[prior[-1]] if len(prior) else None
            if risk_row is None:
                # No risk data at or before this date -- screen nothing rather
                # than screening arbitrarily. Same "absent evidence is not
                # evidence" convention used elsewhere in this pipeline.
                kept_frames.append(g)
                n_after += len(g)
                continue
            risk = pd.to_numeric(g["symbol"].map(risk_row), errors="coerce")
            # Names with no risk value are KEPT: dropping them would screen on
            # data availability rather than on risk.
            cutoff = risk.quantile(1.0 - exclude_riskiest_quantile)
            keep_mask = risk.isna() | (risk <= cutoff)
            kept_frames.append(g[keep_mask.values])
            n_after += int(keep_mask.sum())
        scores_by_date = pd.concat(kept_frames, ignore_index=True)
        logger.info(
            "build_fundamental_portfolio_weights: RISK SCREEN dropped the riskiest %.0f%% of the "
            "candidate pool before score selection (%d -> %d candidate rows across all dates).",
            exclude_riskiest_quantile * 100, n_before, n_after,
        )

    has_sector = "sector" in scores_by_date.columns
    if max_sector_weight is not None and not has_sector:
        logger.warning(
            "build_fundamental_portfolio_weights: max_sector_weight=%.2f requested but no 'sector' "
            "column in scores_by_date -- sector cap will be a no-op for every date.", max_sector_weight,
        )

    if exclude_bottom_quantile is not None:
        if not 0.0 <= exclude_bottom_quantile < 1.0:
            raise ValueError(
                f"exclude_bottom_quantile must be in [0, 1), got {exclude_bottom_quantile}"
            )
        rows = {}
        for date, g in scores_by_date.groupby("date"):
            g = g.dropna(subset=["composite_score"])
            if g.empty:
                rows[date] = pd.Series(dtype=float)
                continue
            n_drop = int(len(g) * exclude_bottom_quantile)
            keep = g.sort_values("composite_score", ascending=False)
            if n_drop > 0:
                keep = keep.iloc[:-n_drop]
            if len(keep) < min_positions:
                # Never let the exclusion cut below the position floor; keep the
                # best `min_positions` instead of returning a degenerate book.
                keep = g.sort_values("composite_score", ascending=False).iloc[:min_positions]
            rows[date] = _assign_weights(
                keep["symbol"].values, date, weighting, risk_panel, max_position_weight,
            )
        out = pd.DataFrame(rows).T
        out.index.name = "date"
        logger.info(
            "build_fundamental_portfolio_weights: EXCLUSION mode -- dropping the worst %.0f%% by "
            "composite_score, holding the rest equal-weighted (mean %.0f names/date). "
            "top_quantile/sector/position caps are not applied in this mode.",
            exclude_bottom_quantile * 100, float((out > 0).sum(axis=1).mean()),
        )
        return out

    rows = {}
    for date, g in scores_by_date.groupby("date"):
        g = g.dropna(subset=["composite_score"])
        universe_n = int(pre_screen_universe.get(date, len(g)))
        n_select = max(min_positions, int(universe_n * top_quantile))
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

        # NOTE: max_position_weight is applied inside _assign_weights by
        # CLIPPING, letting gross exposure shrink rather than renormalizing
        # back to 100% -- renormalizing would defeat the point of the cap by
        # redistributing the capped amount among the same few names.
        rows[date] = _assign_weights(
            selected["symbol"].values, date, weighting, risk_panel, max_position_weight,
        )

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


def apply_beta_rotation(
    weights_daily: pd.DataFrame,
    beta_panel: pd.DataFrame | None,
    regime: pd.Series | None,
    stress_by_regime: dict | None = None,
    rotation_strength: float = 1.0,
) -> pd.DataFrame:
    """COMPOSITIONAL de-risking: in stressed regimes, rotate weight toward
    LOW-BETA names among those already held, while keeping total exposure
    EXACTLY unchanged. Risk comes down through *what* is held, never
    through *how much*.

    **Why this exists, and why it is structurally different from every
    other regime mechanism in this project.** Five separate regime-side
    attempts have now failed (consensus governor, factor-dispersion
    regimes, supervised regime prediction, regime-conditional
    technical_momentum weighting, and the combined-strategy interaction
    effect itself). Every one of them changed the regime LABEL — smoothed
    it, refit it on different features, predicted it with a supervised
    model — while the CONSUMPTION mechanism stayed identical throughout:
    a single scalar exposure multiplier, ``weights x E_t``, applied
    uniformly to everything (``combine_regime_and_fundamentals``). That is
    one degree of freedom with exactly one failure mode: when it is wrong,
    it is wrong about every holding simultaneously.

    The diagnosed drag is ``Cov(E_t, R_p) < 0``: exposure is cut precisely
    when quality names are most dislocated, so the portfolio takes the
    full drawdown at full exposure and captures only ``E_t`` of the
    rebound. This function makes that covariance term structurally
    incapable of being negative, because ``E_t`` is CONSTANT — total
    invested fraction is preserved to machine precision on every single
    day. De-risking still happens; it happens through portfolio beta.

    Mechanically this is the same "reallocate, never cut" shape as
    ``apply_ichimoku_conviction_tilt`` (and deliberately so — that
    docstring's "removing value vs adding value" distinction applies
    identically here), but tilting on trailing market beta in stressed
    regimes rather than on Ichimoku conviction every day.

    Per day t, for the set of held names H (``weights_daily[t] > 0``):

        stress_t   = stress_by_regime[regime_t]        # 0.0 = calm, 1.0 = max stress
        z_i        = (beta_i - mean(beta over H)) / std(beta over H)
        tilt_i     = max(0, 1 - rotation_strength * stress_t * z_i)
        w_i(new)   = w_i * tilt_i, RENORMALIZED so sum(w(new)) == sum(w) exactly

    Note the MINUS sign: high-beta names get tilted DOWN. On a calm day
    ``stress_t = 0``, so ``tilt_i = 1`` for every name and this is an exact
    no-op — the calm-regime portfolio is bit-identical to what it would
    have been without this function, which keeps the comparison against
    the baseline clean.

    Beta is z-scored across held names each day rather than used raw, for
    the same reason ``apply_ichimoku_conviction_tilt`` was fixed to
    z-score: it makes the tilt invariant to the absolute level and spread
    of beta on that day, so only the RELATIVE ordering within the held set
    drives reallocation. A day when every held name happens to be
    high-beta should rotate toward the least-high-beta of them, not
    collapse everything.

    ``beta_panel``: daily (date x symbol) trailing beta, from
    ``fundamental_analysis.orthogonalization.compute_rolling_beta_panel``
    — the same panel the beta-orthogonalization work already builds, reused
    rather than recomputed.

    ``stress_by_regime``: maps regime label -> stress level in [0, 1].
    Defaults to a linear ramp over the observed labels (calmest = 0.0,
    most stressed = 1.0). Deliberately NOT the same numbers as
    ``exposure_by_regime`` — that config says how much to DISINVEST, this
    says how hard to ROTATE, and conflating them would make the two
    mechanisms look more comparable than they are.

    ``rotation_strength`` (default 1.0): how much relative weight moves per
    standard deviation of beta spread, at full stress. At 1.0, a name 1
    standard deviation above the held set's mean beta gets 0x its base
    weight in maximum stress (fully rotated out) before renormalization;
    the ``max(0, ...)`` floor prevents negative weights.

    Names with no beta available that day keep a neutral tilt (z = 0, so
    ``tilt = 1``) rather than being dropped or penalized — same "absent
    evidence is not evidence" convention as ``forensic_gates`` and
    ``orthogonalization.residualize_against_beta``.

    No-op (returns ``weights_daily`` unchanged) if ``beta_panel`` or
    ``regime`` is ``None``, so ``combined`` behaves exactly as it did
    before this existed.

    **Status: experimental, unvalidated on real data as of writing.** Note
    the prior explicitly: this is the sixth regime-side attempt. It is
    worth trying because it is the first one to change the consumption
    mechanism rather than the label, but that reasoning is a hypothesis,
    not evidence.
    """
    if beta_panel is None or regime is None:
        return weights_daily

    if stress_by_regime is None:
        labels = sorted(pd.Series(regime.unique()).dropna().tolist())
        if len(labels) > 1:
            stress_by_regime = {lab: i / (len(labels) - 1) for i, lab in enumerate(labels)}
        else:
            stress_by_regime = {lab: 0.0 for lab in labels}
        logger.info("apply_beta_rotation: no stress_by_regime given, using linear ramp %s", stress_by_regime)

    stress = regime.reindex(weights_daily.index).map(stress_by_regime).fillna(0.0)
    beta = beta_panel.reindex(index=weights_daily.index, columns=weights_daily.columns)

    held = weights_daily > 0
    beta_held = beta.where(held)
    mean_beta = beta_held.mean(axis=1)
    std_beta = beta_held.std(axis=1)

    z = beta_held.sub(mean_beta, axis=0).div(std_beta.replace(0, np.nan), axis=0)
    z = z.fillna(0.0)  # no beta, or a degenerate day with no cross-sectional spread -> neutral

    tilt = (1.0 - z.mul(stress * rotation_strength, axis=0)).clip(lower=0.0)
    tilted = weights_daily.multiply(tilt).where(held, 0.0)

    # Renormalize back to the ORIGINAL daily total, so total exposure is
    # preserved exactly. Days where the tilt zeroed everything (possible
    # only at extreme rotation_strength) fall back to the untilted weights
    # rather than producing an all-cash day this function was never meant
    # to create.
    original_total = weights_daily.sum(axis=1)
    tilted_total = tilted.sum(axis=1)
    degenerate = tilted_total <= 0
    if degenerate.any():
        logger.warning(
            "apply_beta_rotation: %d day(s) had every held name tilted to zero (rotation_strength=%.2f "
            "is very high) -- falling back to untilted weights on those days rather than going to cash.",
            int(degenerate.sum()), rotation_strength,
        )
    scale = (original_total / tilted_total.replace(0, np.nan)).fillna(0.0)
    out = tilted.multiply(scale, axis=0)
    out.loc[degenerate] = weights_daily.loc[degenerate]
    return out


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
