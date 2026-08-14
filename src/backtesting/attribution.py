"""Run each strategy component in isolation, plus combined, and attribute the
combined result back to its regime-detection and fundamental-analysis pieces.

Four returns series are always produced:
  - "benchmark"          — 100% invested in the index, always. The baseline
                            everything else is measured against.
  - "fundamentals_only"  — top-quantile fundamental-score stock selection,
                            fully invested regardless of market regime.
  - "regime_only"        — 100% benchmark, exposure scaled by the GMM/
                            KMeans/HMM-detected regime (no stock selection).
  - "combined"           — fundamentals stock selection, exposure scaled by
                            regime (both signals together) — and, if a
                            geometric crash flag is supplied, ALSO scaled by
                            that overlay on top ("on top of everything" — see
                            strategies.apply_geometric_overlay).

A fifth is produced only when ``geometric_crash_flag`` is passed:
  - "geometric_overlay_only" — 100% benchmark, exposure scaled purely by the
                            geometric wedge-product crash-risk flag
                            (``regime_detection/geometric_signal.py``) —
                            NOT the GMM regime label; computed and applied
                            completely independently of it (see
                            ``regime_detection/pipeline.py``'s docstring).
                            Isolates the geometric signal's own standalone
                            effect for direct comparison against
                            ``regime_only``.

A sixth and seventh are produced only when ``active_regime`` is passed
(the ``regime_detection.consensus_governor`` output — see
``docs/regime_detection_spec.md``):
  - "governed_regime_only"  — same construction as "regime_only" but using
                            ``active_regime`` (entropy-gated, persistence-
                            and hysteresis-governed) instead of the raw
                            per-bar ``regime`` label. Direct, apples-to-
                            apples comparison against "regime_only" to see
                            whether reducing over-switching actually
                            changes performance.
  - "governed_combined"    — same construction as "combined" but with the
                            regime-timing leg driven by ``active_regime``
                            instead of ``regime``. Direct comparison
                            against "combined".

An eighth and ninth are produced only when ``ichimoku_weights`` is passed
(the daily weight matrix from ``adaptive_ichimoku.build_ichimoku_weights``
— see ``docs/backtesting_spec.md``):
  - "ichimoku_only"        — the Ichimoku signal's own weights, backtested
                            standalone (equal-weighted across whichever
                            symbols are currently triple-confirmed
                            bullish). Isolates the signal's own effect,
                            same convention as ``fundamentals_only``/
                            ``regime_only``.
  - "combined_with_ichimoku" — "combined", further scaled by Ichimoku
                            confirmation. Default construction (``ichimoku_mode="breadth_scalar"``,
                            via ``strategies.apply_ichimoku_breadth_scalar``):
                            total exposure scaled by what fraction of the
                            CURRENTLY-HELD names Ichimoku confirms bullish
                            today — every selected name stays in the
                            portfolio, only aggregate exposure moves.
                            ``ichimoku_mode="hard_gate"`` (via
                            ``strategies.apply_ichimoku_gate``) is also
                            available for comparison but is NOT recommended
                            — see that function's docstring for why it
                            produces severe cash drag (beta ~0.19 in the
                            first real backtest) rather than a useful
                            confirmation filter. Direct comparison against
                            "combined" either way, to see whether requiring
                            technical confirmation on top of
                            fundamentals+regime selection helps or costs
                            exposure.

A tenth, "combined_ichimoku_tilted", is produced whenever ``ichimoku_weights``
is passed AND ``ichimoku_tilt_strength`` is nonzero:
  - "combined_ichimoku_tilted" — "combined", REALLOCATED among its
                            currently-held names by relative Ichimoku
                            conviction (``strategies.apply_ichimoku_conviction_tilt``),
                            with TOTAL exposure held exactly fixed to
                            "combined"'s own — a structurally different
                            mechanism from "combined_with_ichimoku" (which
                            can only ever cut exposure): this one can only
                            move capital between names "combined" already
                            selected, never invest less overall, and never
                            add a name the base selection didn't pick.
                            Real-data motivation: ``ichimoku_only``
                            standalone was the single best-performing
                            component found in initial testing, while
                            gating ``combined`` by it produced a large CAGR
                            loss despite better Sharpe/drawdown — this
                            component tests whether the conviction signal's
                            real information is better captured by
                            reallocation than by exposure-cutting.

See docs/backtesting_spec.md for the full methodology and the important
caveat on why this decomposition is additive/approximate, not exact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.engine import align_weights_to_returns
from src.backtesting.engine import run_backtest as run_backtest_custom
from src.backtesting.metrics import cagr, performance_summary
from src.backtesting.strategies import (
    apply_beta_rotation,
    apply_geometric_overlay,
    apply_ichimoku_breadth_scalar,
    apply_ichimoku_conviction_tilt,
    apply_ichimoku_gate,
    build_fundamental_portfolio_weights,
    build_geometric_overlay_weights,
    build_regime_exposure_weights,
    combine_regime_and_fundamentals,
)
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _run_backtest(
    weights: pd.DataFrame,
    returns_panel: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    transaction_cost_bps: float,
    engine: str,
    lag_days: int = 1,
) -> dict[str, pd.Series]:
    """Engine dispatch used by every component backtest below. ``engine`` is
    ``"vectorbt"`` (default, with automatic fallback to the custom engine if
    vectorbt isn't installed — see vbt_engine.py) or ``"custom"`` to force
    engine.py's dependency-free implementation directly. ``lag_days``
    (default 1) is forwarded to whichever engine runs — see
    ``engine.run_backtest``'s docstring for the same-bar look-ahead bug this
    prevents; every weight matrix built anywhere in this project (regime
    exposure, fundamentals selection, technical signals) is computed using
    data through a given day's close and therefore cannot be assumed
    tradeable until the following day.
    """
    if engine == "custom" or prices_panel is None:
        return run_backtest_custom(returns_panel, weights, transaction_cost_bps, lag_days=lag_days)
    from src.backtesting.vbt_engine import run_backtest_with_fallback

    return run_backtest_with_fallback(
        prices_panel, returns_panel, weights, transaction_cost_bps, engine=engine, lag_days=lag_days
    )


def run_component_backtests(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    scores_by_date: pd.DataFrame,
    regime: pd.Series,
    exposure_by_regime: dict,
    top_quantile: float = 0.2,
    min_positions: int = 5,
    transaction_cost_bps: float = 0.0,
    stock_prices: pd.DataFrame | None = None,
    benchmark_prices: pd.Series | None = None,
    engine: str = "vectorbt",
    geometric_crash_flag: pd.Series | None = None,
    crash_exposure_multiplier: float = 0.5,
    lag_days: int = 1,
    active_regime: pd.Series | None = None,
    ichimoku_weights: pd.DataFrame | None = None,
    ichimoku_mode: str = "breadth_scalar",
    ichimoku_confirmation_floor: float = 0.0,
    ichimoku_tilt_strength: float = 0.0,
    max_sector_weight: float | None = None,
    max_position_weight: float | None = None,
    beta_panel: pd.DataFrame | None = None,
    stress_by_regime: dict | None = None,
    rotation_strength: float = 1.0,
    exclude_bottom_quantile: float | None = None,
    weighting: str = "equal",
    risk_panel: pd.DataFrame | None = None,
    exclude_riskiest_quantile: float | None = None,
    vol_target_exposure: pd.Series | None = None,
) -> dict[str, dict[str, pd.Series]]:
    """Run all component backtests and return {component_name: run_backtest output}.

    ``stock_prices``/``benchmark_prices`` (optional): actual price levels,
    not just returns — required for the vectorbt engine (see
    vbt_engine.py). If omitted, this always uses the custom engine
    regardless of ``engine``, since price-based order sizing has no
    returns-only equivalent.

    ``geometric_crash_flag`` (optional): the ``geometric_crash_risk_flag``
    column from ``regime_detection.pipeline.run_pipeline``'s output (present
    only if ``regime_detection.geometric_signal.enabled`` was set). When
    given, adds the standalone ``geometric_overlay_only`` component and
    applies the same overlay multiplicatively on top of ``combined`` — see
    module docstring. When ``None`` (the default / signal disabled),
    behavior is identical to before this overlay existed: 4 components,
    ``combined`` untouched.

    ``active_regime`` (optional): the ``active_regime`` column from
    ``regime_detection.pipeline.run_pipeline``'s output (present only if
    ``regime_detection.consensus_governor.enabled`` was set) — a mix of int
    regime ids and the string ``"transitional"``. When given, adds
    ``governed_regime_only`` and ``governed_combined``, built exactly like
    ``regime_only``/``combined`` but using this series instead of ``regime``
    — see module docstring. When ``None`` (default / governor disabled),
    behavior is unchanged: no governed components, ``regime_only``/
    ``combined`` untouched. Purely additive, same pattern as
    ``geometric_crash_flag`` — the raw ``regime_only``/``combined``
    components are never modified by this.

    ``ichimoku_weights`` (optional): the daily weight matrix from
    ``adaptive_ichimoku.build_ichimoku_weights`` (columns = whichever
    symbols had usable OHLC data — not necessarily the full universe). When
    given, adds ``ichimoku_only`` (standalone) and ``combined_with_ichimoku``
    (``combined`` scaled by Ichimoku confirmation — see ``ichimoku_mode``).
    ``None`` (default) reproduces behavior exactly as before Ichimoku was
    wired in — ``combined`` itself is never modified by this parameter,
    only the new ``combined_with_ichimoku`` component is.

    ``ichimoku_mode`` (default ``"breadth_scalar"``): which
    ``strategies.apply_ichimoku_*`` function builds ``combined_with_ichimoku``.
    ``"breadth_scalar"`` (recommended, default) scales total exposure by
    the fraction of held names Ichimoku confirms — see
    ``strategies.apply_ichimoku_breadth_scalar``. ``"hard_gate"`` zeroes
    out individual unconfirmed names — see ``strategies.apply_ichimoku_gate``'s
    docstring for why this produced severe cash drag (beta ~0.19) in the
    first real backtest; kept only for explicit comparison.

    ``ichimoku_confirmation_floor`` (default 0.0): forwarded to
    ``apply_ichimoku_breadth_scalar`` when ``ichimoku_mode="breadth_scalar"``
    — minimum exposure fraction even on a zero-confirmation day. Ignored
    for ``ichimoku_mode="hard_gate"``.

    ``ichimoku_tilt_strength`` (default 0.0, i.e. off): when nonzero and
    ``ichimoku_weights`` is given, additionally adds a
    ``combined_ichimoku_tilted`` component via
    ``strategies.apply_ichimoku_conviction_tilt`` — see module docstring.
    Unlike ``ichimoku_mode``, this is fully independent of/additional to
    ``combined_with_ichimoku``, not a mode switch on it — both can be
    computed side by side for direct comparison.

    ``max_sector_weight`` / ``max_position_weight`` (both default ``None``,
    i.e. no cap — exact pre-cap behavior): forwarded to
    ``strategies.build_fundamental_portfolio_weights``. Applies to EVERY
    component that builds on the fundamentals selection (``fundamentals_only``,
    ``combined``, and anything downstream of ``combined`` like the Ichimoku
    variants) since they all share the same underlying selected basket —
    this is a property of WHICH stocks get chosen, not a separate overlay.
    See that function's docstring for why this was added: fundamentals_only's
    max drawdown has consistently been worse than the raw benchmark's
    across every real run so far, a concentration problem this directly
    targets.

    ``lag_days`` (default 1): forwarded to every component backtest — see
    ``engine.run_backtest``'s docstring for the same-bar look-ahead bug this
    prevents. Do not set to 0 without a specific reason.
    """
    common_index = stock_returns.index.intersection(benchmark_returns.index).intersection(
        regime.index
    )
    stock_returns = stock_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    regime = regime.loc[common_index]
    stock_prices_aligned = stock_prices.loc[common_index] if stock_prices is not None else None
    benchmark_price_frame = (
        benchmark_prices.loc[common_index].to_frame("benchmark") if benchmark_prices is not None else None
    )
    crash_flag_aligned = (
        geometric_crash_flag.reindex(common_index) if geometric_crash_flag is not None else None
    )
    active_regime_aligned = (
        active_regime.reindex(common_index) if active_regime is not None else None
    )
    ichimoku_weights_aligned = (
        ichimoku_weights.reindex(index=common_index).fillna(0.0) if ichimoku_weights is not None else None
    )

    # --- benchmark: fully invested, no signal ---
    benchmark_weights = align_weights_to_returns(
        pd.DataFrame({"benchmark": 1.0}, index=common_index[:1]),
        common_index,
        pd.Index(["benchmark"]),
    )
    benchmark_result = _run_backtest(
        benchmark_weights, pd.DataFrame({"benchmark": benchmark_returns}),
        benchmark_price_frame, transaction_cost_bps, engine, lag_days,
    )

    # --- regime-only: benchmark exposure scaled by regime, no stock selection ---
    regime_exposure_sparse = build_regime_exposure_weights(regime, exposure_by_regime)
    regime_exposure_daily = align_weights_to_returns(
        regime_exposure_sparse, common_index, pd.Index(["benchmark"])
    )
    regime_only_result = _run_backtest(
        regime_exposure_daily, pd.DataFrame({"benchmark": benchmark_returns}),
        benchmark_price_frame, transaction_cost_bps, engine, lag_days,
    )

    # --- fundamentals-only: top-quantile stock selection, always fully invested ---
    fundamental_weights_sparse = build_fundamental_portfolio_weights(
        scores_by_date, top_quantile=top_quantile, min_positions=min_positions,
        max_sector_weight=max_sector_weight, max_position_weight=max_position_weight,
        exclude_bottom_quantile=exclude_bottom_quantile,
        weighting=weighting, risk_panel=risk_panel,
        exclude_riskiest_quantile=exclude_riskiest_quantile,
    )
    fundamental_weights_daily = align_weights_to_returns(
        fundamental_weights_sparse, common_index, stock_returns.columns
    )
    fundamentals_only_result = _run_backtest(
        fundamental_weights_daily, stock_returns, stock_prices_aligned, transaction_cost_bps, engine, lag_days,
    )

    # --- combined: fundamentals selection, exposure scaled by regime, THEN by
    # the geometric overlay on top (no-op if crash_flag_aligned is None) ---
    regime_exposure_for_stocks = align_weights_to_returns(
        regime_exposure_sparse.rename(columns={"benchmark": "_exposure"}),
        common_index,
        pd.Index(["_exposure"]),
    )["_exposure"]
    combined_weights_daily = combine_regime_and_fundamentals(
        fundamental_weights_daily, regime_exposure_for_stocks
    )
    combined_weights_daily = apply_geometric_overlay(
        combined_weights_daily, crash_flag_aligned, crash_exposure_multiplier
    )
    combined_result = _run_backtest(
        combined_weights_daily, stock_returns, stock_prices_aligned, transaction_cost_bps, engine, lag_days,
    )

    # --- volatility-targeted arms: the no-regime-model alternative to
    # regime exposure scaling. `vol_target_only` is the direct
    # apples-to-apples counterpart of `regime_only` (both scale benchmark
    # exposure, one from clustered states and one from raw trailing vol),
    # and `combined_vol_target` is the counterpart of `combined`. If these
    # match or beat their regime equivalents, the clustering subsystem is
    # not earning its complexity. Purely additive -- regime arms unchanged. ---
    if vol_target_exposure is not None:
        vt = vol_target_exposure.reindex(common_index).ffill().fillna(0.0)
        vt_bench = pd.DataFrame({"benchmark": vt}, index=common_index)
        results_vt_only = _run_backtest(
            vt_bench, pd.DataFrame({"benchmark": benchmark_returns}),
            None, transaction_cost_bps, engine, lag_days,
        )
        vt_for_stocks = pd.DataFrame(
            np.tile(vt.values[:, None], (1, len(stock_returns.columns))),
            index=common_index, columns=stock_returns.columns,
        )
        vt_combined = fundamental_weights_daily * vt_for_stocks

    results = {
        "benchmark": benchmark_result,
        "regime_only": regime_only_result,
        "fundamentals_only": fundamentals_only_result,
        "combined": combined_result,
    }
    if vol_target_exposure is not None:
        results["vol_target_only"] = results_vt_only
        results["combined_vol_target"] = _run_backtest(
            vt_combined, stock_returns, stock_prices_aligned,
            transaction_cost_bps, engine, lag_days,
        )

    # --- fundamentals_beta_rotated: COMPOSITIONAL de-risking. Same
    # fundamentals selection, ALWAYS 100% invested (no regime exposure
    # scaling at all), but rotated toward low-beta names within the
    # selection during stressed regimes. This is the first regime mechanism
    # in this project that changes WHAT is held rather than HOW MUCH --
    # see strategies.apply_beta_rotation's docstring. Compare it against
    # `fundamentals_only` (same selection, no rotation) to isolate the
    # rotation effect, and against `combined` (same regime info consumed as
    # exposure cuts instead) to compare the two mechanisms head to head.
    # Purely additive: `combined` is never modified by this. ---
    if beta_panel is not None:
        beta_rotated_weights = apply_beta_rotation(
            fundamental_weights_daily, beta_panel, regime,
            stress_by_regime=stress_by_regime, rotation_strength=rotation_strength,
        )
        results["fundamentals_beta_rotated"] = _run_backtest(
            beta_rotated_weights, stock_returns, stock_prices_aligned,
            transaction_cost_bps, engine, lag_days,
        )

    # --- geometric_overlay_only: pure benchmark exposure driven by the crash
    # flag ALONE (not the GMM regime) — only produced when the signal was
    # actually supplied, so a disabled signal doesn't clutter the report ---
    if crash_flag_aligned is not None:
        geometric_exposure_sparse = build_geometric_overlay_weights(
            crash_flag_aligned, crash_exposure_multiplier
        )
        geometric_exposure_daily = align_weights_to_returns(
            geometric_exposure_sparse, common_index, pd.Index(["benchmark"])
        )
        results["geometric_overlay_only"] = _run_backtest(
            geometric_exposure_daily, pd.DataFrame({"benchmark": benchmark_returns}),
            benchmark_price_frame, transaction_cost_bps, engine, lag_days,
        )

    # --- governed_regime_only / governed_combined: same construction as
    # regime_only/combined, but driven by the consensus-governor's
    # active_regime (entropy-gated, persistence + hysteresis) instead of
    # the raw per-bar regime label -- only computed when active_regime was
    # actually supplied (regime_detection.consensus_governor.enabled) ---
    if active_regime_aligned is not None:
        governed_exposure_sparse = build_regime_exposure_weights(
            active_regime_aligned, exposure_by_regime
        )
        governed_exposure_daily = align_weights_to_returns(
            governed_exposure_sparse, common_index, pd.Index(["benchmark"])
        )
        results["governed_regime_only"] = _run_backtest(
            governed_exposure_daily, pd.DataFrame({"benchmark": benchmark_returns}),
            benchmark_price_frame, transaction_cost_bps, engine, lag_days,
        )

        governed_exposure_for_stocks = align_weights_to_returns(
            governed_exposure_sparse.rename(columns={"benchmark": "_exposure"}),
            common_index,
            pd.Index(["_exposure"]),
        )["_exposure"]
        governed_combined_weights_daily = combine_regime_and_fundamentals(
            fundamental_weights_daily, governed_exposure_for_stocks
        )
        governed_combined_weights_daily = apply_geometric_overlay(
            governed_combined_weights_daily, crash_flag_aligned, crash_exposure_multiplier
        )
        results["governed_combined"] = _run_backtest(
            governed_combined_weights_daily, stock_returns, stock_prices_aligned,
            transaction_cost_bps, engine, lag_days,
        )

    # --- ichimoku_only / combined_with_ichimoku: standalone Ichimoku signal
    # backtest, plus "combined" gated by Ichimoku confirmation -- only
    # computed when ichimoku_weights was actually supplied
    # (technical_signals.ichimoku.enabled) ---
    if ichimoku_weights_aligned is not None:
        ichimoku_weights_full = ichimoku_weights_aligned.reindex(columns=stock_returns.columns, fill_value=0.0)
        results["ichimoku_only"] = _run_backtest(
            ichimoku_weights_full, stock_returns, stock_prices_aligned, transaction_cost_bps, engine, lag_days,
        )

        combined_with_ichimoku_weights_daily = (
            apply_ichimoku_gate(combined_weights_daily, ichimoku_weights_aligned)
            if ichimoku_mode == "hard_gate"
            else apply_ichimoku_breadth_scalar(
                combined_weights_daily, ichimoku_weights_aligned, floor=ichimoku_confirmation_floor
            )
        )
        results["combined_with_ichimoku"] = _run_backtest(
            combined_with_ichimoku_weights_daily, stock_returns, stock_prices_aligned,
            transaction_cost_bps, engine, lag_days,
        )

        if ichimoku_tilt_strength != 0.0:
            combined_ichimoku_tilted_weights_daily = apply_ichimoku_conviction_tilt(
                combined_weights_daily, ichimoku_weights_aligned, tilt_strength=ichimoku_tilt_strength
            )
            results["combined_ichimoku_tilted"] = _run_backtest(
                combined_ichimoku_tilted_weights_daily, stock_returns, stock_prices_aligned,
                transaction_cost_bps, engine, lag_days,
            )

    return results


def compute_attribution_table(
    component_results: dict[str, dict[str, pd.Series]],
    benchmark_key: str = "benchmark",
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Performance summary (CAGR, Sharpe, drawdown, etc.) for every component,
    plus each component's CAGR spread over the benchmark."""
    bench_returns = (
        benchmark_returns if benchmark_returns is not None else component_results[benchmark_key]["returns"]
    )
    rows = {}
    for name, result in component_results.items():
        summary = performance_summary(result["returns"], benchmark_returns=bench_returns)
        summary["excess_cagr_vs_benchmark"] = summary["cagr"] - cagr(bench_returns)
        rows[name] = summary
    return pd.DataFrame(rows).T


def compute_return_decomposition(component_results: dict[str, dict[str, pd.Series]]) -> dict[str, float]:
    """Additive decomposition (in CAGR terms) of the combined strategy's excess
    return over the benchmark into a fundamentals contribution, a regime-timing
    contribution, and an interaction/residual term.

    combined_excess ≈ fundamentals_contribution + regime_contribution + interaction

    This is a simplified, approximate decomposition (see docs/backtesting_spec.md
    "Attribution methodology" for why an *exact* multiplicative decomposition
    would require Brinson-style geometric attribution instead) — it's meant to
    give a clear, explainable read on "roughly how much did each signal add,"
    not a precision-audited P&L split.

    If ``component_results`` includes ``geometric_overlay_only`` (i.e. the
    geometric crash-risk signal was enabled — see ``run_component_backtests``),
    an additional informational (not folded into the interaction math above)
    ``geometric_overlay_cagr``/``geometric_overlay_contribution`` pair is
    included, showing that standalone signal's own CAGR spread over the
    benchmark — directly comparable to ``regime_contribution`` since both are
    computed the same way (pure benchmark exposure scaled by one signal),
    just from two completely independent signals.

    If ``component_results`` includes ``governed_regime_only``/
    ``governed_combined`` (i.e. ``regime_detection.consensus_governor.enabled``
    was set), two more informational pairs are included:
    ``governed_regime_cagr``/``governed_regime_contribution`` (directly
    comparable to ``regime_contribution`` — same construction, governed
    regime label instead of raw) and ``governed_combined_cagr``/
    ``governed_vs_raw_combined_delta`` (governed ``combined`` minus raw
    ``combined``, i.e. did persistence/hysteresis gating actually change the
    end-to-end result once stock selection is layered on top, not just the
    pure-timing leg in isolation).

    If ``component_results`` includes ``ichimoku_only``/
    ``combined_with_ichimoku`` (i.e. ``technical_signals.ichimoku.enabled``
    was set), two more informational pairs are included: ``ichimoku_cagr``/
    ``ichimoku_contribution`` (the standalone signal's own CAGR spread over
    the benchmark) and ``combined_with_ichimoku_cagr``/
    ``ichimoku_vs_raw_combined_delta`` (gated ``combined`` minus raw
    ``combined`` — did requiring Ichimoku confirmation on top of
    fundamentals+regime selection help or just cost exposure).

    If ``component_results`` includes ``combined_ichimoku_tilted`` (i.e.
    ``ichimoku_tilt_strength`` was nonzero), one more pair is included:
    ``combined_ichimoku_tilted_cagr``/``ichimoku_tilt_vs_raw_combined_delta``
    (tilted ``combined`` minus raw ``combined`` — directly comparable to
    ``ichimoku_vs_raw_combined_delta`` since both measure "combined plus
    Ichimoku" against plain "combined", just via reallocation instead of
    exposure-cutting).
    """
    bench_cagr = cagr(component_results["benchmark"]["returns"])
    fund_cagr = cagr(component_results["fundamentals_only"]["returns"])
    regime_cagr = cagr(component_results["regime_only"]["returns"])
    combined_cagr = cagr(component_results["combined"]["returns"])

    fundamentals_contribution = fund_cagr - bench_cagr
    regime_contribution = regime_cagr - bench_cagr
    combined_excess = combined_cagr - bench_cagr
    interaction = combined_excess - fundamentals_contribution - regime_contribution

    decomposition = {
        "benchmark_cagr": bench_cagr,
        "fundamentals_only_cagr": fund_cagr,
        "regime_only_cagr": regime_cagr,
        "combined_cagr": combined_cagr,
        "combined_excess_cagr": combined_excess,
        "fundamentals_contribution": fundamentals_contribution,
        "regime_contribution": regime_contribution,
        "interaction_effect": interaction,
    }

    if "geometric_overlay_only" in component_results:
        geometric_cagr = cagr(component_results["geometric_overlay_only"]["returns"])
        decomposition["geometric_overlay_cagr"] = geometric_cagr
        decomposition["geometric_overlay_contribution"] = geometric_cagr - bench_cagr

    if "governed_regime_only" in component_results:
        governed_regime_cagr = cagr(component_results["governed_regime_only"]["returns"])
        decomposition["governed_regime_cagr"] = governed_regime_cagr
        decomposition["governed_regime_contribution"] = governed_regime_cagr - bench_cagr
    if "governed_combined" in component_results:
        governed_combined_cagr = cagr(component_results["governed_combined"]["returns"])
        decomposition["governed_combined_cagr"] = governed_combined_cagr
        decomposition["governed_vs_raw_combined_delta"] = governed_combined_cagr - combined_cagr

    if "ichimoku_only" in component_results:
        ichimoku_cagr = cagr(component_results["ichimoku_only"]["returns"])
        decomposition["ichimoku_cagr"] = ichimoku_cagr
        decomposition["ichimoku_contribution"] = ichimoku_cagr - bench_cagr
    if "combined_with_ichimoku" in component_results:
        combined_with_ichimoku_cagr = cagr(component_results["combined_with_ichimoku"]["returns"])
        decomposition["combined_with_ichimoku_cagr"] = combined_with_ichimoku_cagr
        decomposition["ichimoku_vs_raw_combined_delta"] = combined_with_ichimoku_cagr - combined_cagr

    if "combined_ichimoku_tilted" in component_results:
        combined_ichimoku_tilted_cagr = cagr(component_results["combined_ichimoku_tilted"]["returns"])
        decomposition["combined_ichimoku_tilted_cagr"] = combined_ichimoku_tilted_cagr
        decomposition["ichimoku_tilt_vs_raw_combined_delta"] = combined_ichimoku_tilted_cagr - combined_cagr

    return decomposition
