"""Point-in-time (PIT) fundamentals: turn a long-format quarterly-results
history into "what would this snapshot have looked like as of date t",
using only results that were *already public* as of t.

Why this exists
----------------
The fetchers in ``data_fetchers/`` only ever give you a *current* snapshot —
today's P/E, today's ROE, etc. Reusing that same current snapshot at every
historical rebalance date in a backtest is a textbook look-ahead-bias bug: on
1 Jan 2019 the backtest would "know" 2026 fundamentals that didn't exist yet.
This was flagged as the project's top open gap in docs/backtesting_spec.md.

The fix implemented here: fetch *quarterly* history (one row per result,
tagged with the date it became public — see
``screener_fetcher.fetch_quarterly_history``'s ``known_date``), then for any
query date t, take the most recent result with ``known_date <= t`` and
forward-fill it until the next result's ``known_date``. This is exactly
"fill discontinuity points (periods between results) with the previous
result's data" — implemented via ``pd.merge_asof(..., direction="backward")``,
which is a strict "search only backward in time" join and therefore cannot
leak a future value into an earlier query date by construction.

Two important caveats, both already true of the underlying data, not
introduced by this module:
  1. ``known_date`` is a heuristic upper bound (period-end + a statutory
     reporting-lag), not a scraped filing date — see
     ``screener_fetcher.fetch_quarterly_history`` docstring.
  2. Before a symbol's *first* available quarterly result, the PIT snapshot
     is NaN (correctly — there is nothing to forward-fill from), which will
     shrink the effective backtest universe in early years for symbols with
     short scraped history. This is intentional: an all-NaN gap is honest;
     silently backfilling from a symbol's earliest result into years before
     that result existed would itself be look-ahead in the other direction.
"""
from __future__ import annotations

import pandas as pd

from src.common.logging_utils import get_logger
from src.fundamental_analysis.scoring.composite_score import rebalanced_weights

logger = get_logger(__name__)

REQUIRED_HISTORY_COLUMNS = {"symbol", "period_end", "known_date", "field", "value"}


def _validate_history(history_long: pd.DataFrame) -> None:
    missing = REQUIRED_HISTORY_COLUMNS - set(history_long.columns)
    if missing:
        raise ValueError(f"history_long is missing required columns: {missing}")


def build_pit_snapshot(history_long: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    """Single-date PIT snapshot: for every (symbol, field), the most recent
    value whose ``known_date`` is <= ``as_of_date``, forward-filled from the
    last result (NaN if no result was known yet).

    Returns a wide DataFrame indexed by symbol, columns = field names —
    directly joinable onto a SNAPSHOT_SCHEMA-shaped current snapshot via
    ``merge_pit_into_snapshot`` below.
    """
    _validate_history(history_long)
    as_of = pd.Timestamp(as_of_date)
    known = history_long[history_long["known_date"] <= as_of]
    if known.empty:
        logger.warning("No quarterly results known as of %s — snapshot will be all-NaN", as_of.date())
        return pd.DataFrame(columns=sorted(history_long["field"].unique()))

    # For each (symbol, field), take the row with the latest known_date.
    latest_idx = known.groupby(["symbol", "field"])["known_date"].idxmax()
    latest = known.loc[latest_idx]
    snapshot = latest.pivot(index="symbol", columns="field", values="value")
    snapshot.index.name = "symbol"
    return snapshot


def build_pit_panel(
    history_long: pd.DataFrame, rebalance_dates: pd.DatetimeIndex | list
) -> pd.DataFrame:
    """Vectorized multi-date version of ``build_pit_snapshot``: for every
    (rebalance date, symbol), the most recently known value of every field,
    via a per-symbol, per-field ``merge_asof(direction="backward")``.

    Returns long format: columns = [date, symbol, field, value] — one row
    per (rebalance_date, symbol, field). This is the shape
    ``run_pit_fundamental_pipeline`` (below) reshapes per-date and feeds into
    ``fundamental_analysis.pipeline.run_pipeline``.

    ``merge_asof`` is the key correctness mechanism: for a given rebalance
    date t and (symbol, field), it finds the row with the largest
    ``known_date`` that is still <= t — i.e. it can only look backward in
    time, so it is structurally impossible for a later result to leak into
    an earlier rebalance date.
    """
    _validate_history(history_long)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(rebalance_dates).unique()))
    if len(dates) == 0:
        return pd.DataFrame(columns=["date", "symbol", "field", "value"])

    frames = []
    for (symbol, field), group in history_long.sort_values("known_date").groupby(["symbol", "field"]):
        left = pd.DataFrame({"date": dates})
        merged = pd.merge_asof(
            left,
            group[["known_date", "value"]].rename(columns={"known_date": "date"}),
            on="date",
            direction="backward",
        )
        merged["symbol"] = symbol
        merged["field"] = field
        frames.append(merged[["date", "symbol", "field", "value"]])

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "field", "value"])
    return pd.concat(frames, ignore_index=True)


def build_annual_growth_history_pit(
    history_long: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    fields: tuple[str, ...] = ("revenue", "net_income", "eps"),
    min_quarters_for_ttm: int = 4,
) -> pd.DataFrame:
    """Point-in-time multi-year ANNUAL history for the ``growth`` dimension
    (``metrics/growth.py``'s ``compute_growth_metrics``, which needs one row
    per (symbol, fiscal_year) with columns revenue/net_income/eps — NOT the
    single-latest-value shape ``build_pit_panel``/``build_pit_snapshot``
    produce).

    This was previously a real gap: ``run_pit_fundamental_pipeline`` called
    ``run_fundamental_pipeline(..., history=None)`` unconditionally, so the
    ``growth`` dimension (14.25% of ``composite_weights`` by default) was
    silently skipped and renormalized away at every single rebalance date
    in every real backtest — see ``docs/fundamental_analysis_spec.md``'s
    "Known gaps" section and the log line
    ``"growth dimension enabled but no `history` provided — skipping"``.
    There is no separate annual-P&L scrape (``screener_fetcher`` only has
    ``fetch_quarterly_history``), so this builds an approximate annual
    series directly from the SAME quarterly PIT data already fetched for
    the other dimensions, via trailing-twelve-month (TTM) rollups —
    no new scraping, no new network dependency.

    Method, per symbol:
      1. Filter to rows with ``known_date <= as_of_date`` (identical PIT
         discipline to ``build_pit_snapshot``/``build_pit_panel`` above —
         strictly backward-looking, cannot leak a future quarter into an
         earlier "as of" date).
      2. Pivot to one row per ``period_end`` (columns = fields), sorted
         chronologically.
      3. Rolling ``min_quarters_for_ttm``-quarter (default 4) sum ->
         trailing-twelve-month revenue/net_income/eps as of each quarter
         that has a full trailing window available. This assumes
         Screener's quarterly rows are already single-quarter (not
         cumulative) figures — true for its "Quarterly Results" table (see
         ``screener_fetcher.QUARTERLY_FIELD_MAP``) — and that there are no
         gaps in the quarterly sequence; a missing quarter will make the
         nearest TTM windows spanning it understate the true annual figure,
         same caveat as any TTM rollup from raw quarterly prints.
      4. One annual data point per calendar year: the LAST TTM value within
         each ``period_end.year`` (a year-end-TTM proxy for that fiscal
         year), which is the shape ``compute_growth_metrics`` expects
         (``fiscal_year`` sorted ascending, first-vs-last CAGR, YoY std for
         stability).

    Returns columns: symbol, fiscal_year, revenue, net_income, eps — empty
    (but correctly-shaped) if there's not enough PIT history yet as of
    ``as_of_date`` for any symbol to produce a single full TTM window.
    """
    _validate_history(history_long)
    as_of = pd.Timestamp(as_of_date)
    known = history_long[
        (history_long["known_date"] <= as_of) & (history_long["field"].isin(fields))
    ]
    if known.empty:
        return pd.DataFrame(columns=["symbol", "fiscal_year", *fields])

    records = []
    for symbol, g in known.groupby("symbol"):
        wide = g.pivot_table(index="period_end", columns="field", values="value", aggfunc="last").sort_index()
        for f in fields:
            if f not in wide.columns:
                wide[f] = pd.NA
        wide = wide[list(fields)].apply(pd.to_numeric, errors="coerce")

        ttm = wide.rolling(min_quarters_for_ttm, min_periods=min_quarters_for_ttm).sum()
        ttm = ttm.dropna(how="all")
        if ttm.empty:
            continue

        ttm["fiscal_year"] = ttm.index.year
        annual = ttm.groupby("fiscal_year").last().reset_index()
        annual["symbol"] = symbol
        records.append(annual[["symbol", "fiscal_year", *fields]])

    if not records:
        return pd.DataFrame(columns=["symbol", "fiscal_year", *fields])
    return pd.concat(records, ignore_index=True)


def merge_pit_into_snapshot(
    current_snapshot: pd.DataFrame, pit_fields: pd.DataFrame
) -> pd.DataFrame:
    """Overlay PIT-derived quarterly fields (revenue, net_income, eps, ...)
    onto an otherwise-current snapshot (sector, industry, price, and other
    fields Screener/yfinance/Trendlyne don't expose historically at all).

    This is a pragmatic middle ground, not a claim that the whole snapshot is
    now point-in-time: only the columns present in ``pit_fields`` (i.e. the
    ones ``screener_fetcher.QUARTERLY_FIELD_MAP`` actually tracks quarter by
    quarter) are replaced with their as-of-date value; everything else
    (sector classification, shareholding, valuation ratios not derivable
    from revenue/net_income/eps alone, etc.) still comes from the current
    snapshot and remains subject to the same look-ahead caveat as before —
    see docs/backtesting_spec.md's "Known gaps" section, which this
    narrows but does not fully close.
    """
    merged = current_snapshot.copy()
    for col in pit_fields.columns:
        if col in merged.columns:
            merged[col] = pit_fields[col].reindex(merged.index)
        else:
            merged[col] = pit_fields[col].reindex(merged.index)
    return merged


def apply_regime_conditional_weight(
    base_weights: dict[str, float],
    dim: str,
    regime_label: str | int | None,
    multipliers: dict,
    default_multiplier: float = 1.0,
) -> dict[str, float]:
    """Scale ``dim``'s composite weight by a per-regime multiplier before
    rebalancing the rest to compensate — the "trust this dimension more in
    some market conditions than others" mechanism.

    **Motivation**: a walk-forward validation of ``technical_momentum``
    found genuine out-of-sample edge on average, but concentrated unevenly
    across periods, with a real (if imperfect — one flat-market fold broke
    the pattern too) positive correlation between edge size and how much
    the benchmark was trending during that period (see
    ``docs/fundamental_analysis_spec.md``'s technical_momentum section and
    ``scripts/analyze_walk_forward_regimes.py``). Ichimoku is
    architecturally a trend-following signal, so this isn't shocking — but
    it means a single STATIC weight either overpays for the signal during
    choppy/high-vol periods where it's least reliable, or underuses it
    during genuinely trending periods where it's shown real value. This
    function lets the effective weight vary by regime instead.

    ``regime_label``: the CURRENT regime at the rebalance date being
    scored (e.g. ``"low_vol_calm"``/``"moderate_vol"``/``"elevated_vol"``/
    ``"high_vol_stress"`` — ``regime_detection.models.REGIME_LABELS_4``,
    or the raw int ``regime`` column if names aren't available). This is
    NOT a look-ahead risk: the regime label for "today" is exactly as
    available "today" as every other PIT input in this module — it's
    already known, causally-computed, at the point a rebalance actually
    happens.

    ``multipliers``: ``{regime_label: multiplier}``, e.g.
    ``{"low_vol_calm": 1.0, "moderate_vol": 0.6, "elevated_vol": 0.3,
    "high_vol_stress": 0.0}``. A regime not present in ``multipliers``
    falls back to ``default_multiplier`` (1.0 — i.e. "no adjustment") with
    a logged warning, rather than silently defaulting to either extreme.

    ``regime_label=None`` (e.g. no regime data available for this date at
    all) also falls back to ``default_multiplier`` unchanged, logged once.

    Returns the fully rebalanced weights dict for this specific date (via
    ``scoring.composite_score.rebalanced_weights``), ready to pass as
    ``composite_weights`` for that one rebalance date's scoring.
    """
    if regime_label is None:
        logger.warning(
            "apply_regime_conditional_weight: no regime label available for this date -- "
            "using default_multiplier=%.2f (no adjustment) rather than guessing.", default_multiplier,
        )
        multiplier = default_multiplier
    elif regime_label not in multipliers:
        logger.warning(
            "apply_regime_conditional_weight: regime %r not found in configured multipliers %s -- "
            "using default_multiplier=%.2f.", regime_label, list(multipliers.keys()), default_multiplier,
        )
        multiplier = default_multiplier
    else:
        multiplier = multipliers[regime_label]

    base_dim_weight = base_weights.get(dim, 0.0)
    effective_weight = max(0.0, min(base_dim_weight * multiplier, 0.999999))
    return rebalanced_weights(base_weights, dim, effective_weight)


def run_pit_fundamental_pipeline(
    config: dict,
    current_snapshot: pd.DataFrame,
    history_long: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex | list,
    conviction_panel: pd.DataFrame | None = None,
    regime: pd.Series | None = None,
    regime_weight_multipliers: dict | None = None,
) -> pd.DataFrame:
    """Run ``fundamental_analysis.pipeline.run_pipeline`` once per rebalance
    date, each time using the PIT snapshot as-of that date instead of the
    (fixed, "current") snapshot — this is the function
    ``strategies.build_fundamental_portfolio_weights`` expects to have
    produced its ``scores_by_date`` input from, per that module's docstring.

    Also builds and passes a PIT-safe annual growth history
    (``build_annual_growth_history_pit``, TTM-rollups of the same quarterly
    data) at each rebalance date, so the ``growth`` dimension is actually
    computed rather than silently skipped — this was a real, previously
    undiagnosed gap (see that function's docstring). Early rebalance dates
    that don't yet have enough quarterly history for even one full TTM
    window per symbol will still show growth as NaN for those symbols
    (correctly — there's nothing to compute yet), same "honest gap, not a
    silent backfill" convention as the rest of this module.

    ``conviction_panel`` (optional): the daily per-symbol Ichimoku
    conviction panel from
    ``adaptive_ichimoku.build_ichimoku_conviction_panel``, feeding the
    ``technical_momentum`` composite dimension (see
    ``metrics/technical_momentum.py``) if enabled. At each rebalance date,
    the most recent conviction value AT OR BEFORE that date is used
    (``reindex(..., method="ffill")``) — this is PIT-safe by construction,
    since Ichimoku itself is already a purely-causal (backward-looking)
    computation from OHLC data, same as every other PIT-scored input in
    this module. ``None`` (default, or if ``technical_momentum`` isn't
    enabled) reproduces the exact pre-Ichimoku-dimension behavior.

    ``regime`` (optional) + ``regime_weight_multipliers`` (optional):
    when BOTH are given, ``technical_momentum``'s composite weight is
    scaled per rebalance date by the current regime
    (``point_in_time.apply_regime_conditional_weight`` — see that
    function's docstring for the full motivation and multiplier format).
    ``regime`` is a daily Series (regime label per date, e.g. from
    ``regime_detection.pipeline.run_pipeline``'s ``regime_name`` column),
    looked up per rebalance date the same PIT-safe way as
    ``conviction_panel`` (``reindex(..., method="ffill")``). If either is
    ``None`` (default), the composite weight stays static at whatever
    ``config["composite_weights"]`` specifies — exact pre-regime-
    conditioning behavior.

    Returns long format: (date, symbol, composite_score, ...) stacked across
    all rebalance dates.
    """
    from src.fundamental_analysis.data_fetchers.fundamentals_fetcher import SNAPSHOT_SCHEMA
    from src.fundamental_analysis.pipeline import run_pipeline as run_fundamental_pipeline

    # Every metrics/*.py function assumes the full SNAPSHOT_SCHEMA is present
    # (missing = NaN, per fundamentals_fetcher.fetch_fundamentals' contract)
    # — reindex defensively here too, since a caller-supplied current_snapshot
    # (e.g. a partial fixture, or one built by hand) may not already do this.
    base_snapshot = current_snapshot.reindex(columns=SNAPSHOT_SCHEMA)
    for col in current_snapshot.columns:
        if col not in base_snapshot.columns:
            base_snapshot[col] = current_snapshot[col]

    panel_long = build_pit_panel(history_long, rebalance_dates)

    conviction_at_rebalance = None
    if conviction_panel is not None:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(rebalance_dates).unique()))
        conviction_at_rebalance = conviction_panel.sort_index().reindex(dates, method="ffill")

    regime_at_rebalance = None
    if regime is not None and regime_weight_multipliers is not None:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(rebalance_dates).unique()))
        regime_at_rebalance = regime.sort_index().reindex(dates, method="ffill")

    results = []
    n_dates_with_growth = 0
    n_dates_with_conviction = 0
    n_dates_regime_conditioned = 0
    for date, group in panel_long.groupby("date"):
        pit_fields = group.pivot(index="symbol", columns="field", values="value")
        snapshot_as_of = merge_pit_into_snapshot(base_snapshot, pit_fields)
        growth_history_as_of = build_annual_growth_history_pit(history_long, date)
        if not growth_history_as_of.empty:
            n_dates_with_growth += 1

        technical_conviction = None
        if conviction_at_rebalance is not None and date in conviction_at_rebalance.index:
            row = conviction_at_rebalance.loc[date].dropna()
            if not row.empty:
                technical_conviction = row
                n_dates_with_conviction += 1

        run_config = config
        if regime_at_rebalance is not None and date in regime_at_rebalance.index and "technical_momentum" in config.get("composite_weights", {}):
            regime_label = regime_at_rebalance.loc[date]
            regime_label = None if pd.isna(regime_label) else regime_label
            adjusted_weights = apply_regime_conditional_weight(
                config["composite_weights"], "technical_momentum", regime_label, regime_weight_multipliers
            )
            run_config = dict(config)
            run_config["composite_weights"] = adjusted_weights
            n_dates_regime_conditioned += 1

        scored = run_fundamental_pipeline(
            run_config, snapshot_as_of,
            history=growth_history_as_of if not growth_history_as_of.empty else None,
            technical_conviction=technical_conviction,
        )
        scored = scored.reset_index().rename(columns={"index": "symbol"})
        scored["date"] = date
        results.append(scored)

    n_dates = panel_long["date"].nunique()
    logger.info(
        "run_pit_fundamental_pipeline: growth dimension had usable PIT annual history on "
        "%d/%d rebalance dates (early dates with insufficient quarterly history are "
        "expected to be 0/NaN, not a bug -- see build_annual_growth_history_pit's docstring)",
        n_dates_with_growth, n_dates,
    )
    if conviction_panel is not None:
        logger.info(
            "run_pit_fundamental_pipeline: technical_momentum dimension had usable Ichimoku "
            "conviction data on %d/%d rebalance dates",
            n_dates_with_conviction, n_dates,
        )
    if regime_weight_multipliers is not None:
        logger.info(
            "run_pit_fundamental_pipeline: technical_momentum weight was regime-conditioned on "
            "%d/%d rebalance dates (dates without a matching regime label fell back to the "
            "unconditioned base weight, logged individually above if any occurred)",
            n_dates_regime_conditioned, n_dates,
        )

    if not results:
        return pd.DataFrame(columns=["date", "symbol", "composite_score"])
    return pd.concat(results, ignore_index=True)
