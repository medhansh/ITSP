"""Technical-momentum dimension: Ichimoku conviction as a fundamentals
composite-score input, computed at stock-SELECTION time rather than as a
post-selection gate/tilt on an already-built portfolio.

Unlike every other dimension in this package, this one's raw data isn't
part of the current-snapshot/quarterly-history universe
(``data_fetchers/``) — it comes from ``adaptive_ichimoku.build_ichimoku_conviction_panel``,
a daily per-symbol OHLC-derived score. See that function's docstring and
``docs/backtesting_spec.md``'s Ichimoku sections for why this dimension
exists: two post-selection mechanisms (gating and reallocating an
already-selected portfolio) were both confirmed negative on real data,
despite the underlying signal (``ichimoku_only``) being the single
best-performing standalone component found. The working hypothesis this
dimension tests: the signal's edge is a stock-PICKING signal, not a
within-basket timing enhancer, so it belongs in selection, not after it.

**Status: experimental, unvalidated on real data as of writing** — same
caveat as everything else new in this project. Default composite weight
is deliberately small (0.05, same convention as ``options_earnings``, the
other newest/least-validated dimension) — see configs/config.yaml.
"""
from __future__ import annotations

import pandas as pd


def compute_technical_momentum_metrics(technical_conviction: pd.Series | None) -> pd.DataFrame:
    """Wrap a single (symbol -> conviction value) Series into the
    ``fn(snapshot) -> DataFrame`` shape every other dimension module
    returns, for ``fundamental_analysis/pipeline.py``'s uniform join.

    ``technical_conviction`` is one row per symbol: that symbol's Ichimoku
    conviction score (``[0, 1]``, from
    ``adaptive_ichimoku.compute_ichimoku_conviction_score``) as of the
    fundamentals scoring date — see
    ``point_in_time.py``'s per-rebalance-date extraction for how this gets
    built PIT-safely.

    Symbols with no OHLC coverage that date (e.g. a fetch gap, or a symbol
    that hasn't IPO'd yet as of an early rebalance date) are simply absent
    from ``technical_conviction`` or NaN — ``compute_dimension_scores``'
    per-row weight renormalization (see ``scoring/composite_score.py``)
    already handles a dimension being NaN for some symbols gracefully, same
    as every other dimension when its underlying data is incomplete.

    Returns a single-column DataFrame (``ichimoku_conviction``) indexed by
    symbol. If ``technical_conviction`` is ``None``, returns an empty
    DataFrame with that column (all-NaN once joined) — matching the
    "missing input degrades to NaN, not a crash" convention used
    throughout this pipeline (e.g. ``options_earnings``' pre-earnings
    fields when no options history is available).
    """
    if technical_conviction is None:
        return pd.DataFrame(columns=["ichimoku_conviction"])
    return technical_conviction.rename("ichimoku_conviction").to_frame()
