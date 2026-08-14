"""End-to-end fundamental analysis pipeline: snapshot(+history) -> per-dimension
metrics -> sector-relative composite score, for the NIFTY500 universe."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.logging_utils import get_logger
from src.fundamental_analysis.metrics.cashflow_quality import compute_cashflow_quality_metrics
from src.fundamental_analysis.metrics.earnings_surprise import compute_earnings_surprise_metrics
from src.fundamental_analysis.metrics.growth import compute_growth_metrics
from src.fundamental_analysis.metrics.leverage_solvency import compute_leverage_solvency_metrics
from src.fundamental_analysis.metrics.options_earnings import compute_options_earnings_dimension
from src.fundamental_analysis.metrics.ownership_governance import (
    compute_ownership_governance_metrics,
)
from src.fundamental_analysis.metrics.profitability_quality import (
    compute_profitability_quality_metrics,
)
from src.fundamental_analysis.metrics.technical_momentum import compute_technical_momentum_metrics
from src.fundamental_analysis.metrics.valuation import compute_valuation_metrics
from src.fundamental_analysis.scoring.composite_score import (
    compute_composite_score,
    compute_dimension_scores,
)

logger = get_logger(__name__)

DIMENSION_COMPUTERS = {
    "valuation": compute_valuation_metrics,
    "profitability_quality": compute_profitability_quality_metrics,
    "leverage_solvency": compute_leverage_solvency_metrics,
    "cashflow_quality": compute_cashflow_quality_metrics,
    "ownership_governance": compute_ownership_governance_metrics,
    "earnings_surprise": compute_earnings_surprise_metrics,
    # Expects pre_earnings_iv_percentile / pre_earnings_put_call_oi_ratio /
    # implied_move_pct already merged onto `snapshot` — see
    # metrics/options_earnings.py's compute_options_earnings_metrics (which
    # needs options history + an earnings calendar the uniform
    # fn(snapshot)->DataFrame DIMENSION_COMPUTERS signature has no room for)
    # and scripts/run_full_pipeline.py for where that merge happens.
    "options_earnings": compute_options_earnings_dimension,
}


def run_pipeline(
    config: dict[str, Any],
    snapshot: pd.DataFrame,
    history: pd.DataFrame | None = None,
    technical_conviction: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute every enabled dimension's metrics, sector-relative z-scores,
    and the final composite score for each symbol in ``snapshot``.

    ``config`` is the ``fundamental_analysis`` section of configs/config.yaml.
    ``snapshot`` must be indexed by symbol and satisfy
    data_fetchers.fundamentals_fetcher.SNAPSHOT_SCHEMA (missing columns are
    tolerated — the affected metrics just come back NaN).
    ``history`` (optional) is required only if the ``growth`` dimension is
    enabled; see metrics/growth.py for its schema.
    ``technical_conviction`` (optional) is required only if the
    ``technical_momentum`` dimension is enabled — a Series indexed by
    symbol, that symbol's Ichimoku conviction score as of this scoring
    date; see ``metrics/technical_momentum.py`` and
    ``point_in_time.py``'s per-rebalance-date extraction for how this is
    built PIT-safely.
    """
    enabled = config["dimensions"]
    all_metrics = pd.DataFrame(index=snapshot.index)

    for dim, fn in DIMENSION_COMPUTERS.items():
        if not enabled.get(dim, False):
            continue
        logger.info("Computing dimension: %s", dim)
        all_metrics = all_metrics.join(fn(snapshot))

    if enabled.get("growth", False):
        if history is None:
            logger.warning("growth dimension enabled but no `history` provided — skipping")
        else:
            logger.info("Computing dimension: growth")
            growth_metrics = compute_growth_metrics(history)
            all_metrics = all_metrics.join(growth_metrics)

    if enabled.get("technical_momentum", False):
        if technical_conviction is None:
            logger.warning("technical_momentum dimension enabled but no `technical_conviction` provided — skipping")
        else:
            logger.info("Computing dimension: technical_momentum")
            technical_metrics = compute_technical_momentum_metrics(technical_conviction)
            all_metrics = all_metrics.join(technical_metrics)

    sector = snapshot["sector"] if "sector" in snapshot.columns else pd.Series(
        "UNKNOWN", index=snapshot.index
    )
    dimension_scores = compute_dimension_scores(
        all_metrics, sector, sector_relative=config.get("sector_relative", True)
    )
    composite = compute_composite_score(dimension_scores, config["composite_weights"])

    result = all_metrics.join(dimension_scores, rsuffix="_score")
    result["composite_score"] = composite
    result = result.join(snapshot[["sector", "industry"]] if "industry" in snapshot.columns else sector.rename("sector"))
    return result.sort_values("composite_score", ascending=False)
