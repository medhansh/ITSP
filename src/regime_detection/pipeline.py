"""End-to-end regime-detection pipeline: data -> features -> model -> labeled history."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger
from src.regime_detection.data_loader import load_from_csv, load_sector_prices_from_csv
from src.regime_detection.features import build_feature_matrix
from src.regime_detection.models import RegimeModel, describe_regimes

logger = get_logger(__name__)


def run_pipeline(
    config: dict[str, Any],
    price_csv: str,
    sector_price_csv: str | None = None,
    factor_dispersion: pd.Series | None = None,
) -> tuple[pd.DataFrame, RegimeModel]:
    """Fit a regime model and return (labeled history, fitted model).

    ``config`` is the ``regime_detection`` section of configs/config.yaml.
    ``price_csv`` points at a CSV loadable by ``data_loader.load_from_csv`` —
    if it has ``open``/``high``/``low``/``volume`` columns (all optional,
    independently), Parkinson/Garman-Klass range volatility and volume
    features (volume z-score, OBV trend) are automatically added to the
    clustering feature matrix; see ``features.build_feature_matrix``'s
    docstring.

    ``sector_price_csv`` (optional): a multi-sector price CSV (see
    ``data_loader.load_sector_prices_from_csv``); if given (or if
    ``config["geometric_signal"]["enabled"]`` and
    ``config["geometric_signal"]["sector_price_csv"]`` are set), the
    geometric wedge-product crash-risk columns from ``geometric_signal.py``
    are computed and joined onto the returned history.

    **By explicit design, the geometric signal is NOT part of the GMM/KMeans/
    HMM clustering feature set** — it's computed entirely separately, after
    ``model.fit``/``model.predict`` have already produced the regime label,
    and is joined onto the output purely as extra informational/overlay
    columns (``wedge_volume_*``, ``geometric_crash_risk_flag``, etc.). It
    therefore cannot influence ``regime``/``regime_name`` even indirectly.
    The intended use is as a fully independent overlay applied directly in
    ``backtesting/strategies.py`` (``build_geometric_overlay_weights`` /
    ``apply_geometric_overlay``) — see ``docs/regime_detection_spec.md``'s
    "Geometric wedge-product crash-risk signal" section for the reasoning,
    and ``docs/backtesting_spec.md`` for how it flows into the backtest.

    ``factor_dispersion`` (optional): a daily cross-sectional fundamental
    factor-score dispersion series, already aligned/forward-filled to
    ``price_csv``'s daily index (see
    ``factor_dispersion.compute_cross_sectional_dispersion`` +
    ``resample_dispersion_to_daily``). Passed straight through to
    ``build_feature_matrix``, joined by ``config["factor_dispersion"]``'s
    ``use_price_features``/``windows`` settings — see that function's
    docstring for the "add as an extra feature" vs. "replace price-vol
    features entirely" modes. ``None`` (default) reproduces exact
    pre-factor-dispersion behavior, i.e. the existing price-vol-only
    clustering feature set.
    """
    raw = load_from_csv(price_csv)

    dispersion_cfg = config.get("factor_dispersion", {})
    features = build_feature_matrix(
        prices=raw["close"],
        return_windows=config["feature_windows"]["returns"],
        vol_windows=config["feature_windows"]["realized_vol"],
        advances=raw.get("advances"),
        declines=raw.get("declines"),
        breadth_windows=config["feature_windows"].get("breadth"),
        vix=raw.get("vix"),
        open_=raw.get("open"),
        high=raw.get("high"),
        low=raw.get("low"),
        volume=raw.get("volume"),
        range_vol_windows=config["feature_windows"].get("range_vol"),
        volume_zscore_windows=config["feature_windows"].get("volume_zscore"),
        obv_trend_windows=config["feature_windows"].get("obv_trend"),
        factor_dispersion=factor_dispersion,
        factor_dispersion_windows=dispersion_cfg.get("windows"),
        use_price_features=dispersion_cfg.get("use_price_features", True) if factor_dispersion is not None else True,
    )
    logger.info(
        "Built feature matrix: %d rows x %d cols (geometric signal excluded by design; "
        "factor_dispersion %s)", *features.shape,
        "included" if factor_dispersion is not None else "not provided",
    )

    model_cfg = config["model"]
    model = RegimeModel(
        model_type=model_cfg["type"],
        n_regimes=model_cfg["n_regimes"],
        random_state=model_cfg.get("random_state", 42),
    )
    model.fit(features)

    labels = model.predict(features)
    proba = model.predict_proba(features)
    named = describe_regimes(labels)

    result = features.copy()
    result["regime"] = labels
    result["regime_name"] = named
    result = result.join(proba)
    logger.info("Regime distribution:\n%s", labels.value_counts().sort_index())

    result = _attach_geometric_overlay(result, config, sector_price_csv)
    result = _attach_consensus_governor(result, features, labels, model, config)
    return result, model


def _attach_consensus_governor(
    result: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    model: RegimeModel,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Hybrid HMM/GMM + Wasserstein consensus governor (opt-in, additive).

    Purely additive, same pattern as ``_attach_geometric_overlay``: adds
    ``consensus_entropy`` / ``proposed_regime`` / ``active_regime`` /
    ``is_transitional`` / ``wasserstein_proximity_*`` columns without
    touching ``regime``/``regime_name``/``p_regime_*``, which remain exactly
    what the raw model produced. Downstream code that already keys off
    ``regime`` is unaffected unless it opts into ``active_regime`` instead.

    Templates are built from the SAME in-sample ``features``/``labels`` used
    to fit ``model`` above — see ``wasserstein_proximity.py``'s look-ahead
    note. If this pipeline is later called per walk-forward window (as the
    backtest engine should), that in-sample discipline carries through
    automatically since it's just whatever was passed to ``run_pipeline``.
    """
    gov_cfg = config.get("consensus_governor", {})
    if not gov_cfg.get("enabled", False):
        return result

    proba_cols = [c for c in result.columns if c.startswith("p_regime_")]
    if not proba_cols:
        logger.warning("consensus_governor enabled but no p_regime_* columns found; skipping.")
        return result

    template_columns = tuple(
        gov_cfg.get("wasserstein_columns")
        or [c for c in ("return_5d", "return_21d", "realized_vol_21d", "realized_vol_63d") if c in features.columns]
    )
    missing = [c for c in template_columns if c not in features.columns]
    if missing:
        logger.warning(
            "consensus_governor: configured wasserstein_columns %s not in feature set; skipping governor.",
            missing,
        )
        return result

    templates = build_regime_templates(features, labels, columns=template_columns)
    proximity = rolling_wasserstein_proximity(
        features, templates, window=gov_cfg.get("wasserstein_window", 21)
    )
    governed = run_governor_over_history(
        result[proba_cols],
        proximity,
        entropy_limit=gov_cfg.get("entropy_limit", 0.85),
        persistence_window=gov_cfg.get("persistence_window", 5),
        hysteresis_epsilon=gov_cfg.get("hysteresis_epsilon", 0.05),
    )
    logger.info(
        "Consensus governor: %d/%d bars flagged transitional, %d active-regime changes vs raw regime label.",
        int(governed["is_transitional"].sum()),
        len(governed),
        int((governed["active_regime"].astype(str) != result["regime"].astype(str)).sum()),
    )
    return result.join(proximity).join(governed)


def _attach_geometric_overlay(
    result: pd.DataFrame, config: dict[str, Any], sector_price_csv: str | None
) -> pd.DataFrame:
    """Compute the geometric wedge-product crash-risk columns (if configured)
    and left-join them onto the already-finalized regime history. Runs
    strictly after model fit/predict above — see ``run_pipeline``'s docstring
    for why that ordering is the whole point.
    """
    geo_cfg = config.get("geometric_signal", {})
    sector_price_csv = sector_price_csv or (
        geo_cfg.get("sector_price_csv") if geo_cfg.get("enabled") else None
    )
    if not sector_price_csv:
        return result

    sector_prices = load_sector_prices_from_csv(sector_price_csv)
    sector_returns = np.log(sector_prices).diff()
    gcfg = {k: v for k, v in geo_cfg.items() if k not in ("enabled", "sector_price_csv", "sector_tickers")}


    overlay = compute_geometric_crash_features(sector_returns, **gcfg)
    logger.info(
        "Computed geometric overlay columns (%s) independently of the regime model — "
        "joining onto history, not re-fitting anything.", list(overlay.columns),
    )
    return result.join(overlay, how="left")


def save_outputs(result: pd.DataFrame, model: RegimeModel, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path / "regime_history.csv")
    model.save(str(out_path / "regime_model.joblib"))
    logger.info("Saved regime outputs to %s", out_path)
