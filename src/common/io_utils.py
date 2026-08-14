"""Shared I/O helpers: config loading and the NIFTY500 universe list."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (``override`` wins on
    conflicts), without mutating either input. Used by ``load_config`` to
    apply ``configs/config.local.yaml`` on top of the base config.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | Path = "configs/config.yaml",
    local_overrides_path: str | Path | None = "configs/config.local.yaml",
) -> dict[str, Any]:
    """Load the project YAML config, resolving relative to the project root.

    If ``local_overrides_path`` exists (default: ``configs/config.local.yaml``,
    silently skipped if absent — this is opt-in, not required), it's
    deep-merged on top of ``config_path`` — only the keys you actually put
    in the overrides file are changed; everything else comes from the base
    config untouched.

    **Why this exists**: any time the whole project tree gets updated
    (e.g. re-extracting an updated zip), ``configs/config.yaml`` itself
    gets overwritten along with everything else, silently reverting any
    ``enabled: true`` toggles or other local experimentation back to
    shipped defaults — this has genuinely happened multiple times across
    this project's sessions (consensus_governor and ichimoku toggles
    reverting without any error or warning, several runs producing
    identical results to a much earlier run before anyone noticed). Put
    your local toggles in ``configs/config.local.yaml`` instead of editing
    ``configs/config.yaml`` directly, and they'll survive future
    whole-project updates, since that file is yours to keep — e.g.:

        # configs/config.local.yaml
        technical_signals:
          ichimoku:
            enabled: true
        regime_detection:
          consensus_governor:
            enabled: true

    Add ``configs/config.local.yaml`` to your local ``.gitignore`` if you
    don't want it tracked/shared.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if local_overrides_path is not None:
        overrides_path = Path(local_overrides_path)
        if not overrides_path.is_absolute():
            overrides_path = PROJECT_ROOT / overrides_path
        if overrides_path.exists():
            with open(overrides_path, "r") as f:
                overrides = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, overrides)
            logger.info(
                "load_config: applied local overrides from %s on top of %s",
                overrides_path, path,
            )

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    weights = cfg.get("fundamental_analysis", {}).get("composite_weights", {})
    if weights:
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"fundamental_analysis.composite_weights must sum to 1.0, got {total:.4f}"
            )


def load_universe(list_path: str | Path = "data/universe/nifty500_list.csv") -> pd.DataFrame:
    """Load the NIFTY500 constituent list.

    Expected columns: symbol, name, sector, industry, series.
    Comment lines (leading '#') are ignored, so the shipped placeholder file
    (header only) loads cleanly as an empty DataFrame with the right columns.
    """
    path = Path(list_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    df = pd.read_csv(path, comment="#")
    expected = {"symbol", "name", "sector", "industry", "series"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Universe file {path} is missing columns: {missing}")
    return df
