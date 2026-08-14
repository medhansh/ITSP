"""Wasserstein-1 proximity scoring against pre-trained regime templates.

Implements the "geometric distance" half of the Hybrid HMM + Wasserstein
approach (Option 1 in ``docs/regime_detection_research_options.md``): rather
than trusting the HMM's temporal posterior alone, we independently measure
how close the *current rolling empirical distribution* of a small set of
distributional features is to each regime's *training-time* empirical
distribution, using the Wasserstein-1 (earth mover's) distance. That
proximity score is later blended with the HMM posterior by
``state_governor.RegimeStateGovernor``.

Why Wasserstein-1 and not something like KL divergence: it compares
distribution *shapes* directly (it's the L1 distance between CDFs) without
requiring overlapping support or a density estimate, and it responds
immediately to an anomalous draw rather than needing many observations to
update a fitted density — see the research doc's discussion of this being
useful precisely at regime transitions, where by definition there isn't much
in-regime history yet.

Look-ahead-bias note: templates must be built ONLY from the same in-sample
window used to fit the regime model (i.e. call ``build_regime_templates``
with the exact ``features``/``labels`` pair used in ``RegimeModel.fit``).
This module does not enforce that itself — enforcement lives in
``pipeline.py``, mirroring how PIT discipline is enforced by the caller of
``merge_asof`` in the fundamentals pipeline, not by the merge itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

DEFAULT_TEMPLATE_COLUMNS = (
    "return_5d",
    "return_21d",
    "realized_vol_21d",
    "realized_vol_63d",
)


@dataclass
class RegimeTemplates:
    """Per-regime empirical distributions for a fixed set of feature columns.

    ``distributions[regime_id][column]`` is a 1-D numpy array: the training-
    time values of that feature on days labeled with that regime.
    """

    columns: tuple[str, ...]
    distributions: dict[int, dict[str, np.ndarray]]
    regime_ids: tuple[int, ...]


def build_regime_templates(
    features: pd.DataFrame,
    labels: pd.Series,
    columns: tuple[str, ...] | list[str] = DEFAULT_TEMPLATE_COLUMNS,
) -> RegimeTemplates:
    """Build per-regime empirical distribution templates from in-sample data.

    ``features`` and ``labels`` must be the same in-sample rows used to fit
    the ``RegimeModel`` (same index) — see the look-ahead-bias note above.
    """
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(
            f"Template columns not found in features: {missing}. "
            f"Available: {list(features.columns)}"
        )
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must share the same index (in-sample rows).")

    regime_ids = tuple(sorted(labels.unique().tolist()))
    distributions: dict[int, dict[str, np.ndarray]] = {}
    for regime_id in regime_ids:
        mask = labels == regime_id
        distributions[regime_id] = {
            col: features.loc[mask, col].dropna().to_numpy() for col in columns
        }
        for col, arr in distributions[regime_id].items():
            if len(arr) < 5:
                raise ValueError(
                    f"Regime {regime_id} has only {len(arr)} in-sample observations for "
                    f"column {col!r} — too few to build a stable Wasserstein template "
                    "(need >= 5). Consider fewer regimes or a longer training window."
                )

    return RegimeTemplates(columns=tuple(columns), distributions=distributions, regime_ids=regime_ids)


def rolling_wasserstein_proximity(
    features: pd.DataFrame,
    templates: RegimeTemplates,
    window: int = 21,
) -> pd.DataFrame:
    """For each day, compute a proximity score in [0, 1] to each regime template.

    For a rolling window of ``window`` trading days ending on that day, this
    computes the mean Wasserstein-1 distance (averaged across
    ``templates.columns``) between the window's empirical distribution and
    each regime's template distribution, then converts distances to
    proximity weights via ``w_k = 1 - normalize(distance_k)`` (so the
    *closest* template gets the *highest* weight, matching the research
    doc's ``w_k = 1.0 - Normalize(W1(...))`` definition).

    Returns a DataFrame indexed like ``features`` with one column per
    regime id, ``wasserstein_proximity_{regime_id}``. The first
    ``window - 1`` rows are NaN (insufficient history).
    """
    cols = list(templates.columns)
    regime_ids = templates.regime_ids
    n = len(features)
    out = np.full((n, len(regime_ids)), np.nan)

    values = features[cols].to_numpy()
    for i in range(window - 1, n):
        window_slice = values[i - window + 1 : i + 1]
        if np.isnan(window_slice).any():
            continue
        distances = np.zeros(len(regime_ids))
        for k, regime_id in enumerate(regime_ids):
            per_col_dist = [
                wasserstein_distance(window_slice[:, c], templates.distributions[regime_id][col])
                for c, col in enumerate(cols)
            ]
            distances[k] = float(np.mean(per_col_dist))
        span = distances.max() - distances.min()
        if span > 1e-12:
            normalized = (distances - distances.min()) / span
        else:
            normalized = np.zeros_like(distances)
        proximity = 1.0 - normalized
        proximity_sum = proximity.sum()
        out[i] = proximity / proximity_sum if proximity_sum > 1e-12 else np.ones(len(regime_ids)) / len(regime_ids)

    col_names = [f"wasserstein_proximity_{r}" for r in regime_ids]
    return pd.DataFrame(out, index=features.index, columns=col_names)
