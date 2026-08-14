"""Unsupervised models for market-regime classification.

Three interchangeable backends (selected via config):
  - "gmm"    : Gaussian Mixture Model — soft clustering, gives regime probabilities.
  - "kmeans" : hard clustering baseline, fast and simple to sanity-check GMM against.
  - "hmm"    : Gaussian Hidden Markov Model — adds temporal persistence, so regimes
               don't flicker day-to-day the way i.i.d. clustering can.

All backends expose the same fit / predict / predict_proba interface so the
pipeline and downstream strategy code don't need to know which one is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeModel:
    """Common interface wrapping a scaler + an unsupervised backend."""

    def __init__(self, model_type: str = "gmm", n_regimes: int = 4, random_state: int = 42):
        self.model_type = model_type
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = self._build_backend()
        self.feature_names_: list[str] | None = None
        self.regime_order_: np.ndarray | None = None  # maps raw label -> vol-sorted label

    def _build_backend(self):
        if self.model_type == "gmm":
            return GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=self.random_state,
                n_init=5,
            )
        if self.model_type == "kmeans":
            return KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state, n_init=10
            )
        if self.model_type == "hmm":
            try:
                from hmmlearn.hmm import GaussianHMM
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "hmmlearn is required for model_type='hmm'. `pip install hmmlearn`."
                ) from e
            return GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                random_state=self.random_state,
                n_iter=200,
            )
        raise ValueError(f"Unknown model_type: {self.model_type!r}")

    def fit(self, features: pd.DataFrame) -> "RegimeModel":
        self.feature_names_ = list(features.columns)
        x = self.scaler.fit_transform(features.values)
        self.model.fit(x)
        # Order raw cluster labels by realized-volatility-like feature (mean of
        # feature columns whose name contains 'vol') so label 0 is always the
        # calmest regime and the highest label is always the most volatile,
        # regardless of the backend's arbitrary internal ordering.
        raw_labels = self._raw_predict(x)
        vol_cols = [c for c in features.columns if "vol" in c.lower()]
        ranking_series = (
            features[vol_cols].mean(axis=1) if vol_cols else features.mean(axis=1)
        )
        means_by_label = (
            pd.Series(ranking_series.values, index=raw_labels).groupby(level=0).mean()
        )
        self.regime_order_ = means_by_label.sort_values().index.to_numpy()
        return self

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_type == "hmm":
            return self.model.predict(x)
        return self.model.predict(x)

    def _remap(self, raw_labels: np.ndarray) -> np.ndarray:
        # regime_order_[i] == the raw label that should become new label i
        remap = {raw: new for new, raw in enumerate(self.regime_order_)}
        return np.array([remap[r] for r in raw_labels])

    def predict(self, features: pd.DataFrame) -> pd.Series:
        x = self.scaler.transform(features[self.feature_names_].values)
        raw = self._raw_predict(x)
        labels = self._remap(raw)
        return pd.Series(labels, index=features.index, name="regime")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        x = self.scaler.transform(features[self.feature_names_].values)
        if self.model_type == "kmeans":
            # KMeans has no native probability; fall back to a hard one-hot.
            raw = self._raw_predict(x)
            proba = np.eye(self.n_regimes)[raw]
        elif self.model_type == "hmm":
            proba = self.model.predict_proba(x)
        else:
            proba = self.model.predict_proba(x)
        remap = {raw: new for new, raw in enumerate(self.regime_order_)}
        # reorder columns to match the vol-sorted label order
        col_order = [remap[i] for i in range(self.n_regimes)]
        # invert: we need proba columns re-indexed so column `new` = original column `raw`
        inv_remap = {new: raw for raw, new in remap.items()}
        reordered = proba[:, [inv_remap[i] for i in range(self.n_regimes)]]
        cols = [f"p_regime_{i}" for i in range(self.n_regimes)]
        return pd.DataFrame(reordered, index=features.index, columns=cols)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RegimeModel":
        return joblib.load(path)


REGIME_LABELS_4 = {
    0: "low_vol_calm",
    1: "moderate_vol",
    2: "elevated_vol",
    3: "high_vol_stress",
}


def describe_regimes(labels: pd.Series, label_map: dict[int, str] | None = None) -> pd.Series:
    """Map integer regime ids to human-readable names (best-effort, for n_regimes<=4)."""
    n = labels.nunique()
    mapping = label_map or REGIME_LABELS_4
    if n > len(mapping):
        return labels.astype(str)
    return labels.map(mapping)
