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

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


class RegimeModel:
    """Common interface wrapping a scaler + an unsupervised backend."""

    def __init__(self, model_type: str = "gmm", n_regimes: int = 4, random_state: int = 42,
                 eps: float = 1.5, min_cluster_size: int = 50,
                 noise_to_most_stressed: bool = True):
        self.model_type = model_type
        self.n_regimes = n_regimes
        self.random_state = random_state
        # Density-backend params, ignored by gmm/kmeans/hmm/bgmm.
        self.eps = eps
        self.min_cluster_size = min_cluster_size
        # How to treat DBSCAN/HDBSCAN's noise label (-1). True (default) folds
        # noise days into the MOST STRESSED regime, on the reading that a day
        # too unusual to belong to any dense region is a stress day. This is a
        # CHOICE, not a finding -- the alternative (noise as its own calm-side
        # bucket, or excluded) would give different exposure behavior, and the
        # fraction of days affected is logged so the choice stays visible.
        self.noise_to_most_stressed = noise_to_most_stressed
        self.n_states_found_: int | None = None
        self.noise_fraction_: float | None = None
        self._train_labels_: np.ndarray | None = None
        self._train_x_: np.ndarray | None = None
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
        if self.model_type == "bgmm":
            # Bayesian GMM with a Dirichlet-process prior: fit with n_regimes as
            # an UPPER BOUND and let unneeded components shrink toward zero
            # weight, so the data decides how many states are actually
            # populated instead of the count being imposed. This is the
            # sklearn equivalent of the entropy-penalty "automatic k"
            # methods surveyed in the clustering literature, and gives a third
            # independent read on n_regimes alongside BIC (which fell
            # monotonically to the edge of the tested range, i.e. unusable)
            # and silhouette (which preferred n=2).
            from sklearn.mixture import BayesianGaussianMixture

            return BayesianGaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                # Small prior => strong pressure to use FEWER components. Left
                # explicit rather than defaulted because this single value is
                # what makes the arm a test of "how many states does the data
                # want" rather than a rerun of plain GMM. An earlier value of
                # 1/n_regimes was too weak to prune anything (all 10 allowed
                # components stayed populated on real data, which is a silent
                # no-op dressed up as automatic model selection).
                weight_concentration_prior=1e-3,
                random_state=self.random_state,
                max_iter=500,
                n_init=3,
            )
        if self.model_type in ("dbscan", "hdbscan"):
            # Density-based: the only backends here that do NOT take a cluster
            # count at all, and the only ones that can label a day as NOISE
            # rather than forcing it into a state. That is the point of
            # including them. Every k-forcing backend must invent a home for
            # the ~2.9% of days in the smallest state; a density method can
            # instead say those days belong to no regime, which is a
            # materially different (and untested) model of rare stress days.
            if self.model_type == "hdbscan":
                try:
                    from sklearn.cluster import HDBSCAN
                except ImportError as e:  # pragma: no cover
                    raise ImportError(
                        "HDBSCAN requires scikit-learn >= 1.3. Use model_type='dbscan' instead."
                    ) from e
                return HDBSCAN(min_cluster_size=self.min_cluster_size)
            from sklearn.cluster import DBSCAN

            return DBSCAN(eps=self.eps, min_samples=self.min_cluster_size)
        raise ValueError(f"Unknown model_type: {self.model_type!r}")

    def fit(self, features: pd.DataFrame) -> "RegimeModel":
        self.feature_names_ = list(features.columns)
        x = self.scaler.fit_transform(features.values)
        if self.model_type in ("dbscan", "hdbscan"):
            # These backends have NO predict() -- they are fit_predict only, and
            # they emit -1 for noise. Cache the training labels and points so
            # _raw_predict can serve them back, and assign any unseen point to
            # the nearest cluster centroid.
            raw = self.model.fit_predict(x)
            n_noise = int((raw == -1).sum())
            self.noise_fraction_ = float(n_noise / len(raw))
            found = sorted(set(raw.tolist()) - {-1})
            self.n_states_found_ = len(found)
            logger.info(
                "RegimeModel(%s): discovered %d cluster(s) from the data (no k was imposed); "
                "%d/%d days (%.1f%%) labeled NOISE and %s.",
                self.model_type, self.n_states_found_, n_noise, len(raw),
                100 * self.noise_fraction_,
                "folded into the most stressed regime" if self.noise_to_most_stressed
                else "kept as their own separate label",
            )
            if self.n_states_found_ == 0:
                raise ValueError(
                    f"{self.model_type} found no clusters at all (every point is noise). "
                    f"Loosen eps (currently {self.eps}) or lower min_cluster_size "
                    f"(currently {self.min_cluster_size})."
                )
            self._train_labels_, self._train_x_ = raw, x
        else:
            self.model.fit(x)
        # Order raw cluster labels by realized-volatility-like feature (mean of
        # feature columns whose name contains 'vol') so label 0 is always the
        # calmest regime and the highest label is always the most volatile,
        # regardless of the backend's arbitrary internal ordering.
        raw_labels = self._raw_predict(x)
        # Regimes are ordered by "how turbulent" each cluster's mean feature
        # level is, using any column whose name signals that axis: realized/
        # range volatility ("vol") in the default price-based feature set, OR
        # cross-sectional factor-score dispersion ("dispersion") when
        # regime_detection.factor_dispersion.use_price_features=False strips
        # out every vol-named column entirely (see features.py /
        # factor_dispersion.py). Signed delta/"change" columns (e.g.
        # ``vix_change_5d``, ``factor_dispersion_change_5d``) are explicitly
        # EXCLUDED from this ranking set -- they can be negative, so
        # averaging them in with magnitude-like level/z-score columns can
        # break the intended "label 0 = calmest, label N-1 = most turbulent"
        # monotonic ordering. Falls back to the mean of every feature only
        # if no magnitude-like vol/dispersion column is present at all.
        vol_cols = [
            c for c in features.columns
            if ("vol" in c.lower() or "dispersion" in c.lower()) and "change" not in c.lower()
        ]
        ranking_series = (
            features[vol_cols].mean(axis=1) if vol_cols else features.mean(axis=1)
        )
        # Built from OBSERVED labels rather than range(n_regimes): bgmm prunes
        # components it does not need, and the density backends discover their
        # own count, so neither is guaranteed to populate 0..n_regimes-1.
        means_by_label = (
            pd.Series(ranking_series.values, index=raw_labels).groupby(level=0).mean()
        )
        # Noise (-1) is not a regime and must not take part in the calm->stressed
        # ranking; it is remapped separately in _remap.
        means_by_label = means_by_label[means_by_label.index != -1]
        self.regime_order_ = means_by_label.sort_values().index.to_numpy()
        if self.model_type in ("gmm", "bgmm"):
            self.n_states_found_ = len(self.regime_order_)
            if self.model_type == "bgmm" and self.n_states_found_ < self.n_regimes:
                logger.info(
                    "RegimeModel(bgmm): the Dirichlet-process prior pruned %d of the %d allowed "
                    "components -- the data populated only %d state(s).",
                    self.n_regimes - self.n_states_found_, self.n_regimes, self.n_states_found_,
                )
        return self

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_type in ("dbscan", "hdbscan"):
            if self._train_x_ is not None and x.shape == self._train_x_.shape \
                    and np.allclose(x, self._train_x_):
                return self._train_labels_
            # Unseen points: assign to the nearest CLUSTER centroid (noise
            # points excluded from centroid computation, since "nearest to the
            # noise cloud" is not meaningful). Density methods have no native
            # out-of-sample rule, so this is an explicit approximation rather
            # than something the algorithm provides.
            labels = self._train_labels_
            clusters = sorted(set(labels.tolist()) - {-1})
            centroids = np.vstack([self._train_x_[labels == c].mean(axis=0) for c in clusters])
            nearest = np.argmin(
                ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
            )
            return np.array([clusters[i] for i in nearest])
        return self.model.predict(x)

    def _remap(self, raw_labels: np.ndarray) -> np.ndarray:
        # regime_order_[i] == the raw label that should become new label i
        remap = {raw: new for new, raw in enumerate(self.regime_order_)}
        if -1 in raw_labels:
            # Noise days. See noise_to_most_stressed in __init__ for why this is
            # a documented choice rather than a neutral default.
            remap[-1] = (len(self.regime_order_) - 1) if self.noise_to_most_stressed else 0
        return np.array([remap[r] for r in raw_labels])

    def predict(self, features: pd.DataFrame) -> pd.Series:
        x = self.scaler.transform(features[self.feature_names_].values)
        raw = self._raw_predict(x)
        labels = self._remap(raw)
        return pd.Series(labels, index=features.index, name="regime")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        x = self.scaler.transform(features[self.feature_names_].values)
        if self.model_type in ("kmeans", "dbscan", "hdbscan"):
            # No native probability; fall back to a hard one-hot over the
            # states actually observed (not range(n_regimes) -- the density
            # backends discover their own count).
            n_states = len(self.regime_order_)
            new_labels = self._remap(self._raw_predict(x))
            proba = np.eye(n_states)[new_labels]
            cols = [f"p_regime_{i}" for i in range(n_states)]
            return pd.DataFrame(proba, index=features.index, columns=cols)
        elif self.model_type == "hmm":
            proba = self.model.predict_proba(x)
        else:
            proba = self.model.predict_proba(x)
        remap = {raw: new for new, raw in enumerate(self.regime_order_)}
        # Re-index proba columns so column `new` = original column `raw`. Sized
        # by OBSERVED states, since bgmm may have pruned components.
        inv_remap = {new: raw for raw, new in remap.items()}
        n_states = len(self.regime_order_)
        reordered = proba[:, [inv_remap[i] for i in range(n_states)]]
        cols = [f"p_regime_{i}" for i in range(n_states)]
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
