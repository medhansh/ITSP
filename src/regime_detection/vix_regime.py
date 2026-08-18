# Destination: src/regime_detection/vix_regime.py  (modified)
"""Bucketing India VIX into a small number of discrete stress states, plus
the asymmetric hysteresis filter applied to those buckets -- together, the
production regime source for this strategy (see
``build_production_vix_regime`` and ``docs/regime_detection_spec.md``'s
"VIX-bucket regime" section for the walk-forward evidence and methodology
behind this being the default, replacing the GMM(4) price-feature regime
for exposure scaling, beta rotation, and the technical_momentum blend
ladder). GMM's own regime detection (``regime_detection/models.py``) is
still available as an explicit fallback via
``regime_detection.production_regime_source: "gmm"`` in config -- not
deleted, just no longer the default.

**Why 1D.** ``regime_detection/models.py`` clusters on a 14-dimensional
price-feature space and orders the resulting clusters by mean volatility
after the fact. VIX bucketing here clusters on ONE column only (raw VIX
level, optionally log-transformed) -- a genuinely one-dimensional problem,
both cheaper to sweep and directly visualizable/eyeball-checkable as a
histogram with the fitted breakpoints overlaid (a full price-feature
clustering has no such direct visual sanity check). Keeping it strictly 1D
also keeps ``RegimeModel``'s existing calm-to-stressed label ordering
trivially correct: with a single feature column, "mean of every feature"
(the fallback path in ``RegimeModel.fit`` when no column name contains
"vol" or "dispersion") IS the VIX level itself, so bucket 0 is guaranteed
to be the lowest-VIX bucket with no extra plumbing.

**Sweep, don't assume.** ``n_buckets`` is never hardcoded in
``sweep_bucket_counts``/``choose_bucket_count`` -- a 1D GMM fit at each
candidate count, reported via BIC (lower is better) and silhouette (higher
is better) side by side, exactly the two criteria the project's
price-feature ``n_regimes`` sweep already uses (see
``regime_detection_spec.md``'s "Choosing n_regimes" section). BIC is the
default choice criterion since it penalizes complexity directly, where
silhouette on a genuinely continuous 1D variable tends to degenerate
toward the smallest k tested. The shipped production config
(``regime_detection.vix_bucket_regime.n_buckets: 4``) fixes the count
explicitly rather than re-sweeping on every run, matching exactly what was
walk-forward validated -- see that config block's comment.

**Asymmetric hysteresis (``apply_bucket_hysteresis``).** Upgrades to a more
stressed bucket are instant; downgrades require
``min_days_to_downgrade`` consecutive days at the calmer level before
taking effect. This is deliberately NOT the same shape as the confirmed-
negative ``regime_detection.consensus_governor`` (symmetric persistence
that lagged both into and out of every regime change, and lost on every
axis on real data) -- see that config entry, and this function's own
docstring, for the full distinction.

**Look-ahead discipline.** Every function below is a pure fit/predict
transform with no forward-looking dependency by construction, EXCEPT that
``build_production_vix_regime`` fits on ALL available VIX history by
design -- that's what "production" means (use everything known up to
today). If some future workflow needs a train/test-safe version (e.g. a
research re-validation), fit ``fit_vix_buckets`` on a restricted training
slice explicitly rather than calling ``build_production_vix_regime`` on
the full series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.common.logging_utils import get_logger
from src.regime_detection.models import RegimeModel

logger = get_logger(__name__)

DEFAULT_K_RANGE = range(2, 7)  # 2..6 buckets, inclusive


def apply_bucket_hysteresis(bucket_labels: pd.Series, min_days_to_downgrade: int = 0) -> pd.Series:
    """Asymmetric confirmation filter on a bucket-label Series: an UPGRADE
    (bucket rises) is always accepted immediately (no delay), but a
    DOWNGRADE (bucket falls) is only accepted once the raw signal has
    stayed AT OR BELOW the lower level for ``min_days_to_downgrade``
    consecutive days. ``min_days_to_downgrade=0`` (default) is a no-op --
    returns ``bucket_labels`` unchanged.

    **Why asymmetric, and why this is not the same shape as the confirmed-
    negative ``regime_detection.consensus_governor``** (see that block's
    ``configs/config.yaml`` entry -- disabled; real-data result: governed
    exposure lost on CAGR, Sharpe, AND drawdown, attributed to SYMMETRIC
    persistence/hysteresis lagging both into and out of every regime
    change). This filter never delays recognizing INCREASED stress -- only
    the release back down is slowed, which is the standard "quick to
    de-risk, slow to re-risk" risk-management convention.

    **Motivation**: real walk-forward output showed
    ``vix_bucket_contemporaneous``'s OWN annualized turnover exceeding
    GMM's in at least one fold (2022) despite that fold's calmer overall
    conditions -- consistent with the bucket assignment whipsawing across
    a boundary rather than genuinely changing regime. This targets that
    directly, at the bucket-assignment level, independent of any GMM
    comparison or gating.

    Implemented as an explicit state-machine loop (not vectorized) --
    walk-forward fold windows are at most a few thousand rows, so this is
    not a performance concern, and a loop makes the "confirm N days before
    downgrading" logic unambiguous to read and audit.
    """
    if min_days_to_downgrade <= 0:
        return bucket_labels
    values = bucket_labels.to_numpy()
    out = values.copy()
    confirmed = values[0] if len(values) else 0
    pending_downgrade_streak = 0
    for i in range(len(values)):
        raw = values[i]
        if raw > confirmed:
            confirmed = raw           # upgrade: instant, no confirmation needed
            pending_downgrade_streak = 0
        elif raw < confirmed:
            pending_downgrade_streak += 1
            if pending_downgrade_streak >= min_days_to_downgrade:
                confirmed = raw       # downgrade accepted after N consecutive lower days
        else:
            pending_downgrade_streak = 0
        out[i] = confirmed
    return pd.Series(out, index=bucket_labels.index, name=bucket_labels.name)


def _prepare_vix_column(vix: pd.Series, log_transform: bool) -> pd.DataFrame:
    """Single-column feature frame for clustering: raw or log VIX level.

    ``log_transform`` (default False in the public functions below, but
    worth trying): VIX is right-skewed (long right tail during stress
    spikes), so log-space clustering can give more evenly-populated buckets
    at the calm end where most of the density actually sits. Exposed as a
    parameter rather than hardcoded so both can be compared -- not
    validated as superior on real data yet.
    """
    level = vix.dropna()
    if log_transform:
        if (level <= 0).any():
            raise ValueError(
                "log_transform=True but vix contains non-positive values -- "
                "VIX should always be > 0; check the input series."
            )
        level = np.log(level)
    return level.to_frame("vix_level")


def sweep_bucket_counts(
    vix: pd.Series,
    k_range: range | list[int] = DEFAULT_K_RANGE,
    log_transform: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a 1D GMM at each candidate bucket count and report BIC + silhouette.

    Returns a DataFrame indexed by ``k`` with columns ``bic`` (lower =
    better fit per unit of added complexity) and ``silhouette`` (higher =
    better-separated clusters; NaN for k=1, which has no silhouette score).

    ``vix`` should already be restricted to whatever window this sweep is
    allowed to see (see module docstring's look-ahead note) -- this
    function itself has no train/test concept, it just fits what it's
    given.
    """
    k_range = list(k_range)
    if len(k_range) == 0:
        raise ValueError("k_range must contain at least one candidate bucket count.")
    x_df = _prepare_vix_column(vix, log_transform)
    if len(x_df) < max(k_range) * 5:
        logger.warning(
            "sweep_bucket_counts: only %d usable VIX observations for a max candidate "
            "k=%d -- results at the higher end of k_range may be unstable (rule of "
            "thumb: want >=5 obs per candidate bucket).",
            len(x_df), max(k_range),
        )
    scaler = StandardScaler()
    x = scaler.fit_transform(x_df.values)

    rows = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state, n_init=5)
        gmm.fit(x)
        labels = gmm.predict(x)
        bic = gmm.bic(x)
        sil = silhouette_score(x, labels) if k > 1 and len(set(labels.tolist())) > 1 else float("nan")
        rows.append({"k": k, "bic": bic, "silhouette": sil})

    result = pd.DataFrame(rows).set_index("k")
    logger.info("VIX bucket-count sweep (k=%s):\n%s", list(k_range), result)
    return result


def choose_bucket_count(sweep_result: pd.DataFrame, criterion: str = "bic") -> int:
    """Pick a winning bucket count from ``sweep_bucket_counts``'s output.

    ``criterion``: ``"bic"`` (default, lower is better) or ``"silhouette"``
    (higher is better). See module docstring for why BIC is the default and
    why silhouette on a continuum variable tends to degenerate toward the
    smallest k tested -- passing ``criterion="silhouette"`` is there for
    comparison, not because it's expected to win.
    """
    if criterion == "bic":
        chosen = int(sweep_result["bic"].idxmin())
    elif criterion == "silhouette":
        valid = sweep_result["silhouette"].dropna()
        if valid.empty:
            raise ValueError("No valid silhouette scores in sweep_result (all k<=1 or degenerate fits).")
        chosen = int(valid.idxmax())
    else:
        raise ValueError(f"criterion must be 'bic' or 'silhouette', got {criterion!r}")
    logger.info("choose_bucket_count(criterion=%s) -> k=%d", criterion, chosen)
    return chosen


@dataclass
class VixBucketModel:
    """Thin wrapper pairing a fitted ``RegimeModel`` with the metadata
    (log_transform, chosen k) needed to apply it consistently at predict
    time, without the caller having to remember or re-pass those details.
    """
    regime_model: RegimeModel
    n_buckets: int
    log_transform: bool

    def predict(self, vix: pd.Series) -> pd.Series:
        """Bucket labels (0 = calmest .. n_buckets-1 = most stressed) for
        every date in ``vix`` (NaN VIX rows are dropped, not imputed)."""
        x_df = _prepare_vix_column(vix, self.log_transform)
        return self.regime_model.predict(x_df).rename("vix_bucket")


def fit_vix_buckets(
    vix: pd.Series,
    n_buckets: int,
    log_transform: bool = False,
    random_state: int = 42,
) -> VixBucketModel:
    """Fit a 1D GMM with a FIXED bucket count (use ``sweep_bucket_counts`` +
    ``choose_bucket_count`` first to pick ``n_buckets``, or pass a value
    already chosen by ``configs/config.yaml``'s cached sweep result).

    Reuses ``regime_detection.models.RegimeModel`` directly (rather than a
    bespoke GMM wrapper) so the calm->stressed label-ordering convention
    (label 0 = calmest) is identical to the price-feature regime model, and
    so this gets ``.save()``/``.load()`` and the GMM probability outputs for
    free. See module docstring for why a single ``vix_level`` column makes
    that ordering trivially correct here.
    """
    x_df = _prepare_vix_column(vix, log_transform)
    model = RegimeModel(model_type="gmm", n_regimes=n_buckets, random_state=random_state)
    model.fit(x_df)
    logger.info(
        "fit_vix_buckets: n_buckets=%d, log_transform=%s, fit on %d observations "
        "(%s .. %s)", n_buckets, log_transform, len(x_df),
        x_df.index.min().date() if len(x_df) else None,
        x_df.index.max().date() if len(x_df) else None,
    )
    return VixBucketModel(regime_model=model, n_buckets=n_buckets, log_transform=log_transform)


def describe_bucket_edges(model: VixBucketModel, vix: pd.Series) -> pd.DataFrame:
    """Diagnostic table: for the VIX history in ``vix``, the [min, max] raw
    VIX level actually observed in each bucket, plus how many days fell in
    it. Purely for sanity-checking a fitted ``VixBucketModel`` by eye (e.g.
    "does bucket 0 really correspond to VIX roughly under 15") -- not used
    by any other function in this module.
    """
    labels = model.predict(vix)
    aligned = vix.reindex(labels.index)
    df = pd.DataFrame({"vix": aligned, "bucket": labels})
    summary = df.groupby("bucket")["vix"].agg(["min", "max", "mean", "count"])
    return summary.sort_index()


def bucket_labels_to_regime_series(bucket_labels: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a (possibly sparse) integer bucket-label Series onto a
    full daily ``index``, defaulting warm-up dates to 0 (calmest) -- same
    "undefined-before-warm-up defaults to calm, not stressed" convention
    used throughout this project (e.g. ``momentum_reversal_blend.stress_from_volatility``'s
    warm-up handling).
    """
    return bucket_labels.reindex(index, method="ffill").fillna(0).astype(int)


def build_production_vix_regime(
    vix: pd.Series,
    index: pd.DatetimeIndex,
    n_buckets: int = 4,
    log_transform: bool = False,
    random_state: int = 42,
    min_days_to_downgrade: int = 0,
) -> pd.Series:
    """The single entry point production code (``run_full_pipeline.py``,
    paper trading) should call for the VIX-bucket regime: same-day VIX
    bucketing + asymmetric downgrade hysteresis, fit on ALL available VIX
    history (no train/test split -- this is what "production" means: use
    everything known up to today).

    **This is the walk-forward-validated production choice** -- see
    ``docs/regime_detection_spec.md``'s "VIX-bucket regime" section for the
    full methodology and evidence. Default parameters
    (``n_buckets=4, log_transform=False, random_state=42``, with
    ``min_days_to_downgrade`` set by the caller via config, not defaulted
    here to 10) match exactly what was walk-forward tested; changing them
    (especially ``min_days_to_downgrade`` well above ~15-20) should be
    re-validated before shipping, not assumed to keep improving -- a wider
    sweep during development showed the aggregate walk-forward metric kept
    rising past that point while individual folds visibly overfit (see that
    doc section's "known risk" note).

    Returns a full-index-length integer Series (0 = calmest), via
    ``bucket_labels_to_regime_series``.
    """
    vix_clean = vix.dropna()
    bucket_model = fit_vix_buckets(
        vix_clean, n_buckets=n_buckets, log_transform=log_transform, random_state=random_state,
    )
    labels = bucket_model.predict(vix_clean)
    labels = apply_bucket_hysteresis(labels, min_days_to_downgrade=min_days_to_downgrade)
    return bucket_labels_to_regime_series(labels, index)
