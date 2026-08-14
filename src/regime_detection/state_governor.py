"""Entropy-gated consensus state governor for regime transitions.

Second half of the Hybrid HMM + Wasserstein design. Blends the HMM/GMM
posterior with the Wasserstein template-proximity weights from
``wasserstein_proximity.py`` into a single consensus probability vector,
then applies three gating rules (from the research doc's "state governor")
before ever actually changing the *active* trading regime:

1. **Consensus gating** — if the consensus vector's Shannon entropy exceeds
   ``entropy_limit``, the day is flagged "Transitional" outright, regardless
   of which regime has the highest probability.
2. **Persistence governor** — a newly-proposed regime must be the top
   consensus pick for ``persistence_window`` *consecutive* bars before it is
   allowed to become the active regime. This directly targets the
   over-switching / whipsaw problem, which is the same failure mode
   underlying the project's observed under-exposure issue: if the active
   regime keeps flickering, position sizing keeps getting cut even when the
   market never really left a permissive regime.
3. **Hysteresis gating** — even after persistence is satisfied, a
   transition from active state i to candidate state j additionally
   requires j's consensus probability to exceed i's by a safety buffer
   ``hysteresis_epsilon``, so two nearly-tied regimes don't toggle back and
   forth turn by turn.

This module is intentionally a pure, stateful, sequential state machine
(mirrors the research doc's blueprint) rather than a vectorized pandas
operation, because rules 2 and 3 are explicitly path-dependent (they depend
on the *previous* active state and a running candidate counter).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRANSITIONAL = "transitional"


@dataclass
class RegimeStateGovernor:
    """Stateful sequential governor — call ``step`` once per bar, in order."""

    entropy_limit: float = 0.85
    persistence_window: int = 5
    hysteresis_epsilon: float = 0.05
    initial_state: str | int = TRANSITIONAL

    active_state: str | int = field(init=False)
    _candidate: str | int | None = field(default=None, init=False)
    _candidate_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.active_state = self.initial_state

    @staticmethod
    def consensus_probability(hmm_proba: np.ndarray, wasserstein_weights: np.ndarray) -> np.ndarray:
        """P_consensus(k) = hmm_proba[k] * w[k] / sum_j hmm_proba[j] * w[j]."""
        combined = hmm_proba * wasserstein_weights
        total = combined.sum()
        if total <= 1e-12:
            return np.ones_like(hmm_proba) / len(hmm_proba)
        return combined / total

    @staticmethod
    def entropy(consensus: np.ndarray) -> float:
        p = np.clip(consensus, 1e-12, 1.0)
        return float(-np.sum(p * np.log2(p)))

    def step(self, consensus: np.ndarray, regime_ids: list[int]) -> dict:
        """Advance the governor by one bar. Returns a dict of diagnostics + the
        resolved active state for this bar."""
        h = self.entropy(consensus)
        top_idx = int(np.argmax(consensus))
        top_prob = float(consensus[top_idx])
        proposed = regime_ids[top_idx]

        if h > self.entropy_limit:
            proposed = TRANSITIONAL

        # Hysteresis: if proposing a switch away from the current active
        # state, require a safety-buffer margin over the active state's own
        # consensus probability before even considering the switch.
        if proposed != TRANSITIONAL and proposed != self.active_state and self.active_state in regime_ids:
            active_idx = regime_ids.index(self.active_state)
            active_prob = float(consensus[active_idx])
            if not (top_prob > active_prob + self.hysteresis_epsilon):
                proposed = self.active_state

        # Persistence governor.
        if proposed == self.active_state:
            self._candidate = None
            self._candidate_count = 0
        else:
            if proposed == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = proposed
                self._candidate_count = 1
            if self._candidate_count >= self.persistence_window:
                self.active_state = proposed
                self._candidate = None
                self._candidate_count = 0

        return {
            "consensus_entropy": h,
            "proposed_regime": proposed,
            "active_regime": self.active_state,
            "is_transitional": self.active_state == TRANSITIONAL,
            "candidate_count": self._candidate_count,
        }


def run_governor_over_history(
    proba: pd.DataFrame,
    wasserstein_proximity: pd.DataFrame,
    entropy_limit: float = 0.85,
    persistence_window: int = 5,
    hysteresis_epsilon: float = 0.05,
) -> pd.DataFrame:
    """Run the governor sequentially over a full history.

    ``proba`` — columns ``p_regime_0..p_regime_{k-1}`` from
    ``RegimeModel.predict_proba``.
    ``wasserstein_proximity`` — columns ``wasserstein_proximity_0..{k-1}``
    from ``rolling_wasserstein_proximity``, same index as ``proba``.

    Rows where the Wasserstein proximity is not yet available (warm-up
    window) fall back to the raw HMM/GMM posterior (proximity weights of 1
    for every regime), so the governor still produces output from day one —
    it just isn't Wasserstein-reweighted until the rolling window fills.

    Returns a DataFrame indexed like ``proba`` with columns:
    ``consensus_entropy``, ``proposed_regime``, ``active_regime``,
    ``is_transitional``.
    """
    regime_ids = sorted(int(c.replace("p_regime_", "")) for c in proba.columns if c.startswith("p_regime_"))
    proba_cols = [f"p_regime_{r}" for r in regime_ids]
    prox_cols = [f"wasserstein_proximity_{r}" for r in regime_ids]

    governor = RegimeStateGovernor(
        entropy_limit=entropy_limit,
        persistence_window=persistence_window,
        hysteresis_epsilon=hysteresis_epsilon,
    )

    records = []
    for idx in proba.index:
        hmm_p = proba.loc[idx, proba_cols].to_numpy(dtype=float)
        prox_row = wasserstein_proximity.loc[idx, prox_cols].to_numpy(dtype=float) if idx in wasserstein_proximity.index else np.full(len(regime_ids), np.nan)
        if np.isnan(prox_row).any():
            prox_row = np.ones(len(regime_ids))  # neutral weighting during warm-up
        consensus = governor.consensus_probability(hmm_p, prox_row)
        result = governor.step(consensus, regime_ids)
        records.append(result)

    return pd.DataFrame(records, index=proba.index)
