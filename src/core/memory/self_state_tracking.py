"""
SelfStateTracking — FAISS-backed store of past homeostatic state snapshots.

Enables the agent to recall past physiological states and their outcomes.
Feeds depletion rate estimates back to AllostaticController.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core.models.memory_records import SelfStateRecord
from core.models.state import AgentState

log = logging.getLogger(__name__)

_DIM = 6  # [health, saturation, energy, oxygen, threat, resource]


class SelfStateTracking:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._k: int = int(config.get("faiss_k_default", 5))
        self._records: List[SelfStateRecord] = []
        self._vectors: Optional[np.ndarray] = None
        self._index: Optional[Any] = None

    def record(self, state: AgentState, action_taken: Optional[str] = None) -> None:
        vec = state.homeostatic_vector()
        rec = SelfStateRecord(
            step=state.step,
            homeostatic_vector=vec,
            area_id=state.perception.area_id or "unknown",
            action_taken=action_taken,
        )
        self._records.append(rec)
        arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        self._vectors = arr if self._vectors is None else np.vstack([self._vectors, arr])
        self._rebuild_index()

    def update_last_outcome(
        self,
        next_state: AgentState,
    ) -> None:
        """Update the most recent record with outcome deltas."""
        if not self._records:
            return
        rec = self._records[-1]
        prev = rec.homeostatic_vector
        curr = next_state.homeostatic_vector()
        rec.outcome_health_delta = curr[0] - prev[0]
        rec.outcome_saturation_delta = curr[1] - prev[1]

    def get_depletion_rates(self) -> Dict[str, float]:
        """
        Compute average per-step depletion rates from recent history.
        Returns {channel_id: rate} for channels where rate > 0.
        """
        if len(self._records) < 2:
            return {}

        recent = self._records[-20:]
        health_deltas, sat_deltas = [], []

        for i in range(1, len(recent)):
            step_gap = recent[i].step - recent[i - 1].step
            if step_gap <= 0:
                continue
            prev = recent[i - 1].homeostatic_vector
            curr = recent[i].homeostatic_vector
            health_deltas.append((prev[0] - curr[0]) / step_gap)   # depletion = positive
            sat_deltas.append((prev[1] - curr[1]) / step_gap)

        rates = {}
        if health_deltas:
            r = float(np.mean(health_deltas))
            if r > 0:
                rates["health"] = r
        if sat_deltas:
            r = float(np.mean(sat_deltas))
            if r > 0:
                rates["saturation"] = r

        return rates

    def query_similar(self, state: AgentState) -> List[SelfStateRecord]:
        if self._index is None or len(self._records) < self._k:
            return self._records[-self._k:] if self._records else []

        query = np.array(state.homeostatic_vector(), dtype=np.float32).reshape(1, -1)
        try:
            import faiss
            k_actual = min(self._k, len(self._records))
            _, indices = self._index.search(query, k_actual)
            return [self._records[i] for i in indices[0] if 0 <= i < len(self._records)]
        except Exception:
            return self._records[-self._k:]

    def _rebuild_index(self) -> None:
        try:
            import faiss
            if self._vectors is None:
                return
            vecs = self._vectors.astype(np.float32)
            idx = faiss.IndexFlatL2(vecs.shape[1])
            idx.add(vecs)
            self._index = idx
        except Exception:
            self._index = None
