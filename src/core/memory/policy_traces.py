"""
PolicyTraces — FAISS-backed store of policy selection outcomes.

Enables the agent to recall: "When I was in a similar situation and took this
action, what happened?" Used by PolicyGenerator to bias future selections.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core.models.memory_records import PolicyTraceRecord

log = logging.getLogger(__name__)

_DIM = 10  # [health, saturation, energy, threat, resource, pe_mean, urgency, x, z, step_norm]


class PolicyTraces:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._k: int = int(config.get("faiss_k_default", 5))
        self._records: List[PolicyTraceRecord] = []
        self._vectors: Optional[np.ndarray] = None
        self._index: Optional[Any] = None

    def record(self, rec: PolicyTraceRecord) -> None:
        self._records.append(rec)
        vec = np.array(rec.context_vector[:_DIM], dtype=np.float32)
        # Pad if shorter than expected
        if len(vec) < _DIM:
            vec = np.pad(vec, (0, _DIM - len(vec)))
        vec = vec.reshape(1, -1)
        self._vectors = vec if self._vectors is None else np.vstack([self._vectors, vec])
        self._rebuild_index()

    def get_policy_outcome_history(self, policy_id: str) -> float:
        """Return average outcome score for a given policy_id (last 50 records)."""
        relevant = [r for r in self._records if r.policy_id == policy_id][-50:]
        if not relevant:
            return 0.5
        return float(np.mean([r.outcome_score for r in relevant]))

    def query_similar(self, context_vector: List[float]) -> List[PolicyTraceRecord]:
        if self._index is None or len(self._records) < self._k:
            return self._records[-self._k:] if self._records else []

        vec = np.array(context_vector[:_DIM], dtype=np.float32)
        if len(vec) < _DIM:
            vec = np.pad(vec, (0, _DIM - len(vec)))
        query = vec.reshape(1, -1)

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
