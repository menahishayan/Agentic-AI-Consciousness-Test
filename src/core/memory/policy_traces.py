"""
PolicyTraces — FAISS-backed store of policy selection outcomes.

Enables the agent to recall: "When I was in a similar situation and took this
action, what happened?" Used by PolicyGenerator to bias future selections.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
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

    def save(self, path: Path) -> None:
        """
        Persist records to JSON sidecar and FAISS index to disk.
        Two files: policy_traces_records.json + policy_traces.faiss
        """
        if not self._records:
            return
        path.mkdir(parents=True, exist_ok=True)
        try:
            (path / "policy_traces_records.json").write_text(json.dumps([
                {
                    "step": r.step,
                    "policy_id": r.policy_id,
                    "context_vector": r.context_vector,
                    "outcome_score": r.outcome_score,
                    "drive_signals": r.drive_signals,
                    "goal_coherence": r.goal_coherence,
                    "notes": r.notes,
                }
                for r in self._records
            ]))
        except Exception as exc:
            log.warning("PolicyTraces: records save failed: %s", exc)
            return
        try:
            import faiss
            if self._index is not None:
                faiss.write_index(self._index, str(path / "policy_traces.faiss"))
        except Exception as exc:
            log.warning("PolicyTraces: FAISS index save failed: %s", exc)
        log.info("PolicyTraces: saved %d records to %s", len(self._records), path)

    def load(self, path: Path) -> None:
        """
        Load records from JSON sidecar and FAISS index from disk.
        Silently no-ops if files don't exist.
        """
        records_file = path / "policy_traces_records.json"
        index_file = path / "policy_traces.faiss"
        if not records_file.exists():
            return
        try:
            data = json.loads(records_file.read_text())
            self._records = [
                PolicyTraceRecord(
                    step=r["step"],
                    policy_id=r["policy_id"],
                    context_vector=r["context_vector"],
                    outcome_score=r["outcome_score"],
                    drive_signals=r["drive_signals"],
                    goal_coherence=r.get("goal_coherence"),
                    notes=r.get("notes"),
                )
                for r in data
            ]
            if self._records:
                vecs = []
                for r in self._records:
                    vec = np.array(r.context_vector[:_DIM], dtype=np.float32)
                    if len(vec) < _DIM:
                        vec = np.pad(vec, (0, _DIM - len(vec)))
                    vecs.append(vec)
                self._vectors = np.vstack(vecs)
            try:
                import faiss
                if index_file.exists():
                    self._index = faiss.read_index(str(index_file))
                else:
                    self._rebuild_index()
            except Exception:
                self._rebuild_index()
            log.info("PolicyTraces: loaded %d records from %s", len(self._records), path)
        except Exception as exc:
            log.warning("PolicyTraces: load failed: %s", exc)

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
