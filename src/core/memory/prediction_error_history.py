"""
PredictionErrorHistory — FAISS-backed store of per-area prediction error history.

Enables the agent to recall: "When I was in this area, how surprising was the world?"
This feeds into precision weighting in the PredictionErrorCalculator.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.models.memory_records import PredictionErrorRecord

log = logging.getLogger(__name__)

_FEATURE_DIM = 8  # [area_hash_floats(4), pe_health, pe_saturation, pe_resource, pe_threat]


class PredictionErrorHistory:
    """
    FAISS-indexed store of prediction error records per area.
    Enables familiarity computation for precision weighting.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._k: int = int(config.get("faiss_k_default", 5))
        self._epsilon: float = float(config.get("faiss_epsilon", 1e-6))
        self._min_obs: int = int(config.get("pe_min_observations", 5))
        self._ema_alpha: float = float(config.get("pe_ema_alpha", 0.1))

        self._records: List[PredictionErrorRecord] = []
        self._index: Optional[Any] = None  # faiss.IndexFlatL2
        self._vectors: Optional[np.ndarray] = None

    def record(self, rec: PredictionErrorRecord) -> None:
        """Add a new prediction error record and update the FAISS index."""
        self._records.append(rec)
        vec = self._to_vector(rec)

        if self._vectors is None:
            self._vectors = vec.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])

        self._rebuild_index()

    def clear_area(self, area_id: str) -> None:
        """Remove all records for a specific area and rebuild the FAISS index."""
        self._records = [r for r in self._records if r.area_id != area_id]
        if self._records:
            self._vectors = np.vstack([
                self._to_vector(r).reshape(1, -1) for r in self._records
            ])
        else:
            self._vectors = None
        self._rebuild_index()

    def get_area_familiarity(self, area_id: str) -> float:
        """
        Return [0,1] familiarity score for the given area.
        Higher = more observations = more familiar = higher precision.
        """
        area_records = [r for r in self._records if r.area_id == area_id]
        n = len(area_records)
        if n == 0:
            return 0.0
        return float(min(1.0, n / max(self._min_obs, 1)))

    def get_mean_pe(self, area_id: str) -> float:
        """Return mean PE for a specific area, or global mean if unknown."""
        area_records = [r for r in self._records if r.area_id == area_id]
        if not area_records:
            if self._records:
                return float(np.mean([r.mean_pe for r in self._records[-50:]]))
            return 0.5
        return float(np.mean([r.mean_pe for r in area_records[-20:]]))

    def query_similar(
        self,
        area_id: str,
        pe_snapshot: Dict[str, float],
    ) -> List[PredictionErrorRecord]:
        """Find k nearest records by area embedding + PE values."""
        if self._index is None or len(self._records) < self._k:
            return self._records[-self._k:] if self._records else []

        dummy_rec = PredictionErrorRecord(
            area_id=area_id,
            step=0,
            feature_vector=self._area_to_vector(area_id),
            pe_per_channel=pe_snapshot,
            mean_pe=float(np.mean(list(pe_snapshot.values()))) if pe_snapshot else 0.0,
        )
        query = self._to_vector(dummy_rec).reshape(1, -1)

        try:
            import faiss
            k_actual = min(self._k, len(self._records))
            distances, indices = self._index.search(query.astype(np.float32), k_actual)
            return [self._records[i] for i in indices[0] if 0 <= i < len(self._records)]
        except Exception as exc:
            log.debug("FAISS query failed: %s", exc)
            return self._records[-self._k:]

    def save(self, path: Path) -> None:
        """
        Persist records to JSON sidecar and FAISS index to disk.
        Two files: pe_history_records.json + pe_history.faiss
        """
        if not self._records:
            return
        path.mkdir(parents=True, exist_ok=True)
        try:
            (path / "pe_history_records.json").write_text(json.dumps([
                {
                    "area_id": r.area_id,
                    "step": r.step,
                    "feature_vector": r.feature_vector,
                    "pe_per_channel": r.pe_per_channel,
                    "mean_pe": r.mean_pe,
                    "action_id": r.action_id,
                }
                for r in self._records
            ]))
        except Exception as exc:
            log.warning("PredictionErrorHistory: records save failed: %s", exc)
            return
        try:
            import faiss
            if self._index is not None:
                faiss.write_index(self._index, str(path / "pe_history.faiss"))
        except Exception as exc:
            log.warning("PredictionErrorHistory: FAISS index save failed: %s", exc)
        log.info("PredictionErrorHistory: saved %d records to %s", len(self._records), path)

    def load(self, path: Path) -> None:
        """
        Load records from JSON sidecar and FAISS index from disk.
        Silently no-ops if files don't exist.
        """
        records_file = path / "pe_history_records.json"
        index_file = path / "pe_history.faiss"
        if not records_file.exists():
            return
        try:
            data = json.loads(records_file.read_text())
            self._records = [
                PredictionErrorRecord(
                    area_id=r["area_id"],
                    step=r["step"],
                    feature_vector=r["feature_vector"],
                    pe_per_channel=r["pe_per_channel"],
                    mean_pe=r["mean_pe"],
                    action_id=r.get("action_id"),
                )
                for r in data
            ]
            if self._records:
                self._vectors = np.vstack([
                    self._to_vector(r).reshape(1, -1) for r in self._records
                ])
            try:
                import faiss
                if index_file.exists():
                    self._index = faiss.read_index(str(index_file))
                else:
                    self._rebuild_index()
            except Exception:
                self._rebuild_index()
            log.info("PredictionErrorHistory: loaded %d records from %s",
                     len(self._records), path)
        except Exception as exc:
            log.warning("PredictionErrorHistory: load failed: %s", exc)

    def _rebuild_index(self) -> None:
        try:
            import faiss
            if self._vectors is None:
                return
            vecs = self._vectors.astype(np.float32)
            index = faiss.IndexFlatL2(vecs.shape[1])
            index.add(vecs)
            self._index = index
        except Exception:
            self._index = None

    def _to_vector(self, rec: PredictionErrorRecord) -> np.ndarray:
        area_vec = self._area_to_vector(rec.area_id)
        pe_vals = [
            rec.pe_per_channel.get("health", 0.0),
            rec.pe_per_channel.get("saturation", 0.0),
            rec.pe_per_channel.get("resource_level", 0.0),
            rec.pe_per_channel.get("threat_proximity", 0.0),
        ]
        return np.array(area_vec + pe_vals, dtype=np.float32)

    def _area_to_vector(self, area_id: str) -> List[float]:
        """Deterministic 4-float embedding from area_id string."""
        h = hash(area_id) % (10 ** 8)
        return [
            float((h >> 0) & 0xFF) / 255.0,
            float((h >> 8) & 0xFF) / 255.0,
            float((h >> 16) & 0xFF) / 255.0,
            float((h >> 24) & 0xFF) / 255.0,
        ]
