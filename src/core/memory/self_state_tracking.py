from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - import path varies by platform
    faiss = None


logger = logging.getLogger(__name__)


@dataclass
class SelfStateRecord:
    record_id: int
    tick: int
    state_vector: np.ndarray
    channel_deltas: Dict[str, float]
    context_tags: List[str]
    arousal_at_record: float


class SelfStateTracking:
    _DIM = 6

    def __init__(self, k_default: int = 5, epsilon: float = 1e-6) -> None:
        self.k_default = max(1, int(k_default))
        self.epsilon = max(1e-12, float(epsilon))
        self._index = self._build_index()
        self._metadata: Dict[int, SelfStateRecord] = {}
        self._faiss_ids: List[int] = []
        self._id_counter = 0

    def record(
        self,
        state: Any,
        channel_deltas: Dict[str, float],
        context_tags: List[str],
        arousal: float,
    ) -> None:
        vector = self._encode_state(state)
        record_id = int(self._id_counter)
        self._id_counter += 1

        record = SelfStateRecord(
            record_id=record_id,
            tick=self._extract_tick(state),
            state_vector=vector,
            channel_deltas={str(key): float(value) for key, value in dict(channel_deltas).items()},
            context_tags=[str(tag) for tag in list(context_tags)],
            arousal_at_record=float(arousal),
        )
        self._metadata[record_id] = record

        if self._index is None:
            return
        try:
            self._index.add(vector.reshape(1, self._DIM).astype(np.float32))
            self._faiss_ids.append(record_id)
        except Exception as exc:
            logger.exception("SelfStateTracking FAISS add failed: %s", exc)

    def get_depletion_rate(
        self,
        state: Any,
        channel_id: str,
        k: int = 5,
    ) -> Optional[float]:
        if not self._metadata:
            return None

        query = self._encode_state(state)
        neighbours = self._nearest_records(query, k)
        if not neighbours:
            return None

        weights: List[float] = []
        values: List[float] = []
        for distance, record in neighbours:
            if channel_id not in record.channel_deltas:
                continue
            value = float(record.channel_deltas[channel_id])
            weight = 1.0 / (float(distance) + self.epsilon)
            values.append(value)
            weights.append(weight)

        min_required = max(1, int(k) // 2)
        if len(values) < min_required:
            return None
        denom = float(sum(weights))
        if denom <= 0.0:
            return None
        return float(sum(weight * value for weight, value in zip(weights, values)) / denom)

    def get_capability_estimate(
        self,
        state: Any,
        context_tag: str,
        k: int = 5,
    ) -> Optional[float]:
        if not self._metadata:
            return None

        query = self._encode_state(state)
        neighbours = self._nearest_records(query, k)
        if not neighbours:
            return None

        matching = [
            record.arousal_at_record
            for _, record in neighbours
            if context_tag in record.context_tags
        ]
        if not matching:
            return None
        return float(sum(matching) / float(len(matching)))

    def clear(self) -> None:
        self._index = self._build_index()
        self._metadata.clear()
        self._faiss_ids.clear()
        self._id_counter = 0

    def _nearest_records(self, query: np.ndarray, k: int) -> List[Tuple[float, SelfStateRecord]]:
        if not self._metadata:
            return []

        target_k = max(1, int(k))
        if self._index is not None and self._faiss_ids:
            try:
                k_search = min(target_k, len(self._faiss_ids))
                distances, indices = self._index.search(
                    query.reshape(1, self._DIM).astype(np.float32),
                    int(k_search),
                )
                out: List[Tuple[float, SelfStateRecord]] = []
                for distance, index in zip(distances[0], indices[0]):
                    if int(index) < 0 or int(index) >= len(self._faiss_ids):
                        continue
                    record_id = self._faiss_ids[int(index)]
                    record = self._metadata.get(record_id)
                    if record is None:
                        continue
                    out.append((float(distance), record))
                if out:
                    return out
            except Exception as exc:
                logger.exception("SelfStateTracking FAISS search failed: %s", exc)

        # Fallback linear scan for resilience if FAISS fails.
        scored: List[Tuple[float, SelfStateRecord]] = []
        for record in self._metadata.values():
            distance = float(np.sum((query - record.state_vector) ** 2))
            scored.append((distance, record))
        scored.sort(key=lambda item: item[0])
        return scored[:target_k]

    def _build_index(self) -> Optional["faiss.IndexFlatL2"]:
        if faiss is None:
            logger.warning("faiss is unavailable; SelfStateTracking will use linear fallback.")
            return None
        try:
            return faiss.IndexFlatL2(self._DIM)
        except Exception as exc:
            logger.exception("Failed to initialize SelfStateTracking FAISS index: %s", exc)
            return None

    def _encode_state(self, state: Any) -> np.ndarray:
        health = self._clip01(self._extract_component(state, "health"))
        hunger = self._clip01(self._extract_component(state, "hunger"))
        resource_level = self._clip01(self._extract_component(state, "resource_level"))
        threat_proximity = self._clip01(self._extract_component(state, "threat_proximity"))
        oxygen = self._clip01(self._extract_component(state, "oxygen"))
        time_of_day = self._extract_component(state, "time_of_day")
        if time_of_day is None:
            time_of_day = 0.0
        vector = np.asarray(
            [health, hunger, resource_level, threat_proximity, oxygen, float(time_of_day)],
            dtype=np.float32,
        )
        return self._l2_normalize(vector)

    @staticmethod
    def _extract_component(state: Any, key: str) -> Optional[float]:
        raw: Any = None
        if isinstance(state, Mapping):
            raw = state.get(key)
            if raw is None:
                values = state.get("values")
                if isinstance(values, Mapping):
                    raw = values.get(key)
        else:
            raw = getattr(state, key, None)
            if raw is None:
                values = getattr(state, "values", None)
                if isinstance(values, Mapping):
                    raw = values.get(key)

        if isinstance(raw, bool):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_tick(state: Any) -> int:
        tick: Any = None
        if isinstance(state, Mapping):
            tick = state.get("tick")
        else:
            tick = getattr(state, "tick", None)
        if isinstance(tick, int):
            return int(tick)
        return 0

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _clip01(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        return float(max(0.0, min(1.0, value)))
