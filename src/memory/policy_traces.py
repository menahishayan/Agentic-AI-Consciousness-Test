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
class PolicyTraceRecord:
    record_id: int
    tick: int
    channel_a_id: str
    channel_b_id: str
    winner_channel_id: str
    action_tag: str
    outcome_score: float
    context_vector: np.ndarray


class PolicyTraces:
    _DIM = 6

    def __init__(self, episode_length: int = 1000, k_default: int = 5, epsilon: float = 1e-6) -> None:
        self.episode_length = max(1, int(episode_length))
        self.k_default = max(1, int(k_default))
        self.epsilon = max(1e-12, float(epsilon))
        self._index = self._build_index()
        self._metadata: Dict[int, PolicyTraceRecord] = {}
        self._faiss_ids: List[int] = []
        self._id_counter = 0

    def record(
        self,
        channel_a_id: str,
        channel_b_id: str,
        winner_channel_id: str,
        action_tag: str,
        context_vector: np.ndarray,
        outcome_score: float,
        tick: int,
    ) -> None:
        vector = self._encode_context_for_record(context_vector=context_vector, tick=tick)
        record_id = int(self._id_counter)
        self._id_counter += 1

        record = PolicyTraceRecord(
            record_id=record_id,
            tick=int(tick),
            channel_a_id=str(channel_a_id),
            channel_b_id=str(channel_b_id),
            winner_channel_id=str(winner_channel_id),
            action_tag=str(action_tag),
            outcome_score=self._clip01(outcome_score),
            context_vector=vector,
        )
        self._metadata[record_id] = record

        if self._index is None:
            return
        try:
            self._index.add(vector.reshape(1, self._DIM).astype(np.float32))
            self._faiss_ids.append(record_id)
        except Exception as exc:
            logger.exception("PolicyTraces FAISS add failed: %s", exc)

    def get_conflict_resolution_score(
        self,
        channel_a_id: str,
        channel_b_id: str,
        context_vector: np.ndarray,
        k: int = 5,
    ) -> float:
        if not self._metadata:
            return 0.0

        query = self._encode_context_for_query(context_vector)
        neighbours = self._nearest_records(query, k)
        if not neighbours:
            return 0.0

        target_pair = self._pair_key(channel_a_id, channel_b_id)
        weighted_sum = 0.0
        total_weight = 0.0
        for distance, record in neighbours:
            if self._pair_key(record.channel_a_id, record.channel_b_id) != target_pair:
                continue
            if record.winner_channel_id == channel_a_id:
                signed = float(record.outcome_score)
            elif record.winner_channel_id == channel_b_id:
                signed = -float(record.outcome_score)
            else:
                continue
            weight = 1.0 / (float(distance) + self.epsilon)
            weighted_sum += weight * signed
            total_weight += weight

        if total_weight <= 0.0:
            return 0.0
        return float(weighted_sum / total_weight)

    def get_best_action_for_drive(
        self,
        channel_id: str,
        context_vector: np.ndarray,
        k: int = 5,
    ) -> Optional[str]:
        if not self._metadata:
            return None

        query = self._encode_context_for_query(context_vector)
        neighbours = self._nearest_records(query, k)
        if not neighbours:
            return None

        outcomes_by_action: Dict[str, List[float]] = {}
        for _, record in neighbours:
            if record.winner_channel_id != channel_id:
                continue
            outcomes_by_action.setdefault(record.action_tag, []).append(float(record.outcome_score))
        if not outcomes_by_action:
            return None

        scored_actions: List[Tuple[float, str]] = []
        for action_tag, values in outcomes_by_action.items():
            mean_outcome = float(sum(values) / float(len(values)))
            scored_actions.append((mean_outcome, action_tag))
        scored_actions.sort(key=lambda item: (-item[0], item[1]))
        return scored_actions[0][1]

    def clear(self) -> None:
        self._index = self._build_index()
        self._metadata.clear()
        self._faiss_ids.clear()
        self._id_counter = 0

    def _nearest_records(self, query: np.ndarray, k: int) -> List[Tuple[float, PolicyTraceRecord]]:
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
                out: List[Tuple[float, PolicyTraceRecord]] = []
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
                logger.exception("PolicyTraces FAISS search failed: %s", exc)

        # Fallback linear scan for resilience if FAISS fails.
        scored: List[Tuple[float, PolicyTraceRecord]] = []
        for record in self._metadata.values():
            distance = float(np.sum((query - record.context_vector) ** 2))
            scored.append((distance, record))
        scored.sort(key=lambda item: item[0])
        return scored[:target_k]

    def _build_index(self) -> Optional["faiss.IndexFlatL2"]:
        if faiss is None:
            logger.warning("faiss is unavailable; PolicyTraces will use linear fallback.")
            return None
        try:
            return faiss.IndexFlatL2(self._DIM)
        except Exception as exc:
            logger.exception("Failed to initialize PolicyTraces FAISS index: %s", exc)
            return None

    def _encode_context_for_record(self, context_vector: np.ndarray, tick: int) -> np.ndarray:
        values = np.asarray(context_vector, dtype=np.float32).reshape(-1)
        vector = np.zeros((self._DIM,), dtype=np.float32)
        length = min(5, values.shape[0])
        if length > 0:
            vector[:length] = values[:length]
        vector[5] = self._normalize_tick(tick)
        return self._l2_normalize(vector)

    def _encode_context_for_query(self, context_vector: np.ndarray) -> np.ndarray:
        values = np.asarray(context_vector, dtype=np.float32).reshape(-1)
        vector = np.zeros((self._DIM,), dtype=np.float32)
        length = min(self._DIM, values.shape[0])
        if length > 0:
            vector[:length] = values[:length]
        return self._l2_normalize(vector)

    def _normalize_tick(self, tick: int) -> float:
        normalized = (int(tick) % self.episode_length) / float(self.episode_length)
        return self._clip01(normalized)

    @staticmethod
    def _pair_key(channel_a_id: str, channel_b_id: str) -> Tuple[str, str]:
        a = str(channel_a_id)
        b = str(channel_b_id)
        if a <= b:
            return (a, b)
        return (b, a)

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
