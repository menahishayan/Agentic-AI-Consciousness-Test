from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from memory.policy_traces import PolicyTraces
from memory.prediction_error_history import PredictionErrorHistory
from memory.self_state_tracking import SelfStateTracking
from memory.working_memory_buffer import WorkingMemoryBuffer, WorkingMemoryEntry


@dataclass
class MemoryConfig:
    working_memory_capacity: int = 100
    pe_min_observations: int = 5
    pe_ema_alpha: float = 0.1
    faiss_k_default: int = 5
    faiss_epsilon: float = 1e-6
    episode_length: int = 1000


class MemoryManager:
    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config if isinstance(config, MemoryConfig) else MemoryConfig()

        self._working_memory = WorkingMemoryBuffer(
            capacity=self.config.working_memory_capacity,
        )
        self._prediction_error_history = PredictionErrorHistory(
            min_observations=self.config.pe_min_observations,
            ema_alpha=self.config.pe_ema_alpha,
        )
        self._self_state_tracking = SelfStateTracking(
            k_default=self.config.faiss_k_default,
            epsilon=self.config.faiss_epsilon,
        )
        self._policy_traces = PolicyTraces(
            episode_length=self.config.episode_length,
            k_default=self.config.faiss_k_default,
            epsilon=self.config.faiss_epsilon,
        )

    # WorkingMemoryBuffer delegation
    def record_working(self, entry: WorkingMemoryEntry) -> None:
        self._working_memory.record(entry)

    def get_recent(self, n: int, entry_type: Optional[str] = None) -> List[WorkingMemoryEntry]:
        return self._working_memory.get_recent(n=n, entry_type=entry_type)

    def get_active_goals(self) -> List[WorkingMemoryEntry]:
        return self._working_memory.get_active_goals()

    # PredictionErrorHistory delegation
    def record_pe(self, area_id: str, error: Any) -> None:
        self._prediction_error_history.record(area_id=area_id, error=error)

    def get_area_familiarity(self, area_id: str) -> float:
        return self._prediction_error_history.get_area_familiarity(area_id=area_id)

    def get_area_threat_prior(self, area_id: str) -> float:
        return self._prediction_error_history.get_area_threat_prior(area_id=area_id)

    # SelfStateTracking delegation
    def record_state(
        self,
        state: Any,
        channel_deltas: Dict[str, float],
        context_tags: List[str],
        arousal: float,
    ) -> None:
        self._self_state_tracking.record(
            state=state,
            channel_deltas=channel_deltas,
            context_tags=context_tags,
            arousal=arousal,
        )

    def get_depletion_rate(self, state: Any, channel_id: str) -> Optional[float]:
        return self._self_state_tracking.get_depletion_rate(
            state=state,
            channel_id=channel_id,
            k=self.config.faiss_k_default,
        )

    def get_capability_estimate(self, state: Any, context_tag: str) -> Optional[float]:
        return self._self_state_tracking.get_capability_estimate(
            state=state,
            context_tag=context_tag,
            k=self.config.faiss_k_default,
        )

    # PolicyTraces delegation
    def record_trace(
        self,
        channel_a_id: str,
        channel_b_id: str,
        winner_channel_id: str,
        action_tag: str,
        context_vector: np.ndarray,
        outcome_score: float,
        tick: int,
    ) -> None:
        self._policy_traces.record(
            channel_a_id=channel_a_id,
            channel_b_id=channel_b_id,
            winner_channel_id=winner_channel_id,
            action_tag=action_tag,
            context_vector=context_vector,
            outcome_score=outcome_score,
            tick=tick,
        )

    def get_conflict_resolution_score(
        self,
        channel_a_id: str,
        channel_b_id: str,
        context_vector: np.ndarray,
    ) -> float:
        return self._policy_traces.get_conflict_resolution_score(
            channel_a_id=channel_a_id,
            channel_b_id=channel_b_id,
            context_vector=context_vector,
            k=self.config.faiss_k_default,
        )

    def get_best_action_for_drive(
        self,
        channel_id: str,
        context_vector: np.ndarray,
    ) -> Optional[str]:
        return self._policy_traces.get_best_action_for_drive(
            channel_id=channel_id,
            context_vector=context_vector,
            k=self.config.faiss_k_default,
        )

    # Lifecycle
    def clear_episode(self) -> None:
        self._working_memory.clear()

    def clear_all(self) -> None:
        self._working_memory.clear()
        self._prediction_error_history.clear()
        self._self_state_tracking.clear()
        self._policy_traces.clear()

    # Compatibility/read-only accessors
    @property
    def working_memory(self) -> WorkingMemoryBuffer:
        return self._working_memory

    @property
    def prediction_errors(self) -> PredictionErrorHistory:
        return self._prediction_error_history

    @property
    def self_state(self) -> SelfStateTracking:
        return self._self_state_tracking

    @property
    def policy_traces(self) -> PolicyTraces:
        return self._policy_traces
