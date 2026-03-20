from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from core.memory.long_term_memory import LongTermMemory
from core.memory.policy_traces import PolicyTraces
from core.memory.prediction_error_history import PredictionErrorHistory
from core.memory.self_state_tracking import SelfStateTracking
from core.memory.working_memory_buffer import WorkingMemoryBuffer, WorkingMemoryEntry


@dataclass
class MemoryConfig:
    working_memory_capacity: int = 100
    pe_min_observations: int = 5
    pe_ema_alpha: float = 0.1
    faiss_k_default: int = 5
    faiss_epsilon: float = 1e-6
    episode_length: int = 1000
    long_term_memory_path: str = "data/long_term_memory/policies.json"
    long_term_memory_max_score_history: int = 200
    long_term_memory_max_outcome_history: int = 200


class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        long_term_memory_config: Optional[Mapping[str, Any]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.config = config if isinstance(config, MemoryConfig) else MemoryConfig()
        self.logger = logger
        ltm_cfg = dict(long_term_memory_config or {})

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
        self._long_term_memory = LongTermMemory(
            path=str(ltm_cfg.get("path", self.config.long_term_memory_path)),
            max_score_history=int(
                ltm_cfg.get(
                    "max_score_history",
                    self.config.long_term_memory_max_score_history,
                )
            ),
            max_outcome_history=int(
                ltm_cfg.get(
                    "max_outcome_history",
                    self.config.long_term_memory_max_outcome_history,
                )
            ),
        )
        self._self_state_snapshots: List[Dict[str, Any]] = []
        self._policy_trace_events: List[Dict[str, Any]] = []
        self._active_pe_policy_id: str = "bootstrap"
        self._max_self_state_snapshots = 5000
        self._max_policy_trace_events = 5000

    # WorkingMemoryBuffer delegation
    def record_working(self, entry: WorkingMemoryEntry) -> None:
        self._working_memory.record(entry)

    def get_recent(self, n: int, entry_type: Optional[str] = None) -> List[WorkingMemoryEntry]:
        return self._working_memory.get_recent(n=n, entry_type=entry_type)

    def get_active_goals(self) -> List[WorkingMemoryEntry]:
        return self._working_memory.get_active_goals()

    # PredictionErrorHistory delegation
    def set_active_policy_for_pe(self, policy_id: Optional[str]) -> None:
        normalized = self._normalize_policy_id(policy_id)
        self._active_pe_policy_id = normalized if normalized is not None else "bootstrap"

    def record_pe(self, area_id: str, error: Any, policy_id: Optional[str] = None) -> None:
        resolved_policy_id = self._normalize_policy_id(policy_id)
        if resolved_policy_id is None:
            resolved_policy_id = self._active_pe_policy_id
        payload = self._coerce_pe_payload(area_id=area_id, error=error, policy_id=resolved_policy_id)
        self._prediction_error_history.record(area_id=area_id, error=payload)

    def record_prediction_error(self, error: Any, area_id: str = "unknown") -> None:
        self.record_pe(area_id=area_id, error=error)

    def get_area_familiarity(self, area_id: str) -> float:
        return self._prediction_error_history.get_area_familiarity(area_id=area_id)

    def get_area_threat_prior(self, area_id: str) -> float:
        return self._prediction_error_history.get_area_threat_prior(area_id=area_id)

    def query_prediction_errors(
        self,
        policy_id: Optional[str] = None,
        area_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        normalized_policy_id = self._normalize_policy_id(policy_id)
        if normalized_policy_id is not None:
            query["policy_id"] = normalized_policy_id
        if area_id is not None:
            text_area = str(area_id).strip()
            if text_area:
                query["area_id"] = text_area
        if isinstance(limit, int):
            query["limit"] = int(limit)
        return self._prediction_error_history.query(query)

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

    # Runtime snapshots and policy lifecycle
    def snapshot_self_state(self, snapshot: Any) -> None:
        if isinstance(snapshot, Mapping):
            payload = dict(snapshot)
        else:
            payload = {"snapshot": snapshot}
        payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")
        self._self_state_snapshots.append(payload)
        if len(self._self_state_snapshots) > self._max_self_state_snapshots:
            del self._self_state_snapshots[:-self._max_self_state_snapshots]

    def register_policies(self, policies: Any) -> None:
        if isinstance(policies, list):
            self._long_term_memory.upsert_policies(policies)

    def record_policy_selection(
        self,
        policy_id: str,
        score: float,
        components: Any,
        step: Optional[int],
    ) -> None:
        self._long_term_memory.record_policy_selection(
            policy_id=policy_id,
            score=float(score),
            components=components,
            step=step,
        )

    def record_policy_outcome(
        self,
        policy_id: str,
        reward: Any,
        done: Any,
        step: Optional[int],
    ) -> None:
        self._long_term_memory.record_policy_outcome(
            policy_id=policy_id,
            reward=reward,
            done=bool(done),
            step=step,
        )

    def get_policies(self, adapter_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._long_term_memory.get_policies(adapter_folder=adapter_folder)

    def record_policy_trace(self, trace: Any) -> None:
        if isinstance(trace, Mapping):
            payload = dict(trace)
        else:
            payload = {"trace": trace}
        payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")
        self._policy_trace_events.append(payload)
        if len(self._policy_trace_events) > self._max_policy_trace_events:
            del self._policy_trace_events[:-self._max_policy_trace_events]

    def query(self, query: Any) -> Any:
        if not isinstance(query, Mapping):
            return {
                "self_state": list(self._self_state_snapshots),
                "policy_traces": list(self._policy_trace_events),
                "prediction_errors": self._prediction_error_history.query({}),
                "policies": self.get_policies(),
            }

        target = str(query.get("target", "")).strip()
        if target == "self_state":
            phase = query.get("phase")
            step = query.get("step")
            limit = query.get("limit")
            out = list(self._self_state_snapshots)
            if phase is not None:
                out = [
                    item
                    for item in out
                    if isinstance(item, Mapping) and item.get("phase") == phase
                ]
            if step is not None:
                out = [
                    item
                    for item in out
                    if isinstance(item, Mapping) and item.get("step") == step
                ]
            if isinstance(limit, int) and limit >= 0:
                return out[-limit:]
            return out
        if target == "policy_traces":
            policy_id = self._normalize_policy_id(query.get("policy_id"))
            limit = query.get("limit")
            out = list(self._policy_trace_events)
            if policy_id is not None:
                out = [
                    item
                    for item in out
                    if isinstance(item, Mapping)
                    and self._normalize_policy_id(item.get("policy_id")) == policy_id
                ]
            if isinstance(limit, int) and limit >= 0:
                return out[-limit:]
            return out
        if target == "prediction_errors":
            return self._prediction_error_history.query(query)
        if target == "policies":
            return self.get_policies(adapter_folder=query.get("adapter_folder"))
        return {
            "self_state": list(self._self_state_snapshots),
            "policy_traces": list(self._policy_trace_events),
            "prediction_errors": self._prediction_error_history.query({}),
            "policies": self.get_policies(),
        }

    # Lifecycle
    def clear_episode(self) -> None:
        self._working_memory.clear()
        self._self_state_snapshots.clear()
        self._policy_trace_events.clear()
        self._active_pe_policy_id = "bootstrap"

    def clear_all(self) -> None:
        self._working_memory.clear()
        self._prediction_error_history.clear()
        self._self_state_tracking.clear()
        self._policy_traces.clear()
        self._self_state_snapshots.clear()
        self._policy_trace_events.clear()
        self._active_pe_policy_id = "bootstrap"

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

    @property
    def long_term_memory(self) -> LongTermMemory:
        return self._long_term_memory

    @staticmethod
    def _normalize_policy_id(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text

    @staticmethod
    def _coerce_pe_payload(area_id: str, error: Any, policy_id: str) -> Dict[str, Any]:
        payload: Dict[str, Any]
        if isinstance(error, Mapping):
            payload = dict(error)
        else:
            payload = {
                "magnitude": getattr(error, "magnitude", None),
                "source": getattr(error, "source", None),
                "channel": getattr(error, "channel", None),
                "tick": getattr(error, "tick", None),
            }
        payload["area_id"] = str(area_id)
        payload["policy_id"] = str(policy_id)
        return payload
