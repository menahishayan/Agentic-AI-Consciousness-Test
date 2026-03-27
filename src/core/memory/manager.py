"""
MemoryManager — coordinates all memory subsystems.

Provides a unified interface for the rest of the brain to access memory
without directly instantiating individual memory classes. AgentLoop calls
the manager; the manager delegates to the appropriate subsystem.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.memory.long_term_memory import LongTermMemory
from core.memory.policy_traces import PolicyTraces
from core.memory.prediction_error_history import PredictionErrorHistory
from core.memory.self_state_tracking import SelfStateTracking
from core.memory.working_memory_buffer import WorkingMemoryBuffer
from core.models.memory_records import PolicyTraceRecord, PredictionErrorRecord
from core.models.signals import PredictionErrorBatch
from core.models.state import AgentState

log = logging.getLogger(__name__)


class MemoryManager:
    """
    Facade over all memory subsystems. Brain layers access memory exclusively
    through this class.

    Subsystems:
      - WorkingMemoryBuffer: rolling recent states
      - PredictionErrorHistory: FAISS PE history by area
      - SelfStateTracking: FAISS homeostatic state history
      - PolicyTraces: FAISS policy outcome history
      - LongTermMemory: JSON-persisted cross-episode policy history
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        mem_cfg = config.get("memory", {})

        self._working = WorkingMemoryBuffer(
            capacity=int(mem_cfg.get("working_memory_capacity", 100))
        )
        self._pe_history = PredictionErrorHistory(mem_cfg)
        self._self_state = SelfStateTracking(mem_cfg)
        self._policy_traces = PolicyTraces(mem_cfg)
        self._long_term = LongTermMemory(mem_cfg)

    # ------------------------------------------------------------------
    # WorkingMemory
    # ------------------------------------------------------------------

    def record_state(self, state: AgentState, action: Optional[str] = None) -> None:
        self._working.record(state)
        self._self_state.record(state, action_taken=action)

    def get_recent_states(self, n: int = 10) -> List[AgentState]:
        return self._working.get_recent(n)

    # ------------------------------------------------------------------
    # Prediction Error History
    # ------------------------------------------------------------------

    def record_prediction_error(
        self,
        state: AgentState,
        pe_batch: PredictionErrorBatch,
        action: Optional[str] = None,
    ) -> None:
        area_id = state.perception.area_id or "unknown"
        pe_per_channel = {e.channel: e.magnitude for e in pe_batch.errors}
        rec = PredictionErrorRecord(
            area_id=area_id,
            step=state.step,
            feature_vector=self._pe_history._area_to_vector(area_id),
            pe_per_channel=pe_per_channel,
            mean_pe=pe_batch.mean_magnitude,
            action_id=action,
        )
        self._pe_history.record(rec)

    def get_area_familiarity(self, area_id: str) -> float:
        return self._pe_history.get_area_familiarity(area_id)

    def get_area_mean_pe(self, area_id: str) -> float:
        return self._pe_history.get_mean_pe(area_id)

    # ------------------------------------------------------------------
    # SelfStateTracking
    # ------------------------------------------------------------------

    def update_state_outcome(self, next_state: AgentState) -> None:
        self._self_state.update_last_outcome(next_state)

    def get_depletion_rates(self) -> Dict[str, float]:
        return self._self_state.get_depletion_rates()

    # ------------------------------------------------------------------
    # PolicyTraces
    # ------------------------------------------------------------------

    def record_policy_trace(
        self,
        state: AgentState,
        policy_id: str,
        outcome_score: float,
        drive_signals: Optional[Dict[str, float]] = None,
    ) -> None:
        # Build context vector: homeostatic + step normalized
        hv = state.homeostatic_vector()
        step_norm = min(1.0, state.step / 1000.0)
        x = state.position.x or 0.0
        z = state.position.z or 0.0
        context_vec = hv + [step_norm, x / 50.0, z / 50.0, 0.0]  # pad to 10-dim

        rec = PolicyTraceRecord(
            step=state.step,
            policy_id=policy_id,
            context_vector=context_vec[:10],
            outcome_score=outcome_score,
            drive_signals=drive_signals or {},
        )
        self._policy_traces.record(rec)

    def get_policy_outcome_history(self, policy_id: str) -> float:
        return self._policy_traces.get_policy_outcome_history(policy_id)

    # ------------------------------------------------------------------
    # LongTermMemory
    # ------------------------------------------------------------------

    def record_episode_outcome(
        self,
        policy_id: str,
        score: float,
        outcome: str = "partial",
    ) -> None:
        self._long_term.record_outcome(policy_id, score, outcome)

    def get_ltm_success_rate(self, policy_id: str) -> float:
        return self._long_term.get_success_rate(policy_id)

    def log_summary(self) -> None:
        records = self._long_term.get_all_records()
        for pid, rec in records.items():
            log.info(
                "LTM[%s]: selections=%d, success_rate=%.2f",
                pid, rec.total_selections, rec.success_rate,
            )
