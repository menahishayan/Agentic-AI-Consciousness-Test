"""
MemoryManager — coordinates all memory subsystems.

Provides a unified interface for the rest of the brain to access memory
without directly instantiating individual memory classes. AgentLoop calls
the manager; the manager delegates to the appropriate subsystem.
"""
from __future__ import annotations

import logging
from pathlib import Path
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

        # Persistence path — same directory as LongTermMemory so all memory
        # artefacts live together under data/long_term_memory/.
        self._persist_path = Path(mem_cfg.get("long_term_memory_path", "data/long_term_memory"))

        # Auto-load FAISS stores on startup so prior episodes are immediately
        # available for retrieval. SelfStateTracking is skipped — its vectors
        # include broken position coordinates and will be persisted once
        # dead-reckoning is fixed.
        self._pe_history.load(self._persist_path)
        self._policy_traces.load(self._persist_path)

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

    def reset_area_familiarity(self, area_id: str) -> None:
        """
        Clear PE history records for a specific area.

        Called at episode start for the agent's initial area so that
        area_familiarity = 0 on step 0 every episode, making area_novelty = 1.0
        and allowing the epistemic value of turns to dominate — producing a
        natural orienting scan without hardcoded behaviour.
        """
        self._pe_history.clear_area(area_id)

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
        notes: Optional[str] = None,
    ) -> None:
        # Context vector: homeostatic state (6) + step_norm (1) + padding (3).
        # Position is intentionally excluded — dead-reckoning accumulates unbounded
        # error so spatial coords would corrupt FAISS similarity distances.
        # Queries are purely drive-state based, which is what the LLM needs.
        hv = state.homeostatic_vector()
        step_norm = min(1.0, state.step / 1000.0)
        context_vec = hv + [step_norm, 0.0, 0.0, 0.0]  # 10-dim, position slots zeroed

        rec = PolicyTraceRecord(
            step=state.step,
            policy_id=policy_id,
            context_vector=context_vec[:10],
            outcome_score=outcome_score,
            drive_signals=drive_signals or {},
            notes=notes,
        )
        self._policy_traces.record(rec)

    def get_policy_outcome_history(self, policy_id: str) -> float:
        return self._policy_traces.get_policy_outcome_history(policy_id)

    def query_similar_traces(self, state: AgentState, k: int = 3) -> list:
        """
        Return up to k PolicyTraceRecords from past situations most similar
        to the current homeostatic + position state.

        Uses the same context vector as record_policy_trace so FAISS distances
        are meaningful. Called by PolicyGenerator before each LLM prompt to
        inject episodic memory as a prior.
        """
        hv = state.homeostatic_vector()
        step_norm = min(1.0, state.step / 1000.0)
        context_vec = hv + [step_norm, 0.0, 0.0, 0.0]  # position excluded (see record_policy_trace)
        records = self._policy_traces.query_similar(context_vec)
        return records[:k]

    def save_faiss_stores(self) -> None:
        """
        Persist FAISS-backed stores to disk.  Called at episode end.

        PolicyTraces and PredictionErrorHistory are persisted.
        SelfStateTracking is intentionally skipped until dead-reckoning
        position coordinates are fixed (spatial vectors are currently corrupt).
        """
        self._pe_history.save(self._persist_path)
        self._policy_traces.save(self._persist_path)

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
