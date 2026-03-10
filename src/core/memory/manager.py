from __future__ import annotations

from typing import Any, Mapping, Optional

from core.memory.long_term_memory import LongTermMemory
from core.memory.policy_traces import PolicyTraces
from core.memory.prediction_error import PredictionErrorHistory
from core.memory.self_state import SelfStateMemory
from core.memory.working_memory import WorkingMemoryBuffer
from core.observability.logger import RunLogger


class MemoryManager:
    def __init__(
        self,
        working_memory: Optional[WorkingMemoryBuffer] = None,
        self_state: Optional[SelfStateMemory] = None,
        prediction_errors: Optional[PredictionErrorHistory] = None,
        policy_traces: Optional[PolicyTraces] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        long_term_memory_config: Optional[Mapping[str, Any]] = None,
        logger: Optional[RunLogger] = None,
    ) -> None:
        ltm_config = dict(long_term_memory_config or {})
        self.working_memory = working_memory or WorkingMemoryBuffer()
        self.self_state = self_state or SelfStateMemory()
        self.prediction_errors = prediction_errors or PredictionErrorHistory()
        self.policy_traces = policy_traces or PolicyTraces()
        self.long_term_memory = long_term_memory or LongTermMemory(
            path=str(ltm_config.get("path", "data/long_term_memory/policies.json")),
            max_score_history=int(ltm_config.get("max_score_history", 200)),
            max_outcome_history=int(ltm_config.get("max_outcome_history", 200)),
        )
        self.logger = logger

    def snapshot_self_state(self, snapshot: Any) -> None:
        step: Optional[int] = None
        if isinstance(snapshot, Mapping):
            raw_step = snapshot.get("step")
            if isinstance(raw_step, int):
                step = raw_step
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "self_state",
                    "operation": "write",
                    "record": snapshot,
                },
                step=step,
            )
        self.self_state.record(snapshot)

    def record_prediction_error(self, error: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "prediction_error",
                    "operation": "write",
                    "record": error,
                }
            )
        self.prediction_errors.record(error)

    def record_policy_trace(self, trace: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "policy_trace",
                    "operation": "write",
                    "record": trace,
                }
            )
        self.policy_traces.record(trace)

    def register_policies(self, policies: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "long_term_memory",
                    "operation": "upsert_policies",
                    "record": policies,
                }
            )
        if isinstance(policies, list):
            self.long_term_memory.upsert_policies(policies)

    def record_policy_selection(
        self,
        policy_id: str,
        score: float,
        components: Any,
        step: Optional[int],
    ) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "long_term_memory",
                    "operation": "policy_selection",
                    "record": {
                        "policy_id": policy_id,
                        "score": score,
                        "components": components,
                        "step": step,
                    },
                }
            )
        self.long_term_memory.record_policy_selection(
            policy_id=policy_id,
            score=score,
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
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "long_term_memory",
                    "operation": "policy_outcome",
                    "record": {
                        "policy_id": policy_id,
                        "reward": reward,
                        "done": bool(done),
                        "step": step,
                    },
                }
            )
        self.long_term_memory.record_policy_outcome(
            policy_id=policy_id,
            reward=reward,
            done=done,
            step=step,
        )

    def get_policies(self, adapter_folder: Optional[str] = None) -> Any:
        return self.long_term_memory.get_policies(adapter_folder=adapter_folder)

    def query(self, query: Any) -> Any:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "generic",
                    "operation": "query",
                    "query": query,
                }
            )
        if isinstance(query, Mapping):
            target = query.get("target")
            if target == "policy_traces":
                return self.policy_traces.query(query)
            if target == "prediction_errors":
                return self.prediction_errors.query(query)
            if target == "self_state":
                return self.self_state.query(query)
            if target == "policies":
                return self.get_policies(adapter_folder=query.get("adapter_folder"))
        return {
            "self_state": self.self_state.query({}),
            "policy_traces": self.policy_traces.query({}),
            "prediction_errors": self.prediction_errors.query({}),
            "policies": self.get_policies(),
        }
