from __future__ import annotations

from typing import Any, Optional

from core.memory.episodic import EpisodicMemory
from core.memory.policy_traces import PolicyTraces
from core.memory.prediction_error import PredictionErrorHistory
from core.memory.procedural import ProceduralMemory
from core.memory.semantic import SemanticMemory
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
        episodic: Optional[EpisodicMemory] = None,
        semantic: Optional[SemanticMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        logger: Optional[RunLogger] = None,
    ) -> None:
        self.working_memory = working_memory or WorkingMemoryBuffer()
        self.self_state = self_state or SelfStateMemory()
        self.prediction_errors = prediction_errors or PredictionErrorHistory()
        self.policy_traces = policy_traces or PolicyTraces()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural or ProceduralMemory()
        self.logger = logger

    def snapshot_self_state(self, snapshot: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "self_state",
                    "operation": "write",
                    "record": snapshot,
                }
            )
        raise NotImplementedError("Self-state snapshotting not implemented.")

    def record_prediction_error(self, error: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "prediction_error",
                    "operation": "write",
                    "record": error,
                }
            )
        raise NotImplementedError("Prediction error recording not implemented.")

    def record_policy_trace(self, trace: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "policy_trace",
                    "operation": "write",
                    "record": trace,
                }
            )
        raise NotImplementedError("Policy trace recording not implemented.")

    def store_episode(self, episode: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "episodic",
                    "operation": "write",
                    "record": episode,
                }
            )
        raise NotImplementedError("Episodic memory storage not implemented.")

    def store_semantic(self, entry: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "semantic",
                    "operation": "write",
                    "record": entry,
                }
            )
        raise NotImplementedError("Semantic memory storage not implemented.")

    def store_procedural(self, skill: Any) -> None:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "procedural",
                    "operation": "write",
                    "record": skill,
                }
            )
        raise NotImplementedError("Procedural memory storage not implemented.")

    def query(self, query: Any) -> Any:
        if self.logger is not None:
            self.logger.memory_event(
                {
                    "type": "generic",
                    "operation": "query",
                    "query": query,
                }
            )
        raise NotImplementedError("Memory queries not implemented.")
