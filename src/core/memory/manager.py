from __future__ import annotations

from typing import Any, Optional

from core.memory.episodic import EpisodicMemory
from core.memory.policy_traces import PolicyTraces
from core.memory.prediction_error import PredictionErrorHistory
from core.memory.procedural import ProceduralMemory
from core.memory.semantic import SemanticMemory
from core.memory.self_state import SelfStateMemory
from core.memory.working_memory import WorkingMemoryBuffer


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
    ) -> None:
        self.working_memory = working_memory or WorkingMemoryBuffer()
        self.self_state = self_state or SelfStateMemory()
        self.prediction_errors = prediction_errors or PredictionErrorHistory()
        self.policy_traces = policy_traces or PolicyTraces()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural or ProceduralMemory()

    def snapshot_self_state(self, snapshot: Any) -> None:
        raise NotImplementedError("Self-state snapshotting not implemented.")

    def record_prediction_error(self, error: Any) -> None:
        raise NotImplementedError("Prediction error recording not implemented.")

    def record_policy_trace(self, trace: Any) -> None:
        raise NotImplementedError("Policy trace recording not implemented.")

    def store_episode(self, episode: Any) -> None:
        raise NotImplementedError("Episodic memory storage not implemented.")

    def store_semantic(self, entry: Any) -> None:
        raise NotImplementedError("Semantic memory storage not implemented.")

    def store_procedural(self, skill: Any) -> None:
        raise NotImplementedError("Procedural memory storage not implemented.")

    def query(self, query: Any) -> Any:
        raise NotImplementedError("Memory queries not implemented.")
