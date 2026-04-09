from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentMessage:
    """
    The sole inter-layer communication primitive.

    All brain layers communicate exclusively through GlobalWorkspace using
    this message type. No layer may hold a direct reference to another layer.

    kind values:
        "vital_state"       - HomeostasisState snapshot from VitalStateMonitor
        "drive_signal"      - DriveSignal batch from AllostaticController
        "prediction_error"  - PredictionErrorBatch from PredictionErrorCalculator
        "goal"              - Task goal from environment adapter (via AgentLoop init)
        "policy_proposal"   - Candidate policies from PolicyGenerator
        "arousal_valence"   - ArousalValence from ArousalValenceSystem
        "metacognitive"     - Urgency broadcast from MetacognitiveMonitor
        "world_model"       - WorldModelUpdate from WorldModelGenerator
    """

    sender: str
    kind: str
    payload: Any
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.0  # higher = more urgent for metacognitive broadcast ordering

    def __repr__(self) -> str:
        return f"AgentMessage(sender={self.sender!r}, kind={self.kind!r}, step={self.step})"
