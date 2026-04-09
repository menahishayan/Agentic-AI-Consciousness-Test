from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DriveChannel:
    """
    Defines a single homeostatic drive channel.
    Owned by the adapter (via get_drive_channels()) — NOT hardcoded in brain layers.
    """
    channel_id: str
    setpoint: float                     # Desired value [0,1]
    critical_threshold: float           # Value below which urgency spikes
    recovery_cost_ticks: int            # Estimated ticks to recover from critical
    suggested_action_tags: List[str]    # Policy tags that address this channel
    weight: float = 1.0                 # Relative importance vs other channels


@dataclass
class DriveSignal:
    """Single drive channel urgency signal published to GlobalWorkspace."""
    channel_id: str
    current_value: float
    setpoint: float
    urgency: float              # [0,1] computed by AllostaticController
    ticks_to_critical: Optional[int] = None
    suggested_action_tags: List[str] = field(default_factory=list)


@dataclass
class DriveSignalBatch:
    """Published as kind='drive_signal' to GlobalWorkspace."""
    signals: List[DriveSignal]
    max_urgency: float = 0.0
    dominant_channel: Optional[str] = None

    def __post_init__(self) -> None:
        if self.signals:
            # Dominant channel = highest urgency signal (may be negative when above setpoint)
            dominant = max(self.signals, key=lambda s: s.urgency)
            # max_urgency is clamped to [0,1]: negative urgency (over-satisfied drives)
            # is a surplus signal, not a pressure signal. Leaking negative values poisons
            # urgency_boost in ArousalValenceSystem and idle EFE scoring in FreeEnergyMinimizer.
            self.max_urgency = max(0.0, min(1.0, dominant.urgency))
            self.dominant_channel = dominant.channel_id


@dataclass
class PredictionError:
    """Prediction error for a single sensory channel."""
    channel: str
    expected: float
    observed: float
    magnitude: float            # Precision-weighted error magnitude
    precision: float            # Context-dependent confidence in prediction
    source: str = "unknown"     # "proprioceptive" | "visual" | "threat"


@dataclass
class PredictionErrorBatch:
    """Published as kind='prediction_error' to GlobalWorkspace."""
    errors: List[PredictionError]
    max_magnitude: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        if self.errors:
            self.max_magnitude = max(e.magnitude for e in self.errors)

    @property
    def mean_magnitude(self) -> float:
        """Mean PE across perceptual channels only — motor efference copy excluded.

        Motor PE (wall collisions, stuck detection) is a proprioceptive signal on
        a separate LC-NE pathway and should not contaminate perceptual surprise.
        Consumers that need motor PE should use the motor_pe property.
        """
        perceptual = [e.magnitude for e in self.errors if e.channel != "motor"]
        return sum(perceptual) / len(perceptual) if perceptual else 0.0

    @property
    def motor_pe(self) -> float:
        """Motor efference copy PE — dedicated LC-NE pathway signal."""
        for e in self.errors:
            if e.channel == "motor":
                return e.magnitude
        return 0.0


@dataclass
class ArousalValence:
    """Published as kind='arousal_valence' to GlobalWorkspace."""
    arousal: float          # [0,1] activation/alertness level
    valence: float          # [-1,1] positive=wellbeing, negative=distress
    learning_rate_mod: float = 1.0  # Modulates memory update rates


@dataclass
class ActionProposal:
    """A candidate action from PolicyGenerator."""
    policy_id: str
    action: str
    expected_outcome: str
    score: float
    drive_tags: List[str] = field(default_factory=list)
    cost: float = 0.0
    rationale: Optional[str] = None


@dataclass
class Goal:
    """Task-level goal injected from adapter. Persists across all steps."""
    goal_id: str
    description: str
    priority: float
    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldModelUpdate:
    """Published as kind='world_model' after WorldModelGenerator.update()."""
    action_id: str
    channel_deltas: Dict[str, float]    # channel → observed delta
    predicted_deltas: Dict[str, float]  # channel → predicted delta before step
    step: int = 0
