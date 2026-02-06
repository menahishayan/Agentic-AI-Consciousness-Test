from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ArousalValence:
    arousal: Optional[float] = None
    valence: Optional[float] = None


@dataclass
class ThreatSignal:
    source: Optional[str] = None
    severity: Optional[float] = None
    description: Optional[str] = None


@dataclass
class Goal:
    goal_id: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[float] = None


@dataclass
class ActionProposal:
    action_id: Optional[str] = None
    action: Optional[Any] = None
    expected_outcome: Optional[str] = None
    cost: Optional[float] = None


@dataclass
class Prediction:
    description: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class PredictionError:
    expected: Optional[Any] = None
    observed: Optional[Any] = None
    magnitude: Optional[float] = None


@dataclass
class WorkspaceBroadcast:
    messages: List[Any] = field(default_factory=list)
    priority: Optional[float] = None


@dataclass
class UncertaintyEstimate:
    confidence: Optional[float] = None
    variance: Optional[float] = None
