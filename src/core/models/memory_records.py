from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PredictionErrorRecord:
    """Stored in PredictionErrorHistory FAISS index."""
    area_id: str
    step: int
    feature_vector: List[float]     # Area embedding for FAISS
    pe_per_channel: Dict[str, float]
    mean_pe: float
    action_id: Optional[str] = None


@dataclass
class SelfStateRecord:
    """Stored in SelfStateTracking FAISS index."""
    step: int
    homeostatic_vector: List[float]  # [health, saturation, energy, oxygen, threat, resource]
    area_id: str
    action_taken: Optional[str] = None
    outcome_health_delta: Optional[float] = None
    outcome_saturation_delta: Optional[float] = None


@dataclass
class PolicyTraceRecord:
    """Stored in PolicyTraces FAISS index."""
    step: int
    policy_id: str
    context_vector: List[float]     # Flattened context for FAISS
    outcome_score: float            # [0,1] how well the policy performed
    drive_signals: Dict[str, float] # channel_id → urgency at time of selection
    goal_coherence: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class LongTermPolicyRecord:
    """Persisted in JSON — survives restarts."""
    policy_id: str
    score_history: List[float] = field(default_factory=list)
    outcome_history: List[str] = field(default_factory=list)   # "success"|"failure"|"partial"
    total_selections: int = 0
    total_successes: int = 0

    @property
    def success_rate(self) -> float:
        if not self.score_history:
            return 0.5
        return sum(self.score_history[-50:]) / len(self.score_history[-50:])
