"""
WorldModelGenerator — Layer 2, Predictive Processing

Action-conditional EMA transition model. Predicts how homeostatic and
perceptual channels will change given the last action taken.

Based on Seth's controlled hallucination / generative model:
  - The brain predicts; sensory data corrects the prediction
  - Prediction error = divergence between expectation and observation
  - Higher-precision contexts amplify small errors; novel contexts dampen large ones

Architecture:
  - Maintains delta_table[(action_id, channel)] = EMA of observed deltas
  - predict(state, action_id) → predicted next channel values
  - update(prev_state, action_id, next_state) → update the EMA model
  - observation_count[(action_id, channel)] → used to compute precision

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.state import AgentState


class WorldModelGenerator:
    """
    Online, action-conditional transition model over homeostatic channels.

    For each (action_id, channel) pair:
      - Tracks EMA of observed state deltas
      - Tracks observation count for precision estimation

    Phase 1 (low observations): Model has no expectations → low precision
    Phase 2 (sufficient observations): Model has learned expectations → high precision
    """

    _CHANNELS = [
        "health", "saturation", "energy", "oxygen",
        "resource_level", "threat_proximity",
        "terrain_novelty", "entity_density",
        "motor_efficiency",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        wm = config.get("world_model", {})
        self._alpha: float = float(wm.get("alpha", 0.1))                               # EMA learning rate
        self._confidence_threshold: int = int(wm.get("action_confidence_threshold", 20))  # obs before trusting model
        self._min_precision: float = float(wm.get("min_precision", 0.3))

        # EMA delta per (action_id, channel)
        self._delta_table: Dict[Tuple[str, str], float] = {}
        # Observation count per (action_id, channel)
        self._obs_count: Dict[Tuple[str, str], int] = {}

    def predict(
        self,
        state: AgentState,
        action_id: Optional[str],
    ) -> Dict[str, float]:
        """
        Predict next-step channel values given current state and action.

        Returns:
            Dict mapping channel_id → predicted float value [0,1]
        """
        current = self._state_to_channels(state)
        predicted: Dict[str, float] = {}

        for ch in self._CHANNELS:
            base = current.get(ch, 0.0)
            if action_id is not None:
                delta = self._delta_table.get((action_id, ch), 0.0)
            else:
                delta = 0.0
            predicted[ch] = float(min(1.0, max(0.0, base + delta)))

        return predicted

    def update(
        self,
        prev_state: AgentState,
        action_id: Optional[str],
        next_state: AgentState,
        workspace: Optional[GlobalWorkspace] = None,
        step: int = 0,
    ) -> Dict[str, float]:
        """
        Update the transition model with observed transition.

        Returns:
            Dict mapping channel_id → observed delta (for logging)
        """
        if action_id is None:
            return {}

        prev = self._state_to_channels(prev_state)
        curr = self._state_to_channels(next_state)

        observed_deltas: Dict[str, float] = {}

        for ch in self._CHANNELS:
            key = (action_id, ch)
            prev_val = prev.get(ch, 0.0)
            curr_val = curr.get(ch, 0.0)
            delta = curr_val - prev_val
            observed_deltas[ch] = delta

            # EMA update
            if key not in self._delta_table:
                self._delta_table[key] = delta
                self._obs_count[key] = 1
            else:
                old = self._delta_table[key]
                self._delta_table[key] = old + self._alpha * (delta - old)
                self._obs_count[key] = self._obs_count.get(key, 0) + 1

        if workspace is not None:
            from core.models.signals import WorldModelUpdate
            workspace.publish(AgentMessage(
                sender="WorldModelGenerator",
                kind="world_model",
                payload=WorldModelUpdate(
                    action_id=action_id,
                    channel_deltas=observed_deltas,
                    predicted_deltas={
                        ch: self._delta_table.get((action_id, ch), 0.0)
                        for ch in self._CHANNELS
                    },
                    step=step,
                ),
                step=step,
            ))

        return observed_deltas

    def get_precision(self, action_id: str, channel: str) -> float:
        """
        Return precision (confidence) for a given (action, channel) pair.
        Scales with observation count: 0 obs → min_precision, >=threshold → 1.0
        """
        count = self._obs_count.get((action_id, channel), 0)
        if count == 0:
            return self._min_precision
        return float(min(1.0, self._min_precision + (1.0 - self._min_precision) * count / self._confidence_threshold))

    def get_expected_delta(self, action_id: str, channel: str) -> float:
        """Return the EMA-learned expected delta for (action, channel)."""
        return self._delta_table.get((action_id, channel), 0.0)

    def _state_to_channels(self, state: AgentState) -> Dict[str, float]:
        h = state.homeostasis
        r = state.resources
        p = state.perception
        motor_efficiency = float(state.raw_metadata.get("motor_efficiency", 1.0))
        return {
            "health":            h.health or 0.0,
            "saturation":        h.saturation or 0.0,
            "energy":            h.energy or 0.0,
            "oxygen":            h.oxygen or 1.0,
            "resource_level":    r.resource_level or 0.5,
            "threat_proximity":  r.threat_proximity or 0.0,
            "terrain_novelty":   p.terrain_novelty or 0.0,
            "entity_density":    p.entity_density or 0.0,
            "motor_efficiency":  motor_efficiency,
        }
