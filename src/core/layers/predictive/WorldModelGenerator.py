"""
WorldModelGenerator — Layer 2, Predictive Processing

Action-conditional EMA transition model. Predicts how homeostatic and
perceptual channels will change given the last action taken.

Based on Seth's controlled hallucination / generative model:
  - The brain predicts; sensory data corrects the prediction
  - Prediction error = divergence between expectation and observation
  - Higher-precision contexts amplify small errors; novel contexts dampen large ones

Architecture:
  - Maintains delta_table[(action_id, heading_bucket, channel)] = EMA of observed deltas
  - predict(state, action_id) → predicted next channel values
  - update(prev_state, action_id, next_state) → update the EMA model
  - observation_count[(action_id, heading_bucket, channel)] → used to compute precision

Heading bucket: heading_degrees is discretized into 8 compass directions (0–7,
each spanning 45°). This allows the model to learn direction-dependent motor
consequences — e.g. move_forward into a wall scores differently per compass
direction, so motor PE spikes only for the heading where the wall was encountered.

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.state import AgentState


def _heading_bucket(heading_deg: float) -> int:
    """
    Discretize a heading in degrees to one of 8 compass buckets (0–7).
    Each bucket spans 45°: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW.
    """
    return int((heading_deg % 360.0) / 45.0) % 8


class WorldModelGenerator:
    """
    Online, action-conditional transition model over homeostatic channels.

    For each (action_id, heading_bucket, channel) triple:
      - Tracks EMA of observed state deltas
      - Tracks observation count for precision estimation

    The heading bucket conditions the model on facing direction so wall-contact
    motor consequences are learned per compass sector rather than globally.

    Phase 1 (low observations): Model has no expectations → low precision
    Phase 2 (sufficient observations): Model has learned expectations → high precision
    """

    _CHANNELS = [
        "health", "saturation", "energy", "oxygen",
        "resource_level", "threat_proximity",
        "terrain_novelty", "entity_density",
        "motor_efficiency",
        "position_delta_norm",
        "heading_change",
        "food_distance",   # forward raycast distance to food; 1.0 = no food in view
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        wm = config.get("world_model", {})
        self._alpha: float = float(wm.get("alpha", 0.1))                               # EMA learning rate
        self._confidence_threshold: int = int(wm.get("action_confidence_threshold", 20))  # obs before trusting model
        self._min_precision: float = float(wm.get("min_precision", 0.3))

        # EMA delta per (action_id, heading_bucket, channel)
        self._delta_table: Dict[Tuple[str, int, str], float] = {}
        # Observation count per (action_id, heading_bucket, channel)
        self._obs_count: Dict[Tuple[str, int, str], int] = {}

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
        hb = _heading_bucket(state.position.heading or 0.0)
        predicted: Dict[str, float] = {}

        for ch in self._CHANNELS:
            base = current.get(ch, 0.0)
            if action_id is not None:
                delta = self._delta_table.get((action_id, hb, ch), 0.0)
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

        hb = _heading_bucket(prev_state.position.heading or 0.0)
        prev = self._state_to_channels(prev_state)
        curr = self._state_to_channels(next_state)

        observed_deltas: Dict[str, float] = {}

        for ch in self._CHANNELS:
            key = (action_id, hb, ch)
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
                        ch: self._delta_table.get((action_id, hb, ch), 0.0)
                        for ch in self._CHANNELS
                    },
                    step=step,
                ),
                step=step,
            ))

        return observed_deltas

    def get_precision(self, action_id: str, channel: str, heading_deg: float = 0.0) -> float:
        """
        Return precision (confidence) for a given (action, heading_bucket, channel) triple.
        Scales with observation count: 0 obs → min_precision, >=threshold → 1.0
        """
        hb = _heading_bucket(heading_deg)
        count = self._obs_count.get((action_id, hb, channel), 0)
        if count == 0:
            return self._min_precision
        return float(min(1.0, self._min_precision + (1.0 - self._min_precision) * count / self._confidence_threshold))

    def get_expected_delta(self, action_id: str, channel: str, heading_deg: float = 0.0) -> float:
        """Return the EMA-learned expected delta for (action, heading_bucket, channel)."""
        hb = _heading_bucket(heading_deg)
        return self._delta_table.get((action_id, hb, channel), 0.0)

    def _state_to_channels(self, state: AgentState) -> Dict[str, float]:
        h = state.homeostasis
        r = state.resources
        p = state.perception
        motor_efficiency = float(state.raw_metadata.get("motor_efficiency", 1.0))
        position_delta_norm = float(state.raw_metadata.get("position_delta_norm", 0.0))
        # heading_change stores normalised heading (heading/360) so that
        # update()'s delta = curr - prev naturally captures per-step turning:
        #   turn_left  → Δ ≈ +0.125  (45°/360°)
        #   turn_right → Δ ≈ −0.125
        #   forward/idle → Δ ≈ 0
        # Wrap artefacts near 0/360 are rare and smoothed out by EMA.
        heading_change = float((state.position.heading or 0.0) / 360.0)
        rh = state.perception.raycast_hits
        food_distance = 1.0
        if rh:
            food_ray = next(
                (r for r in rh if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None
            )
            if food_ray:
                food_distance = float(food_ray.get("distance", 1.0))

        return {
            "health":               h.health or 0.0,
            "saturation":           h.saturation or 0.0,
            "energy":               h.energy or 0.0,
            "oxygen":               h.oxygen or 1.0,
            "resource_level":       r.resource_level or 0.5,
            "threat_proximity":     r.threat_proximity or 0.0,
            "terrain_novelty":      p.terrain_novelty or 0.0,
            "entity_density":       p.entity_density or 0.0,
            "motor_efficiency":     motor_efficiency,
            "position_delta_norm":  position_delta_norm,
            "heading_change":       heading_change,
            "food_distance":        food_distance,
        }
