"""
PredictionErrorCalculator — Layer 2, Predictive Processing

Computes precision-weighted prediction error per channel. The error
signal is the difference between WorldModelGenerator predictions and
actual observed values, weighted by the agent's confidence in its model.

Design rationale (from PEC_Design_Rationale.docx):
  - Precision weighting: high-confidence predictions → amplified errors
  - Per-channel decomposition with source tagging
  - EMA baseline as implicit Bayesian prior
  - Sigma-clipping to suppress outliers

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.layers.predictive.WorldModelGenerator import WorldModelGenerator
from core.models.signals import PredictionError, PredictionErrorBatch
from core.models.state import AgentState


_CHANNEL_SOURCES = {
    "health":            "proprioceptive",
    "saturation":        "proprioceptive",
    "energy":            "proprioceptive",
    "oxygen":            "proprioceptive",
    "motor_efficiency":  "proprioceptive",
    "motor":             "proprioceptive",
    "resource_level":    "visual",
    "threat_proximity":  "threat",
    "terrain_novelty":   "visual",
    "entity_density":    "visual",
    "food_distance":     "exteroceptive",
}

# Actions that command forward/backward translation
_MOVEMENT_ACTIONS = {"move_forward", "move_backward"}


class PredictionErrorCalculator:
    """
    Computes per-channel prediction errors weighted by model precision.

    The error signal drives:
      - Arousal modulation (sustained high PE → higher arousal)
      - LLM trigger gating (PE streak → call for deliberation)
      - Memory familiarity updates
    """

    def __init__(
        self,
        world_model: WorldModelGenerator,
        config: Dict[str, Any],
    ) -> None:
        self._world_model = world_model
        pe = config.get("perceptual_prediction_error", {})

        self._alpha: float = float(pe.get("alpha", 0.1))              # EMA for baseline
        self._epsilon: float = float(pe.get("epsilon", 0.01))         # stability floor
        self._sigma_clip: float = float(pe.get("sigma_clip", 3.0))    # outlier suppression
        self._default_precision: float = float(pe.get("default_precision", 0.5))
        self._min_precision: float = float(pe.get("min_precision", 0.3))
        # LC-NE gain: arousal multiplies effective precision so high-arousal states
        # produce faster belief updating. Yu & Dayan (2005): NE signals unexpected
        # uncertainty and transiently boosts sensory gain.
        # arousal=0.0 → ×1.0; arousal=1.0 → ×1.5
        self._arousal_precision_gain: float = float(pe.get("arousal_precision_gain", 0.5))

        # EMA baseline per channel (represents "normal" value history)
        self._baseline: Dict[str, float] = {}
        # EMA of squared deviations for variance estimation
        self._variance: Dict[str, float] = {}

    def update(
        self,
        predicted: Dict[str, float],
        observed: AgentState,
        last_action: Optional[str],
        workspace: GlobalWorkspace,
        step: int,
        area_familiarity: float = 0.5,
        arousal: float = 0.0,
    ) -> PredictionErrorBatch:
        """
        Compute prediction error between predicted and actual state channels.

        Args:
            predicted: channel → predicted value from WorldModelGenerator
            observed: Actual next AgentState
            last_action: Action taken to transition (for precision lookup)
            workspace: GlobalWorkspace for publishing
            step: Current episode step
            area_familiarity: [0,1] from memory — modulates precision
        """
        actual = self._state_to_channels(observed)
        errors: List[PredictionError] = []

        for ch, obs_val in actual.items():
            pred_val = predicted.get(ch, self._baseline.get(ch, obs_val))

            # Update EMA baseline
            prev_baseline = self._baseline.get(ch, obs_val)
            self._baseline[ch] = prev_baseline + self._alpha * (obs_val - prev_baseline)

            # Update variance estimate
            deviation = abs(obs_val - self._baseline[ch])
            prev_var = self._variance.get(ch, self._epsilon)
            self._variance[ch] = prev_var + self._alpha * (deviation ** 2 - prev_var)
            sigma = (self._variance[ch] + self._epsilon) ** 0.5

            # Raw error: deviation of observed from predicted
            raw_error = obs_val - pred_val
            normalized_error = abs(raw_error) / (sigma + self._epsilon)

            # Sigma clipping — suppress outliers
            if normalized_error > self._sigma_clip:
                normalized_error = self._sigma_clip

            # Precision = familiarity × model confidence for this action-channel,
            # then scaled by LC-NE arousal pathway: high arousal → sharper updating.
            # Aston-Jones & Cohen (2005): LC-NE release transiently boosts cortical gain.
            model_precision = (
                self._world_model.get_precision(last_action, ch)
                if last_action else self._default_precision
            )
            base_precision = area_familiarity * 0.5 + model_precision * 0.5
            arousal_multiplier = 1.0 + arousal * self._arousal_precision_gain
            precision = max(self._min_precision, base_precision * arousal_multiplier)

            # Precision-weighted magnitude: novel areas dampen errors; familiar amplify
            magnitude = normalized_error * precision

            source = _CHANNEL_SOURCES.get(ch, "unknown")

            errors.append(PredictionError(
                channel=ch,
                expected=pred_val,
                observed=obs_val,
                magnitude=float(magnitude),
                precision=float(precision),
                source=source,
            ))

        # Motor PE — proprioceptive efference copy channel.
        # When a movement action was taken but the agent didn't move, the
        # motor prediction was violated. This feeds the LC-NE arousal pathway.
        # Cite: Friston (2010) predictive coding of motor commands.
        motor_pe_magnitude = 0.0
        if last_action in _MOVEMENT_ACTIONS:
            actual_delta = float(observed.raw_metadata.get("position_delta_norm", 0.0))
            motor_pe_magnitude = abs(1.0 - actual_delta)  # expected full movement

        errors.append(PredictionError(
            channel="motor",
            expected=1.0 if last_action in _MOVEMENT_ACTIONS else 0.0,
            observed=float(observed.raw_metadata.get("position_delta_norm", 0.0)),
            magnitude=float(motor_pe_magnitude),
            precision=float(self._default_precision),
            source="proprioceptive",
        ))

        batch = PredictionErrorBatch(errors=errors, step=step)

        workspace.publish(AgentMessage(
            sender="PredictionErrorCalculator",
            kind="prediction_error",
            payload=batch,
            step=step,
            priority=batch.max_magnitude,
        ))

        return batch

    def _state_to_channels(self, state: AgentState) -> Dict[str, float]:
        h = state.homeostasis
        r = state.resources
        p = state.perception
        motor_efficiency = float(state.raw_metadata.get("motor_efficiency", 1.0))
        rh = state.perception.raycast_hits
        food_distance = 1.0
        if rh:
            food_ray = next(
                (r for r in rh if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None
            )
            if food_ray:
                food_distance = float(food_ray.get("distance", 1.0))

        return {
            "health":            h.health or 0.0,
            "saturation":        h.saturation or 0.0,
            "energy":            h.energy or 0.0,
            "oxygen":            h.oxygen or 1.0,
            "motor_efficiency":  motor_efficiency,
            "resource_level":    r.resource_level or 0.5,
            "threat_proximity":  r.threat_proximity or 0.0,
            "terrain_novelty":   p.terrain_novelty or 0.0,
            "entity_density":    p.entity_density or 0.0,
            "food_distance":     food_distance,
        }
