"""
ArousalValenceSystem — Layer 1, Interoceptive Foundation

Computes arousal (activation level) and valence (hedonic tone) from
homeostatic state and prediction error. These are the emotional signals
that modulate cognition throughout the architecture.

Based on Seth's theory: emotions are not representations of external events
but of the agent's own body state — interoceptive predictions about
physiological condition.

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.signals import ArousalValence, DriveSignalBatch, PredictionErrorBatch


class ArousalValenceSystem:
    """
    Computes and publishes ArousalValence to GlobalWorkspace each step.

    Arousal is high when: drive urgency is high, prediction errors are high,
    or threats are near. It modulates learning rates and attention.

    Valence is positive when: homeostatic needs are met (fed, safe, healthy).
    Negative valence signals distress and triggers more urgent action selection.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        av = config.get("arousal_valence", {})

        # Arousal weights
        self._w_health: float = float(av.get("w_health", 0.35))
        self._w_saturation: float = float(av.get("w_saturation", 0.25))
        self._w_threat: float = float(av.get("w_threat", 0.30))
        self._w_pred_err: float = float(av.get("w_pred_err", 0.10))
        # Motor PE: high when the agent is blocked / unable to translate intent to movement
        self._w_motor: float = float(av.get("w_motor", 0.20))

        # Valence weights
        self._v_health: float = float(av.get("v_health", 0.30))
        self._v_saturation: float = float(av.get("v_saturation", 0.25))
        self._v_resource: float = float(av.get("v_resource", 0.20))
        self._v_safety: float = float(av.get("v_safety", 0.25))
        # Motor PE penalty: stuck + failing drives → more negative valence
        self._v_motor_pe: float = float(av.get("v_motor_pe", 0.15))

    def update(
        self,
        vitals: Dict[str, Optional[float]],
        drive_batch: Optional[DriveSignalBatch],
        pe_batch: Optional[PredictionErrorBatch],
        workspace: GlobalWorkspace,
        step: int,
    ) -> ArousalValence:
        """
        Compute arousal and valence and publish to workspace.

        Args:
            vitals: channel_id → current value from VitalStateMonitor
            drive_batch: Latest drive signals (urgency values)
            pe_batch: Latest prediction error batch
            workspace: GlobalWorkspace for publishing
            step: Current episode step
        """
        # Extract motor PE once — used by both arousal and valence
        motor_pe = 0.0
        if pe_batch:
            for err in pe_batch.errors:
                if err.channel == "motor_efficiency":
                    motor_pe = float(err.magnitude)
                    break

        arousal = self._compute_arousal(vitals, drive_batch, pe_batch, motor_pe)
        valence = self._compute_valence(vitals, motor_pe)
        learning_rate_mod = self._learning_rate_mod(arousal)

        av = ArousalValence(
            arousal=arousal,
            valence=valence,
            learning_rate_mod=learning_rate_mod,
        )

        workspace.publish(AgentMessage(
            sender="ArousalValenceSystem",
            kind="arousal_valence",
            payload=av,
            step=step,
            priority=arousal,
        ))

        return av

    def _compute_arousal(
        self,
        vitals: Dict[str, Optional[float]],
        drive_batch: Optional[DriveSignalBatch],
        pe_batch: Optional[PredictionErrorBatch],
        motor_pe: float = 0.0,
    ) -> float:
        """
        Arousal = weighted combination of:
          - Health deficit (low health → high arousal)
          - Saturation deficit (hungry → high arousal)
          - Threat proximity (danger → high arousal)
          - Prediction error (surprise → high arousal)
          - Motor PE (blocked movement → high arousal, organic stuck detection)
        """
        health = vitals.get("health") or 0.5
        saturation = vitals.get("saturation") or 0.5
        threat = vitals.get("threat_proximity") or 0.0

        # Invert — low health/saturation drives high arousal
        health_comp = (1.0 - health) * self._w_health
        saturation_comp = (1.0 - saturation) * self._w_saturation
        threat_comp = threat * self._w_threat

        # Prediction error contribution (mean across all channels)
        pe_comp = 0.0
        if pe_batch and pe_batch.mean_magnitude > 0.0:
            pe_comp = min(pe_batch.mean_magnitude, 1.0) * self._w_pred_err

        # Motor PE: persistent wall-blocking drives arousal up organically
        motor_comp = min(motor_pe, 1.0) * self._w_motor

        # Also factor in max drive urgency
        max_urgency = drive_batch.max_urgency if drive_batch else 0.0
        urgency_boost = max_urgency * 0.15

        raw = health_comp + saturation_comp + threat_comp + pe_comp + motor_comp + urgency_boost
        return float(min(1.0, raw))

    def _compute_valence(
        self,
        vitals: Dict[str, Optional[float]],
        motor_pe: float = 0.0,
    ) -> float:
        """
        Valence in [-1, 1]:
          +1 = fully satisfied (healthy, fed, safe, resourced)
          -1 = severe distress (dying, starving, threatened, or blocked)

        Negative valence + high arousal is the affect state that signals
        "this is not working, something must change" (Seth/Friston: high
        expected free energy under the current policy).
        """
        health = vitals.get("health") or 0.0
        saturation = vitals.get("saturation") or 0.0
        resource = vitals.get("resource_level") or 0.0
        threat = vitals.get("threat_proximity") or 0.0

        safety = 1.0 - threat

        positive = (
            health * self._v_health
            + saturation * self._v_saturation
            + resource * self._v_resource
            + safety * self._v_safety
        )

        # Motor PE penalty: stuck + unable to act toward goals → negative affect
        # Combined with high arousal this marks "current strategy is failing"
        positive -= min(motor_pe, 1.0) * self._v_motor_pe

        # Map to [-1, 1]
        valence = (positive - 0.5) * 2.0
        return float(max(-1.0, min(1.0, valence)))

    def _learning_rate_mod(self, arousal: float) -> float:
        """
        Higher arousal → faster learning (more weight to surprising events).
        Capped to prevent instability.
        Returns multiplier in [0.5, 2.0].
        """
        return float(min(2.0, max(0.5, 0.5 + arousal * 1.5)))
