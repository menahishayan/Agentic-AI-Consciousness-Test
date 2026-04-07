"""
ArousalValenceSystem — Layer 1, Interoceptive Foundation

Computes arousal (activation level) and valence (hedonic tone) from
prediction error and homeostatic state change. These are the affective
signals that modulate cognition throughout the architecture.

Based on Seth's theory: emotions are not representations of external events
but of the agent's own body state — interoceptive predictions about
physiological condition.

Arousal and valence are orthogonal dimensions:

  Arousal = surprise/engagement intensity. Driven by PE magnitude, threat,
    threat rate-of-change, positive-surprise (food found), and the LC-NE
    motor pathway. High arousal = high uncertainty = deeper inference.

  Valence = hedonic quality as reward prediction error (RPE).
    Positive when outcomes are better than expected (food found,
    no threat). Negative when worse than expected (threat, starvation
    below passive depletion). Centered at 0 (not at homeostatic baseline).

    The RPE formulation means valence is event-driven:
      - Food collection  → sharp positive spike
      - Passive depletion → near-zero (expected, not surprising)
      - Threat encounter  → negative hit
      - Food in view      → moderate positive (incentive salience)

This separation gives genuinely independent arousal/valence dynamics:
  - High arousal + positive valence: appetitive excitement (food found)
  - High arousal + negative valence: fear/distress (threat or blocked)
  - Low arousal + positive valence: calm satiation
  - Low arousal + negative valence: lethargy (slowly depleting, nothing happening)

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

    Arousal is high when: PE is high, threats are near or rising, food is
    suddenly found, or movement is blocked (LC-NE pathway).

    Valence is event-driven RPE: positive on better-than-expected outcomes
    (food found, food visible), negative on threat or homeostatic loss
    exceeding the expected passive depletion rate.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        av = config.get("arousal_valence", {})
        homeo = config.get("adapter_config", {}).get("homeostatic", {})

        # ── Arousal weights ──────────────────────────────────────────────
        # Health/saturation deliberately excluded — drive state enters only
        # via urgency_boost to keep arousal and valence orthogonal.
        self._w_threat: float = float(av.get("w_threat", 0.30))
        self._w_pred_err: float = float(av.get("w_pred_err", 0.35))
        # Threat rate-of-change: rapid onset → fast arousal spike.
        # Corr (2004): reinforcement sensitivity theory — rapid danger onset.
        self._w_threat_roc: float = float(av.get("w_threat_roc", 0.20))
        # Positive surprise: food found (resource_level jump) = arousing event.
        # Mirrors the negative-surprise PE path but for appetitive outcomes.
        self._w_food_pe: float = float(av.get("w_food_pe", 0.40))
        # LC-NE pathway gain: sustained motor PE boosts arousal.
        # Aston-Jones & Cohen (2005): adaptive gain theory of LC-NE.
        self._lc_ne_gain: float = float(av.get("lc_ne_gain", 0.4))

        # ── Valence (RPE) parameters ─────────────────────────────────────
        # Expected passive loss per step — used to centre the RPE around 0.
        # Taken from adapter config so the AV system never hard-codes game rates.
        self._health_depletion_rate: float = float(homeo.get("health_depletion_rate", 0.002))
        self._sat_depletion_rate: float = float(homeo.get("saturation_depletion_rate", 0.002))
        # Scale factor converts small per-step deltas to [-1,1] range.
        # At default 3.0: food collection (health+0.3) → health_rpe ≈ 0.9.
        self._rpe_scale: float = float(av.get("rpe_scale", 3.0))
        # Anticipatory valence: food visible → moderate positive (incentive salience).
        # Berridge & Kringelbach (2015): wanting (incentive salience) is hedonic.
        self._anticipation_weight: float = float(av.get("anticipation_weight", 0.4))
        # Threat hit: penalty fires only above threshold to ignore residual noise.
        self._threat_hit_threshold: float = float(av.get("threat_hit_threshold", 0.3))
        self._threat_hit_weight: float = float(av.get("threat_hit_weight", 0.5))

        # ── Persisted state ──────────────────────────────────────────────
        self._prev_threat: float = 0.0
        self._prev_resource: float = 0.0
        # None = uninitialized; first step yields zero RPE (no prior to compare).
        self._prev_health: Optional[float] = None
        self._prev_saturation: Optional[float] = None

    def update(
        self,
        vitals: Dict[str, Optional[float]],
        drive_batch: Optional[DriveSignalBatch],
        pe_batch: Optional[PredictionErrorBatch],
        workspace: GlobalWorkspace,
        step: int,
        raycast_hits: Optional[List] = None,
    ) -> ArousalValence:
        """
        Compute arousal and valence and publish to workspace.

        Args:
            vitals:       channel_id → current value from VitalStateMonitor
            drive_batch:  Latest drive signals (urgency values)
            pe_batch:     Latest prediction error batch
            workspace:    GlobalWorkspace for publishing
            step:         Current episode step
            raycast_hits: Forward raycast hits (for food proximity anticipation)
        """
        # Extract motor PE from the efference copy channel (LC-NE input).
        motor_pe = 0.0
        if pe_batch:
            for err in pe_batch.errors:
                if err.channel == "motor":
                    motor_pe = float(err.magnitude)
                    break

        current_health = float(vitals.get("health") or 0.0)
        current_sat = float(vitals.get("saturation") or 0.0)
        current_resource = float(vitals.get("resource_level") or 0.0)

        arousal = self._compute_arousal(
            vitals, drive_batch, pe_batch, motor_pe,
            current_resource=current_resource,
        )
        valence = self._compute_valence(
            vitals,
            current_health=current_health,
            current_saturation=current_sat,
            raycast_hits=raycast_hits,
        )

        # Persist current state for next step's delta computation.
        self._prev_health = current_health
        self._prev_saturation = current_sat
        self._prev_resource = current_resource

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
        current_resource: float = 0.0,
    ) -> float:
        """
        Arousal = engagement/surprise intensity, independent of hedonic sign.

        Components:
          PE mean        — perceptual/proprioceptive surprise
          Threat         — danger proximity
          Threat RoC     — rapid threat onset spike (Corr 2004)
          Food PE        — positive surprise from resource_level jump (food found)
          LC-NE          — motor PE: stuck detection (Aston-Jones & Cohen 2005)
          Urgency boost  — indirect homeostatic pressure (clamped ≥ 0)
        """
        threat = float(vitals.get("threat_proximity") or 0.0)

        threat_comp = threat * self._w_threat

        threat_roc = max(0.0, threat - self._prev_threat)
        threat_roc_comp = threat_roc * self._w_threat_roc
        self._prev_threat = threat

        pe_comp = 0.0
        if pe_batch and pe_batch.mean_magnitude > 0.0:
            pe_comp = min(pe_batch.mean_magnitude, 1.0) * self._w_pred_err

        # Positive surprise: resource_level jumped this step → food found.
        # Seeing / reaching food is an arousing event regardless of valence sign.
        resource_jump = max(0.0, current_resource - self._prev_resource)
        food_pe = resource_jump * self._w_food_pe

        # Indirect homeostatic pressure — clamped so surplus drives don't
        # produce negative arousal.
        max_urgency = drive_batch.max_urgency if drive_batch else 0.0
        urgency_boost = max(0.0, max_urgency) * 0.15

        base_arousal = threat_comp + threat_roc_comp + pe_comp + food_pe + urgency_boost

        lc_ne_contribution = min(motor_pe, 1.0) * self._lc_ne_gain

        return float(min(1.0, max(0.0, base_arousal + lc_ne_contribution)))

    def _compute_valence(
        self,
        vitals: Dict[str, Optional[float]],
        current_health: float = 0.0,
        current_saturation: float = 0.0,
        raycast_hits: Optional[List] = None,
    ) -> float:
        """
        Valence in [-1, 1] as reward prediction error (RPE).

        Positive when outcomes exceed expected passive depletion:
          food collection → health/sat delta >> expected_loss → spike to +1
          food visible    → anticipatory incentive salience
        Negative when outcomes worse than expected:
          threat encounter → direct penalty
          homeostatic loss > expected → mild negative
        Near zero when passive depletion is proceeding normally (expected outcome).

        Removes:
          - Absolute health/saturation weights (correlated with valence → bad)
          - safety = 1 - threat (anti-correlated with arousal → bad)
          - motor_pe penalty (already in arousal via LC-NE)
        """
        # Delta vs prior step; zero on first step (no prior to compare).
        prev_h = self._prev_health if self._prev_health is not None else current_health
        prev_s = self._prev_saturation if self._prev_saturation is not None else current_saturation

        delta_health = current_health - prev_h
        delta_sat = current_saturation - prev_s

        # Expected passive loss this step (negative values)
        expected_health_delta = -self._health_depletion_rate
        expected_sat_delta = -self._sat_depletion_rate

        # Signed RPE: actual - expected. Scaled so food collection → ~1.0.
        # food collection: delta_health≈+0.3, expected=-0.002 → rpe=(0.302)*3=0.91
        # passive tick:    delta_health≈-0.002, expected=-0.002 → rpe≈0.0
        health_rpe = (delta_health - expected_health_delta) * self._rpe_scale
        sat_rpe = (delta_sat - expected_sat_delta) * self._rpe_scale

        # Anticipatory valence: food visible ahead → moderate positive.
        food_proximity = 0.0
        if raycast_hits:
            r = raycast_hits[0]
            if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti"):
                food_proximity = 1.0 - float(r.get("distance", 1.0))
        anticipation = food_proximity * self._anticipation_weight

        # Threat hit: direct negative penalty (fires only above threshold).
        threat = float(vitals.get("threat_proximity") or 0.0)
        threat_hit = -threat * self._threat_hit_weight if threat > self._threat_hit_threshold else 0.0

        valence_raw = health_rpe * 0.35 + sat_rpe * 0.30 + anticipation + threat_hit
        return float(max(-1.0, min(1.0, valence_raw)))

    def _learning_rate_mod(self, arousal: float) -> float:
        """
        Higher arousal → faster learning (more weight to surprising events).
        Returns multiplier in [0.5, 2.0].
        """
        return float(min(2.0, max(0.5, 0.5 + arousal * 1.5)))
