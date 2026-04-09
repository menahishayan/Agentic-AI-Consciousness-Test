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
    suddenly found, or movement is blocked (LC-NE pathway). Tonic stress
    (sustained PE EMA) raises the arousal baseline; fatigue suppresses it.

    Valence is event-driven RPE: positive on better-than-expected outcomes
    (food found, food visible), negative on threat, homeostatic loss
    exceeding expected passive depletion, frustration, anxiety, fatigue,
    or chronic stress. Contentment adds a small positive baseline when
    all drives are near setpoint.
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

        # ── Frustration: motor PE → negative valence ─────────────────────
        # Panksepp (1998): RAGE system. Blocked movement = high motor PE
        # already raises arousal (LC-NE); this completes the state by adding
        # negative valence so high-arousal + negative-valence = frustration.
        self._frustration_weight: float = float(av.get("frustration_weight", 0.30))
        # Raised from 0.30 → 0.50: routine blocked turns produce motor PE ~0.3–0.4,
        # which was triggering frustration (high arousal + negative valence) on nearly
        # every step during stuck sequences, causing anti-correlated AV signals.
        self._frustration_pe_threshold: float = float(av.get("frustration_pe_threshold", 0.50))

        # ── Contentment: quiescent regulatory success ────────────────────
        # Seth (2021): low urgency across all drives = interoceptive target met.
        # Quiescence is positively valenced, not merely zero-valenced.
        self._contentment_floor: float = float(av.get("contentment_floor", 0.20))
        self._contentment_baseline: float = float(av.get("contentment_baseline", 0.15))

        # ── Anxiety: high PE without locatable source ────────────────────
        # Davis & Whalen (2001): amygdala CeA → anxious state without explicit
        # conditioned stimulus. Distinct from fear: no proximal threat detected.
        self._anxiety_pe_threshold: float = float(av.get("anxiety_pe_threshold", 0.40))
        self._anxiety_threat_ceiling: float = float(av.get("anxiety_threat_ceiling", 0.15))
        self._anxiety_weight: float = float(av.get("anxiety_weight", 0.20))

        # ── Fatigue: energy depletion × motor struggle ───────────────────
        # Boksem & Tops (2008): mental fatigue reduces arousal and impairs
        # valence. Unlike frustration (arousal spike), fatigue suppresses
        # arousal and adds negative valence — a dual drag on both axes.
        self._fatigue_energy_threshold: float = float(av.get("fatigue_energy_threshold", 0.30))
        self._fatigue_motor_threshold: float = float(av.get("fatigue_motor_threshold", 0.20))
        self._fatigue_valence_weight: float = float(av.get("fatigue_valence_weight", 0.25))
        self._fatigue_arousal_suppression: float = float(av.get("fatigue_arousal_suppression", 0.20))

        # ── Stress (tonic): EMA of PE magnitude ─────────────────────────
        # McEwen (1998): allostatic load accumulates from sustained high PE
        # without resolution. Slow alpha → ~20-step time constant so this
        # is a tonic baseline, not a phasic spike.
        self._stress_ema_alpha: float = float(av.get("stress_ema_alpha", 0.05))
        self._stress_arousal_weight: float = float(av.get("stress_arousal_weight", 0.15))
        self._stress_valence_weight: float = float(av.get("stress_valence_weight", 0.20))
        self._stress_ema: float = 0.0

        # ── Arousal temporal smoothing ───────────────────────────────────
        # Arousal EMA decay. 0.50 = ~1-step half-life, so routine per-step PE
        # dissipates quickly and doesn't accumulate into a tonic floor that
        # chronically inflates the epistemic weight in FEM.
        # 0.85 was too slow: arousal sat at 0.25–0.29 all run, biasing EFE
        # toward turns (high epistemic) over move_forward (high pragmatic).
        self._arousal_decay: float = float(av.get("arousal_decay", 0.50))
        self._prev_arousal: float = 0.0

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
        pe_mean = 0.0
        if pe_batch:
            for err in pe_batch.errors:
                if err.channel == "motor":
                    motor_pe = float(err.magnitude)
                    break
            pe_mean = pe_batch.mean_magnitude

        current_health = float(vitals.get("health") or 0.0)
        current_sat = float(vitals.get("saturation") or 0.0)
        current_resource = float(vitals.get("resource_level") or 0.0)

        # Tonic stress EMA: update before computing AV so it's current this step.
        self._stress_ema = (
            self._stress_ema_alpha * pe_mean
            + (1.0 - self._stress_ema_alpha) * self._stress_ema
        )

        arousal = self._compute_arousal(
            vitals, drive_batch, pe_batch, motor_pe,
            current_resource=current_resource,
            stress_ema=self._stress_ema,
        )
        valence = self._compute_valence(
            vitals,
            current_health=current_health,
            current_saturation=current_sat,
            raycast_hits=raycast_hits,
            motor_pe=motor_pe,
            drive_batch=drive_batch,
            stress_ema=self._stress_ema,
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
        stress_ema: float = 0.0,
    ) -> float:
        """
        Arousal = engagement/surprise intensity, independent of hedonic sign.

        Components:
          PE mean          — perceptual/proprioceptive surprise
          Threat           — danger proximity
          Threat RoC       — rapid threat onset spike (Corr 2004)
          Food PE          — positive surprise from resource_level jump (food found)
          LC-NE            — motor PE: stuck detection (Aston-Jones & Cohen 2005)
          Urgency boost    — indirect homeostatic pressure (clamped ≥ 0)
          Stress (tonic)   — EMA of PE raises baseline (McEwen 1998)
          Fatigue suppress — energy depletion dampens arousal (Boksem & Tops 2008)
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
        resource_jump = max(0.0, current_resource - self._prev_resource)
        food_pe = resource_jump * self._w_food_pe

        # Indirect homeostatic pressure — clamped so surplus drives don't
        # produce negative arousal.
        max_urgency = drive_batch.max_urgency if drive_batch else 0.0
        urgency_boost = max(0.0, max_urgency) * 0.15

        base_arousal = threat_comp + threat_roc_comp + pe_comp + food_pe + urgency_boost

        lc_ne_contribution = min(motor_pe, 1.0) * self._lc_ne_gain

        # Tonic stress raises the arousal baseline (sustained allostatic load).
        stress_contribution = stress_ema * self._stress_arousal_weight

        raw = base_arousal + lc_ne_contribution + stress_contribution

        # Fatigue suppression: low energy + sustained motor struggle → dampened
        # arousal. Unlike frustration which spikes arousal, fatigue saps it.
        energy_urgency = self._energy_urgency(vitals, drive_batch)
        if (energy_urgency > self._fatigue_energy_threshold
                and motor_pe > self._fatigue_motor_threshold):
            raw -= self._fatigue_arousal_suppression * min(energy_urgency, 1.0)

        # Temporal smoothing: sympathetic arousal rises fast, decays slowly.
        # EMA with high decay keeps the signal as a sustained envelope rather
        # than a step-wise mirror of instantaneous PE.
        smoothed = self._prev_arousal * self._arousal_decay + raw * (1.0 - self._arousal_decay)
        self._prev_arousal = smoothed
        return float(min(1.0, max(0.0, smoothed)))

    def _compute_valence(
        self,
        vitals: Dict[str, Optional[float]],
        current_health: float = 0.0,
        current_saturation: float = 0.0,
        raycast_hits: Optional[List] = None,
        motor_pe: float = 0.0,
        drive_batch: Optional[DriveSignalBatch] = None,
        stress_ema: float = 0.0,
    ) -> float:
        """
        Valence in [-1, 1] as reward prediction error (RPE).

        Positive contributors:
          health/sat RPE   — better-than-expected homeostatic outcome
          anticipation     — food visible, scaled by saturation urgency (craving)
          contentment      — all drives near setpoint (regulatory quiescence)

        Negative contributors:
          threat hit       — danger proximity penalty above threshold
          frustration      — motor PE above threshold (blocked movement, RAGE)
          anxiety          — high mean PE + no threat + no food (diffuse dread)
          fatigue          — energy depletion + motor struggle
          stress (tonic)   — EMA of PE drains valence baseline
        """
        # Delta vs prior step; zero on first step (no prior to compare).
        prev_h = self._prev_health if self._prev_health is not None else current_health
        prev_s = self._prev_saturation if self._prev_saturation is not None else current_saturation

        delta_health = current_health - prev_h
        delta_sat = current_saturation - prev_s

        # Expected passive loss this step (negative values).
        expected_health_delta = -self._health_depletion_rate
        expected_sat_delta = -self._sat_depletion_rate

        # Signed RPE: actual - expected. Scaled so food collection → ~1.0.
        # food collection: delta_health≈+0.3, expected=-0.002 → rpe=(0.302)*3=0.91
        # passive tick:    delta_health≈-0.002, expected=-0.002 → rpe≈0.0
        health_rpe = (delta_health - expected_health_delta) * self._rpe_scale
        sat_rpe = (delta_sat - expected_sat_delta) * self._rpe_scale

        # Per-channel urgency for craving and fatigue.
        sat_urgency = 0.0
        energy_urgency = self._energy_urgency(vitals, drive_batch)
        max_urgency = 0.0
        if drive_batch:
            for s in drive_batch.signals:
                if s.channel_id == "saturation":
                    sat_urgency = max(0.0, s.urgency)
            max_urgency = drive_batch.max_urgency

        # Urgency-weighted anticipatory craving: food visible + hunger → wanting.
        # Berridge & Robinson (1998): incentive salience scales with deprivation.
        food_detected = False
        food_proximity = 0.0
        if raycast_hits:
            food_ray = next(
                (r for r in raycast_hits if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")),
                None,
            )
            if food_ray:
                food_detected = True
                food_proximity = 1.0 - float(food_ray.get("distance", 1.0))
        anticipation = food_proximity * self._anticipation_weight * sat_urgency

        # Threat hit: direct negative penalty (fires only above threshold).
        threat = float(vitals.get("threat_proximity") or 0.0)
        threat_hit = -threat * self._threat_hit_weight if threat > self._threat_hit_threshold else 0.0

        # Frustration: motor PE above threshold completes the negative-valence half.
        # Arousal (LC-NE) + negative valence = high-arousal distress, not curiosity.
        frustration = 0.0
        if motor_pe > self._frustration_pe_threshold:
            frustration = -min(motor_pe, 1.0) * self._frustration_weight

        # Contentment: all drives near setpoint → regulatory success = positive.
        # Seth (2021): quiescence is a target state, not mere absence of distress.
        contentment = 0.0
        if drive_batch and max_urgency < self._contentment_floor:
            contentment = self._contentment_baseline

        # Anxiety: sustained unexplained PE → diffuse dread.
        # Uses _stress_ema (tonic, ~20-step window) not instantaneous pe_mean,
        # so anxiety accumulates over time rather than firing/un-firing every
        # step in sync with PE oscillations. Fires only when all three hold:
        #   - tonic stress is elevated (unexplained PE persisting over time)
        #   - no proximal threat (otherwise this is fear, not anxiety)
        #   - no food visible (otherwise a positive-valence explanation exists)
        anxiety = 0.0
        if (stress_ema > self._anxiety_pe_threshold
                and threat < self._anxiety_threat_ceiling
                and not food_detected):
            excess = min((stress_ema - self._anxiety_pe_threshold)
                         / max(1.0 - self._anxiety_pe_threshold, 1e-6), 1.0)
            anxiety = -self._anxiety_weight * excess

        # Fatigue: energy depletion + motor struggle → negative valence penalty.
        # Complements arousal suppression; together they produce a state of
        # flagging motivation that neither fear nor frustration captures.
        fatigue_valence = 0.0
        if (energy_urgency > self._fatigue_energy_threshold
                and motor_pe > self._fatigue_motor_threshold):
            fatigue_valence = -self._fatigue_valence_weight * min(energy_urgency, 1.0)

        # Tonic stress: sustained high PE chronically drains valence baseline.
        stress_penalty = -stress_ema * self._stress_valence_weight

        valence_raw = (
            health_rpe * 0.35
            + sat_rpe * 0.30
            + anticipation
            + threat_hit
            + frustration
            + contentment
            + anxiety
            + fatigue_valence
            + stress_penalty
        )
        return float(max(-1.0, min(1.0, valence_raw)))

    def _energy_urgency(
        self,
        vitals: Dict[str, Optional[float]],
        drive_batch: Optional[DriveSignalBatch],
    ) -> float:
        """
        Return energy urgency in [0, 1], preferring the drive_batch signal.
        Falls back to 1 - vitals["energy"] when no explicit energy drive channel
        exists (energy_urgency = 0 when drive_batch has no 'energy' channel and
        vitals has no 'energy' key, which is the safe default).
        """
        if drive_batch:
            for s in drive_batch.signals:
                if s.channel_id == "energy":
                    return max(0.0, min(1.0, s.urgency))
        raw_energy = vitals.get("energy")
        if raw_energy is not None:
            return max(0.0, 1.0 - float(raw_energy))
        return 0.0

    def _learning_rate_mod(self, arousal: float) -> float:
        """
        Higher arousal → faster learning (more weight to surprising events).
        Returns multiplier in [0.5, 2.0].
        """
        return float(min(2.0, max(0.5, 0.5 + arousal * 1.5)))
