"""
FreeEnergyMinimizer — Layer 3, Action Selection

Scores action proposals on how well they minimise expected free energy.
Structure follows Friston's EFE decomposition (Friston et al. 2017):

  G(π) = pragmatic_value  *  w_pragmatic   (allostatic drive urgency relief)
         + epistemic_value  *  w_epistemic   (information gain / uncertainty reduction)
         - motor_cost       *  w_motor_cost  (metabolic cost of action)

Idle:
  score = 1.0 − max(max_urgency, mean_PE)
  → rest scores highest when the agent is satisfied AND unsurprised.
  → Seth's beast machine at allostatic equilibrium.

Turn actions (epistemic):
  epistemic = area_novelty + EPISTEMIC_FORAGING_BASELINE
  (scanning novel directions reduces directional uncertainty)

Movement actions:
  epistemic = mean_PE + EPISTEMIC_FORAGING_BASELINE
  (moving through space resolves spatial prediction errors)

The foraging baseline ensures active actions always out-score idle when drives
are satisfied. An agent that only moves when starving is a thermostat, not a
beast machine. Epistemic foraging under low-urgency is the canonical active
inference behaviour (Friston et al. 2017; Seth & Bayne 2022).

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.models.signals import DriveSignalBatch, PredictionErrorBatch

# Actions that scan unseen directions — pure epistemic value
_EPISTEMIC_ACTIONS = {"turn_left", "turn_right"}

# Intrinsic epistemic foraging baseline added to all active (non-idle) actions.
# Ensures active actions always beat idle when drives are satisfied — epistemic
# foraging under allostatic equilibrium (Friston et al. 2017; Seth & Bayne 2022).
# At default weights (w_epistemic=0.35): 0.15 × 0.35 = 0.0525 score advantage
# over idle, which scores at most 1.0 - 0.0 = 1.0 only when urgency AND pe are 0.
_EPISTEMIC_FORAGING_BASELINE = 0.15

# Per-policy metabolic cost in [0, 1]
_MOTOR_COST: Dict[str, float] = {
    "idle":          0.0,
    "turn_left":     0.1,
    "turn_right":    0.1,
    "move_forward":  0.2,
    "move_backward": 0.25,
}


class FreeEnergyMinimizer:
    """
    Computes expected-free-energy scores for candidate action proposals.

    Score components:
      pragmatic   0.50 — allostatic drive urgency relief
      epistemic   0.35 — information gain (novelty × direction / PE × movement)
      motor_cost  0.15 — metabolic cost (subtracted)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        pg = config.get("policy_generator", {})
        weights = pg.get("weights", {})
        self._w_pragmatic  = float(weights.get("allostatic_urgency", 0.50))
        self._w_epistemic  = float(weights.get("epistemic_gain",     0.35))
        self._w_motor_cost = float(weights.get("motor_cost",         0.15))
        # Idle urgency penalty: reduces idle EFE proportional to drive urgency so
        # that "do nothing" cannot score near 1.0 while drives are actively depleting.
        # At urgency=0.20 with penalty=2.0: idle drops from 0.80 → 0.40.
        self._idle_urgency_penalty: float = float(weights.get("idle_urgency_penalty", 2.0))
        # Absolute food-proximity bonus for move_forward when food is directly ahead.
        # Independent of urgency level — food in view has intrinsic EFE value even
        # when drives are only moderately depleted. Complements the urgency-scaled
        # pragmatic boost already present.
        self._food_proximity_bonus: float = float(weights.get("food_proximity_bonus", 0.6))
        # Consecutive steps with motor_eff < 0.3 — penalty only fires after 2+
        # to avoid suppressing move_forward on the first stale proprioceptive obs.
        self._motor_fail_streak: int = 0

    def score(
        self,
        policies: List[Dict[str, Any]],
        drive_batch: Optional[DriveSignalBatch],
        pe_batch: Optional[PredictionErrorBatch],
        area_familiarity: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> Dict[str, float]:
        """
        Score each policy and return {policy_id: score}.

        Args:
            policies:         Available policy descriptors
            drive_batch:      Current drive urgency signals
            pe_batch:         Current prediction error batch
            area_familiarity: [0,1] from memory — 0=novel, 1=well-visited
            context:          Full step context — used for motor failure penalty

        Returns:
            Dict mapping policy_id → EFE score [0, 1]
        """
        area_novelty = 1.0 - float(area_familiarity)
        scores: Dict[str, float] = {}

        max_urgency = drive_batch.max_urgency if drive_batch else 0.0
        pe_mean = pe_batch.mean_magnitude if pe_batch else 0.0

        # Valence precision scaler on the pragmatic term.
        # Negative valence (threat/deprivation) → drive relief is more valuable → amplify.
        # Positive valence (satiated/safe) → epistemic foraging can dominate → suppress.
        # Scale is centered at 1.0 and moves ±0.3 across the [-1, 1] valence range:
        #   valence=-1.0 → ×1.3  |  valence=0.0 → ×1.0  |  valence=+1.0 → ×0.7
        # Seth (2021): valence modulates the precision of interoceptive predictions.
        valence = float(context.get("valence", 0.0)) if context else 0.0
        valence_precision = max(0.7, min(1.3, 1.0 - 0.3 * valence))

        # Motor failure penalty: if the last action failed (efficiency < 0.3),
        # scale its EFE to discourage repeating a blocked action.
        motor_eff = float(context.get("motor_efficiency", 1.0)) if context else 1.0
        last_action = context.get("last_action") if context else None

        # Update consecutive-failure streak; penalty requires streak ≥ 2 so a
        # single stale obs (e.g. the first step after reset) doesn't fire it.
        if motor_eff < 0.3:
            self._motor_fail_streak += 1
        else:
            self._motor_fail_streak = 0

        # tag → max urgency across all drive signals
        urgency_by_tag: Dict[str, float] = {}
        if drive_batch:
            for signal in drive_batch.signals:
                for tag in signal.suggested_action_tags:
                    urgency_by_tag[tag] = max(urgency_by_tag.get(tag, 0.0), signal.urgency)

        for policy in policies:
            pid = policy["policy_id"]

            if pid == "idle":
                # Allostatic equilibrium: rest is appropriate when quiet and unsurprised.
                # The base score decays as urgency/PE rise. The additional urgency
                # penalty ensures idle cannot hold near 1.0 while drives are actively
                # depleting — without it, idle outscores all movement actions until
                # urgency exceeds ~0.5, which is too late given slow depletion rates.
                score = 1.0 - max(max_urgency, pe_mean)
                score -= max(0.0, max_urgency) * self._idle_urgency_penalty
                scores[pid] = float(max(0.0, min(1.0, score)))
                continue

            drive_tags = policy.get("drive_tags", [])

            # Pragmatic: urgency-weighted drive tag match
            pragmatic = max(
                (urgency_by_tag.get(tag, 0.0) for tag in drive_tags),
                default=0.0,
            )

            # Exteroceptive pragmatic boost for move_forward when food is directly ahead.
            # Encodes E[drive_relief | food_visible, move_forward] without requiring the
            # world model to have learned this yet. Proximity 0=max range, 1=contact;
            # scaled by max_urgency so it only dominates when drives actually need relief.
            if pid == "move_forward" and context:
                raycast = (context.get("raycast_hits") or [{}])[0]
                if raycast.get("hit_tag") in ("GoodGoal", "GoodGoalMulti"):
                    proximity = 1.0 - float(raycast.get("distance", 1.0))
                    pragmatic = max(pragmatic, proximity * max_urgency)

            # Epistemic: information gain from this action class.
            # The foraging baseline ensures active actions always beat idle at
            # allostatic equilibrium — epistemic foraging, not just drive relief.
            if pid in _EPISTEMIC_ACTIONS:
                epistemic = area_novelty + _EPISTEMIC_FORAGING_BASELINE
            else:
                epistemic = pe_mean + _EPISTEMIC_FORAGING_BASELINE

            motor_cost = _MOTOR_COST.get(pid, 0.2)

            combined = (
                pragmatic * self._w_pragmatic * valence_precision
                + epistemic * self._w_epistemic
                - motor_cost * self._w_motor_cost
            )

            # Absolute food-proximity bonus for move_forward when food is directly ahead.
            # The urgency-scaled pragmatic boost above is already present; this term
            # adds intrinsic EFE value for approaching visible food independent of
            # current urgency — critical when drives are only moderately depleted but
            # food is in view (without this, idle outscores move_forward until ~urgency=0.4).
            if pid == "move_forward" and context:
                _rc = (context.get("raycast_hits") or [{}])[0]
                if _rc.get("hit_tag") in ("GoodGoal", "GoodGoalMulti"):
                    _prox = 1.0 - float(_rc.get("distance", 1.0))
                    combined += _prox * self._food_proximity_bonus

            # Motor failure penalty: soften EFE of the last action if it was blocked
            # for at least 2 consecutive steps (streak guard stops spurious first-obs firing).
            # motor_eff=0.0 → combined*0.30; motor_eff=0.29 → combined*0.50.
            if pid == last_action and motor_eff < 0.3 and self._motor_fail_streak >= 2:
                combined = combined * (0.3 + 0.7 * motor_eff)

            scores[pid] = float(max(0.0, min(1.0, combined)))

        return scores
