"""
FreeEnergyMinimizer — Layer 3, Action Selection

Scores action proposals on how well they minimise expected free energy.
Structure follows Friston's EFE decomposition (Friston et al. 2017):

  G(π) = pragmatic_value  *  w_pragmatic   (allostatic drive urgency relief)
         + epistemic_value  *  w_epistemic   (information gain / uncertainty reduction)
         - motor_cost       *  w_motor_cost  (metabolic cost of action)

Idle:
  score = (1 − urgency) × (1 − area_novelty) − urgency_penalty
  → rest scores highest when the agent is satisfied AND in a familiar area.
  → caps at 0.5 when area is novel, letting epistemic turns dominate.

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
# At w_epistemic=0.70: 0.60 × 0.70 = 0.42 for move_forward; 0.42 + area_novelty×0.70
# for turns. Combined with the product-form idle cap (≤0.5 at novelty≥0.5), turns
# beat idle at equilibrium → agent scans until food is found → move_forward wins
# via food-proximity bonus once food enters the ray fan.
_EPISTEMIC_FORAGING_BASELINE = 0.60

# Angular windows [lo, hi] (degrees) within which food detection benefits each action.
# Food inside the window earns a proximity bonus; outside → no bonus for that action.
_ACTION_FOOD_WINDOWS: Dict[str, Optional[tuple]] = {
    "move_forward":  (-20,  +20),  # food near centre → approach
    "turn_left":     (-80,  -10),  # food on left  → turn toward it
    "turn_right":    (+10,  +80),  # food on right → turn toward it
    "move_backward": None,         # no food bonus
    "idle":          None,         # handled separately
}

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
        # Stuck-turn detector: tracks how many consecutive steps the same food
        # angle has been seen while the agent is turning. When a turn action fails
        # to change the food angle for 5+ steps the food_prox turn bonus is halved
        # — the agent is spinning next to food it can't reach via turning alone.
        self._stuck_turn_angle: Optional[float] = None
        self._stuck_turn_streak: int = 0

    def _food_bonus_for_action(
        self,
        food_rays: List[Dict[str, Any]],
        action_id: str,
        turn_bonus_scale: float = 1.0,
    ) -> float:
        """Return food-proximity bonus for action_id given the current ray fan.

        Selects the closest food ray inside the action's angular window and
        returns proximity = 1 - distance (0 = far, 1 = touching).
        turn_bonus_scale halves the turn bonus when the stuck-turn detector fires.
        """
        window = _ACTION_FOOD_WINDOWS.get(action_id)
        if window is None or not food_rays:
            return 0.0
        lo, hi = window
        relevant = [r for r in food_rays if lo <= float(r.get("angle_deg", 0.0)) <= hi]
        if not relevant:
            return 0.0
        best = min(relevant, key=lambda r: r.get("distance", 1.0))
        proximity = 1.0 - float(best.get("distance", 1.0))
        if action_id in _EPISTEMIC_ACTIONS:
            return proximity * self._food_proximity_bonus * 0.7 * turn_bonus_scale
        return proximity * self._food_proximity_bonus

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

        # Update consecutive move_forward failure streak.
        # Only increment on move_forward; turns/backward don't count as failures.
        # Penalty requires streak ≥ 2 to avoid firing on the first stale obs.
        if last_action == "move_forward" and motor_eff < 0.3:
            self._motor_fail_streak += 1
        elif last_action == "move_forward":
            self._motor_fail_streak = 0
        # On turn/backward/idle actions: leave streak unchanged — the wall is
        # still there; resetting would immediately re-allow move_forward.

        # tag → max urgency across all drive signals
        urgency_by_tag: Dict[str, float] = {}
        if drive_batch:
            for signal in drive_batch.signals:
                for tag in signal.suggested_action_tags:
                    urgency_by_tag[tag] = max(urgency_by_tag.get(tag, 0.0), signal.urgency)

        # Find the most-aligned food ray across the full fan — used for directional bonuses.
        # Most-aligned (smallest |angle|) rather than closest (smallest distance) because
        # the directional bonuses key off angle: food at 80° closest is less actionable
        # than food at 20° slightly further. Alignment determines which action to boost.
        all_rays = context.get("raycast_hits") or [] if context else []
        food_candidates = [r for r in all_rays if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")]
        food_ray = (
            min(food_candidates, key=lambda r: abs(r.get("angle_deg", 90)))
            if food_candidates else None
        )

        # Stuck-turn detector: if the food angle hasn't changed for 5+ steps while
        # the agent is turning, the turn bonus is halved to encourage move_forward instead.
        current_food_angle = float(food_ray.get("angle_deg", 0.0)) if food_ray else None
        if current_food_angle is not None and last_action in ("turn_left", "turn_right"):
            if (self._stuck_turn_angle is not None
                    and abs(current_food_angle - self._stuck_turn_angle) < 5.0):
                self._stuck_turn_streak += 1
            else:
                self._stuck_turn_streak = 0
            self._stuck_turn_angle = current_food_angle
        else:
            self._stuck_turn_streak = 0
            self._stuck_turn_angle = current_food_angle
        _turn_bonus_scale = 0.5 if self._stuck_turn_streak >= 5 else 1.0

        for policy in policies:
            pid = policy["policy_id"]

            if pid == "idle":
                # Idle is only appropriate when BOTH allostatic AND epistemic demands
                # are satisfied. Using a product forces both factors to be high:
                #   urgency=0, area_novelty=0.0 (fully familiar) → score=1.0  (correct: rest)
                #   urgency=0, area_novelty=0.5 (novel)          → score=0.5  (exploration beats rest)
                #   urgency=0.5, area_novelty=0.5                → score=0.25 (drives beat rest)
                # This caps idle at 0.5 whenever the area is novel, making epistemic
                # turns competitive at allostatic equilibrium without needing urgency.
                allostatic_sat = 1.0 - max_urgency
                effective_novelty = max(0.15, area_novelty)  # floor: idle never fully wins
                epistemic_sat  = 1.0 - effective_novelty
                score = allostatic_sat * epistemic_sat
                # Urgency penalty: fire whenever drives are above zero, not just at
                # extreme urgency. At urgency=0.20 with penalty=2.0: -0.40.
                score -= max(0.0, max_urgency) * self._idle_urgency_penalty
                # Food visible → idling is never appropriate regardless of urgency level.
                if food_candidates:
                    score -= 0.6
                scores[pid] = float(max(0.0, min(1.0, score)))
                continue

            drive_tags = policy.get("drive_tags", [])

            # Pragmatic: urgency-weighted drive tag match
            pragmatic = max(
                (urgency_by_tag.get(tag, 0.0) for tag in drive_tags),
                default=0.0,
            )

            # Urgency-scaled pragmatic boost when food is visible in the forward ray.
            # Scaled by max_urgency so it only dominates when drives need relief.
            if pid == "move_forward" and food_ray and abs(float(food_ray.get("angle_deg", 0.0))) < 15:
                proximity = 1.0 - float(food_ray.get("distance", 1.0))
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

            # Food-proximity bonus via window-based per-action method.
            # Close food (dist < 0.10): alignment irrelevant — just move.
            #   move_forward gets 1.5× bonus; turns suppressed (spinning next to
            #   adjacent food is counterproductive).
            # Far food (dist ≥ 0.10): delegate to _food_bonus_for_action which
            #   checks each action's angular window against all food rays.
            if food_candidates:
                food_dist_min = min(r.get("distance", 1.0) for r in food_candidates)
                if food_dist_min < 0.01:
                    if pid == "move_forward":
                        combined += (1.0 - food_dist_min) * self._food_proximity_bonus * 1.5
                    # No turn bonus when food is adjacent.
                else:
                    combined += self._food_bonus_for_action(
                        food_candidates, pid, _turn_bonus_scale
                    )

            # Motor failure penalty: soften EFE of the last action if it was blocked
            # for at least 2 consecutive steps (streak guard stops spurious first-obs firing).
            # motor_eff=0.0 → combined*0.30; motor_eff=0.29 → combined*0.50.
            #
            # EXEMPTION: do not penalise move_forward when food is adjacent (dist < 0.15).
            # The food item itself causes the motor stall — penalising move_forward in this
            # state causes the agent to retreat from food it is already touching.
            food_adj = (food_ray is not None
                        and float(food_ray.get("distance", 1.0)) < 0.015)
            if (pid == last_action
                    and motor_eff < 0.3
                    and self._motor_fail_streak >= 2
                    and not (pid == "move_forward" and food_adj)):
                combined = combined * (0.3 + 0.7 * motor_eff)

            scores[pid] = float(max(0.0, min(1.0, combined)))

        return scores
