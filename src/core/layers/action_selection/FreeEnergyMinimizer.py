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
  epistemic = area_novelty  (scanning novel directions reduces directional uncertainty)
  Cite: Friston et al. (2017) active inference; Seth & Bayne (2022) curiosity.

Movement actions:
  epistemic = mean_PE  (moving through space resolves spatial prediction errors)

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.models.signals import DriveSignalBatch, PredictionErrorBatch

# Actions that scan unseen directions — pure epistemic value
_EPISTEMIC_ACTIONS = {"turn_left", "turn_right"}

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

    def score(
        self,
        policies: List[Dict[str, Any]],
        drive_batch: Optional[DriveSignalBatch],
        pe_batch: Optional[PredictionErrorBatch],
        area_familiarity: float = 0.5,
        **_kwargs,
    ) -> Dict[str, float]:
        """
        Score each policy and return {policy_id: score}.

        Args:
            policies:         Available policy descriptors
            drive_batch:      Current drive urgency signals
            pe_batch:         Current prediction error batch
            area_familiarity: [0,1] from memory — 0=novel, 1=well-visited

        Returns:
            Dict mapping policy_id → EFE score [0, 1]
        """
        area_novelty = 1.0 - float(area_familiarity)
        scores: Dict[str, float] = {}

        max_urgency = drive_batch.max_urgency if drive_batch else 0.0
        pe_mean = pe_batch.mean_magnitude if pe_batch else 0.0

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
                # The agent rests only when it has nothing to resolve.
                score = 1.0 - max(max_urgency, pe_mean)
                scores[pid] = float(max(0.0, min(1.0, score)))
                continue

            drive_tags = policy.get("drive_tags", [])

            # Pragmatic: urgency-weighted drive tag match
            pragmatic = max(
                (urgency_by_tag.get(tag, 0.0) for tag in drive_tags),
                default=0.0,
            )

            # Epistemic: information gain from this action class
            if pid in _EPISTEMIC_ACTIONS:
                # Turn actions scan unseen directions — pure novelty-driven IG
                epistemic = area_novelty
            else:
                # Movement: resolves spatial prediction errors
                epistemic = pe_mean

            motor_cost = _MOTOR_COST.get(pid, 0.2)

            combined = (
                pragmatic * self._w_pragmatic
                + epistemic * self._w_epistemic
                - motor_cost * self._w_motor_cost
            )
            scores[pid] = float(max(0.0, min(1.0, combined)))

        return scores
