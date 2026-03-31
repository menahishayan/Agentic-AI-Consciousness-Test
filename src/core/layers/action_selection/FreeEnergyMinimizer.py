"""
FreeEnergyMinimizer — Layer 3, Action Selection

Scores action proposals on how well they minimize free energy:
  - Reduce prediction error (epistemic value)
  - Maintain homeostatic setpoints (pragmatic value)

Based on Friston's free energy principle:
  - Agents minimize surprise about their own body states
  - Actions that reduce both PE and homeostatic deficit are preferred

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.models.signals import DriveSignalBatch, PredictionErrorBatch


class FreeEnergyMinimizer:
    """
    Computes free-energy-based scores for candidate action proposals.

    Score = pragmatic_value * w1 + epistemic_value * w2

    pragmatic_value: How well does this action address drive urgencies?
    epistemic_value: How much does this action reduce prediction errors?
    """

    # Actions that provide angular information gain (scan new directions)
    _EPISTEMIC_ACTIONS = {"turn_left", "turn_right"}

    def __init__(self, config: Dict[str, Any]) -> None:
        pg = config.get("policy_generator", {})
        weights = pg.get("weights", {})
        self._w_pragmatic = float(weights.get("allostatic_survival_fit", 0.2))
        self._w_epistemic = float(weights.get("prediction_error", 0.4))
        self._w_coherence = float(weights.get("goal_coherence", 0.6))
        # Epistemic bonus weight for turning in novel/unvisited areas.
        # Corresponds to the information-gain (curiosity) term in Friston's
        # expected free energy: G = pragmatic + epistemic.
        self._w_turn_novelty = float(weights.get("turn_novelty", 0.15))

    def score(
        self,
        policies: List[Dict[str, Any]],
        drive_batch: Optional[DriveSignalBatch],
        pe_batch: Optional[PredictionErrorBatch],
        goal_coherence_scores: Optional[Dict[str, float]] = None,
        area_familiarity: float = 0.5,
    ) -> Dict[str, float]:
        """
        Score each policy and return {policy_id: score}.

        Args:
            policies: Available policy descriptors
            drive_batch: Current drive urgency signals
            pe_batch: Current prediction error batch
            goal_coherence_scores: Optional pre-computed coherence scores
            area_familiarity: [0,1] from memory — 0=novel, 1=well-visited

        Returns:
            Dict mapping policy_id → combined score [0, 1]
        """
        # area_novelty is the epistemic complement of familiarity
        area_novelty = 1.0 - float(area_familiarity)
        scores: Dict[str, float] = {}

        # Build urgency index: tag → max urgency
        urgency_by_tag: Dict[str, float] = {}
        if drive_batch:
            for signal in drive_batch.signals:
                for tag in signal.suggested_action_tags:
                    urgency_by_tag[tag] = max(urgency_by_tag.get(tag, 0.0), signal.urgency)

        # Channel-level PE: which channels are most surprising?
        pe_by_channel: Dict[str, float] = {}
        if pe_batch:
            for error in pe_batch.errors:
                pe_by_channel[error.channel] = error.magnitude

        for policy in policies:
            pid = policy["policy_id"]

            # Pragmatic value: urgency match on drive_tags
            pragmatic = 0.0
            drive_tags = policy.get("drive_tags", [])
            if drive_tags:
                pragmatic = max(urgency_by_tag.get(tag, 0.0) for tag in drive_tags)

            # Epistemic value: does this policy address high-PE channels?
            epistemic = 0.0
            if pe_by_channel and drive_tags:
                relevant_pe = [pe_by_channel.get(tag, 0.0) for tag in drive_tags]
                epistemic = max(relevant_pe) if relevant_pe else 0.0

            # Goal coherence
            coherence = (
                goal_coherence_scores.get(pid, 0.5)
                if goal_coherence_scores else 0.5
            )

            # Epistemic bonus (Friston's G information-gain term):
            # Turning in a novel/unvisited area reduces uncertainty about what
            # lies in unseen directions — directly citable as curiosity-driven
            # active inference (Seth & Bayne 2022; Friston et al. 2017).
            turn_novelty_bonus = (
                area_novelty * self._w_turn_novelty
                if pid in self._EPISTEMIC_ACTIONS else 0.0
            )

            combined = (
                pragmatic * self._w_pragmatic
                + epistemic * self._w_epistemic
                + coherence * self._w_coherence
                + turn_novelty_bonus
            )

            scores[pid] = float(min(1.0, combined))

        return scores
