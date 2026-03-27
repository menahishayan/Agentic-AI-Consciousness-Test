"""
GoalCoherenceChecker — Layer 4, Metacognitive Monitor

Scores how coherent each policy proposal is with the active goals.
Uses token-overlap between policy description/tags and goal description.

This is a lightweight semantic alignment check — not a full NLU system.
It ensures the agent doesn't pursue actions that are misaligned with
its declared task objectives.

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from core.models.signals import Goal


class GoalCoherenceChecker:
    """
    Computes token-overlap coherence scores between policies and goals.

    Score = (shared tokens) / (total unique goal tokens)
    Range: [0, 1] — 1 means perfect token alignment
    """

    def __init__(self) -> None:
        self._stop_words: Set[str] = {
            "a", "an", "the", "to", "and", "or", "in", "of", "for", "is",
            "are", "that", "this", "with", "on", "at", "by", "it", "its",
        }

    def score_all(
        self,
        policies: List[Dict[str, Any]],
        goals: List[Goal],
    ) -> Dict[str, float]:
        """
        Score each policy against the active goal set.

        Args:
            policies: Policy descriptor dicts
            goals: Active Goal objects

        Returns:
            Dict mapping policy_id → coherence score [0, 1]
        """
        if not goals:
            return {p["policy_id"]: 0.5 for p in policies}

        goal_tokens = self._extract_tokens_from_goals(goals)
        scores: Dict[str, float] = {}

        for policy in policies:
            policy_tokens = self._extract_tokens_from_policy(policy)
            score = self._overlap_score(policy_tokens, goal_tokens)
            scores[policy["policy_id"]] = score

        return scores

    def _extract_tokens_from_goals(self, goals: List[Goal]) -> Set[str]:
        tokens: Set[str] = set()
        for goal in goals:
            tokens.update(self._tokenize(goal.description))
            tokens.update(self._tokenize(goal.task_id))
        return tokens

    def _extract_tokens_from_policy(self, policy: Dict[str, Any]) -> Set[str]:
        tokens: Set[str] = set()
        tokens.update(self._tokenize(policy.get("policy_id", "")))
        tokens.update(self._tokenize(policy.get("description", "")))
        for tag in policy.get("tags", []):
            tokens.update(self._tokenize(tag))
        for tag in policy.get("drive_tags", []):
            tokens.update(self._tokenize(tag))
        return tokens

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        return {w for w in words if w not in self._stop_words and len(w) > 1}

    def _overlap_score(self, policy_tokens: Set[str], goal_tokens: Set[str]) -> float:
        if not goal_tokens:
            return 0.5
        shared = len(policy_tokens & goal_tokens)
        return float(min(1.0, shared / len(goal_tokens))) if goal_tokens else 0.5
