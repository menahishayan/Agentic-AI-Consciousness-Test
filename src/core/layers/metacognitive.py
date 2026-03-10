from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class InformationIntegrationHub:
    def integrate(self, signals: Any) -> Any:
        raise NotImplementedError("Information integration not implemented.")


class UncertaintyEstimator:
    def estimate(self, signals: Any) -> Any:
        raise NotImplementedError("Uncertainty estimation not implemented.")


class GoalCoherenceChecker:
    def check(
        self,
        goals: Any,
        policy_descriptor: Any,
        context: Any = None,
    ) -> Dict[str, Any]:
        policy_tokens = self._policy_tokens(policy_descriptor)
        if not policy_tokens:
            return {
                "coherence_score": 0.0,
                "reason": "no_policy_tokens",
                "goal_scores": [],
            }

        goal_items = self._normalize_goals(goals)
        if not goal_items:
            return {
                "coherence_score": None,
                "reason": "no_goals",
                "goal_scores": [],
            }

        weighted = 0.0
        total_weight = 0.0
        details: List[Dict[str, Any]] = []
        for description, priority in goal_items:
            goal_tokens = self._tokenize(description)
            if not goal_tokens:
                continue
            overlap = goal_tokens.intersection(policy_tokens)
            union = goal_tokens.union(policy_tokens)
            score = float(len(overlap)) / float(len(union)) if union else 0.0
            weighted += score * priority
            total_weight += priority
            details.append(
                {
                    "goal": description,
                    "priority": priority,
                    "score": score,
                    "overlap": sorted(overlap),
                }
            )

        if total_weight <= 0:
            return {
                "coherence_score": None,
                "reason": "no_weighted_goals",
                "goal_scores": details,
            }

        return {
            "coherence_score": max(0.0, min(1.0, weighted / total_weight)),
            "goal_scores": details,
            "policy_tokens": sorted(policy_tokens),
        }

    def _policy_tokens(self, descriptor: Any) -> set[str]:
        if not isinstance(descriptor, Mapping):
            descriptor = {}

        token_sources: List[str] = []
        for key in ("policy_id", "callable_name", "description"):
            value = descriptor.get(key)
            if value is not None:
                token_sources.append(str(value))

        tags = descriptor.get("tags")
        if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
            for tag in tags:
                token_sources.append(str(tag))

        tokens: set[str] = set()
        for source in token_sources:
            tokens.update(self._tokenize(source))
        return tokens

    def _normalize_goals(self, goals: Any) -> List[Tuple[str, float]]:
        if goals is None:
            return []
        if not isinstance(goals, list):
            goals = [goals]

        normalized: List[Tuple[str, float]] = []
        for goal in goals:
            description: Optional[str] = None
            priority = 1.0

            if isinstance(goal, Mapping):
                raw_description = goal.get("description") or goal.get("goal")
                if raw_description is not None:
                    description = str(raw_description)
                raw_priority = goal.get("priority")
                if isinstance(raw_priority, (int, float)):
                    priority = max(0.0, float(raw_priority))
            else:
                if hasattr(goal, "description"):
                    raw_description = getattr(goal, "description")
                    if raw_description is not None:
                        description = str(raw_description)
                if hasattr(goal, "priority"):
                    raw_priority = getattr(goal, "priority")
                    if isinstance(raw_priority, (int, float)):
                        priority = max(0.0, float(raw_priority))

            if description:
                normalized.append((description, priority))

        return normalized

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        parts = re.split(r"[^a-zA-Z0-9_]+", text.lower())
        return {part for part in parts if part}
