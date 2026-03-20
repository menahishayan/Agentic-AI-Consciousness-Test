from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.layers.action_selection import PolicyGenerator
from core.memory.manager import MemoryManager
from core.models.signals import ActionProposal


class _NeutralGoalChecker:
    def check(self, goals: Any, policy_descriptor: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        _ = goals
        _ = policy_descriptor
        _ = context
        return {"coherence_score": 0.5}


class _NeutralPredictionErrorCalculator:
    def compute(self, policy_id: str, context: Any = None, memory_manager: Any = None) -> Dict[str, Any]:
        _ = policy_id
        _ = context
        _ = memory_manager
        return {"prediction_error_score": 0.5}


class _DriveAwareAdapter:
    def reset(self) -> Any:
        return None

    def step(self, action: Any) -> Any:
        return None

    def close(self) -> None:
        return None

    def get_available_vitals(self) -> List[str]:
        return []

    def get_available_policies(self) -> List[Dict[str, Any]]:
        return [
            {
                "policy_id": "dummy:policy_seek_food",
                "callable_name": "policy_seek_food",
                "description": "Collect and eat food.",
                "tags": ["collect", "food", "eat", "survival"],
                "drive_tags": ["hunger", "resource_level"],
            },
            {
                "policy_id": "dummy:policy_explore",
                "callable_name": "policy_explore",
                "description": "Explore distant terrain.",
                "tags": ["explore", "move", "terrain"],
                "drive_tags": ["resource_level"],
            },
        ]

    def policy_seek_food(self) -> str:
        return "food_action"

    def policy_explore(self) -> str:
        return "explore_action"


class _InvalidPolicyAdapter(_DriveAwareAdapter):
    def get_available_policies(self) -> List[Dict[str, Any]]:
        return [
            {
                "policy_id": "dummy:policy_missing_tags",
                "callable_name": "policy_seek_food",
                "description": "Invalid because tags are empty.",
                "tags": [],
            }
        ]


class _NoPolicyContractAdapter:
    def reset(self) -> Any:
        return None

    def step(self, action: Any) -> Any:
        _ = action
        return None

    def close(self) -> None:
        return None

    def policy_hidden(self) -> str:
        return "hidden_action"


def _memory_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(
        long_term_memory_config={
            "path": str(tmp_path / "policies.json"),
            "max_score_history": 20,
            "max_outcome_history": 20,
        }
    )


def test_drive_urgency_alignment_prioritizes_matching_policy(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={
            "weights": {
                "goal_coherence": 0.5,
                "prediction_error": 0.5,
                "allostatic_survival_fit": 0.2,
                "allostatic_urgency_alignment": 0.8,
            },
            "fallback_scores": {
                "goal_coherence": 0.5,
                "prediction_error": 0.5,
                "allostatic_survival_fit": 0.5,
                "allostatic_urgency_alignment": 0.5,
            },
        },
    )

    proposal = generator.propose_action(
        goals=[],
        context={
            "step": 1,
            "drive_signals": {
                "signals": [
                    {"channel_id": "hunger", "urgency": 0.95},
                    {"channel_id": "safety", "urgency": 0.20},
                ],
                "highest_urgency": 0.95,
            },
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_seek_food"
    assert proposal.action == "food_action"


def test_policy_discovery_rejects_descriptors_without_tags(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_InvalidPolicyAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )

    proposal = generator.propose_action(
        goals=[],
        context={"step": 1, "drive_signals": {"signals": []}},
    )

    assert proposal is None
    assert memory_manager.get_policies(adapter_folder="dummy") == []


def test_reflection_fallback_is_disabled_when_adapter_contract_is_missing(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_NoPolicyContractAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )

    proposal = generator.propose_action(
        goals=[],
        context={"step": 1, "drive_signals": {"signals": []}},
    )

    assert proposal is None
