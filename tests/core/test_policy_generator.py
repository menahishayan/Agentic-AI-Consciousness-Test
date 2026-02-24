from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.layers.action_selection import PolicyGenerator
from core.memory.manager import MemoryManager


class _GoalChecker:
    def check(self, goals: Any, policy_descriptor: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        name = policy_descriptor.get("callable_name")
        if name == "policy_b":
            return {"coherence_score": 0.9}
        return {"coherence_score": 0.1}


class _PredictionErrorCalculator:
    def compute(self, policy_id: str, context: Any = None, memory_manager: Any = None) -> Dict[str, Any]:
        if policy_id.endswith(":policy_b"):
            return {"prediction_error_score": 0.1}
        return {"prediction_error_score": 0.9}


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


class _DummyAdapter:
    def reset(self) -> Any:
        return None

    def step(self, action: Any) -> Any:
        return None

    def close(self) -> None:
        return None

    def sample_action(self) -> str:
        return "fallback"

    def get_available_vitals(self) -> List[str]:
        return []

    def policy_a(self) -> str:
        return "action_a"

    def policy_b(self) -> str:
        return "action_b"


class _FailingAdapter:
    def reset(self) -> Any:
        return None

    def step(self, action: Any) -> Any:
        return None

    def close(self) -> None:
        return None

    def get_available_vitals(self) -> List[str]:
        return []

    def policy_a_fail(self, required: str) -> str:
        return required

    def policy_b_ok(self, step: int) -> Dict[str, int]:
        return {"step": step}


class _AllostaticAdapter:
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
                "description": "Collect and eat food for survival.",
                "tags": ["collect", "food", "eat", "survival"],
            },
            {
                "policy_id": "dummy:policy_explore",
                "callable_name": "policy_explore",
                "description": "Explore distant terrain.",
                "tags": ["explore", "move", "terrain"],
            },
        ]

    def policy_seek_food(self) -> str:
        return "food_action"

    def policy_explore(self) -> str:
        return "explore_action"


def _memory_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(
        long_term_memory_config={
            "path": str(tmp_path / "policies.json"),
            "max_score_history": 20,
            "max_outcome_history": 20,
        }
    )


def test_policy_generator_discovers_public_callables_and_selects_best(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DummyAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_GoalChecker(),
        prediction_error_calculator=_PredictionErrorCalculator(),
        config={
            "weights": {"goal_coherence": 0.6, "prediction_error": 0.4},
            "fallback_scores": {"goal_coherence": 0.5, "prediction_error": 0.5},
            "discovery": {
                "reserved_methods": [
                    "reset",
                    "step",
                    "close",
                    "sample_action",
                    "get_available_vitals",
                    "get_available_policies",
                ]
            },
        },
    )

    proposal = generator.propose_action(
        goals=[{"description": "prefer b", "priority": 1.0}],
        context={"step": 1, "info": {}, "obs": None, "workspace_messages": []},
    )

    assert proposal is not None
    assert proposal.action_id == "dummy:policy_b"
    assert proposal.action == "action_b"

    discovered = memory_manager.get_policies(adapter_folder="dummy")
    policy_ids = {item["policy_id"] for item in discovered}
    assert "dummy:policy_a" in policy_ids
    assert "dummy:policy_b" in policy_ids


def test_policy_generator_skips_failed_invocation_and_uses_next_candidate(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_FailingAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_GoalChecker(),
        prediction_error_calculator=_PredictionErrorCalculator(),
        config={
            "weights": {"goal_coherence": 0.5, "prediction_error": 0.5},
            "fallback_scores": {"goal_coherence": 0.5, "prediction_error": 0.5},
            "discovery": {
                "reserved_methods": [
                    "reset",
                    "step",
                    "close",
                    "get_available_vitals",
                    "get_available_policies",
                ]
            },
        },
    )

    proposal = generator.propose_action(
        goals=[],
        context={"step": 7, "info": {}, "obs": None, "workspace_messages": []},
    )

    assert proposal is not None
    assert proposal.action_id == "dummy:policy_b_ok"
    assert proposal.action == {"step": 7}

    traces = memory_manager.policy_traces.query({})
    assert any(
        isinstance(item, dict) and item.get("policy_id") == "dummy:policy_a_fail"
        for item in traces
    )


def test_policy_generator_allostatic_bias_prioritizes_survival_policy(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_AllostaticAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={
            "weights": {
                "goal_coherence": 0.5,
                "prediction_error": 0.5,
                "allostatic_survival_fit": 0.8,
                "allostatic_urgency_alignment": 0.8,
            },
            "fallback_scores": {
                "goal_coherence": 0.5,
                "prediction_error": 0.5,
                "allostatic_survival_fit": 0.5,
                "allostatic_urgency_alignment": 0.5,
            },
            "discovery": {
                "reserved_methods": [
                    "reset",
                    "step",
                    "close",
                    "get_available_vitals",
                    "get_available_policies",
                ]
            },
        },
    )

    allostatic_assessment = {
        "source": "heuristic",
        "needs": [
            {
                "need_id": "restore_calories",
                "urgency": 0.9,
                "time_to_critical_steps": 80,
                "actions": ["collect", "eat"],
                "resources": ["food"],
                "evidence": ["food=4"],
            }
        ],
        "policy_bias_tags": ["survival", "food", "eat"],
    }

    proposal = generator.propose_action(
        goals=[],
        context={
            "step": 2,
            "info": {},
            "obs": None,
            "workspace_messages": [],
            "allostatic_assessment": allostatic_assessment,
        },
    )

    assert proposal is not None
    assert proposal.action_id == "dummy:policy_seek_food"
    assert proposal.action == "food_action"


def test_policy_generator_baseline_unchanged_when_allostatic_weights_zero(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DummyAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_GoalChecker(),
        prediction_error_calculator=_PredictionErrorCalculator(),
        config={
            "weights": {
                "goal_coherence": 0.6,
                "prediction_error": 0.4,
                "allostatic_survival_fit": 0.0,
                "allostatic_urgency_alignment": 0.0,
            },
            "fallback_scores": {
                "goal_coherence": 0.5,
                "prediction_error": 0.5,
                "allostatic_survival_fit": 0.5,
                "allostatic_urgency_alignment": 0.5,
            },
            "discovery": {
                "reserved_methods": [
                    "reset",
                    "step",
                    "close",
                    "sample_action",
                    "get_available_vitals",
                    "get_available_policies",
                ]
            },
        },
    )

    proposal = generator.propose_action(
        goals=[{"description": "prefer b", "priority": 1.0}],
        context={
            "step": 3,
            "info": {},
            "obs": None,
            "workspace_messages": [],
            "allostatic_assessment": {
                "source": "llm",
                "needs": [
                    {
                        "need_id": "collect_a_resource",
                        "urgency": 1.0,
                        "actions": ["policy_a"],
                        "resources": ["resource_a"],
                        "evidence": ["forced-bias"],
                    }
                ],
                "policy_bias_tags": ["policy_a", "resource_a"],
            },
        },
    )

    assert proposal is not None
    assert proposal.action_id == "dummy:policy_b"
