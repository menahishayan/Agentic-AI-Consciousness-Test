from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.llm.types import LLMRequest, LLMResponse
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


class _StaticLLMClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: List[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.text)


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


def test_llm_arbitrator_selects_from_response(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 1, "rationale": "index 1 aligns", "drive_conflict_detected": false, "confidence": 0.88}'
    )
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
        llm_client=llm_client,
    )

    proposal = generator.propose_action(
        goals=[],
        context={
            "step": 3,
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
    assert proposal.action_id == "dummy:policy_explore"
    assert proposal.action == "explore_action"
    assert len(llm_client.requests) == 1

    traces = memory_manager.query({"target": "policy_traces", "limit": 10})
    assert any(
        isinstance(trace, dict)
        and trace.get("operation") == "arbitrate"
        and trace.get("status") == "selected"
        and trace.get("policy_id") == "dummy:policy_explore"
        for trace in traces
    )


def test_llm_parse_failure_falls_back_to_urgency(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(text="{selected_index: not-json}")
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
        llm_client=llm_client,
    )

    proposal = generator.propose_action(
        goals=[],
        context={
            "step": 4,
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
