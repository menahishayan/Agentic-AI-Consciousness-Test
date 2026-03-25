from __future__ import annotations

from pathlib import Path
import time
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


class _NoopFirstAdapter(_DriveAwareAdapter):
    def get_available_policies(self) -> List[Dict[str, Any]]:
        return [
            {
                "policy_id": "dummy:policy_noop",
                "callable_name": "policy_noop",
                "description": "Do nothing.",
                "tags": ["noop", "idle"],
                "drive_tags": ["noop"],
            },
            {
                "policy_id": "dummy:policy_collect_milk",
                "callable_name": "policy_collect_milk",
                "description": "Collect milk from nearby animals.",
                "tags": ["collect", "milk", "gather"],
                "drive_tags": ["resource_level"],
            },
        ]

    def policy_noop(self) -> str:
        return "noop_action"

    def policy_collect_milk(self) -> str:
        return "collect_milk_action"


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


def _wait_for(predicate: Any, timeout_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


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
                    {"channel_id": "resource_level", "urgency": 0.92},
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
                    {"channel_id": "resource_level", "urgency": 0.90},
                ],
                "highest_urgency": 0.95,
            },
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_seek_food"
    assert proposal.action == "food_action"


def test_skill_plan_bias_prefers_queue_head_when_not_in_emergency(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={"skill_plan_bias": 0.25},
    )

    proposal = generator.propose_action(
        goals=[{"goal_id": "task:harvest_milk", "description": "Harvest milk", "priority": 1.0}],
        context={
            "step": 6,
            "skill_plan": {
                "head_policy_id": "dummy:policy_explore",
                "remaining_policy_ids": ["dummy:policy_explore", "dummy:policy_seek_food"],
                "remaining_count": 2,
            },
            "allostatic_assessment": {
                "needs": [
                    {"need_id": "stabilize_hunger", "urgency": 0.6, "irreversible": False},
                ]
            },
            "drive_signals": {
                "signals": [
                    {"channel_id": "hunger", "urgency": 0.60},
                    {"channel_id": "resource_level", "urgency": 0.55},
                ],
            },
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_explore"


def test_skill_plan_bias_is_disabled_by_irreversible_emergency(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={
            "skill_plan_bias": 0.25,
            "skill_plan_emergency_urgency_threshold": 0.85,
        },
    )

    proposal = generator.propose_action(
        goals=[{"goal_id": "task:harvest_milk", "description": "Harvest milk", "priority": 1.0}],
        context={
            "step": 7,
            "skill_plan": {
                "head_policy_id": "dummy:policy_explore",
                "remaining_policy_ids": ["dummy:policy_explore", "dummy:policy_seek_food"],
                "remaining_count": 2,
            },
            "allostatic_assessment": {
                "needs": [
                    {"need_id": "stabilize_health", "urgency": 0.9, "irreversible": True},
                ]
            },
            "drive_signals": {
                "signals": [
                    {"channel_id": "hunger", "urgency": 0.60},
                    {"channel_id": "resource_level", "urgency": 0.55},
                ],
            },
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_seek_food"


def test_llm_prompt_includes_skill_plan_context(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 0, "rationale": "follow skill plan", "drive_conflict_detected": false, "confidence": 0.9}'
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
        goals=[{"goal_id": "task:harvest_milk", "description": "Harvest milk", "priority": 1.0}],
        context={
            "step": 8,
            "skill_plan": {
                "head_policy_id": "dummy:policy_explore",
                "remaining_policy_ids": ["dummy:policy_explore", "dummy:policy_seek_food"],
                "remaining_count": 2,
                "metadata": [{"phase_index": 0, "intent_tokens": ["milk", "use"]}],
            },
            "drive_signals": {
                "signals": [
                    {"channel_id": "hunger", "urgency": 0.95},
                    {"channel_id": "resource_level", "urgency": 0.90},
                ]
            },
            "allostatic_assessment": {"needs": []},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert len(llm_client.requests) == 1
    prompt = llm_client.requests[0].messages[1].content
    assert "skill_plan.head_policy_id: dummy:policy_explore" in prompt


def test_selective_gate_skips_llm_when_no_trigger_conditions(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 1, "rationale": "unused", "drive_conflict_detected": false, "confidence": 0.88}'
    )
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={"llm_reeval_interval": 10},
        llm_client=llm_client,
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001

    proposal = generator.propose_action(
        goals=[],
        context={
            "step": 1,
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_seek_food"
    assert len(llm_client.requests) == 0


def test_goal_directed_fallback_skips_noop_when_urgency_scores_are_flat(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_NoopFirstAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
        llm_client=None,
    )

    proposal = generator.propose_action(
        goals=[{"goal_id": "task:collect_milk", "description": "Collect milk"}],
        context={"step": 1, "drive_signals": {"signals": []}},
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_collect_milk"


def test_goal_change_without_pending_result_triggers_sync_llm(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 1, "rationale": "async result", "drive_conflict_detected": false, "confidence": 0.88}'
    )
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={"llm_reeval_interval": 10},
        llm_client=llm_client,
    )

    proposal = generator.propose_action(
        goals=[{"goal_id": "new_goal"}],
        context={
            "step": 1,
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_explore"
    assert len(llm_client.requests) == 1


def test_goal_change_with_pending_result_runs_async_and_uses_cached_policy(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 0, "rationale": "refresh", "drive_conflict_detected": false, "confidence": 0.7}'
    )
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={"llm_reeval_interval": 10},
        llm_client=llm_client,
    )
    generator._pending_llm_result = {"policy_id": "dummy:policy_explore"}  # noqa: SLF001

    follow_up = generator.propose_action(
        goals=[{"goal_id": "new_goal"}],
        context={
            "step": 1,
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
        },
    )

    assert isinstance(follow_up, ActionProposal)
    assert follow_up.action_id == "dummy:policy_explore"
    assert _wait_for(lambda: len(llm_client.requests) == 1)


def test_sustained_prediction_error_triggers_sync_llm_call(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 1, "rationale": "switch strategy", "drive_conflict_detected": false, "confidence": 0.91}'
    )
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={
            "pe_high_threshold": 0.6,
            "pe_streak_threshold": 2,
            "llm_reeval_interval": 50,
        },
        llm_client=llm_client,
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001

    first = generator.propose_action(
        goals=[],
        context={
            "step": 1,
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.9},
        },
    )
    second = generator.propose_action(
        goals=[],
        context={
            "step": 2,
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.9},
        },
    )

    assert isinstance(first, ActionProposal)
    assert first.action_id == "dummy:policy_seek_food"
    assert isinstance(second, ActionProposal)
    assert second.action_id == "dummy:policy_explore"
    assert len(llm_client.requests) == 1


def test_skill_gap_triggers_sync_llm_when_no_matching_policy(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 1, "rationale": "no direct skill match", "drive_conflict_detected": false, "confidence": 0.77}'
    )
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={"llm_reeval_interval": 50},
        llm_client=llm_client,
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001

    proposal = generator.propose_action(
        goals=[],
        context={
            "step": 1,
            "drive_signals": {"signals": [{"channel_id": "oxygen", "urgency": 0.95}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_explore"
    assert len(llm_client.requests) == 1
