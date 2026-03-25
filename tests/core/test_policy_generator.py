from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List

from core.llm.types import LLMRequest, LLMResponse
from core.layers.action_selection import PolicyGenerator
from core.memory import MemoryManager, WorkingMemoryEntry
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


class _StopwordOverlapAdapter(_DriveAwareAdapter):
    def get_available_policies(self) -> List[Dict[str, Any]]:
        return [
            {
                "policy_id": "dummy:policy_craft_or_smelt",
                "callable_name": "policy_craft_or_smelt",
                "description": "Craft or smelt resources.",
                "tags": ["craft", "smelt"],
                "drive_tags": ["resource_level"],
            },
            {
                "policy_id": "dummy:policy_use_bucket",
                "callable_name": "policy_use_bucket",
                "description": "Use a bucket to collect milk.",
                "tags": ["bucket", "use", "collect"],
                "drive_tags": ["resource_level"],
            },
        ]

    def policy_craft_or_smelt(self) -> str:
        return "craft_or_smelt_action"

    def policy_use_bucket(self) -> str:
        return "use_bucket_action"


class _PlanCycleAdapter(_DriveAwareAdapter):
    def get_available_policies(self) -> List[Dict[str, Any]]:
        return [
            {
                "policy_id": "dummy:policy_craft",
                "callable_name": "policy_craft",
                "description": "Craft an item.",
                "tags": ["craft"],
                "drive_tags": ["resource_level"],
            },
            {
                "policy_id": "dummy:policy_use",
                "callable_name": "policy_use",
                "description": "Use an interactable target.",
                "tags": ["use", "interact"],
                "drive_tags": ["resource_level"],
            },
            {
                "policy_id": "dummy:policy_move_forward",
                "callable_name": "policy_move_forward",
                "description": "Move forward to explore.",
                "tags": ["move", "explore"],
                "drive_tags": ["resource_level"],
            },
        ]

    def policy_craft(self) -> str:
        return "craft_action"

    def policy_use(self) -> str:
        return "use_action"

    def policy_move_forward(self) -> str:
        return "move_forward_action"


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


def test_llm_prompt_includes_observation_and_learning_context(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    memory_manager.record_working(
        WorkingMemoryEntry(
            tick=1,
            entry_type="transition",
            payload={
                "policy_id": "dummy:policy_seek_food",
                "reward": 0.0,
                "done": False,
                "prev_facts": {"has_bucket": False},
                "next_facts": {"has_bucket": False},
                "inventory_progress": {"changed": False, "slot_delta": 0},
                "bucket_progress": {"changed": False, "bucket_count_delta": 0},
            },
            priority=0.6,
        )
    )
    memory_manager.record_working(
        WorkingMemoryEntry(
            tick=2,
            entry_type="transition",
            payload={
                "policy_id": "dummy:policy_explore",
                "reward": 1.0,
                "done": False,
                "prev_facts": {"has_bucket": False},
                "next_facts": {"has_bucket": True},
                "inventory_progress": {"changed": True, "slot_delta": 1},
                "bucket_progress": {"changed": True, "bucket_count_delta": 1},
            },
            priority=0.6,
        )
    )
    memory_manager.record_pe(
        "plains:0:0",
        {
            "policy_id": "dummy:policy_seek_food",
            "channel": "resource_level",
            "magnitude": 0.7,
            "tick": 1,
        },
        policy_id="dummy:policy_seek_food",
    )

    llm_client = _StaticLLMClient(
        text='{"selected_index": 0, "rationale": "follow context", "drive_conflict_detected": true, "confidence": 0.8}'
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
        goals=[{"goal_id": "task:harvest_milk"}],
        context={
            "step": 8,
            "world_facts": {
                "biome": "forest",
                "position": {"x": 10.0, "y": 64.0, "z": 3.0},
                "nearby_crafting_table": True,
                "nearby_cow": False,
                "has_bucket": False,
                "bucket_count": 0,
                "inventory_non_air_slots": 2,
                "inventory_total_quantity": 4,
                "inventory_fullness": 0.06,
            },
            "drive_signals": {
                "signals": [
                    {"channel_id": "hunger", "urgency": 0.95},
                    {"channel_id": "resource_level", "urgency": 0.92},
                ]
            },
            "allostatic_assessment": {"needs": []},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert len(llm_client.requests) == 1
    prompt = llm_client.requests[0].messages[1].content
    assert "OBSERVATION CONTEXT" in prompt
    assert "biome: forest" in prompt
    assert "nearby_crafting_table: True" in prompt
    assert "LEARNING CONTEXT" in prompt
    assert "transitions_found: 2" in prompt
    assert "prediction_errors_found: 1" in prompt


def test_learning_context_summarizes_repeated_no_progress_failures(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    for tick in range(1, 4):
        memory_manager.record_working(
            WorkingMemoryEntry(
                tick=tick,
                entry_type="transition",
                payload={
                    "policy_id": "dummy:policy_seek_food",
                    "reward": 0.0,
                    "done": False,
                    "prev_facts": {"has_bucket": False},
                    "next_facts": {"has_bucket": False},
                    "inventory_progress": {"changed": False, "slot_delta": 0},
                    "bucket_progress": {"changed": False, "bucket_count_delta": 0},
                },
                priority=0.6,
            )
        )

    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )

    _, prompt = generator._build_arbitration_prompt(  # noqa: SLF001
        policies=generator.discover_policies(),
        goals=[],
        context={"step": 5, "drive_signals": {"signals": []}},
    )

    assert "no_progress_streak: 3" in prompt
    assert (
        "no_progress_summary: policy=dummy:policy_seek_food failed 3 times with 0 reward and no inventory/bucket change"
        in prompt
    )


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


def test_goal_directed_fallback_filters_stopword_overlap_noise(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_StopwordOverlapAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )

    proposal = generator.propose_action(
        goals="using or bucket",
        context={
            "step": 3,
            "drive_signals": {"signals": []},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_use_bucket"


def test_exhausted_skill_plan_cycles_phase_policies_round_robin(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_PlanCycleAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )

    skill_plan = {
        "head_policy_id": None,
        "remaining_policy_ids": [],
        "remaining_count": 0,
        "metadata": [
            {
                "phase_index": 0,
                "intent_tokens": ["bucket", "craft", "collect"],
                "selected_policy_id": "dummy:policy_craft",
            },
            {
                "phase_index": 1,
                "intent_tokens": ["cow", "milk", "harvest", "interact", "use"],
                "selected_policy_id": "dummy:policy_use",
            },
            {
                "phase_index": 2,
                "intent_tokens": ["explore", "move", "search"],
                "selected_policy_id": "dummy:policy_move_forward",
            },
        ],
    }

    selected_ids: List[str] = []
    for step in (3, 4, 5):
        proposal = generator.propose_action(
            goals=[],
            context={
                "step": step,
                "skill_plan": skill_plan,
                "drive_signals": {"signals": []},
            },
        )
        assert isinstance(proposal, ActionProposal)
        selected_ids.append(proposal.action_id)

    assert selected_ids == [
        "dummy:policy_craft",
        "dummy:policy_use",
        "dummy:policy_move_forward",
    ]


def test_periodic_reeval_is_sync_when_skill_plan_is_exhausted(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text='{"selected_index": 1, "rationale": "refresh strategy", "drive_conflict_detected": false, "confidence": 0.8}'
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
            "step": 10,
            "skill_plan": {
                "head_policy_id": None,
                "remaining_policy_ids": [],
                "remaining_count": 0,
                "metadata": [],
            },
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_explore"
    assert len(llm_client.requests) == 1


def test_should_call_llm_marks_exhausted_periodic_as_urgent(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={"llm_reeval_interval": 10},
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001

    policies = generator.discover_policies()
    gate = generator._should_call_llm(  # noqa: SLF001
        policies=policies,
        goals=[],
        context={
            "step": 10,
            "skill_plan": {
                "head_policy_id": None,
                "remaining_policy_ids": [],
                "remaining_count": 0,
                "metadata": [{"phase_index": 0, "intent_tokens": ["bucket"]}],
            },
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
        },
    )

    assert gate["should_call"] is True
    assert gate["urgent"] is True
    assert "periodic_reeval_plan_exhausted" in gate["reasons"]
    assert "periodic_reeval" not in gate["reasons"]


def test_should_call_llm_keeps_periodic_non_urgent_with_active_plan(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={"llm_reeval_interval": 10},
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001

    policies = generator.discover_policies()
    gate = generator._should_call_llm(  # noqa: SLF001
        policies=policies,
        goals=[],
        context={
            "step": 10,
            "skill_plan": {
                "head_policy_id": "dummy:policy_seek_food",
                "remaining_policy_ids": ["dummy:policy_seek_food"],
                "remaining_count": 1,
                "metadata": [{"phase_index": 0, "intent_tokens": ["collect"]}],
            },
            "drive_signals": {"signals": [{"channel_id": "hunger", "urgency": 0.4}]},
        },
    )

    assert gate["should_call"] is True
    assert gate["urgent"] is False
    assert gate["reasons"] == ["periodic_reeval"]


def test_goal_directed_fallback_uses_skill_plan_intent_tokens(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_StopwordOverlapAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )

    proposal = generator.propose_action(
        goals=[{"goal": "do or act"}],
        context={
            "step": 3,
            "skill_plan": {
                "head_policy_id": None,
                "remaining_policy_ids": [],
                "remaining_count": 0,
                "metadata": [{"phase_index": 0, "intent_tokens": ["bucket"]}],
            },
            "drive_signals": {"signals": []},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_use_bucket"


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


def test_llm_plan_is_stored_and_reused_between_calls(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text=(
            '{"reasoning":"plan","rationale":"explore first","confidence":0.8,'
            '"window_size":30,"interrupt_conditions":["health < 0.5"],'
            '"phases":[{"action":"dummy:policy_explore","until":"cow_visible == true OR steps_elapsed >= 3"},'
            '{"action":"dummy:policy_seek_food","until":"has_bucket == true"}]}'
        )
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

    first = generator.propose_action(
        goals=[{"goal_id": "task:harvest_milk"}],
        context={
            "step": 0,
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
            "world_facts": {"has_bucket": False, "nearby_cow": False, "position": {"x": 0.0, "z": 0.0}},
            "arousal_valence_state": {"arousal": 0.2},
        },
    )
    second = generator.propose_action(
        goals=[{"goal_id": "task:harvest_milk"}],
        context={
            "step": 1,
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
            "world_facts": {"has_bucket": False, "nearby_cow": False, "position": {"x": 1.0, "z": 0.0}},
            "arousal_valence_state": {"arousal": 0.2},
        },
    )

    assert isinstance(first, ActionProposal)
    assert isinstance(second, ActionProposal)
    assert first.action_id == "dummy:policy_explore"
    assert second.action_id == "dummy:policy_explore"
    assert len(llm_client.requests) == 1

    active_plan_entries = memory_manager.get_recent(1, entry_type="active_plan")
    assert active_plan_entries
    assert active_plan_entries[0].payload.get("status") == "active"


def test_plan_interrupt_condition_triggers_sync_replan(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text=(
            '{"reasoning":"plan","rationale":"explore first","confidence":0.8,'
            '"window_size":30,"interrupt_conditions":["health < 0.5"],'
            '"phases":[{"action":"dummy:policy_explore","until":"steps_elapsed >= 5"}]}'
        )
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

    first = generator.propose_action(
        goals=[{"goal_id": "task:harvest_milk"}],
        context={
            "step": 0,
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
            "world_facts": {"has_bucket": False, "nearby_cow": False, "position": {"x": 0.0, "z": 0.0}},
            "arousal_valence_state": {"arousal": 0.2},
        },
    )
    second = generator.propose_action(
        goals=[{"goal_id": "task:harvest_milk"}],
        context={
            "step": 1,
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.2, "current_value": 0.4}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
            "world_facts": {"has_bucket": False, "nearby_cow": False, "position": {"x": 1.0, "z": 0.0}},
            "arousal_valence_state": {"arousal": 0.2},
        },
    )

    assert isinstance(first, ActionProposal)
    assert isinstance(second, ActionProposal)
    assert len(llm_client.requests) == 2


def test_high_arousal_compresses_plan_window_for_replanning(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001
    generator._store_active_plan(  # noqa: SLF001
        {
            "plan_id": "plan_test",
            "status": "active",
            "created_step": 0,
            "window_size": 60,
            "interrupt_conditions": [],
            "phases": [
                {
                    "phase_index": 0,
                    "policy_id": "dummy:policy_explore",
                    "until": "",
                    "max_steps": None,
                }
            ],
            "current_phase_index": 0,
            "phase_started_step": 0,
            "history": [],
        },
        step=0,
    )

    gate = generator._should_call_llm(  # noqa: SLF001
        policies=generator.discover_policies(),
        goals=[],
        context={
            "step": 20,
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
            "arousal_valence_state": {"arousal": 0.9},
        },
    )

    assert gate["should_call"] is True
    assert "plan_window_elapsed" in gate["reasons"]


def test_zero_displacement_stall_triggers_urgent_replan(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001
    generator._store_active_plan(  # noqa: SLF001
        {
            "plan_id": "plan_stall",
            "status": "active",
            "created_step": 0,
            "window_size": 60,
            "interrupt_conditions": [],
            "phases": [{"phase_index": 0, "policy_id": "dummy:policy_explore", "until": ""}],
            "current_phase_index": 0,
            "phase_started_step": 0,
            "initial_plan_x": 10.0,
            "history": [],
        },
        step=0,
    )

    gate = generator._should_call_llm(  # noqa: SLF001
        policies=generator.discover_policies(),
        goals=[],
        context={
            "step": 25,
            "world_facts": {"position": {"x": 10.2, "z": 0.0}},
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
        },
    )

    assert gate["should_call"] is True
    assert gate["urgent"] is True
    assert "plan_stall_zero_displacement" in gate["reasons"]


def test_plan_window_elapsed_escalates_to_urgent_after_streak(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    generator = PolicyGenerator(
        adapter=_DriveAwareAdapter(),
        adapter_folder="dummy",
        memory_manager=memory_manager,
        goal_checker=_NeutralGoalChecker(),
        prediction_error_calculator=_NeutralPredictionErrorCalculator(),
        config={},
    )
    generator._last_goals_fingerprint = generator._goals_fingerprint([])  # noqa: SLF001
    generator._store_active_plan(  # noqa: SLF001
        {
            "plan_id": "plan_window",
            "status": "active",
            "created_step": 0,
            "window_size": 5,
            "interrupt_conditions": [],
            "phases": [{"phase_index": 0, "policy_id": "dummy:policy_explore", "until": ""}],
            "current_phase_index": 0,
            "phase_started_step": 0,
            "initial_plan_x": 0.0,
            "history": [],
        },
        step=0,
    )

    generator._plan_window_elapsed_nonurgent_streak = 2  # noqa: SLF001
    nonurgent_gate = generator._should_call_llm(  # noqa: SLF001
        policies=generator.discover_policies(),
        goals=[],
        context={
            "step": 12,
            "world_facts": {"position": {"x": 3.0, "z": 0.0}},
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
        },
    )
    assert nonurgent_gate["urgent"] is False
    assert "plan_window_elapsed" in nonurgent_gate["reasons"]

    generator._plan_window_elapsed_nonurgent_streak = 3  # noqa: SLF001
    urgent_gate = generator._should_call_llm(  # noqa: SLF001
        policies=generator.discover_policies(),
        goals=[],
        context={
            "step": 13,
            "world_facts": {"position": {"x": 3.0, "z": 0.0}},
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
        },
    )
    assert urgent_gate["urgent"] is True
    assert "plan_window_elapsed_escalated" in urgent_gate["reasons"]


def test_movement_plan_window_is_clamped_and_prompt_has_guidance(tmp_path: Path) -> None:
    memory_manager = _memory_manager(tmp_path)
    llm_client = _StaticLLMClient(
        text=(
            '{"reasoning":"movement plan","rationale":"explore","confidence":0.7,'
            '"window_size":5,"interrupt_conditions":[],"phases":['
            '{"action":"dummy:policy_explore","until":"steps_elapsed >= 30"}]}'
        )
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
        goals=[{"goal_id": "task:harvest_milk"}],
        context={
            "step": 0,
            "drive_signals": {"signals": [{"channel_id": "health", "urgency": 0.1, "current_value": 0.9}]},
            "perceptual_prediction_error": {"aggregate_magnitude": 0.1},
            "world_facts": {"position": {"x": 0.0, "z": 0.0}, "nearby_cow": False, "has_bucket": False},
            "arousal_valence_state": {"arousal": 0.2},
        },
    )

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_id == "dummy:policy_explore"
    assert llm_client.requests
    system_prompt = llm_client.requests[0].messages[0].content
    assert "movement/exploration phases should usually use window_size >= 20" in system_prompt

    plan_entries = memory_manager.get_recent(1, entry_type="active_plan")
    assert plan_entries
    assert int(plan_entries[0].payload.get("window_size", 0)) >= 20
