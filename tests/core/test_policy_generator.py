"""Tests for PolicyGenerator LLM gating and fallback."""
import pytest
from core.layers.action_selection.PolicyGenerator import PolicyGenerator
from core.coordination.workspace import GlobalWorkspace
from core.models.signals import DriveChannel, DriveSignal, DriveSignalBatch, Goal


_POLICIES = [
    {"policy_id": "move_forward", "callable_name": "move_forward",
     "tags": ["navigation"], "drive_tags": ["energy"]},
    {"policy_id": "idle", "callable_name": "idle",
     "tags": ["rest"], "drive_tags": []},
    {"policy_id": "move_backward", "callable_name": "move_backward",
     "tags": ["avoidance"], "drive_tags": ["safety"]},
]

_GOALS = [Goal(goal_id="basic_food", description="find food", priority=1.0, task_id="basic_food")]


def _batch(urgency_map: dict) -> DriveSignalBatch:
    signals = [
        DriveSignal(
            channel_id=cid, current_value=0.3,
            setpoint=0.8, urgency=urg,
            suggested_action_tags=["navigation"]
        )
        for cid, urg in urgency_map.items()
    ]
    return DriveSignalBatch(signals=signals)


def test_urgency_fallback_selects_matching_policy():
    gen = PolicyGenerator(llm_client=None, config={})
    ws = GlobalWorkspace()
    context = {
        "drive_batch": _batch({"energy": 0.9}),
        "pe_batch": None,
        "arousal_valence": None,
        "policies": _POLICIES,
    }
    result = gen.propose_action(_POLICIES, _GOALS, context, ws, step=0)
    # High energy urgency should select move_forward (has drive_tag=energy)
    assert result == "move_forward"


def test_no_llm_always_uses_fallback():
    gen = PolicyGenerator(llm_client=None, config={})
    ws = GlobalWorkspace()
    context = {
        "drive_batch": _batch({"energy": 0.95, "safety": 0.95}),
        "pe_batch": None,
        "arousal_valence": None,
        "policies": _POLICIES,
    }
    result = gen.propose_action(_POLICIES, _GOALS, context, ws, step=0)
    assert result in {p["policy_id"] for p in _POLICIES}


def test_single_policy_always_selected():
    gen = PolicyGenerator(llm_client=None, config={})
    ws = GlobalWorkspace()
    result = gen.propose_action(
        [_POLICIES[0]], _GOALS, {"drive_batch": None, "pe_batch": None, "policies": [_POLICIES[0]]},
        ws, step=0
    )
    assert result == "move_forward"


def test_empty_policies_returns_none():
    gen = PolicyGenerator(llm_client=None, config={})
    ws = GlobalWorkspace()
    result = gen.propose_action([], _GOALS, {"drive_batch": None, "pe_batch": None, "policies": []}, ws, step=0)
    assert result is None


class _MockLLM:
    """Minimal mock so _should_call_llm doesn't short-circuit."""
    pass


def test_trigger_conditions_drive_conflict():
    gen = PolicyGenerator(llm_client=_MockLLM(), config={"policy_generator": {"llm_conflict_threshold": 0.8}})
    goals = _GOALS
    context = {
        "drive_batch": _batch({"energy": 0.9, "safety": 0.9}),
        "pe_batch": None,
        "policies": _POLICIES,
    }
    trigger, reason = gen._should_call_llm(goals, context, step=5)
    assert trigger == "urgent"
    assert reason == "drive_conflict"


def test_trigger_conditions_pe_streak():
    gen = PolicyGenerator(llm_client=_MockLLM(), config={"policy_generator": {"pe_streak_threshold": 3}})
    gen._pe_streak = 4
    trigger, reason = gen._should_call_llm(_GOALS, {"drive_batch": None, "pe_batch": None, "policies": _POLICIES}, step=5)
    assert trigger == "urgent"
    assert reason == "pe_streak"


def test_result_published_to_workspace():
    gen = PolicyGenerator(llm_client=None, config={})
    ws = GlobalWorkspace()
    context = {"drive_batch": None, "pe_batch": None, "arousal_valence": None, "policies": _POLICIES}
    gen.propose_action(_POLICIES, _GOALS, context, ws, step=0)
    msgs = ws.get_by_kind("policy_proposal")
    assert len(msgs) == 1
    assert msgs[0].payload["selected"] in {p["policy_id"] for p in _POLICIES}
