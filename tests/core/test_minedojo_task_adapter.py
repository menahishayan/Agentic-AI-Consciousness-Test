from __future__ import annotations

from typing import Any, Dict, List

import pytest

from core.adapters.minedojo.env_adapter import MineDojoAdapter
from core.adapters.minedojo.remote_adapter import RemoteMineDojoAdapter
from core.adapters.minedojo.task_profiles import task_goal_payload


class _FakeActionSpace:
    nvec = [3, 3, 4, 11, 11, 8, 5, 9]

    def no_op(self) -> List[int]:
        return [0 for _ in self.nvec]

    def sample(self) -> List[int]:
        return [0 for _ in self.nvec]


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = _FakeActionSpace()

    def reset(self) -> Any:
        return None, {}

    def step(self, action: Any) -> Any:
        _ = action
        return None, 0.0, False, {}

    def close(self) -> None:
        return None


def _local_adapter() -> MineDojoAdapter:
    return MineDojoAdapter(
        _FakeEnv(),
        config={
            "task_id": "harvest_milk",
            "include_voyager_policies": False,
            "max_inventory_slots": 36,
        },
    )


def _remote_adapter_without_socket() -> RemoteMineDojoAdapter:
    adapter = RemoteMineDojoAdapter.__new__(RemoteMineDojoAdapter)
    adapter._config = {"task_id": "harvest_milk", "max_inventory_slots": 36}
    adapter._task_id = "harvest_milk"
    adapter._task_goal = task_goal_payload("harvest_milk")
    adapter._max_inventory_slots = 36
    adapter._last_info = {}
    adapter._nvec = [3, 3, 4, 25, 25, 8, 101, 9]
    adapter._noop_action = [0, 0, 0, 0, 0, 0, 0, 0]
    adapter._skill_plan_queue = []
    adapter._skill_plan_metadata = []
    adapter._last_skill_plan_signature = None
    adapter._skill_plan_exhausted = False
    adapter._policies = adapter._build_policies()
    return adapter


def test_local_resource_level_uses_inventory_slots_and_hunger() -> None:
    adapter = _local_adapter()

    empty_inventory: Dict[str, Any] = {"inventory": [{"name": "air", "quantity": 0} for _ in range(36)]}
    partial_inventory = {
        "inventory": [{"name": "dirt", "quantity": 1} for _ in range(18)]
        + [{"name": "air", "quantity": 0} for _ in range(18)]
    }
    full_inventory = {"inventory": [{"name": "dirt", "quantity": 1} for _ in range(36)]}

    empty_score = adapter.estimate_resource_level(
        hunger=0.5,
        lighting={},
        nearby={},
        inventory_state=empty_inventory,
    )
    partial_score = adapter.estimate_resource_level(
        hunger=0.5,
        lighting={},
        nearby={},
        inventory_state=partial_inventory,
    )
    full_score = adapter.estimate_resource_level(
        hunger=0.5,
        lighting={},
        nearby={},
        inventory_state=full_inventory,
    )
    fallback_score = adapter.estimate_resource_level(
        hunger=0.3,
        lighting={},
        nearby={},
        inventory_state={"inventory": "malformed"},
    )

    assert empty_score == pytest.approx(0.2)
    assert partial_score == pytest.approx(0.5)
    assert full_score == pytest.approx(0.8)
    assert fallback_score == pytest.approx(0.3)


def test_local_harvest_profile_drive_channels_and_policy_tags() -> None:
    adapter = _local_adapter()
    channels = adapter.get_drive_channels()

    resource_channel = next(channel for channel in channels if channel.id == "resource_level")
    assert resource_channel.suggested_action_tag == "harvest"

    policies = adapter.get_available_policies()
    attack = next(policy for policy in policies if policy.get("policy_id") == "minedojo:action_space:attack")
    assert "resource_level" not in set(attack.get("drive_tags", []))
    assert any(
        "resource_level" in set(policy.get("drive_tags", []))
        and "use" in set(policy.get("tags", []))
        for policy in policies
    )


def test_local_skill_plan_queue_builds_and_advances() -> None:
    adapter = _local_adapter()
    goals = adapter.get_task_goals()

    plan = adapter.get_skill_plan(goals=goals, context={})
    assert plan["head_policy_id"] is not None
    assert plan["remaining_count"] >= 1

    head = plan["head_policy_id"]
    adapter.notify_policy_selected(head, context={})
    updated = adapter.get_skill_plan(goals=goals, context={})
    assert updated["remaining_count"] == max(0, plan["remaining_count"] - 1)


def _first_phase_intent_tokens(plan: Dict[str, Any]) -> List[str]:
    metadata = plan.get("metadata")
    if not isinstance(metadata, list) or not metadata:
        return []
    first = metadata[0]
    if not isinstance(first, dict):
        return []
    raw_tokens = first.get("intent_tokens")
    if not isinstance(raw_tokens, list):
        return []
    return [str(token) for token in raw_tokens]


def test_local_harvest_milk_plan_reorders_by_world_facts() -> None:
    adapter = _local_adapter()
    goals = adapter.get_task_goals()

    craft_first = adapter.get_skill_plan(
        goals=goals,
        context={
            "world_facts": {
                "has_bucket": False,
                "nearby_crafting_table": True,
                "nearby_cow": False,
            }
        },
    )
    explore_first = adapter.get_skill_plan(
        goals=goals,
        context={
            "world_facts": {
                "has_bucket": False,
                "nearby_crafting_table": False,
                "nearby_cow": False,
            }
        },
    )
    use_first = adapter.get_skill_plan(
        goals=goals,
        context={
            "world_facts": {
                "has_bucket": True,
                "nearby_crafting_table": False,
                "nearby_cow": True,
            }
        },
    )

    assert _first_phase_intent_tokens(craft_first) == ["bucket", "craft", "collect"]
    assert _first_phase_intent_tokens(explore_first) == ["explore", "move", "search"]
    assert _first_phase_intent_tokens(use_first) == ["cow", "milk", "harvest", "interact", "use"]


def test_missing_cow_signal_defaults_to_not_nearby_for_local_plan() -> None:
    adapter = _local_adapter()
    goals = adapter.get_task_goals()

    plan = adapter.get_skill_plan(
        goals=goals,
        context={
            "world_facts": {
                "has_bucket": True,
                "nearby_crafting_table": False,
            }
        },
    )

    assert _first_phase_intent_tokens(plan) == ["explore", "move", "search"]


def test_remote_parity_resource_drive_and_tags_without_socket() -> None:
    adapter = _remote_adapter_without_socket()

    channels = adapter.get_drive_channels()
    resource_channel = next(channel for channel in channels if channel.id == "resource_level")
    assert resource_channel.suggested_action_tag == "harvest"

    policies = adapter.get_available_policies()
    attack = next(policy for policy in policies if policy.get("policy_id") == "minedojo:action_space:attack")
    assert "resource_level" not in set(attack.get("drive_tags", []))
    assert any(
        "resource_level" in set(policy.get("drive_tags", []))
        and "use" in set(policy.get("tags", []))
        for policy in policies
    )

    score = adapter.estimate_resource_level(
        hunger=0.4,
        inventory_state={"inventory": [{"name": "air", "quantity": 0}] * 36},
    )
    malformed = adapter.estimate_resource_level(
        hunger=0.4,
        inventory_state={"inventory": "bad"},
    )
    assert score == pytest.approx(0.16)
    assert malformed == pytest.approx(0.4)


def test_remote_harvest_milk_plan_reorders_and_regenerates_by_world_facts() -> None:
    adapter = _remote_adapter_without_socket()
    goals = adapter.get_task_goals()

    first = adapter.get_skill_plan(
        goals=goals,
        context={
            "world_facts": {
                "has_bucket": False,
                "nearby_crafting_table": True,
                "nearby_cow": False,
            }
        },
    )
    first_signature = adapter._last_skill_plan_signature  # noqa: SLF001

    second = adapter.get_skill_plan(
        goals=goals,
        context={
            "world_facts": {
                "has_bucket": True,
                "nearby_crafting_table": False,
                "nearby_cow": True,
            }
        },
    )
    second_signature = adapter._last_skill_plan_signature  # noqa: SLF001

    assert _first_phase_intent_tokens(first) == ["bucket", "craft", "collect"]
    assert _first_phase_intent_tokens(second) == ["cow", "milk", "harvest", "interact", "use"]
    assert first_signature != second_signature


def test_missing_cow_signal_defaults_to_not_nearby_for_remote_plan() -> None:
    adapter = _remote_adapter_without_socket()
    goals = adapter.get_task_goals()

    plan = adapter.get_skill_plan(
        goals=goals,
        context={
            "world_facts": {
                "has_bucket": True,
                "nearby_crafting_table": False,
            }
        },
    )

    assert _first_phase_intent_tokens(plan) == ["explore", "move", "search"]
