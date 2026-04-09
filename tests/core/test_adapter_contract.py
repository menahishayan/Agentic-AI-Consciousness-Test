"""Tests for the IoT stub adapter (no AnimalAI required) and adapter contract."""
import pytest
from core.adapters.base import AbstractEnvironmentAdapter
from core.adapters.iot.env_adapter import IoTStubAdapter, create_adapter
from core.adapters.loader import build_adapter


def test_iot_adapter_is_abstract_subclass():
    adapter = IoTStubAdapter({})
    assert isinstance(adapter, AbstractEnvironmentAdapter)


def test_iot_reset_returns_agent_state():
    from core.models.state import AgentState
    adapter = IoTStubAdapter({})
    state = adapter.reset()
    assert isinstance(state, AgentState)


def test_iot_step_returns_state_and_done():
    from core.models.state import AgentState
    adapter = IoTStubAdapter({})
    adapter.reset()
    state, done = adapter.step("idle")
    assert isinstance(state, AgentState)
    assert isinstance(done, bool)


def test_iot_get_drive_channels():
    adapter = IoTStubAdapter({})
    channels = adapter.get_drive_channels()
    assert len(channels) > 0
    for ch in channels:
        assert ch.channel_id
        assert 0.0 <= ch.setpoint <= 1.0
        assert 0.0 <= ch.critical_threshold <= 1.0


def test_iot_get_task_goal():
    adapter = IoTStubAdapter({})
    goal = adapter.get_task_goal()
    assert "description" in goal
    assert "priority" in goal
    assert "task_id" in goal


def test_iot_get_available_policies():
    adapter = IoTStubAdapter({})
    policies = adapter.get_available_policies()
    assert len(policies) > 0
    for p in policies:
        assert "policy_id" in p
        assert "callable_name" in p
        assert "tags" in p
        assert len(p["tags"]) > 0


def test_loader_builds_iot():
    adapter = build_adapter("iot", {})
    assert isinstance(adapter, AbstractEnvironmentAdapter)


def test_iot_estimate_methods_return_floats():
    adapter = IoTStubAdapter({})
    state = adapter.reset()
    assert 0.0 <= adapter.estimate_resource_level(state) <= 1.0
    assert 0.0 <= adapter.estimate_threat_proximity(state) <= 1.0
    assert 0.0 <= adapter.estimate_entity_density(state) <= 1.0
    assert 0.0 <= adapter.estimate_terrain_novelty(state) <= 1.0
    assert isinstance(adapter.build_area_id(state), str)
