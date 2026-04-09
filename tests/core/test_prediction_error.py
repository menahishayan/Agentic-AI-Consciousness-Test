"""Tests for WorldModelGenerator and PredictionErrorCalculator."""
import pytest
from core.layers.predictive.WorldModelGenerator import WorldModelGenerator
from core.layers.predictive.PredictionErrorCalculator import PredictionErrorCalculator
from core.coordination.workspace import GlobalWorkspace
from core.models.state import AgentState, HomeostasisState, ResourceState, PerceptionState, PositionState


def _make_state(health=0.8, saturation=0.7, step=0):
    return AgentState(
        homeostasis=HomeostasisState(health=health, saturation=saturation, energy=0.75),
        resources=ResourceState(resource_level=0.6, threat_proximity=0.1),
        perception=PerceptionState(area_id="x0z0", terrain_novelty=0.2, entity_density=0.1),
        step=step,
    )


def test_world_model_predict_returns_channels():
    wm = WorldModelGenerator({})
    state = _make_state()
    predicted = wm.predict(state, "move_forward")
    assert "health" in predicted
    assert "saturation" in predicted
    assert all(0.0 <= v <= 1.0 for v in predicted.values())


def test_world_model_learns_from_transitions():
    wm = WorldModelGenerator({})
    # Simulate: move_forward always decreases health by 0.01
    for i in range(25):
        prev = _make_state(health=0.9 - i * 0.01, step=i)
        next_ = _make_state(health=0.89 - i * 0.01, step=i + 1)
        wm.update(prev, "move_forward", next_)

    delta = wm.get_expected_delta("move_forward", "health")
    assert delta < 0.0  # Should have learned health decreases


def test_world_model_precision_grows_with_observations():
    wm = WorldModelGenerator({})
    s1 = _make_state()
    s2 = _make_state(health=0.79)
    p_before = wm.get_precision("move_forward", "health")

    for _ in range(25):
        wm.update(s1, "move_forward", s2)

    p_after = wm.get_precision("move_forward", "health")
    assert p_after > p_before


def test_pe_calc_publishes_to_workspace():
    wm = WorldModelGenerator({})
    calc = PredictionErrorCalculator(wm, {})
    ws = GlobalWorkspace()
    state = _make_state()
    predicted = wm.predict(state, None)
    calc.update(predicted, state, None, ws, step=0)
    msgs = ws.get_by_kind("prediction_error")
    assert len(msgs) == 1


def test_pe_batch_has_correct_channels():
    wm = WorldModelGenerator({})
    calc = PredictionErrorCalculator(wm, {})
    ws = GlobalWorkspace()
    state = _make_state()
    predicted = wm.predict(state, None)
    batch = calc.update(predicted, state, None, ws, step=0)
    channels = {e.channel for e in batch.errors}
    assert "health" in channels
    assert "saturation" in channels
