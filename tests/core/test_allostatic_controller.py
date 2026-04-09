"""Tests for AllostaticController."""
import pytest
from core.layers.interoceptive.AllostaticController import AllostaticController
from core.coordination.workspace import GlobalWorkspace
from core.models.signals import DriveChannel


def _make_channels():
    return [
        DriveChannel("health", setpoint=0.8, critical_threshold=0.2,
                     recovery_cost_ticks=100, suggested_action_tags=["navigation"]),
        DriveChannel("saturation", setpoint=0.7, critical_threshold=0.25,
                     recovery_cost_ticks=150, suggested_action_tags=["exploration"]),
    ]


def test_urgency_zero_at_setpoint():
    ctrl = AllostaticController(_make_channels(), {})
    ws = GlobalWorkspace()
    batch = ctrl.update({"health": 0.85, "saturation": 0.75}, ws, step=0)
    health_sig = next(s for s in batch.signals if s.channel_id == "health")
    assert health_sig.urgency == pytest.approx(0.0, abs=0.05)


def test_urgency_high_at_critical():
    ctrl = AllostaticController(_make_channels(), {})
    ws = GlobalWorkspace()
    batch = ctrl.update({"health": 0.1, "saturation": 0.1}, ws, step=0)
    health_sig = next(s for s in batch.signals if s.channel_id == "health")
    assert health_sig.urgency > 0.7


def test_batch_published_to_workspace():
    ctrl = AllostaticController(_make_channels(), {})
    ws = GlobalWorkspace()
    ctrl.update({"health": 0.5, "saturation": 0.5}, ws, step=1)
    msgs = ws.get_by_kind("drive_signal")
    assert len(msgs) == 1


def test_dominant_channel_is_highest_urgency():
    ctrl = AllostaticController(_make_channels(), {})
    ws = GlobalWorkspace()
    batch = ctrl.update({"health": 0.1, "saturation": 0.9}, ws, step=0)
    assert batch.dominant_channel == "health"


def test_depletion_rate_from_history():
    ctrl = AllostaticController(_make_channels(), {})
    ws = GlobalWorkspace()
    # Feed declining health values
    for i in range(15):
        ctrl.update({"health": 1.0 - i * 0.05, "saturation": 0.8}, ws, step=i)
    rate = ctrl._estimate_depletion_rate("health")
    assert rate is not None and rate > 0.0
