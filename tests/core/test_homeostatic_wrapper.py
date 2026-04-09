"""Tests for HomeostaticWrapper."""
import pytest
from core.adapters.animalai.homeostatic_wrapper import HomeostaticWrapper


def _cfg(**kwargs):
    cfg = {
        "health_depletion_rate": 0.01,
        "saturation_depletion_rate": 0.02,
        "food_health_restore": 0.3,
        "food_saturation_restore": 0.5,
        "hazard_health_penalty": 0.2,
        "death_threshold": 0.0,
    }
    cfg.update(kwargs)
    return {"homeostatic": cfg}


def test_passive_depletion():
    w = HomeostaticWrapper(_cfg())
    w.step(0.0, False)
    s = w.get_state()
    assert s["health"] == pytest.approx(0.99, abs=1e-4)
    assert s["saturation"] == pytest.approx(0.98, abs=1e-4)


def test_food_restores():
    w = HomeostaticWrapper(_cfg())
    # Deplete first
    for _ in range(50):
        w.step(0.0, False)
    pre = w.health
    w.step(1.0, False)
    assert w.health > pre
    assert w.saturation > 0.0


def test_hazard_penalizes():
    w = HomeostaticWrapper(_cfg())
    pre = w.health
    w.step(-1.0, False)
    assert w.health < pre


def test_death_at_zero():
    w = HomeostaticWrapper(_cfg(health_depletion_rate=1.1))
    w.step(0.0, False)
    assert w.health == 0.0
    assert not w.is_alive


def test_reset_restores():
    w = HomeostaticWrapper(_cfg())
    for _ in range(100):
        w.step(0.0, False)
    w.reset()
    s = w.get_state()
    assert s["health"] == pytest.approx(1.0)
    assert s["saturation"] == pytest.approx(1.0)


def test_energy_composite():
    w = HomeostaticWrapper(_cfg())
    w.step(0.0, False)
    s = w.get_state()
    expected_energy = s["saturation"] * 0.7 + s["health"] * 0.3
    assert s["energy"] == pytest.approx(expected_energy, abs=1e-4)
