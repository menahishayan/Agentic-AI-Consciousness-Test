from __future__ import annotations

import pytest

from core.layers.predictive import WorldModelGenerator


def _obs(
    *,
    health: float = 0.8,
    hunger: float = 0.7,
    resource_level: float = 0.6,
    oxygen: float = 0.9,
    threat_proximity: float = 0.1,
    entity_density: float = 0.2,
    terrain_novelty: float = 0.3,
) -> dict[str, float]:
    return {
        "health": health,
        "hunger": hunger,
        "resource_level": resource_level,
        "oxygen": oxygen,
        "threat_proximity": threat_proximity,
        "entity_density": entity_density,
        "terrain_novelty": terrain_novelty,
    }


def test_zero_initialized_model_predicts_no_change() -> None:
    model = WorldModelGenerator(alpha=0.1, min_precision=0.3, confidence_threshold=10)
    observation = _obs(health=0.65, resource_level=0.4)

    predicted = model.predict(observation, "dummy:policy_noop")

    assert predicted["health"] == pytest.approx(observation["health"], abs=1e-9)
    assert predicted["resource_level"] == pytest.approx(observation["resource_level"], abs=1e-9)
    assert model.confidence("dummy:policy_noop", "health") == pytest.approx(0.3, abs=1e-9)


def test_ema_delta_converges_for_repeated_action_channel_transition() -> None:
    model = WorldModelGenerator(alpha=0.2, min_precision=0.3, confidence_threshold=10)
    previous = _obs(health=0.8)
    current = _obs(health=0.6)

    for _ in range(25):
        model.update(previous, "dummy:policy_take_damage", current)

    predicted = model.predict(previous, "dummy:policy_take_damage")
    assert predicted["health"] == pytest.approx(0.6, abs=0.02)


def test_different_actions_learn_different_channel_deltas() -> None:
    model = WorldModelGenerator(alpha=0.3, min_precision=0.3, confidence_threshold=10)
    previous = _obs(resource_level=0.5)
    gain = _obs(resource_level=0.7)
    loss = _obs(resource_level=0.3)

    for _ in range(15):
        model.update(previous, "dummy:policy_gather", gain)
        model.update(previous, "dummy:policy_attack", loss)

    gather_pred = model.predict(previous, "dummy:policy_gather")
    attack_pred = model.predict(previous, "dummy:policy_attack")

    assert gather_pred["resource_level"] > attack_pred["resource_level"]
    assert gather_pred["resource_level"] == pytest.approx(0.7, abs=0.03)
    assert attack_pred["resource_level"] == pytest.approx(0.3, abs=0.03)


def test_confidence_scales_with_observation_count_and_threshold() -> None:
    model = WorldModelGenerator(alpha=0.1, min_precision=0.3, confidence_threshold=5)
    previous = _obs()
    current = _obs(health=0.7)

    assert model.confidence("dummy:policy_noop", "health") == pytest.approx(0.3, abs=1e-9)

    model.update(previous, "dummy:policy_noop", current)
    assert model.confidence("dummy:policy_noop", "health") == pytest.approx(0.3, abs=1e-9)

    model.update(previous, "dummy:policy_noop", current)
    assert model.confidence("dummy:policy_noop", "health") == pytest.approx(0.4, abs=1e-9)

    for _ in range(3):
        model.update(previous, "dummy:policy_noop", current)

    assert model.confidence("dummy:policy_noop", "health") == pytest.approx(1.0, abs=1e-9)
    assert model.confidence("dummy:policy_noop", "hunger") == pytest.approx(1.0, abs=1e-9)
