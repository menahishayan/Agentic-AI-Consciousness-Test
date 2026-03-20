from __future__ import annotations

import numpy as np
import pytest

from core.layers.interoceptive import (
    ArousalHomeostaticState as HomeostaticState,
    ArousalValenceConfig,
    ArousalValenceSystem,
    PredictionError,
)


class _StubBus:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _StubTracker:
    def __init__(self) -> None:
        self.records = []

    def record(self, snapshot) -> None:
        self.records.append(snapshot)


def _homeo(
    *,
    health: float,
    hunger: float,
    resource_level: float,
    threat_proximity: float,
    oxygen: float,
    tick: int,
) -> HomeostaticState:
    return HomeostaticState(
        health=health,
        hunger=hunger,
        resource_level=resource_level,
        threat_proximity=threat_proximity,
        oxygen=oxygen,
        tick=tick,
    )


def test_threat_spike_raises_arousal_and_survival_bias() -> None:
    system = ArousalValenceSystem()
    state = system.update(
        homeostatic_state=_homeo(
            health=0.2,
            hunger=0.1,
            resource_level=0.2,
            threat_proximity=0.9,
            oxygen=0.3,
            tick=1,
        ),
        prediction_error=PredictionError(magnitude=1.0, source="threat", tick=1),
    )

    assert state.arousal > 0.7
    assert state.valence < 0.0
    assert state.policy_bias.survival_weight > state.policy_bias.exploration_weight


def test_satiation_recovery_increases_valence() -> None:
    system = ArousalValenceSystem()
    low_hunger = system.update(
        homeostatic_state=_homeo(
            health=0.8,
            hunger=0.1,
            resource_level=0.7,
            threat_proximity=0.1,
            oxygen=0.8,
            tick=1,
        ),
        prediction_error=PredictionError(magnitude=0.1, source="visual", tick=1),
    )
    recovered = system.update(
        homeostatic_state=_homeo(
            health=0.8,
            hunger=0.9,
            resource_level=0.7,
            threat_proximity=0.1,
            oxygen=0.8,
            tick=2,
        ),
        prediction_error=PredictionError(magnitude=0.1, source="visual", tick=2),
    )

    assert (recovered.valence - low_hunger.valence) > 0.4


def test_temporal_decay_reduces_arousal() -> None:
    system = ArousalValenceSystem()
    system.update(
        homeostatic_state=_homeo(
            health=0.0,
            hunger=0.0,
            resource_level=0.0,
            threat_proximity=1.0,
            oxygen=0.0,
            tick=1,
        ),
        prediction_error=PredictionError(magnitude=1.0, source="threat", tick=1),
    )
    system.apply_decay(dt=30)
    state = system.get_current_state()

    assert state.arousal < 0.5
    assert state.arousal > system.config.resting_arousal


def test_learning_rate_mod_bounds() -> None:
    low_system = ArousalValenceSystem()
    low = low_system.update(
        homeostatic_state=_homeo(
            health=1.0,
            hunger=1.0,
            resource_level=1.0,
            threat_proximity=0.0,
            oxygen=1.0,
            tick=1,
        )
    )
    assert low.learning_rate_mod == pytest.approx(0.5, abs=1e-6)

    high_system = ArousalValenceSystem()
    high = high_system.update(
        homeostatic_state=_homeo(
            health=0.0,
            hunger=0.0,
            resource_level=0.0,
            threat_proximity=1.0,
            oxygen=0.0,
            tick=1,
        ),
        prediction_error=PredictionError(magnitude=1.0, source="threat", tick=1),
    )
    assert high.learning_rate_mod == pytest.approx(2.0, abs=1e-6)


def test_policy_bias_weights_always_sum_to_one() -> None:
    system = ArousalValenceSystem()
    rng = np.random.default_rng(seed=7)

    for tick, row in enumerate(rng.random((150, 6)), start=1):
        state = system.update(
            homeostatic_state=_homeo(
                health=float(row[0]),
                hunger=float(row[1]),
                resource_level=float(row[2]),
                threat_proximity=float(row[3]),
                oxygen=float(row[4]),
                tick=tick,
            ),
            prediction_error=PredictionError(
                magnitude=float(row[5]),
                source="visual",
                tick=tick,
            ),
        )
        total = state.policy_bias.survival_weight + state.policy_bias.exploration_weight
        assert total == pytest.approx(1.0, abs=1e-9)


def test_reset_restores_resting_arousal() -> None:
    config = ArousalValenceConfig(resting_arousal=0.2)
    system = ArousalValenceSystem(config=config)
    system.update(
        homeostatic_state=_homeo(
            health=0.0,
            hunger=0.0,
            resource_level=0.2,
            threat_proximity=1.0,
            oxygen=0.1,
            tick=1,
        ),
        prediction_error=PredictionError(magnitude=0.9, source="threat", tick=1),
    )

    system.reset()
    state = system.get_current_state()
    assert state.arousal == pytest.approx(config.resting_arousal, abs=1e-6)


def test_update_publishes_state_and_records_high_arousal() -> None:
    message_bus = _StubBus()
    tracker = _StubTracker()
    system = ArousalValenceSystem(message_bus=message_bus, self_state_tracker=tracker)
    state = system.update(
        homeostatic_state=_homeo(
            health=0.1,
            hunger=0.2,
            resource_level=0.2,
            threat_proximity=0.9,
            oxygen=0.2,
            tick=1,
        ),
        prediction_error=PredictionError(magnitude=1.0, source="threat", tick=1),
    )

    assert message_bus.messages
    message = message_bus.messages[-1]
    assert message.kind == "homeostatic.arousal_valence"
    assert isinstance(message.payload, dict)
    assert message.payload["arousal"] == pytest.approx(state.arousal, abs=1e-6)
    assert tracker.records
    assert tracker.records[-1]["phase"] == "arousal_valence_alert"
