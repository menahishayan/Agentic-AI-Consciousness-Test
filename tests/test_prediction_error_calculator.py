from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pytest

from perceptual.prediction_error_calculator import (
    ObservationSnapshot,
    PEConfig,
    PredictionErrorCalculator,
)


class _StubBus:
    def __init__(self) -> None:
        self.messages: List[object] = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _StubHistory:
    def __init__(self, familiarities: Dict[str, float], fail_lookup: bool = False) -> None:
        self.familiarities = dict(familiarities)
        self.fail_lookup = fail_lookup
        self.records: List[object] = []

    def get_area_familiarity(self, area_id: str) -> float:
        if self.fail_lookup:
            raise RuntimeError("lookup failure")
        return float(self.familiarities.get(area_id, 0.5))

    def record(self, area_id: str, error: object) -> None:
        self.records.append((area_id, error))


def _obs(
    *,
    tick: int,
    area_id: str = "area_a",
    health: float = 0.8,
    hunger: float = 0.8,
    resource_level: float = 0.8,
    oxygen: float = 0.8,
    threat_proximity: float = 0.1,
    entity_density: float = 0.0,
    terrain_novelty: float = 0.1,
    last_action: str = "noop",
) -> ObservationSnapshot:
    return ObservationSnapshot(
        health=health,
        hunger=hunger,
        resource_level=resource_level,
        oxygen=oxygen,
        threat_proximity=threat_proximity,
        entity_density=entity_density,
        terrain_novelty=terrain_novelty,
        area_id=area_id,
        last_action=last_action,
        tick=tick,
    )


def _error_by_channel(batch) -> Dict[str, object]:
    return {err.channel: err for err in batch.errors}


def test_no_prediction_at_tick_zero_returns_empty_batch() -> None:
    calc = PredictionErrorCalculator(config=PEConfig())
    batch = calc.update(_obs(tick=0))

    assert batch.errors == []
    assert batch.aggregate_magnitude == pytest.approx(0.0, abs=1e-9)


def test_stable_observations_converge_to_low_prediction_error() -> None:
    calc = PredictionErrorCalculator(config=PEConfig())
    calc.update(_obs(tick=0))

    batch = None
    for tick in range(1, 21):
        batch = calc.update(_obs(tick=tick))

    assert batch is not None
    assert batch.aggregate_magnitude < 0.1


def test_sudden_mob_spawn_produces_large_entity_density_raw_pe() -> None:
    calc = PredictionErrorCalculator(config=PEConfig())
    calc.update(_obs(tick=0, entity_density=0.0))
    for tick in range(1, 12):
        calc.update(_obs(tick=tick, entity_density=0.0))

    batch = calc.update(_obs(tick=12, entity_density=0.8))
    err = _error_by_channel(batch)["entity_density"]
    assert err.raw_magnitude > 0.5


def test_health_drop_makes_proprioceptive_source_dominant() -> None:
    calc = PredictionErrorCalculator(config=PEConfig())
    calc.update(_obs(tick=0, health=0.9))
    for tick in range(1, 8):
        calc.update(_obs(tick=tick, health=0.9))

    batch = calc.update(_obs(tick=8, health=0.5))
    assert batch.dominant_source == "proprioceptive"


def test_precision_scaling_familiar_area_amplifies_weighted_pe() -> None:
    familiar_history = _StubHistory({"familiar": 1.0})
    novel_history = _StubHistory({"novel": 0.0})

    familiar_calc = PredictionErrorCalculator(config=PEConfig(), pe_history=familiar_history)
    novel_calc = PredictionErrorCalculator(config=PEConfig(), pe_history=novel_history)

    familiar_calc.update(_obs(tick=0, area_id="familiar", entity_density=0.0))
    novel_calc.update(_obs(tick=0, area_id="novel", entity_density=0.0))

    familiar_batch = familiar_calc.update(_obs(tick=1, area_id="familiar", entity_density=0.6))
    novel_batch = novel_calc.update(_obs(tick=1, area_id="novel", entity_density=0.6))

    familiar_err = _error_by_channel(familiar_batch)["entity_density"]
    novel_err = _error_by_channel(novel_batch)["entity_density"]

    assert familiar_err.raw_magnitude == pytest.approx(novel_err.raw_magnitude, abs=1e-9)
    assert familiar_err.magnitude > novel_err.magnitude


def test_precision_fallback_when_history_unavailable() -> None:
    calc = PredictionErrorCalculator(config=PEConfig(default_precision=0.5), pe_history=None)
    calc.update(_obs(tick=0, threat_proximity=0.0))

    batch = calc.update(_obs(tick=1, threat_proximity=0.7))
    assert batch.errors
    assert all(err.precision == pytest.approx(0.5, abs=1e-9) for err in batch.errors)


def test_baseline_variance_converges_under_noisy_stable_input() -> None:
    rng = np.random.default_rng(seed=3)
    calc = PredictionErrorCalculator(config=PEConfig(alpha=0.1, epsilon=0.01))
    variances: List[float] = []

    for tick in range(50):
        noise = float(rng.normal(0.0, 0.02))
        health = float(np.clip(0.6 + noise, 0.0, 1.0))
        calc.update(_obs(tick=tick, health=health))
        current_var = calc.get_variance().get("health")
        if current_var is not None:
            variances.append(float(current_var))

    assert len(variances) >= 20
    last_window = variances[-10:]
    assert np.isfinite(last_window).all()
    assert max(last_window) < 0.02
    assert (max(last_window) - min(last_window)) < 0.01


def test_dominant_source_identifies_highest_weighted_error_source() -> None:
    calc = PredictionErrorCalculator(config=PEConfig())
    calc.update(_obs(tick=0, threat_proximity=0.0, entity_density=0.0))

    batch = calc.update(_obs(tick=1, threat_proximity=1.0, entity_density=0.2))
    assert batch.dominant_source == "threat"


def test_reset_restarts_one_tick_delay_behavior() -> None:
    calc = PredictionErrorCalculator(config=PEConfig())
    calc.update(_obs(tick=0))
    batch = calc.update(_obs(tick=1, entity_density=0.4))
    assert batch.errors

    calc.reset()
    reset_batch = calc.update(_obs(tick=2, entity_density=0.1))
    assert reset_batch.errors == []
    assert reset_batch.aggregate_magnitude == pytest.approx(0.0, abs=1e-9)


def test_integration_publishes_batch_and_records_each_error() -> None:
    bus = _StubBus()
    history = _StubHistory({"area_a": 0.7})
    calc = PredictionErrorCalculator(config=PEConfig(), pe_history=history, message_bus=bus)

    calc.update(_obs(tick=0, area_id="area_a"))
    batch = calc.update(_obs(tick=1, area_id="area_a", entity_density=0.5))

    assert bus.messages
    message = bus.messages[-1]
    assert getattr(message, "kind", None) == "perceptual.prediction_error"

    payload = getattr(message, "payload", None)
    assert isinstance(payload, dict)
    assert payload["aggregate_magnitude"] == pytest.approx(batch.aggregate_magnitude, abs=1e-9)

    assert len(history.records) == len(batch.errors)
    assert all(area_id == "area_a" for area_id, _ in history.records)

    serialized = [asdict(err) for _, err in history.records]
    assert all("channel" in item for item in serialized)
