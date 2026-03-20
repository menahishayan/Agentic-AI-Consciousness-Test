from __future__ import annotations

from typing import Dict, List

import pytest

from homeostatic.allostatic_controller import (
    AllostaticConfig,
    AllostaticController,
    DriveChannel,
    HomeostaticHistory,
    HomeostaticState,
)


class _StubSelfStateMemory:
    def __init__(self, rates: Dict[str, float] | None = None) -> None:
        self.rates = dict(rates or {})

    def get_depletion_rate(self, channel_id: str, context_hash: str):
        _ = context_hash
        return self.rates.get(channel_id)


class _StubPEHistory:
    def __init__(self, priors: Dict[str, float] | None = None) -> None:
        self.priors = dict(priors or {})

    def get_area_threat_prior(self, area_id: str) -> float:
        return float(self.priors.get(area_id, 0.0))


class _StubBus:
    def __init__(self) -> None:
        self.messages: List[object] = []

    def publish(self, message) -> None:
        self.messages.append(message)


def _channel(
    cid: str,
    *,
    threshold: float = 0.3,
    irreversible: bool = False,
    recovery_cost_ticks: int = 10,
    tag: str = "maintain",
) -> DriveChannel:
    return DriveChannel(
        id=cid,
        setpoint=0.8,
        critical_threshold=threshold,
        irreversible=irreversible,
        recovery_cost_ticks=recovery_cost_ticks,
        suggested_action_tag=tag,
    )


def _history(
    values_by_channel_oldest_first: Dict[str, List[float]],
    channels: List[DriveChannel],
    *,
    context_hash: str = "ctx",
) -> HomeostaticHistory:
    length = max(len(series) for series in values_by_channel_oldest_first.values())
    snapshots_oldest_first: List[HomeostaticState] = []
    for tick in range(length):
        values = {}
        for channel_id, series in values_by_channel_oldest_first.items():
            idx = min(tick, len(series) - 1)
            values[channel_id] = float(series[idx])
        snapshots_oldest_first.append(
            HomeostaticState(values=values, tick=tick, context_hash=context_hash)
        )
    snapshots_newest_first = list(reversed(snapshots_oldest_first))
    return HomeostaticHistory(snapshots=snapshots_newest_first, channels=channels, tick=length - 1)


def _signal_by_channel(output) -> Dict[str, object]:
    return {signal.channel_id: signal for signal in output.signals}


def test_stable_channel_emits_no_signal() -> None:
    channel = _channel("drive_a")
    values = [0.8] * 20
    history = _history({"drive_a": values}, [channel])
    controller = AllostaticController(AllostaticConfig(), [channel])

    output = controller.update(history)

    assert output.signals == []
    assert output.highest_urgency == pytest.approx(0.0, abs=1e-9)


def test_declining_channel_within_horizon_emits_signal_with_correct_projection() -> None:
    channel = _channel("drive_decline")
    values = [0.9 - 0.02 * i for i in range(20)]  # oldest -> newest
    history = _history({"drive_decline": values}, [channel])
    controller = AllostaticController(AllostaticConfig(planning_horizon=50), [channel])

    output = controller.update(history)

    assert output.signals
    signal = output.signals[0]
    expected_ttc = (values[-1] - channel.critical_threshold) / 0.02
    assert abs(signal.ticks_to_critical - expected_ttc) / expected_ttc < 0.10


def test_declining_channel_outside_horizon_emits_no_signal() -> None:
    channel = _channel("drive_slow")
    values = [0.9 - 0.002 * i for i in range(20)]
    history = _history({"drive_slow": values}, [channel])
    controller = AllostaticController(AllostaticConfig(planning_horizon=50), [channel])

    output = controller.update(history)

    assert output.signals == []


def test_irreversibility_bonus_increases_urgency() -> None:
    reversible = _channel("a", irreversible=False)
    irreversible = _channel("b", irreversible=True)
    values = [0.9 - 0.02 * i for i in range(20)]
    history = _history({"a": values, "b": values}, [reversible, irreversible])
    controller = AllostaticController(AllostaticConfig(), [reversible, irreversible])

    output = controller.update(history)
    signals = _signal_by_channel(output)

    assert signals["b"].urgency > signals["a"].urgency


def test_memory_correction_applies_and_raises_confidence() -> None:
    channel = _channel("drive_mem")
    values = [0.9 - 0.02 * i for i in range(20)]
    history = _history({"drive_mem": values}, [channel], context_hash="same_ctx")

    base_controller = AllostaticController(AllostaticConfig(), [channel], self_state_memory=None)
    mem_controller = AllostaticController(
        AllostaticConfig(),
        [channel],
        self_state_memory=_StubSelfStateMemory({"drive_mem": -0.05}),
    )

    base = base_controller.update(history).signals[0]
    corrected = mem_controller.update(history).signals[0]

    assert corrected.projection_confidence == pytest.approx(0.75, abs=1e-9)
    assert corrected.ticks_to_critical < base.ticks_to_critical


def test_memory_absent_degrades_gracefully() -> None:
    channel = _channel("drive_no_mem")
    values = [0.65 - 0.03 * i for i in range(5)]
    history = _history({"drive_no_mem": values}, [channel])

    controller = AllostaticController(AllostaticConfig(history_window=20), [channel], self_state_memory=None)
    output = controller.update(history)

    assert output.signals
    assert output.signals[0].projection_confidence == pytest.approx(0.5, abs=1e-9)


def test_threat_prior_compresses_ttc_for_irreversible_channel() -> None:
    channel = _channel("drive_threat", irreversible=True)
    values = [0.9 - 0.02 * i for i in range(20)]
    history = _history({"drive_threat": values}, [channel])

    low = AllostaticController(
        AllostaticConfig(threat_prior_weight=0.3),
        [channel],
        pe_history=_StubPEHistory({"zone": 0.0}),
    ).update(history, area_id="zone").signals[0]

    high = AllostaticController(
        AllostaticConfig(threat_prior_weight=0.3),
        [channel],
        pe_history=_StubPEHistory({"zone": 1.0}),
    ).update(history, area_id="zone").signals[0]

    assert high.ticks_to_critical < low.ticks_to_critical


def test_prioritisation_order_is_descending_by_urgency() -> None:
    channels = [_channel("a"), _channel("b", irreversible=True), _channel("c")]
    history = _history(
        {
            "a": [0.9 - 0.015 * i for i in range(20)],
            "b": [0.9 - 0.02 * i for i in range(20)],
            "c": [0.9 - 0.01 * i for i in range(20)],
        },
        channels,
    )
    controller = AllostaticController(AllostaticConfig(), channels)

    output = controller.update(history)
    urgencies = [s.urgency for s in output.signals]

    assert urgencies == sorted(urgencies, reverse=True)


def test_tiebreak_uses_recovery_cost_when_policy_traces_unknown() -> None:
    low_cost = _channel("low", recovery_cost_ticks=5)
    high_cost = _channel("high", recovery_cost_ticks=20)
    values = [0.9 - 0.02 * i for i in range(20)]
    history = _history({"low": values, "high": values}, [low_cost, high_cost])

    controller = AllostaticController(
        AllostaticConfig(recovery_weight_factor=0.0, urgency_tie_epsilon=0.2),
        [low_cost, high_cost],
    )
    output = controller.update(history)

    assert output.signals
    assert output.signals[0].channel_id == "high"


def test_planning_horizon_boundary_behavior() -> None:
    config = AllostaticConfig(planning_horizon=50)
    exact = _channel("exact", threshold=0.2)
    outside = _channel("outside", threshold=0.2)

    exact_values = [0.89 - 0.01 * i for i in range(20)]   # newest = 0.70 => ttc 50
    outside_values = [0.90 - 0.01 * i for i in range(20)] # newest = 0.71 => ttc 51
    history = _history({"exact": exact_values, "outside": outside_values}, [exact, outside])

    output = AllostaticController(config, [exact, outside]).update(history)
    ids = [signal.channel_id for signal in output.signals]

    assert "exact" in ids
    assert "outside" not in ids


def test_reset_clears_stale_state() -> None:
    channel = _channel("drive_reset")
    declining = _history({"drive_reset": [0.9 - 0.02 * i for i in range(20)]}, [channel])
    stable = _history({"drive_reset": [0.8 for _ in range(20)]}, [channel])

    controller = AllostaticController(AllostaticConfig(), [channel])
    first = controller.update(declining)
    assert first.signals

    controller.reset()
    second = controller.update(stable)
    assert second.signals == []


def test_update_publishes_drive_signals_and_preserves_action_tag() -> None:
    bus = _StubBus()
    channel = _channel("drive_pub", tag="seek_resource")
    history = _history({"drive_pub": [0.9 - 0.02 * i for i in range(20)]}, [channel])
    controller = AllostaticController(AllostaticConfig(), [channel], message_bus=bus)

    output = controller.update(history)

    assert output.signals
    assert output.signals[0].suggested_action_tag == "seek_resource"

    assert bus.messages
    message = bus.messages[-1]
    assert getattr(message, "kind", None) == "homeostatic.drive_signals"
