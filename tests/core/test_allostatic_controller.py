from __future__ import annotations

from core.layers.interoceptive import AllostaticController
from homeostatic.allostatic_controller import (
    AllostaticConfig,
    DriveChannel,
    HomeostaticHistory,
    HomeostaticState,
)


def _history() -> HomeostaticHistory:
    channel = DriveChannel(
        id="hunger",
        setpoint=0.8,
        critical_threshold=0.2,
        irreversible=False,
        recovery_cost_ticks=20,
        suggested_action_tag="eat",
    )
    snapshots = [
        HomeostaticState(values={"hunger": 0.4}, tick=2, context_hash="zone"),
        HomeostaticState(values={"hunger": 0.5}, tick=1, context_hash="zone"),
        HomeostaticState(values={"hunger": 0.6}, tick=0, context_hash="zone"),
    ]
    return HomeostaticHistory(snapshots=snapshots, channels=[channel], tick=2)


def test_interoceptive_allostatic_controller_reexports_drive_based_controller() -> None:
    controller = AllostaticController(config=AllostaticConfig(), channels=_history().channels)
    output = controller.update(_history(), area_id="zone")
    assert output.signals
    assert output.signals[0].channel_id == "hunger"
    assert output.signals[0].suggested_action_tag == "eat"


def test_drive_controller_update_output_shape() -> None:
    controller = AllostaticController(config=AllostaticConfig(), channels=_history().channels)
    output = controller.update(_history(), area_id="zone")
    assert output.highest_urgency >= 0.0
    assert output.tick == 2
