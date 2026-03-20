from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from core.layers.interoceptive import (
    DriveSignal,
    PrioritisedDriveSignals,
    VitalStateMonitor,
)
from core.memory import MemoryManager
from core.models.state import AgentState
from core.runtime.loop import AgentLoop


class _DriveAwareDummyAdapter:
    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> Any:
        self._step = 0
        return None, {
            "life": 20,
            "food": 18,
            "air": 300,
            "is_alive": True,
            "is_dead": False,
            "biome_name": "plains",
            "xpos": 0.0,
            "ypos": 64.0,
            "zpos": 0.0,
            "light_level": 10.0,
            "can_see_sky": True,
        }

    def step(self, action: Any) -> Any:
        _ = action
        self._step += 1
        life = max(0, 20 - self._step)
        return None, 1.0, False, {
            "life": life,
            "food": 18,
            "air": 300,
            "is_alive": life > 0,
            "is_dead": life <= 0,
            "biome_name": "plains",
            "xpos": float(self._step),
            "ypos": 64.0,
            "zpos": 0.0,
            "light_level": 10.0,
            "can_see_sky": True,
        }

    def close(self) -> None:
        return None

    def sample_action(self) -> str:
        return "noop"

    def get_available_vitals(self) -> List[str]:
        return ["life", "food", "air", "is_alive", "is_dead"]

    def get_available_policies(self) -> List[Dict[str, Any]]:
        return [
            {
                "policy_id": "dummy:policy_noop",
                "callable_name": "policy_noop",
                "description": "Stay stable and survive.",
                "tags": ["survival", "retreat", "eat"],
                "drive_tags": ["health", "hunger", "safety"],
            }
        ]

    def policy_noop(self) -> str:
        return "noop"

    def estimate_resource_level(
        self,
        *,
        hunger: float,
        lighting: Mapping[str, Any],
        nearby: Mapping[str, Any],
        inventory_state: Mapping[str, Any],
        state: Any = None,
        info: Any = None,
        obs: Any = None,
    ) -> float:
        _ = lighting
        _ = nearby
        _ = inventory_state
        _ = state
        _ = info
        _ = obs
        return max(0.0, min(1.0, float(hunger)))

    def estimate_threat_proximity(
        self,
        *,
        messages: List[Any],
        health: float,
        lighting: Mapping[str, Any],
        homeostasis: Mapping[str, Any],
        state: Any = None,
        info: Any = None,
        obs: Any = None,
    ) -> float:
        _ = messages
        _ = lighting
        _ = homeostasis
        _ = state
        _ = info
        _ = obs
        return max(0.0, min(1.0, 1.0 - float(health)))

    def build_area_id(
        self,
        *,
        state: Any = None,
        info: Any = None,
        obs: Any = None,
        step: Optional[int] = None,
    ) -> str:
        _ = state
        _ = obs
        _ = step
        info_map = info if isinstance(info, Mapping) else {}
        return f"{info_map.get('biome_name', 'unknown')}:0:0:4"

    def estimate_entity_density(self, *, state: Any = None, info: Any = None, obs: Any = None) -> float:
        _ = state
        _ = info
        _ = obs
        return 0.1 + 0.05 * self._step

    def estimate_terrain_novelty(self, *, state: Any = None, info: Any = None, obs: Any = None) -> float:
        _ = state
        _ = info
        _ = obs
        return 0.25


class _CountingMapper:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        raw_obs: Any,
        info: Any,
        vital_state_monitor: Optional[VitalStateMonitor] = None,
    ) -> AgentState:
        _ = raw_obs
        monitor = vital_state_monitor or VitalStateMonitor()
        payload = info if isinstance(info, Mapping) else {}
        monitor.update(payload)
        self.calls += 1
        return AgentState.from_info(dict(payload))


def _drive_signal(
    channel_id: str,
    *,
    urgency: float,
    projected_value: float = 0.5,
    tick: int = 0,
) -> DriveSignal:
    return DriveSignal(
        channel_id=channel_id,
        current_value=0.5,
        projected_value=projected_value,
        ticks_to_critical=10.0,
        urgency=urgency,
        projection_confidence=0.8,
        suggested_action_tag=channel_id,
        tick=tick,
    )


def test_step_zero_maps_observation_once_before_action() -> None:
    adapter = _DriveAwareDummyAdapter()
    mapper = _CountingMapper()
    loop = AgentLoop(
        adapter=adapter,
        observation_mapper=mapper,
        memory_manager=MemoryManager(),
        adapter_folder="dummy",
        policy_config={},
    )

    loop.run_step(0)

    # One pre-action mapping + one post-step mapping.
    assert mapper.calls == 2


def test_prediction_error_is_attributed_to_previous_selected_policy() -> None:
    adapter = _DriveAwareDummyAdapter()
    mapper = _CountingMapper()
    memory_manager = MemoryManager()
    loop = AgentLoop(
        adapter=adapter,
        observation_mapper=mapper,
        memory_manager=memory_manager,
        adapter_folder="dummy",
        policy_config={},
    )

    loop.run_step(0)
    loop.run_step(1)

    records = memory_manager.query_prediction_errors(policy_id="dummy:policy_noop", limit=100)
    assert records
    assert all(record.get("policy_id") == "dummy:policy_noop" for record in records)
    assert any(float(record.get("magnitude", 0.0)) >= 0.0 for record in records)

    snapshots = memory_manager.query({"target": "self_state", "phase": "arousal_valence_pre_action"})
    step_one = [
        snap for snap in snapshots if isinstance(snap, Mapping) and snap.get("step") == 1
    ]
    assert step_one
    arousal_payload = step_one[-1].get("arousal_valence")
    assert isinstance(arousal_payload, Mapping)
    assert float(arousal_payload.get("arousal", 0.0)) >= 0.0


def test_drive_allostasis_uses_rolling_history_and_emits_signals() -> None:
    adapter = _DriveAwareDummyAdapter()
    mapper = _CountingMapper()
    memory_manager = MemoryManager()
    loop = AgentLoop(
        adapter=adapter,
        observation_mapper=mapper,
        memory_manager=memory_manager,
        adapter_folder="dummy",
        policy_config={},
    )

    loop.run_step(0)
    loop.run_step(1)

    assert loop._last_drive_signals is not None  # noqa: SLF001
    assert len(loop._homeostatic_history) >= 2  # noqa: SLF001
    assert isinstance(loop._last_drive_signals.signals, list)  # noqa: SLF001
    assert memory_manager.self_state._metadata  # noqa: SLF001
    assert memory_manager.policy_traces._metadata  # noqa: SLF001


def test_policy_trace_winner_channel_is_policy_drive_not_top_signal() -> None:
    class _HungerTaggedAdapter(_DriveAwareDummyAdapter):
        def get_available_policies(self) -> List[Dict[str, Any]]:
            return [
                {
                    "policy_id": "dummy:policy_noop",
                    "callable_name": "policy_noop",
                    "description": "Prefer hunger recovery for testing.",
                    "tags": ["eat", "collect"],
                    "drive_tags": ["hunger"],
                }
            ]

    loop = AgentLoop(
        adapter=_HungerTaggedAdapter(),
        observation_mapper=_CountingMapper(),
        memory_manager=MemoryManager(),
        adapter_folder="dummy",
        policy_config={},
    )
    signals = PrioritisedDriveSignals(
        signals=[
            _drive_signal("health", urgency=0.95, projected_value=0.2, tick=10),
            _drive_signal("hunger", urgency=0.70, projected_value=0.1, tick=10),
            _drive_signal("safety", urgency=0.30, projected_value=0.4, tick=10),
        ],
        highest_urgency=0.95,
        tick=10,
    )

    loop._record_policy_trace(  # noqa: SLF001
        selected_policy_id="dummy:policy_noop",
        drive_signals=signals,
        reward=0.0,
        step=10,
    )

    records = list(loop.memory_manager.policy_traces._metadata.values())  # noqa: SLF001
    assert len(records) == 1
    trace = records[0]
    assert trace.winner_channel_id == "hunger"
    assert trace.channel_a_id == "hunger"
    assert trace.channel_b_id == "health"
    assert trace.outcome_score == 0.0


def test_policy_trace_skips_single_signal_conflicts() -> None:
    loop = AgentLoop(
        adapter=_DriveAwareDummyAdapter(),
        observation_mapper=_CountingMapper(),
        memory_manager=MemoryManager(),
        adapter_folder="dummy",
        policy_config={},
    )
    single = PrioritisedDriveSignals(
        signals=[_drive_signal("health", urgency=0.9, tick=2)],
        highest_urgency=0.9,
        tick=2,
    )

    loop._record_policy_trace(  # noqa: SLF001
        selected_policy_id="dummy:policy_noop",
        drive_signals=single,
        reward=1.0,
        step=2,
    )

    assert loop.memory_manager.policy_traces._metadata == {}  # noqa: SLF001


def test_reward_to_outcome_score_handles_sparse_non_negative_rewards() -> None:
    assert AgentLoop._reward_to_outcome_score(0.0) == 0.0  # noqa: SLF001
    assert AgentLoop._reward_to_outcome_score(0.25) == 0.25  # noqa: SLF001
    assert AgentLoop._reward_to_outcome_score(2.0) == 1.0  # noqa: SLF001
