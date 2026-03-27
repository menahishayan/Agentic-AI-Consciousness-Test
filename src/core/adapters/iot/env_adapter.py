"""
IoT Stub Adapter — demonstrates plug-and-play extensibility.

To use this adapter, set in config.json:
    "adapter_folder": "iot"

A real IoT adapter would:
  - Connect to sensor streams (MQTT, REST, gRPC) as the environment
  - Map sensor readings (battery, temperature, signal strength, proximity) to AgentState
  - Map policy IDs to actuator commands (motor speed, valve open/close, alert trigger)
  - HomeostaticWrapper would track: battery depletion, thermal state, connectivity health
  - The brain remains identical — only this adapter changes

This stub returns synthetic random observations to allow the brain to run without hardware.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

from core.adapters.base import AbstractEnvironmentAdapter
from core.models.signals import DriveChannel
from core.models.state import (
    AgentState,
    HomeostasisState,
    PerceptionState,
    PositionState,
    ResourceState,
)

_POLICIES = [
    {"policy_id": "poll_sensors",      "callable_name": "poll_sensors",    "tags": ["sensing"],    "drive_tags": ["energy"]},
    {"policy_id": "reduce_activity",   "callable_name": "reduce_activity",  "tags": ["rest"],       "drive_tags": ["energy"]},
    {"policy_id": "send_alert",        "callable_name": "send_alert",       "tags": ["communication"], "drive_tags": ["safety"]},
    {"policy_id": "recharge",          "callable_name": "recharge",         "tags": ["maintenance"], "drive_tags": ["energy"]},
    {"policy_id": "idle",              "callable_name": "idle",             "tags": ["rest"],       "drive_tags": []},
]

_DRIVE_CHANNELS = [
    DriveChannel(
        channel_id="energy",
        setpoint=0.7,
        critical_threshold=0.2,
        recovery_cost_ticks=100,
        suggested_action_tags=["maintenance", "rest"],
        weight=1.0,
    ),
    DriveChannel(
        channel_id="safety",
        setpoint=0.9,
        critical_threshold=0.4,
        recovery_cost_ticks=10,
        suggested_action_tags=["communication"],
        weight=1.2,
    ),
]


class IoTStubAdapter(AbstractEnvironmentAdapter):
    """Synthetic IoT sensor environment for brain-only testing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._step = 0
        self._battery = 1.0
        self._signal_strength = 0.8
        self._rng = random.Random(config.get("seed", 42))

    def reset(self) -> AgentState:
        self._step = 0
        self._battery = 1.0
        self._signal_strength = 0.8
        return self._build_state()

    def step(self, action_id: str) -> Tuple[AgentState, bool]:
        self._step += 1
        # Battery depletes; recharge action restores it
        if action_id == "recharge":
            self._battery = min(1.0, self._battery + 0.3)
        else:
            self._battery = max(0.0, self._battery - 0.005)
        # Signal fluctuates
        self._signal_strength = max(0.0, min(1.0, self._signal_strength + self._rng.gauss(0, 0.05)))
        done = self._battery <= 0.0
        return self._build_state(), done

    def close(self) -> None:
        pass

    def get_available_vitals(self) -> List[str]:
        return ["energy", "safety"]

    def get_available_policies(self) -> List[Dict[str, Any]]:
        return _POLICIES

    def get_drive_channels(self) -> List[DriveChannel]:
        return _DRIVE_CHANNELS

    def get_task_goal(self) -> Dict[str, Any]:
        return {
            "description": "maintain sensor uptime and report anomalies",
            "priority": 1.0,
            "task_id": "iot_monitor",
        }

    def estimate_resource_level(self, state: AgentState) -> float:
        return self._battery

    def estimate_threat_proximity(self, state: AgentState) -> float:
        return max(0.0, 1.0 - self._signal_strength)

    def build_area_id(self, state: AgentState) -> str:
        return f"zone_{int(self._step / 50)}"

    def estimate_entity_density(self, state: AgentState) -> float:
        return 0.1

    def estimate_terrain_novelty(self, state: AgentState) -> float:
        return 0.0

    def _build_state(self) -> AgentState:
        return AgentState(
            homeostasis=HomeostasisState(
                health=self._battery,
                energy=self._battery,
                saturation=self._signal_strength,
                is_alive=self._battery > 0.0,
            ),
            position=PositionState(x=0.0, y=0.0, z=0.0),
            perception=PerceptionState(
                visual_features=[self._battery, self._signal_strength],
                area_id=f"zone_{int(self._step / 50)}",
                terrain_novelty=0.0,
                entity_density=0.1,
            ),
            resources=ResourceState(
                resource_level=self._battery,
                threat_proximity=max(0.0, 1.0 - self._signal_strength),
            ),
            step=self._step,
        )


def create_adapter(config: Dict[str, Any]) -> IoTStubAdapter:
    return IoTStubAdapter(config)
