"""
Headless Simulation Adapter — pure Python, no external game engine required.

Simulates Animal AI semantics:
  - 2D arena (configurable size) with food items at fixed positions
  - Agent navigates with 5 discrete actions: move_forward, move_backward,
    turn_left, turn_right, idle
  - Food items consumed when agent gets within interaction_radius
  - Hazard zone near arena boundary causes negative reward
  - HomeostaticWrapper tracks health/saturation depletion

Use this adapter for development and testing without Unity/Animal AI installed.
Switch to "animalai" adapter_folder when the Unity environment is available.

All brain behavior is identical — only this adapter file changes.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.adapters.animalai.homeostatic_wrapper import HomeostaticWrapper
from core.adapters.base import AbstractEnvironmentAdapter
from core.models.signals import DriveChannel
from core.models.state import (
    AgentState,
    HomeostasisState,
    PerceptionState,
    PositionState,
    ResourceState,
)

_DRIVE_CHANNELS = [
    DriveChannel(
        channel_id="health",
        setpoint=0.8,
        critical_threshold=0.2,
        recovery_cost_ticks=200,
        suggested_action_tags=["exploration", "navigation"],
        weight=1.2,
    ),
    DriveChannel(
        channel_id="saturation",
        setpoint=0.7,
        critical_threshold=0.25,
        recovery_cost_ticks=150,
        suggested_action_tags=["exploration", "navigation"],
        weight=1.0,
    ),
    DriveChannel(
        channel_id="energy",
        setpoint=0.65,
        critical_threshold=0.2,
        recovery_cost_ticks=180,
        suggested_action_tags=["navigation", "rest"],
        weight=0.9,
    ),
    DriveChannel(
        channel_id="safety",
        setpoint=0.85,
        critical_threshold=0.3,
        recovery_cost_ticks=30,
        suggested_action_tags=["avoidance"],
        weight=1.3,
    ),
]

_POLICIES = [
    {"policy_id": "move_forward",  "callable_name": "move_forward",  "tags": ["navigation", "exploration"], "drive_tags": ["energy", "resource_level"], "description": "Move forward to explore and find food"},
    {"policy_id": "move_backward", "callable_name": "move_backward", "tags": ["navigation", "avoidance"],   "drive_tags": ["safety"],                  "description": "Move backward to avoid hazards"},
    {"policy_id": "turn_left",     "callable_name": "turn_left",     "tags": ["navigation", "orientation"], "drive_tags": [],                           "description": "Turn left to change direction"},
    {"policy_id": "turn_right",    "callable_name": "turn_right",    "tags": ["navigation", "orientation"], "drive_tags": [],                           "description": "Turn right to change direction"},
    {"policy_id": "idle",          "callable_name": "idle",          "tags": ["rest"],                      "drive_tags": [],                           "description": "Stay still and observe the environment"},
]


class FoodItem:
    def __init__(self, x: float, z: float, value: float = 1.0):
        self.x = x
        self.z = z
        self.value = value
        self.consumed = False


class HeadlessSimAdapter(AbstractEnvironmentAdapter):
    """
    Pure Python simulation of an Animal AI-like food-finding arena.

    Arena: arena_size × arena_size 2D plane
    Agent: starts at center, moves with step_size per action, turns with turn_angle_deg
    Food: n_food items placed randomly (respawn after all consumed)
    Hazard zone: within hazard_margin of arena boundary → negative reward
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        sim = config.get("simulation", {})

        self._arena_size: float = float(sim.get("arena_size", 30.0))
        self._step_size: float = float(sim.get("step_size", 1.0))
        self._turn_angle: float = math.radians(float(sim.get("turn_angle_deg", 15.0)))
        self._interaction_radius: float = float(sim.get("interaction_radius", 2.0))
        self._hazard_margin: float = float(sim.get("hazard_margin", 2.0))
        self._n_food: int = int(sim.get("n_food", 5))
        self._seed: Optional[int] = sim.get("seed")

        self._rng = random.Random(self._seed)
        self._homeostatic = HomeostaticWrapper(config)

        # Agent state
        self._x: float = 0.0
        self._z: float = 0.0
        self._heading: float = 0.0   # radians, 0 = +z direction
        self._step_count: int = 0

        # Food items
        self._food: List[FoodItem] = []

        # Terrain novelty tracking: set of visited grid cells
        self._visited: set = set()
        self._last_area_id: str = "x0z0"

    # ------------------------------------------------------------------
    # AbstractEnvironmentAdapter
    # ------------------------------------------------------------------

    def reset(self) -> AgentState:
        self._x = self._arena_size / 2.0
        self._z = self._arena_size / 2.0
        self._heading = 0.0
        self._step_count = 0
        self._homeostatic.reset()
        self._visited.clear()
        self._food = self._spawn_food()
        return self._build_state(reward=0.0)

    def step(self, action_id: str) -> Tuple[AgentState, bool]:
        self._step_count += 1
        reward = self._apply_action(action_id)
        self._homeostatic.step(reward, env_done=False)
        state = self._build_state(reward=reward)
        done = not self._homeostatic.is_alive
        return state, done

    def close(self) -> None:
        pass

    def get_available_vitals(self) -> List[str]:
        return ["health", "saturation", "energy"]

    def get_available_policies(self) -> List[Dict[str, Any]]:
        return _POLICIES

    def get_drive_channels(self) -> List[DriveChannel]:
        return _DRIVE_CHANNELS

    def get_task_goal(self) -> Dict[str, Any]:
        return {
            "description": "find and consume food items to maintain health and saturation",
            "priority": 1.0,
            "task_id": "basic_food",
        }

    def estimate_resource_level(self, state: AgentState) -> float:
        nearby = sum(1 for f in self._food if not f.consumed and self._dist(f.x, f.z) < self._arena_size * 0.4)
        saturation = (state.homeostasis.saturation if state and state.homeostasis.saturation is not None else 0.5)
        return float(min(1.0, nearby / max(self._n_food, 1) * 0.5 + saturation * 0.5))

    def estimate_threat_proximity(self, state: AgentState) -> float:
        dist_to_wall = min(self._x, self._z, self._arena_size - self._x, self._arena_size - self._z)
        if dist_to_wall < self._hazard_margin:
            return float(1.0 - dist_to_wall / self._hazard_margin)
        return 0.0

    def build_area_id(self, state: AgentState) -> str:
        gx = int(self._x / 5.0)
        gz = int(self._z / 5.0)
        return f"x{gx}z{gz}"

    def estimate_entity_density(self, state: AgentState) -> float:
        visible = sum(1 for f in self._food if not f.consumed and self._dist(f.x, f.z) < 10.0)
        return float(min(1.0, visible / max(self._n_food, 1)))

    def estimate_terrain_novelty(self, state: AgentState) -> float:
        area = self.build_area_id(state)
        if area not in self._visited:
            return 1.0
        return 0.1

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _apply_action(self, action_id: str) -> float:
        """Apply action, return raw reward."""
        if action_id == "move_forward":
            self._x += self._step_size * math.sin(self._heading)
            self._z += self._step_size * math.cos(self._heading)
        elif action_id == "move_backward":
            self._x -= self._step_size * math.sin(self._heading)
            self._z -= self._step_size * math.cos(self._heading)
        elif action_id == "turn_left":
            self._heading = (self._heading - self._turn_angle) % (2 * math.pi)
        elif action_id == "turn_right":
            self._heading = (self._heading + self._turn_angle) % (2 * math.pi)
        # idle: no movement

        # Clamp to arena bounds
        self._x = max(0.0, min(self._arena_size, self._x))
        self._z = max(0.0, min(self._arena_size, self._z))

        # Track visited areas
        area = f"x{int(self._x / 5)}z{int(self._z / 5)}"
        self._visited.add(area)
        self._last_area_id = area

        # Check food interaction
        reward = 0.0
        for food in self._food:
            if not food.consumed and self._dist(food.x, food.z) <= self._interaction_radius:
                food.consumed = True
                reward += food.value

        # Hazard zone
        dist_to_wall = min(
            self._x, self._z,
            self._arena_size - self._x,
            self._arena_size - self._z,
        )
        if dist_to_wall < self._hazard_margin:
            reward -= 0.1 * (1.0 - dist_to_wall / self._hazard_margin)

        # Respawn food if all consumed
        if all(f.consumed for f in self._food):
            self._food = self._spawn_food()

        return reward

    def _dist(self, fx: float, fz: float) -> float:
        return math.sqrt((self._x - fx) ** 2 + (self._z - fz) ** 2)

    def _spawn_food(self) -> List[FoodItem]:
        margin = self._hazard_margin + 1.0
        items = []
        for _ in range(self._n_food):
            fx = self._rng.uniform(margin, self._arena_size - margin)
            fz = self._rng.uniform(margin, self._arena_size - margin)
            items.append(FoodItem(fx, fz, value=1.0))
        return items

    def _build_state(self, reward: float) -> AgentState:
        hs = self._homeostatic.get_state()
        area_id = self._last_area_id
        novelty = 0.0 if area_id in self._visited else 1.0

        # Synthetic visual features based on environment state
        nearest_food_dist = min(
            (self._dist(f.x, f.z) for f in self._food if not f.consumed),
            default=self._arena_size,
        )
        food_proximity = max(0.0, 1.0 - nearest_food_dist / self._arena_size)
        threat = self.estimate_threat_proximity(None)  # type: ignore[arg-type]
        # 11-dim feature: r_mean, r_var, g_mean, g_var, b_mean, b_var, edge, hist×4
        visual_features = [
            threat * 0.8,                          # red channel (hazard proxy)
            threat * 0.1,
            food_proximity * 0.7,                  # green channel (food proxy)
            food_proximity * 0.05,
            0.3,                                   # blue (sky/neutral)
            0.02,
            min(0.5, food_proximity + threat),     # edge density
            0.4, 0.3, 0.2, 0.1,                   # brightness histogram
        ]

        dist_to_wall = min(self._x, self._z, self._arena_size - self._x, self._arena_size - self._z)
        threat_prox = float(max(0.0, 1.0 - dist_to_wall / self._hazard_margin)) if dist_to_wall < self._hazard_margin else 0.0
        # Resource: food proximity + current saturation
        nearest_food_dist_for_resource = min(
            (self._dist(f.x, f.z) for f in self._food if not f.consumed),
            default=self._arena_size,
        )
        food_prox_resource = max(0.0, 1.0 - nearest_food_dist_for_resource / self._arena_size)
        resource = float(min(1.0, food_prox_resource * 0.4 + (hs["saturation"] * 0.6)))

        return AgentState(
            homeostasis=HomeostasisState(
                health=hs["health"],
                saturation=hs["saturation"],
                energy=hs["energy"],
                is_alive=self._homeostatic.is_alive,
            ),
            position=PositionState(
                x=round(self._x, 2),
                y=0.0,
                z=round(self._z, 2),
                heading=round(self._heading, 4),
            ),
            perception=PerceptionState(
                visual_features=visual_features,
                detected_objects=[f"food_{i}" for i, f in enumerate(self._food) if not f.consumed and self._dist(f.x, f.z) < 8.0],
                area_id=area_id,
                terrain_novelty=novelty,
                entity_density=float(min(1.0, sum(1 for f in self._food if not f.consumed and self._dist(f.x, f.z) < 10.0) / max(self._n_food, 1))),
            ),
            resources=ResourceState(
                resource_level=resource,
                threat_proximity=threat_prox,
            ),
            step=self._step_count,
            raw_metadata={"reward": reward, "n_food_remaining": sum(1 for f in self._food if not f.consumed)},
        )


def create_adapter(config: Dict[str, Any]) -> HeadlessSimAdapter:
    return HeadlessSimAdapter(config)
