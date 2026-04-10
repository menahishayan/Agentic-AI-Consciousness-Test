"""
Headless Simulation Adapter — pure Python, no external game engine required.

Simulates Animal AI semantics:
  - 2D arena (configurable size) with food items at fixed positions
  - Agent navigates with 5 discrete actions: move_forward, move_backward,
    turn_left, turn_right, idle
  - Food items consumed when agent gets within interaction_radius
  - Hazard zone near arena boundary causes negative reward
  - HomeostaticWrapper tracks health/saturation depletion

Ray geometry:
  7 synthetic raycasts at the same angles used by the Animal AI
  RayPerceptionSensorComponent3D (raysPerSide=3, centre-first ordering):
    [0.0°, +23.3°, -23.3°, +46.6°, -46.6°, +70.0°, -70.0°]
  Each ray reports the first object hit (food or wall) within max_ray_range.
  Distance is normalised to [0, 1] (0 = at agent, 1 = max range / no hit).
  hit_tag uses the same canonical strings as observation_mapper.py:
    "GoodGoal", "wall", or None.

Motor efficiency:
  For move_forward / move_backward: actual displacement / step_size.
  Wall collisions produce efficiency ≈ 0 (position clamped at boundary).
  Turns and idle always report 1.0 (they are not "stuck" actions).

Stuck detection (mirrors AnimalAI adapter semantics):
  stuck_steps increments on consecutive move_forward failures (eff < 0.3).
  Resets on a successful move_forward. Unchanged for non-forward actions
  (the wall is still there; resetting on a turn would mask the block).

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

# Ray angles matching Animal AI's RayPerceptionSensorComponent3D
# (AlternatingRayOrder=1, centre-first, raysPerSide=3, maxRayDegrees=70)
_RAY_ANGLES_DEG: List[float] = [0.0, 23.3, -23.3, 46.6, -46.6, 70.0, -70.0]

# Stuck detection threshold — mirrors FreeEnergyMinimizer and loop.py semantics
_STUCK_EFFICIENCY_THRESHOLD: float = 0.3


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
        # Default matches Unity prefab: 3 frames × (1/30 s) × 100°/s = 10°
        self._turn_angle: float = math.radians(float(sim.get("turn_angle_deg", 10.0)))
        self._interaction_radius: float = float(sim.get("interaction_radius", 2.0))
        self._hazard_margin: float = float(sim.get("hazard_margin", 2.0))
        self._n_food: int = int(sim.get("n_food", 5))
        self._seed: Optional[int] = sim.get("seed")

        # Max ray range for synthetic raycasts — use arena diagonal to cover
        # all directions regardless of agent position.
        self._max_ray_range: float = self._arena_size * math.sqrt(2.0)
        # Food detection half-width perpendicular to ray.
        # Food items in Unity are ~1 unit radius; use interaction_radius * 0.6
        # so nearby food is reliably seen while avoiding spurious cross-detections.
        self._ray_food_width: float = max(1.0, self._interaction_radius * 0.6)

        self._rng = random.Random(self._seed)
        self._homeostatic = HomeostaticWrapper(config)
        # Health restoration per food item consumed — headless only (Unity normally
        # owns health; sync_health() is the seam we use to drive depletion here).
        hc = config.get("homeostatic", {})
        self._food_health_restore: float = float(hc.get("food_health_restore", 0.15))

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

        # Motor efficiency and stuck detection
        self._motor_efficiency: float = 1.0
        self._stuck_steps: int = 0

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
        self._motor_efficiency = 1.0
        self._stuck_steps = 0
        return self._build_state(reward=0.0)

    def step(self, action_id: str) -> Tuple[AgentState, bool]:
        self._step_count += 1
        prev_x, prev_z = self._x, self._z
        reward = self._apply_action(action_id)

        # Motor efficiency: actual vs. expected displacement for move actions.
        # Turns and idle always report 1.0 — they are not "stuck" actions.
        if action_id in ("move_forward", "move_backward"):
            actual_disp = math.sqrt(
                (self._x - prev_x) ** 2 + (self._z - prev_z) ** 2
            )
            self._motor_efficiency = (
                min(1.0, actual_disp / self._step_size) if self._step_size > 0 else 1.0
            )
            # Stuck counter: only move_forward can be "stuck" (moving into a wall).
            # Backward actions hitting a wall are unusual but treated consistently.
            if action_id == "move_forward":
                if self._motor_efficiency < _STUCK_EFFICIENCY_THRESHOLD:
                    self._stuck_steps += 1
                else:
                    self._stuck_steps = 0
            # Note: move_backward failure leaves stuck_steps unchanged
            # (matches FreeEnergyMinimizer: "turns/backward don't count as failures")
        else:
            self._motor_efficiency = 1.0
            # Non-forward actions: leave stuck_steps unchanged.
            # The wall is still there; resetting would mask the block.

        self._homeostatic.step(reward, env_done=False)

        # Health depletion — headless only. Unity is not available so we drive
        # health via sync_health(), which applies health_depletion_rate each call:
        #   new_health = proxy - health_depletion_rate
        # Passing current health achieves passive depletion per step.
        # Food reward adds a restoration bonus on top before the depletion is applied.
        health_proxy = self._homeostatic.health
        if reward > 0.0:
            health_proxy = min(1.0, health_proxy + min(reward, 1.0) * self._food_health_restore)
        self._homeostatic.sync_health(health_proxy)

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

    def _cast_rays(self) -> List[Dict[str, Any]]:
        """
        Compute 7 synthetic raycasts at the standard Animal AI ray angles.

        Each ray is cast from the agent's current position in the direction
        (heading + angle_deg). The first object hit within max_ray_range is
        reported: food ("GoodGoal") or arena wall ("wall"). Clear rays return
        hit_tag=None, distance=1.0.

        Food detection: a food item is "on" a ray when its perpendicular
        distance to the ray line is < _ray_food_width AND it is in front of
        the agent (along-ray projection > 0). The closest such food is used.

        Wall detection: parametric ray-AABB intersection against the four
        arena boundary planes. Only forward intersections (t > 0) are kept.

        Distance is normalised by max_ray_range so values are in [0, 1],
        matching the convention used by FreeEnergyMinimizer and PolicyGenerator.
        """
        rays: List[Dict[str, Any]] = []
        for angle_deg in _RAY_ANGLES_DEG:
            ray_heading = self._heading + math.radians(angle_deg)
            # Ray direction unit vector (2D: x and z axes)
            dx = math.sin(ray_heading)
            dz = math.cos(ray_heading)

            # --- Food detection ---
            best_food_t: Optional[float] = None
            for food in self._food:
                if food.consumed:
                    continue
                # Vector from agent to food
                fx = food.x - self._x
                fz = food.z - self._z
                # Along-ray projection (must be positive to be in front)
                along = fx * dx + fz * dz
                if along <= 0.0:
                    continue
                # Perpendicular distance to ray line
                perp = abs(fx * dz - fz * dx)
                if perp < self._ray_food_width:
                    if best_food_t is None or along < best_food_t:
                        best_food_t = along

            # --- Wall detection ---
            # Ray: (self._x + t*dx, self._z + t*dz)
            # Planes: x=0, x=arena_size, z=0, z=arena_size
            wall_t = self._max_ray_range
            for wall_pos, component_d, component_pos in [
                (0.0,               dx, self._x),
                (self._arena_size,  dx, self._x),
                (0.0,               dz, self._z),
                (self._arena_size,  dz, self._z),
            ]:
                if abs(component_d) < 1e-10:
                    continue
                t = (wall_pos - component_pos) / component_d
                if 0.0 < t < wall_t:
                    wall_t = t

            # --- Determine hit ---
            if best_food_t is not None and best_food_t < wall_t:
                hit_tag: Optional[str] = "GoodGoal"
                raw_dist = best_food_t
            elif wall_t < self._max_ray_range:
                hit_tag = "wall"
                raw_dist = wall_t
            else:
                hit_tag = None
                raw_dist = self._max_ray_range

            dist_norm = min(1.0, raw_dist / self._max_ray_range)
            rays.append({
                "hit_tag": hit_tag,
                "distance": dist_norm,
                "angle_deg": angle_deg,
            })

        return rays

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

        # Synthetic raycasts — primary sensory signal for all downstream layers
        rays = self._cast_rays()

        # detected_objects: unique hit tags present in the ray fan.
        # Uses canonical tag strings matching observation_mapper.py ("GoodGoal", "wall")
        # so FreeEnergyMinimizer and PolicyGenerator comparisons are correct.
        detected_objects: List[str] = list({
            r["hit_tag"] for r in rays if r.get("hit_tag") is not None
        })

        # Synthetic visual features (proxy — no real camera).
        # 17-dim layout matching observation_mapper._extract_visual_features:
        #   [0-5] R/G/B mean+var, [6] edge_density, [7-10] brightness hist×4,
        #   [11-16] left/center/right (green_dom, edge_density) × 3 regions
        nearest_food_dist = min(
            (self._dist(f.x, f.z) for f in self._food if not f.consumed),
            default=self._arena_size,
        )
        food_proximity = max(0.0, 1.0 - nearest_food_dist / self._arena_size)
        threat = self.estimate_threat_proximity(None)  # type: ignore[arg-type]
        visual_features = [
            threat * 0.8, threat * 0.1,            # R mean, var
            food_proximity * 0.7, food_proximity * 0.05,  # G mean, var
            0.3, 0.02,                              # B mean, var
            min(0.5, food_proximity + threat),      # edge_density
            0.4, 0.3, 0.2, 0.1,                    # brightness histogram ×4
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,          # directional features ×6 (not simulated)
        ]

        # Resource and threat estimates from raycasts (matches observation_mapper logic)
        food_ray = next(
            (r for r in rays if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")),
            None,
        )
        resource_level = (
            float(max(0.0, 1.0 - food_ray["distance"])) if food_ray else 0.1
        )

        dist_to_wall = min(
            self._x, self._z,
            self._arena_size - self._x,
            self._arena_size - self._z,
        )
        threat_prox = (
            float(max(0.0, 1.0 - dist_to_wall / self._hazard_margin))
            if dist_to_wall < self._hazard_margin else 0.0
        )

        # Only populate raycast_hits when at least one ray has a real hit or
        # a cleared distance less than max range — mirrors observation_mapper.py
        # behaviour (None = no real obs arrived this step).
        any_real_hit = any(
            r.get("hit_tag") is not None or r.get("distance", 1.0) < 1.0
            for r in rays
        )
        raycast_hits_out = rays if any_real_hit else None

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
                detected_objects=detected_objects,
                area_id=area_id,
                terrain_novelty=novelty,
                entity_density=float(
                    min(1.0, sum(
                        1 for f in self._food
                        if not f.consumed and self._dist(f.x, f.z) < 10.0
                    ) / max(self._n_food, 1))
                ),
                raycast_hits=raycast_hits_out,
            ),
            resources=ResourceState(
                resource_level=resource_level,
                threat_proximity=threat_prox,
            ),
            step=self._step_count,
            raw_metadata={
                "reward": reward,
                "n_food_remaining": sum(1 for f in self._food if not f.consumed),
                "motor_efficiency": self._motor_efficiency,
                "stuck_steps": self._stuck_steps,
            },
        )


def create_adapter(config: Dict[str, Any]) -> HeadlessSimAdapter:
    return HeadlessSimAdapter(config)
