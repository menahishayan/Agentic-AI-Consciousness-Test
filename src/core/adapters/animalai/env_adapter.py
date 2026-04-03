"""
Animal AI Environment Adapter

Implements AbstractEnvironmentAdapter for the Animal AI testbed.
Wraps AnimalAIEnvironment with HomeostaticWrapper to add physiological depletion.

The brain never imports this file. It receives AgentState and dispatches
policy_id strings, both of which are environment-agnostic types.

To use a different environment, implement a new adapter folder and change
adapter_folder in config.json.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.adapters.animalai.action_mapper import get_policy_descriptors, policy_id_to_action
from core.adapters.animalai.homeostatic_wrapper import HomeostaticWrapper
from core.adapters.animalai.observation_mapper import _PositionTracker, _extract_local_speed, _parse_raycasts, map_obs
from core.adapters.base import AbstractEnvironmentAdapter
from core.models.signals import DriveChannel
from core.models.state import AgentState

log = logging.getLogger(__name__)

# Actions that command translational movement — used for stuck detection
_MOVEMENT_ACTIONS = {"move_forward", "move_back"}

# Local forward speed below which the agent is considered not moving
_SPEED_STUCK_THRESHOLD = 0.05

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


class AnimalAIAdapter(AbstractEnvironmentAdapter):
    """
    Adapter for the Animal AI Unity-based testbed.

    Manages the Animal AI environment lifecycle and exposes a clean
    AbstractEnvironmentAdapter interface to the brain.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._step_count = 0
        self._last_action: Optional[str] = None

        # Homeostatic simulation layer
        self._homeostatic = HomeostaticWrapper(config)

        # Position integrator
        self._position_tracker = _PositionTracker()

        # Lazy-loaded environment — only created on first reset()
        self._env: Optional[Any] = None
        self._behavior_name: Optional[str] = None

        # Arena config path
        arena_path = config.get("arena_config", "animalai_configs/basic_food.yaml")
        if config.get("resolve_filename", True):
            arena_path = str(Path(arena_path).resolve())
        self._arena_config_path = arena_path

        self._worker_id = int(config.get("worker_id", 1))
        self._base_port = int(config.get("base_port", 5005))
        self._additional_args: List[str] = list(config.get("additional_args", []))

        # Optional: path to the Unity binary. None = connect to already-running instance.
        file_name = config.get("file_name")
        if file_name and config.get("resolve_filename", True):
            file_name = str(Path(file_name).resolve())
        self._file_name: Optional[str] = file_name

        # Velocity scaling: Unity physics units/s → arena units/decision-step
        # decision_period = frames per decision step (default 5 at 30Hz ≈ 0.167s/step)
        physics_fps = float(config.get("physics_fps", 30.0))
        decision_period = float(config.get("decision_period", 5.0))
        self._velocity_dt: float = decision_period / physics_fps

        # Expected full-throttle forward speed in Unity physics units/sec.
        # = arena step_size (metres/step) / dt (seconds/step)
        # e.g. step_size=1.0, dt=0.167 → expected_speed=6.0 Unity units/sec
        # motor_efficiency = raw_fwd / expected_speed:
        #   free movement at ~16 u/s → clamped to 1.0
        #   wall-blocked at ~0 u/s   → near 0.0
        sim_cfg = config.get("simulation", {})
        step_size = float(sim_cfg.get("step_size", 1.0))
        self._expected_speed: float = step_size / max(self._velocity_dt, 1e-6)

        # Dead-reckoning heading — integrated from turn actions each step
        self._heading: float = 0.0
        self._turn_angle_deg: float = float(sim_cfg.get("turn_angle_deg", 45.0))

        # Proprioceptive stuck detection
        self._stuck_adapter_count: int = 0
        self._stuck_adapter_threshold: int = int(config.get("stuck_threshold", 5))

        # One-shot obs shape diagnostic — logged on first step to verify raycast presence
        self._obs_shapes_logged: bool = False

    # ------------------------------------------------------------------
    # AbstractEnvironmentAdapter interface
    # ------------------------------------------------------------------

    def reset(self) -> AgentState:
        self._step_count = 0
        self._heading = 0.0
        self._homeostatic.reset()
        self._position_tracker.reset()
        self._stuck_adapter_count = 0

        env = self._get_or_create_env()
        env.reset()
        obs, info = self._get_obs()
        state = map_obs(obs, info, self._homeostatic, self._step_count, self._position_tracker)
        return state

    def step(self, action_id: str) -> Tuple[AgentState, bool]:
        self._step_count += 1
        self._last_action = action_id

        env = self._get_or_create_env()
        action_vec = policy_id_to_action(action_id)

        try:
            self._send_action(env, action_vec)
        except Exception as exc:
            log.warning("Failed to send action '%s': %s", action_id, exc)

        obs, info = self._get_obs()
        raw_reward = float(info.get("raw_reward", 0.0))
        env_done = bool(info.get("env_done", False))

        # Integrate heading from turn actions (dead-reckoning)
        if action_id == "turn_left":
            self._heading = (self._heading + self._turn_angle_deg) % 360.0
        elif action_id == "turn_right":
            self._heading = (self._heading - self._turn_angle_deg) % 360.0
        info["heading"] = self._heading

        # Compute motor_efficiency BEFORE dt-scaling, using raw Unity units/sec.
        # Normalised by expected full-throttle speed so the signal is [0, 1]:
        #   free movement → 1.0,  wall-blocked → ~0.0
        raw_fwd = float(info.get("local_speed_forward", 0.0))
        motor_efficiency = float(min(max(raw_fwd / self._expected_speed, 0.0), 1.0))
        info["motor_efficiency"] = motor_efficiency
        # position_delta_norm = fraction of expected displacement actually achieved.
        # Used by PredictionErrorCalculator to compute motor PE (efference copy channel).
        info["position_delta_norm"] = motor_efficiency

        # Scale Unity physics velocity (units/s) to displacement per decision step
        info["local_speed_forward"] = raw_fwd * self._velocity_dt
        info["local_speed_right"] = info.get("local_speed_right", 0.0) * self._velocity_dt

        # Stuck detection: movement commanded but local forward speed is near zero
        local_fwd = info.get("local_speed_forward", 0.0)
        if action_id in _MOVEMENT_ACTIONS and abs(local_fwd) < _SPEED_STUCK_THRESHOLD:
            self._stuck_adapter_count += 1
        else:
            self._stuck_adapter_count = 0
        motor_stuck = self._stuck_adapter_count >= self._stuck_adapter_threshold
        info["motor_stuck"] = motor_stuck

        self._homeostatic.step(raw_reward, env_done)

        state = map_obs(obs, info, self._homeostatic, self._step_count, self._position_tracker)
        # Termination is homeostatic collapse only.
        # env_done (task object reached) is logged as an event but does not end
        # the episode — the agent continues to act in the environment.
        if env_done:
            log.info("Step %d: env_done signal received (task object reached) — episode continues",
                     self._step_count)
        done = not self._homeostatic.is_alive
        return state, done

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:
                log.warning("Error closing Animal AI env: %s", exc)
            finally:
                self._env = None

    def get_available_vitals(self) -> List[str]:
        return ["health", "saturation", "energy"]

    def get_available_policies(self) -> List[Dict[str, Any]]:
        return get_policy_descriptors()

    def get_drive_channels(self) -> List[DriveChannel]:
        return _DRIVE_CHANNELS

    def get_task_goal(self) -> Dict[str, Any]:
        return {
            "description": "find and consume food items to maintain health and saturation",
            "priority": 1.0,
            "task_id": "basic_food",
        }

    def estimate_resource_level(self, state: AgentState) -> float:
        features = state.perception.visual_features or []
        if len(features) < 6:
            return state.homeostasis.saturation or 0.5
        g_mean = features[2]
        r_mean, b_mean = features[0], features[4]
        green_dom = max(0.0, g_mean - (r_mean + b_mean) / 2.0)
        saturation = state.homeostasis.saturation or 0.5
        return float(min(green_dom * 2.0 * 0.4 + saturation * 0.6, 1.0))

    def estimate_threat_proximity(self, state: AgentState) -> float:
        features = state.perception.visual_features or []
        if len(features) < 6:
            return 0.0
        r_mean, g_mean, b_mean = features[0], features[2], features[4]
        red_anomaly = max(0.0, r_mean - (g_mean + b_mean) / 2.0)
        return float(min(red_anomaly * 2.5, 1.0))

    def build_area_id(self, state: AgentState) -> str:
        x = state.position.x or 0.0
        z = state.position.z or 0.0
        return f"x{int(x / 5)}z{int(z / 5)}"

    def estimate_entity_density(self, state: AgentState) -> float:
        features = state.perception.visual_features or []
        if len(features) < 11:
            return 0.0
        edge = features[6]
        brightness = sum(features[0:6:2]) / 3.0
        mid = 1.0 - abs(brightness - 0.5) * 2.0
        return float(min(edge * 0.7 + mid * 0.3, 1.0))

    def estimate_terrain_novelty(self, state: AgentState) -> float:
        return state.perception.terrain_novelty or 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_env(self) -> Any:
        if self._env is None:
            self._env = self._build_env()
        return self._env

    def _build_env(self) -> Any:
        """
        Build and return the AnimalAIEnvironment instance.
        Deferred to avoid importing animalai at module level (allows tests
        without Animal AI installed to mock this method).
        """
        try:
            from animalai.environment import AnimalAIEnvironment  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "animalai package not installed. Run: pip install animalai==5.0.1 (requires Python 3.10.12)"
            ) from exc

        if self._file_name:
            log.info("Launching Animal AI binary: %s (arena: %s)", self._file_name, self._arena_config_path)
        else:
            log.info("Connecting to running Animal AI instance (arena: %s)", self._arena_config_path)
        env = AnimalAIEnvironment(
            file_name=self._file_name,
            arenas_configurations=self._arena_config_path,
            worker_id=self._worker_id,
            base_port=self._base_port,
            additional_args=self._additional_args or None,
        )
        # Discover behavior name
        specs = env.behavior_specs
        if not specs:
            raise RuntimeError("Animal AI returned no behavior specs.")
        self._behavior_name = next(iter(specs))
        log.info("Behavior name: %s", self._behavior_name)
        return env

    def _get_obs(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Read latest observation from the environment buffer."""
        env = self._env
        if env is None or self._behavior_name is None:
            return None, {}

        try:
            decision_steps, terminal_steps = env.get_steps(self._behavior_name)
        except Exception as exc:
            log.warning("Failed to get steps: %s", exc)
            return None, {"env_done": True}

        # Check terminal (episode ended)
        if len(terminal_steps) > 0:
            agent_id = list(terminal_steps.agent_id)[0]
            obs_list = terminal_steps[agent_id].obs
            raw_reward = float(terminal_steps[agent_id].reward)
            self._log_obs_shapes(obs_list)
            visual_obs = self._extract_visual(obs_list)
            fwd, right, up = _extract_local_speed(obs_list)
            raycast_hits = _parse_raycasts(obs_list)
            return visual_obs, {
                "raw_reward": raw_reward,
                "env_done": True,
                "local_speed_forward": fwd,
                "local_speed_right": right,
                "local_speed_up": up,
                "raycast_hits": raycast_hits,
            }

        if len(decision_steps) == 0:
            return None, {"raw_reward": 0.0, "env_done": False}

        agent_id = list(decision_steps.agent_id)[0]
        obs_list = decision_steps[agent_id].obs
        raw_reward = float(decision_steps[agent_id].reward)
        self._log_obs_shapes(obs_list)
        visual_obs = self._extract_visual(obs_list)
        fwd, right, up = _extract_local_speed(obs_list)
        raycast_hits = _parse_raycasts(obs_list)
        return visual_obs, {
            "raw_reward": raw_reward,
            "env_done": False,
            "local_speed_forward": fwd,
            "local_speed_right": right,
            "local_speed_up": up,
            "raycast_hits": raycast_hits,
        }

    def _log_obs_shapes(self, obs_list: List[Any]) -> None:
        """Log obs_list shapes once on the first step to confirm raycast presence."""
        if self._obs_shapes_logged:
            return
        self._obs_shapes_logged = True
        shapes = [np.asarray(o).shape for o in obs_list if hasattr(o, '__len__')]
        log.info("obs_list shapes (step 1): %s", shapes)
        print(f"[AnimalAI] obs_list shapes: {shapes}")

    def _extract_visual(self, obs_list: List[Any]) -> Optional[np.ndarray]:
        """Extract the first visual observation from the observation list."""
        for obs in obs_list:
            arr = np.asarray(obs)
            if arr.ndim >= 3:
                return arr
        return None

    def _send_action(self, env: Any, action_vec: np.ndarray) -> None:
        """Send action to the Animal AI environment."""
        from mlagents_envs.base_env import ActionTuple  # type: ignore

        action_tuple = ActionTuple(discrete=action_vec.reshape(1, -1))
        env.set_actions(self._behavior_name, action_tuple)
        env.step()


def create_adapter(config: Dict[str, Any]) -> AnimalAIAdapter:
    """Entry point called by loader.build_adapter()."""
    return AnimalAIAdapter(config)
