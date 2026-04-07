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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set AAI_DEBUG=1 (or "true"/"yes") in the environment to enable per-step
# diagnostic logging for all five observation gates.
# Example: AAI_DEBUG=1 python run.py
DEBUG: bool = os.getenv("AAI_DEBUG", "").lower() in ("1", "true", "yes")
# Path for debug output file. Defaults to debug.log in the working directory.
# Override with: AAI_DEBUG_LOG=/path/to/file.log AAI_DEBUG=1 python run.py
_DEBUG_LOG_PATH: str = os.getenv("AAI_DEBUG_LOG", "debug.log")


def _dbg(msg: str) -> None:
    """Append a debug line to the debug log file (line-buffered)."""
    with open(_DEBUG_LOG_PATH, "a", buffering=1) as _f:
        _f.write(msg + "\n")

from core.adapters.animalai.action_mapper import get_policy_descriptors, policy_id_to_action
from core.adapters.animalai.homeostatic_wrapper import HomeostaticWrapper
from core.adapters.animalai.observation_mapper import _PositionTracker, _extract_local_speed, _extract_unity_health, _extract_world_pos, _parse_raycasts, map_obs
from core.adapters.base import AbstractEnvironmentAdapter
from core.models.signals import DriveChannel
from core.models.state import AgentState

log = logging.getLogger(__name__)


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

        # Timing: decision_period physics frames per policy step at physics_fps.
        # velocity_dt converts Unity units/s → arena units/step for dead-reckoning.
        physics_fps = float(config.get("physics_fps", 30.0))
        decision_period = float(config.get("decision_period", 5.0))
        self._velocity_dt: float = decision_period / physics_fps  # seconds per step

        sim_cfg = config.get("simulation", {})
        self._step_size: float = float(sim_cfg.get("step_size", 1.0))
        # Expected full-throttle forward speed in Unity units/s.
        # motor_efficiency = measured_speed / expected_speed → [0, 1].
        self._expected_speed: float = self._step_size / max(self._velocity_dt, 1e-6)

        # Dead-reckoning heading — integrated from turn actions each step
        self._heading: float = 0.0
        self._turn_angle_deg: float = float(sim_cfg.get("turn_angle_deg", 45.0))

        # Proprioceptive stuck detection
        self._stuck_adapter_count: int = 0
        self._stuck_adapter_threshold: int = int(config.get("stuck_threshold", 5))

        # One-shot obs shape diagnostic — logged on first step to verify raycast presence
        self._obs_shapes_logged: bool = False

        # Previous step's world position (x, z) for position-delta motor efficiency.
        # None until first obs arrives.
        self._prev_wx: Optional[float] = None
        self._prev_wz: Optional[float] = None

    # ------------------------------------------------------------------
    # AbstractEnvironmentAdapter interface
    # ------------------------------------------------------------------

    def reset(self) -> AgentState:
        self._step_count = 0
        self._heading = 0.0
        self._homeostatic.reset()
        self._position_tracker.reset()
        self._stuck_adapter_count = 0
        self._prev_wx = None
        self._prev_wz = None

        env = self._get_or_create_env()
        env.reset()

        if DEBUG:
            _dbg(f"\n[DBG RESET] Behavior: {self._behavior_name}")
            try:
                spec = self._env.behavior_specs[self._behavior_name]
                _dbg(f"  action_spec:  {spec.action_spec}")
                _dbg(f"  obs_specs:    {spec.observation_specs}")
            except Exception as _e:
                _dbg(f"  (could not read behavior_specs: {_e})")
            _dbg(f"  velocity_dt={self._velocity_dt:.4f}s  "
                  f"expected_speed={self._expected_speed:.4f} Unity units/s")
            _dbg(f"  (if motor_efficiency always=0, expected_speed may be miscalibrated — "
                  f"divide one observed raw_fwd by {self._expected_speed:.3f} to diagnose)")
            sim_cfg = self._config.get("simulation", {})
            arena_size = float(sim_cfg.get("arena_size", 30.0))
            _dbg(f"  arena_size={arena_size}  (positions should be in [0, {arena_size}])")

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

        # Ground-truth motor signal from Unity proprioceptive obs (index 7: linearVelocity.magnitude).
        # _get_obs() already populated info["local_speed_forward"] from _extract_local_speed().
        raw_fwd = float(info.get("local_speed_forward", 0.0))

        if DEBUG:
            wx_dbg = info.get("x", 0.0)
            wy_dbg = info.get("y", 0.0)
            wz_dbg = info.get("z", 0.0)
            _rc_list = info.get("raycast_hits") or []
            rc_dbg = _rc_list[0] if _rc_list else {}
            rr_dbg = info.get("raw_reward", 0.0)
            _arena = float(self._config.get("simulation", {}).get("arena_size", 30.0))
            _dbg(f"[DBG STEP {self._step_count}] parsers →")
            _dbg(f"  speed:    fwd={raw_fwd:.4f}  "
                  f"right={info.get('local_speed_right', 0.0):.4f}  "
                  f"up={info.get('local_speed_up', 0.0):.4f}  "
                  f"({'non-zero' if raw_fwd != 0 else 'ALWAYS 0 = motor broken'})")
            _dbg(f"  world_pos: x={wx_dbg:.3f} y={wy_dbg:.3f} z={wz_dbg:.3f}  "
                  f"({'in_arena' if 0 < wx_dbg < _arena and 0 < wz_dbg < _arena else 'OUT OF ARENA — garbage?'})")
            _dbg(f"  rays({len(_rc_list)}):  fwd hit_tag={rc_dbg.get('hit_tag')!r} "
                  f"distance={rc_dbg.get('distance', 1.0):.4f}  "
                  f"({'no-hit sentinel' if rc_dbg.get('hit_tag') is None and rc_dbg.get('distance', 1.0) == 1.0 else 'hit detected'})")
            for _ri, _r in enumerate(_rc_list[1:], 1):
                if _r.get("hit_tag") is not None or _r.get("distance", 1.0) < 1.0:
                    _dbg(f"    ray[{_ri}] angle={_r.get('angle_deg', 0.0):+.1f}°  "
                          f"hit_tag={_r.get('hit_tag')!r}  distance={_r.get('distance', 1.0):.4f}")
            if raw_fwd == 0.0 and action_id in ("move_forward", "move_backward"):
                _dbg(f"  *** WARN: movement action sent but speed=0 — "
                      f"proprioceptive obs missing or wrong index")
            if not (0 < wx_dbg < _arena):
                _dbg(f"  *** WARN: world_pos.x={wx_dbg:.3f} out of arena — "
                      f"_extract_world_pos reading wrong array indices")
            if rc_dbg.get("hit_tag") is None and rc_dbg.get("distance", 1.0) == 1.0:
                _dbg(f"  *** WARN: raycast always returning no-hit sentinel — "
                      f"_parse_raycasts format mismatch (check n=7 path)")
            _dbg(f"  raw_reward={rr_dbg:.6f}  env_done={info.get('env_done', False)}")

        # Motor efficiency from consecutive world-position delta.
        # Using position rather than speed avoids the 2.8× inflation from Unity's
        # velocity magnitude including vertical bounce. Position delta over one step
        # divided by expected step size gives exactly the fraction of intended
        # displacement achieved: ~1.0 at full speed, ~0.0 at wall contact.
        curr_wx = float(info.get("x", 0.0))
        curr_wz = float(info.get("z", 0.0))

        if action_id in ("turn_left", "turn_right", "idle"):
            motor_efficiency = 1.0
            position_delta_norm = 0.0
            self._stuck_adapter_count = 0
        else:
            if self._prev_wx is not None and self._prev_wz is not None:
                dx = curr_wx - self._prev_wx
                dz = curr_wz - self._prev_wz
                actual_disp = float(np.sqrt(dx * dx + dz * dz))
                motor_efficiency = float(min(1.0, actual_disp / max(self._step_size, 1e-6)))
            else:
                # No prior position yet (first step after reset) — assume full movement.
                motor_efficiency = 1.0
            position_delta_norm = motor_efficiency

            if motor_efficiency < 0.3:
                self._stuck_adapter_count += 1
            else:
                self._stuck_adapter_count = 0

        self._prev_wx = curr_wx
        self._prev_wz = curr_wz

        info["motor_efficiency"] = motor_efficiency
        # position_delta_norm: fraction of expected displacement achieved this step.
        # Used by PredictionErrorCalculator for the motor PE (efference copy channel).
        info["position_delta_norm"] = position_delta_norm

        motor_stuck = self._stuck_adapter_count >= self._stuck_adapter_threshold
        info["motor_stuck"] = motor_stuck
        info["stuck_steps"] = self._stuck_adapter_count

        if DEBUG:
            _dbg(f"[DBG STEP {self._step_count}] motor signals →")
            _dbg(f"  action={action_id!r}  raw_fwd={raw_fwd:.4f}  "
                  f"expected_speed={self._expected_speed:.4f}  "
                  f"motor_efficiency={motor_efficiency:.4f}  "
                  f"position_delta_norm={position_delta_norm:.4f}  "
                  f"stuck_count={self._stuck_adapter_count}  motor_stuck={motor_stuck}")
            if action_id == "move_forward" and motor_efficiency == 0.0:
                _dbg(f"  *** WARN: motor_efficiency=0 on move_forward — "
                      f"expected_speed={self._expected_speed:.3f} but raw_fwd={raw_fwd:.4f}. "
                      f"Check _velocity_dt={self._velocity_dt:.4f}s and "
                      f"step_size={self._step_size} (stuck_count={self._stuck_adapter_count})")
            if action_id in ("turn_left", "turn_right"):
                _dbg(f"  heading: {self._heading:.1f}°  "
                      f"(should have changed by ±{self._turn_angle_deg}°)")

        # Scale to arena units/step for position dead-reckoning.
        info["local_speed_forward"] = raw_fwd * self._velocity_dt
        info["local_speed_right"] = 0.0

        self._homeostatic.step(raw_reward, env_done)

        # Sync HomeostaticWrapper health from Unity's ground-truth obs (index 0).
        # Unity is authoritative — it tracks food collection and hazard penalties
        # directly. The wrapper's simulated depletion diverges within a few steps.
        # Use the raw obs_list (threaded through info) rather than the extracted
        # visual array — _extract_unity_health needs a list of 1D sensor arrays.
        _raw_obs_list = info.pop("_obs_list", [])
        unity_health = _extract_unity_health(_raw_obs_list)
        if unity_health is not None:
            self._homeostatic.sync_health(unity_health)

        state = map_obs(obs, info, self._homeostatic, self._step_count, self._position_tracker)

        if DEBUG:
            h = state.homeostasis
            pos = state.position
            perc = state.perception
            res = state.resources
            _dbg(f"[DBG STEP {self._step_count}] AgentState →")
            _dbg(f"  homeostasis: health={h.health:.4f} sat={h.saturation:.4f} "
                  f"energy={h.energy:.4f} alive={h.is_alive}")
            _dbg(f"  position:    x={pos.x:.3f} z={pos.z:.3f} "
                  f"heading={pos.heading:.1f}°  vx={pos.velocity_x:.4f}")
            _dbg(f"  perception:  area_id={perc.area_id}  "
                  f"terrain_novelty={perc.terrain_novelty:.4f}  "
                  f"entity_density={perc.entity_density:.4f}")
            rc_s = perc.raycast_hits
            if rc_s:
                _dbg(f"  raycast_hits: tag={rc_s[0].get('hit_tag')!r}  "
                      f"dist={rc_s[0].get('distance', 1.0):.4f}")
            else:
                _dbg(f"  raycast_hits: None  *** WARN: no raycast in state")
            _dbg(f"  resources:   resource_level={res.resource_level:.4f}  "
                  f"threat_proximity={res.threat_proximity:.4f}")
            vf = perc.visual_features or []
            if vf:
                all_zero = all(v == 0.0 for v in vf)
                _dbg(f"  visual_features[0:6]={[round(v, 3) for v in vf[:6]]}  "
                      f"({'ALL ZERO — visual broken' if all_zero else 'ok'})")

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
            useRayCasts=True,
            raysPerSide=3,    # must match m_RaysPerDirection in AAI3Agent.prefab
            rayMaxDegrees=70, # must match m_MaxRayDegrees in AAI3Agent.prefab
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
            if DEBUG:
                self._debug_gate1(obs_list, raw_reward, env_done=True)
            visual_obs = self._extract_visual(obs_list)
            fwd, right, up = _extract_local_speed(obs_list)
            wx, wy, wz = _extract_world_pos(obs_list)
            raycast_hits = _parse_raycasts(obs_list)
            return visual_obs, {
                "raw_reward": raw_reward,
                "env_done": True,
                "local_speed_forward": fwd,
                "local_speed_right": right,
                "local_speed_up": up,
                "x": wx,
                "y": wy,
                "z": wz,
                "raycast_hits": raycast_hits,
                "_obs_list": obs_list,
            }

        if len(decision_steps) == 0:
            return None, {"raw_reward": 0.0, "env_done": False}

        agent_id = list(decision_steps.agent_id)[0]
        obs_list = decision_steps[agent_id].obs
        raw_reward = float(decision_steps[agent_id].reward)
        self._log_obs_shapes(obs_list)
        if DEBUG:
            self._debug_gate1(obs_list, raw_reward, env_done=False)
        visual_obs = self._extract_visual(obs_list)
        fwd, right, up = _extract_local_speed(obs_list)
        wx, wy, wz = _extract_world_pos(obs_list)
        raycast_hits = _parse_raycasts(obs_list)
        return visual_obs, {
            "raw_reward": raw_reward,
            "env_done": False,
            "local_speed_forward": fwd,
            "local_speed_right": right,
            "local_speed_up": up,
            "x": wx,
            "y": wy,
            "z": wz,
            "raycast_hits": raycast_hits,
            "_obs_list": obs_list,
        }

    def _debug_gate1(self, obs_list: List[Any], raw_reward: float, env_done: bool) -> None:
        """Gate 1: Print raw obs_list contents received from Unity this step."""
        step = self._step_count
        _dbg(f"\n[DBG STEP {step}] obs_list has {len(obs_list)} arrays")
        for i, obs in enumerate(obs_list):
            arr = np.asarray(obs)
            _dbg(f"  obs[{i}]: shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
            if arr.ndim == 1 and len(arr) <= 20:
                _dbg(f"    values: {arr.round(4).tolist()}")
            elif arr.ndim == 3:
                _dbg(f"    visual: H={arr.shape[0]} W={arr.shape[1]} C={arr.shape[2]}  "
                      f"normalized={'yes' if arr.max() <= 1.0 else 'no'}")
        _dbg(f"  raw_reward={raw_reward:.6f}  env_done={env_done}")

    def _log_obs_shapes(self, obs_list: List[Any]) -> None:
        """Log obs_list shapes once on the first step to confirm raycast presence."""
        if self._obs_shapes_logged:
            return
        self._obs_shapes_logged = True
        shapes = [np.asarray(o).shape for o in obs_list if hasattr(o, '__len__')]
        log.info("obs_list shapes (step 1): %s", shapes)

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
