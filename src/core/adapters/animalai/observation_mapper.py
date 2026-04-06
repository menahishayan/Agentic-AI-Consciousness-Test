"""
Observation Mapper — converts raw Animal AI observations into AgentState.

Animal AI 4 provides two observation arrays per step:

  1. Visual observation: numpy array of shape (H, W, 3) or (3, H, W) — RGB camera

  2. Agent proprioceptive vector: 1D float array of exactly 8 elements from
     TrainingAgent.CollectObservations (VectorSensor):
       [0]   health           (Unity units, 0–100)
       [1–3] localVel.x/y/z  (local-space velocity, Unity units/s)
       [4–6] localPos.x/y/z  (world position, Unity units)
       [7]   speed_magnitude  (Rigidbody.linearVelocity.magnitude, Unity units/s)
     Index 7 is the proprioceptive speed signal used for motor PE computation:
       free motion → ~step_size/dt (≈18 units/s at default config)
       wall contact → ~0 units/s

  NOTE: Directional raycast sensors are not yet added to the ML-Agents obs pipeline.
  Resource/threat estimation falls back to visual heuristics until a
  RayPerceptionSensorComponent3D is attached to the agent prefab.

We extract a compact feature vector from the visual observation without GPU:
  - Downsample to 21×21
  - Per-channel (R, G, B) mean and variance → 6 floats
  - Edge density via simple gradient magnitude → 1 float
  - Brightness histogram (4 buckets) → 4 floats
  - Directional split (left/center/right thirds): green dominance + edge density per region → 6 floats
  Total: 17-dim visual_features

  Index map:
    [0] R mean,  [1] R var
    [2] G mean,  [3] G var
    [4] B mean,  [5] B var
    [6] edge_density
    [7–10] brightness histogram (4 buckets)
    [11] left green dominance,  [12] left edge density
    [13] center green dominance, [14] center edge density
    [15] right green dominance,  [16] right edge density

Area ID is derived from a discretized position grid.
Terrain novelty and entity density are heuristically estimated from visual features.
Raycast-based resource/threat estimates take priority over visual heuristics when available.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.adapters.animalai.homeostatic_wrapper import HomeostaticWrapper
from core.models.state import (
    AgentState,
    HomeostasisState,
    PerceptionState,
    PositionState,
    ResourceState,
)

# Downsampled resolution for feature extraction
_THUMB_SIZE = 21
# Number of brightness histogram buckets
_HIST_BUCKETS = 4

# Agent proprioceptive observation layout (TrainingAgent.CollectObservations VectorSensor).
# 8 floats: health(1) + localVel(3) + worldPos_normalised(3) + speed_magnitude(1)
_AGENT_OBS_LEN = 8
_AGENT_OBS_SPEED_IDX = 7   # linearVelocity.magnitude — raw Unity units/s

# TrainingAgent._arenaSize used to normalise world position before sending.
# Must match the C# constant (currently 30f).
_ARENA_SIZE_UNITY = 30.0

# Raycast sensor layout — emitted by a RayPerceptionSensorComponent3D on the agent prefab
# as a SEPARATE array in obs_list, distinct from the 8-float proprioceptive obs.
# Default Animal AI single-ray config: 1 ray × 6 detectable tags + 1 distance = 7 floats.
#   distance: 1.0 = max range / no hit; <1.0 = object at that fraction of range.
# NOTE: RayPerceptionSensorComponent3D is not yet attached to the agent prefab; until
# it is, _parse_raycasts will find no 7-float array and return the no-hit sentinel.
_AAI_RAY_TAGS = ["GoodGoal", "GoodGoalMulti", "BadGoal", "BadGoalMulti", "wall", "ramp"]
_AAI_RAY_OBS_LEN = len(_AAI_RAY_TAGS) + 1  # 7


def map_obs(
    visual_obs: Optional[np.ndarray],
    info: Dict[str, Any],
    homeostatic: HomeostaticWrapper,
    step: int,
    position_tracker: "_PositionTracker",
) -> AgentState:
    """
    Build an AgentState from raw Animal AI step output.

    Args:
        visual_obs: RGB frame from Animal AI camera (H×W×3 or 3×H×W), or None
        info: Step info dict from Animal AI environment
        homeostatic: HomeostaticWrapper holding current physiological state
        step: Current episode step number
        position_tracker: Stateful integrator for position estimation
    """
    # Extract visual features
    visual_features = _extract_visual_features(visual_obs)

    # Estimate position
    pos = position_tracker.update(info)

    # Build area ID from discretized position
    area_id = _build_area_id(pos.x or 0.0, pos.z or 0.0)

    # Compute derived perception signals from visual features
    terrain_novelty = _estimate_terrain_novelty(visual_features)
    entity_density = _estimate_entity_density(visual_features)

    # Raycast — single-ray dict from AAI4's (7,) vector obs
    raycast: Dict[str, Any] = info.get("raycast_hits") or {"hit_tag": None, "distance": 1.0}
    hit_tag = raycast.get("hit_tag")
    detected_objects: List[str] = [hit_tag] if hit_tag else []

    # Resource/threat estimates: raycasts take priority over visual heuristics
    resource_level = _estimate_resource(visual_features, raycast)
    threat_proximity = _estimate_threat(visual_features, raycast)

    hs = homeostatic.get_state()

    return AgentState(
        homeostasis=HomeostasisState(
            health=hs["health"],
            saturation=hs["saturation"],
            energy=hs["energy"],
            is_alive=homeostatic.is_alive,
        ),
        position=pos,
        perception=PerceptionState(
            visual_features=visual_features,
            detected_objects=detected_objects,
            area_id=area_id,
            terrain_novelty=terrain_novelty,
            entity_density=entity_density,
            raycast_hits=[raycast] if hit_tag or raycast["distance"] < 1.0 else None,
        ),
        resources=ResourceState(
            resource_level=resource_level,
            threat_proximity=threat_proximity,
        ),
        step=step,
        raw_metadata={
            "raw_reward": info.get("raw_reward", 0.0),
            "env_done": info.get("env_done", False),
            "motor_stuck": info.get("motor_stuck", False),
            "motor_efficiency": info.get("motor_efficiency", 1.0),
            "position_delta_norm": info.get("position_delta_norm", 0.0),
        },
    )


def _extract_visual_features(obs: Optional[np.ndarray]) -> List[float]:
    """Extract compact 11-dim feature vector from RGB frame."""
    if obs is None:
        return [0.0] * 17

    arr = np.asarray(obs, dtype=np.float32)

    # Normalize to [0,1]
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Ensure (H, W, 3) layout
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)

    if arr.ndim != 3 or arr.shape[2] != 3:
        return [0.0] * 17

    # Downsample to thumbnail for fast processing
    h, w = arr.shape[:2]
    step_h = max(1, h // _THUMB_SIZE)
    step_w = max(1, w // _THUMB_SIZE)
    thumb = arr[::step_h, ::step_w, :][:_THUMB_SIZE, :_THUMB_SIZE, :]

    # Per-channel stats (mean + variance) → 6 values
    channel_features: List[float] = []
    for c in range(3):
        ch = thumb[:, :, c].flatten()
        channel_features.append(float(ch.mean()))
        channel_features.append(float(ch.var()))

    # Edge density via gradient magnitude → 1 value
    gray = thumb.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float(min((gx + gy) / 2.0, 1.0))

    # Brightness histogram (4 buckets) → 4 values
    brightness = gray.flatten()
    hist, _ = np.histogram(brightness, bins=_HIST_BUCKETS, range=(0.0, 1.0), density=True)
    hist_features = [float(v) / max(float(hist.max()), 1e-6) for v in hist]

    # Directional features: left/center/right thirds → green dominance + edge density each
    # thumb width = up to _THUMB_SIZE=21 columns; split into [0:7], [7:14], [14:]
    w_thumb = thumb.shape[1]
    third = max(1, w_thumb // 3)
    region_slices = [
        thumb[:, :third, :],
        thumb[:, third:third * 2, :],
        thumb[:, third * 2:, :],
    ]
    directional_features: List[float] = []
    for region in region_slices:
        if region.size == 0:
            directional_features.extend([0.0, 0.0])
            continue
        r_m = float(region[:, :, 0].mean())
        g_m = float(region[:, :, 1].mean())
        b_m = float(region[:, :, 2].mean())
        green_dom = float(max(0.0, g_m - (r_m + b_m) / 2.0))
        gray_r = region.mean(axis=2)
        reg_edge = float(min(
            (np.abs(np.diff(gray_r, axis=1)).mean() + np.abs(np.diff(gray_r, axis=0)).mean()) / 2.0,
            1.0,
        ))
        directional_features.extend([green_dom, reg_edge])

    return channel_features + [edge_density] + hist_features + directional_features


def _extract_local_speed(obs_list: List[Any]) -> Tuple[float, float, float]:
    """
    Extract proprioceptive speed from Animal AI's vector observation.

    TrainingAgent.CollectObservations emits an 8-float VectorSensor array:
      [health, localVel.x, localVel.y, localVel.z, pos.x, pos.y, pos.z, speed_mag]

    Index 7 (linearVelocity.magnitude) is the authoritative proprioceptive speed:
      free motion  → ~step_size / velocity_dt (≈18 Unity units/s at default config)
      wall contact → ~0 Unity units/s

    Returns (speed_magnitude, 0.0, 0.0) so callers receive forward speed in index 0.
    Right and up components are not needed for the current motor PE pipeline.
    """
    for obs in obs_list:
        arr = np.asarray(obs)
        if arr.ndim == 1 and len(arr) == _AGENT_OBS_LEN:
            return float(arr[_AGENT_OBS_SPEED_IDX]), 0.0, 0.0
    return 0.0, 0.0, 0.0


def _extract_world_pos(obs_list: List[Any]) -> Tuple[float, float, float]:
    """
    Extract world position from proprioceptive obs indices [4, 5, 6] and un-normalise.

    TrainingAgent.CollectObservations divides transform.position by _arenaSize before
    sending, so multiply by _ARENA_SIZE_UNITY to recover Unity world coordinates.
    """
    for obs in obs_list:
        arr = np.asarray(obs)
        if arr.ndim == 1 and len(arr) == _AGENT_OBS_LEN:
            x = float(arr[4]) * _ARENA_SIZE_UNITY
            y = float(arr[5]) * _ARENA_SIZE_UNITY
            z = float(arr[6]) * _ARENA_SIZE_UNITY
            return x, y, z
    return 0.0, 0.0, 0.0


def _parse_raycasts(obs_list: List[Any]) -> Dict[str, Any]:
    """
    Parse the directional raycast observation from obs_list.

    A RayPerceptionSensorComponent3D on the agent prefab emits a separate 1D float
    array of exactly _AAI_RAY_OBS_LEN (7) floats — distinct from the 8-float
    proprioceptive obs so there is no collision risk:
        [GoodGoal, GoodGoalMulti, BadGoal, BadGoalMulti, wall, ramp, distance]
    First 6: one-hot tag encoding (>0.5 = hit). Last: normalised distance.

    If no 7-float array is present (sensor not yet attached to the prefab), returns
    the no-hit sentinel {"hit_tag": None, "distance": 1.0} so the rest of the
    pipeline degrades gracefully to visual heuristics.
    """
    for obs in obs_list:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1 and len(arr) == _AAI_RAY_OBS_LEN:
            one_hot = arr[:len(_AAI_RAY_TAGS)]
            distance = float(arr[len(_AAI_RAY_TAGS)])
            best_idx = int(np.argmax(one_hot))
            hit_tag: Optional[str] = (
                _AAI_RAY_TAGS[best_idx] if float(one_hot[best_idx]) > 0.5 else None
            )
            return {"hit_tag": hit_tag, "distance": distance}
    return {"hit_tag": None, "distance": 1.0}


def _build_area_id(x: float, z: float) -> str:
    """Discretize position to 5-unit grid cells."""
    gx = int(x / 5.0)
    gz = int(z / 5.0)
    return f"x{gx}z{gz}"


def _estimate_terrain_novelty(features: List[float]) -> float:
    """High variance + high edge density → potentially novel terrain."""
    if len(features) < 11:
        return 0.5
    avg_variance = sum(features[1::2][:3]) / 3.0   # channel variances
    edge = features[6]
    novelty = (avg_variance * 0.6 + edge * 0.4)
    return float(min(novelty * 2.0, 1.0))


def _estimate_entity_density(features: List[float]) -> float:
    """High edge density with mid-range brightness suggests entities nearby."""
    if len(features) < 11:
        return 0.0
    edge = features[6]
    mean_brightness = sum(features[0::2][:3]) / 3.0
    mid_brightness = 1.0 - abs(mean_brightness - 0.5) * 2.0
    density = edge * 0.7 + mid_brightness * 0.3
    return float(min(density, 1.0))


def _estimate_resource(
    features: List[float],
    raycast: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Estimate food resource level.
    Primary: GoodGoal/GoodGoalMulti raycast hit (accurate, directional).
    Fallback: green channel dominance from visual features (heuristic).
    """
    if raycast is not None:
        hit_tag = raycast.get("hit_tag")
        if hit_tag in ("GoodGoal", "GoodGoalMulti"):
            # 1.0 = food right next to agent, fades linearly with distance
            return float(max(0.0, 1.0 - raycast["distance"]))
        # Raycast active but no food visible
        return 0.1

    # Visual fallback
    if len(features) < 6:
        return 0.5
    r_mean, g_mean, b_mean = features[0], features[2], features[4]
    green_dominance = max(0.0, g_mean - (r_mean + b_mean) / 2.0)
    return float(min(green_dominance * 3.0 + 0.1, 1.0))


def _estimate_threat(
    features: List[float],
    raycast: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Estimate threat proximity.
    Primary: BadGoal/BadGoalMulti raycast hit (accurate). Walls are lower-weight.
    Fallback: red channel anomaly from visual features (heuristic).
    """
    if raycast is not None:
        hit_tag = raycast.get("hit_tag")
        dist = raycast.get("distance", 1.0)
        if hit_tag in ("BadGoal", "BadGoalMulti"):
            return float(1.0 - dist)
        elif hit_tag == "wall":
            # Walls are always nearby in bounded arenas — lower weight
            return float((1.0 - dist) * 0.25)
        return 0.0

    # Visual fallback
    if len(features) < 6:
        return 0.0
    r_mean, g_mean, b_mean = features[0], features[2], features[4]
    red_anomaly = max(0.0, r_mean - (g_mean + b_mean) / 2.0)
    return float(min(red_anomaly * 2.0, 1.0))


class _PositionTracker:
    """
    Integrates velocity to estimate position when Animal AI doesn't
    directly expose world coordinates. Falls back to zero if unavailable.
    """

    def __init__(self) -> None:
        self._x = 0.0
        self._z = 0.0

    def reset(self) -> None:
        self._x = 0.0
        self._z = 0.0

    def update(self, info: Dict[str, Any]) -> PositionState:
        x = info.get("x", self._x)
        y = info.get("y", 0.0)
        z = info.get("z", self._z)
        heading = info.get("heading", 0.0)
        # Prefer real local speeds from Animal AI's vector obs; fall back to
        # the velocity_x/z keys set by the adapter as a proxy.
        vx = info.get("local_speed_forward", info.get("velocity_x", 0.0))
        vz = info.get("local_speed_right", info.get("velocity_z", 0.0))

        heading_rad = math.radians(heading)
        self._x += vx * math.cos(heading_rad)
        self._z += vx * math.sin(heading_rad)

        return PositionState(
            x=float(x),
            y=float(y),
            z=float(z),
            heading=float(heading),
            velocity_x=float(vx),
            velocity_z=float(vz),
        )
