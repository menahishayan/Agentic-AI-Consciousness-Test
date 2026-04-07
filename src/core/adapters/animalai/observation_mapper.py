"""
Observation Mapper — converts raw Animal AI observations into AgentState.

Animal AI 4 provides two observation arrays per step:

  1. Visual observation: numpy array of shape (H, W, 3) or (3, H, W) — RGB camera

  2. Agent proprioceptive vector: 1D float array of exactly 10 elements from
     TrainingAgent.CollectObservations (VectorSensor):
       [0]   health              (normalised 0–1)
       [1–3] localVel.x/y/z     (local-space velocity, Unity units/s)
       [4–6] localPos.x/y/z     (world position normalised by _arenaSize)
       [7]   speed_magnitude     (Rigidbody.linearVelocity.magnitude, Unity units/s)
       [8]   ray_hit_fraction    (forward ray distance, 0=agent 1=no hit / max range)
       [9]   ray_tag_index       (float index: 0=none, 1=GoodGoal, 2=GoodGoalMulti,
                                  3=BadGoal, 4=Immovable/wall, 5=OuterWall/wall)
     Indices 8–9 are injected by CollectObservations via CollectRaycastObservations()
     since Animal AI 5.x does not forward RayPerceptionSensor output to Python obs_list.

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
# 10 floats: health(1) + localVel(3) + worldPos_normalised(3) + speed_magnitude(1) + ray(2)
_AGENT_OBS_LEN = 10
_AGENT_OBS_SPEED_IDX = 7        # linearVelocity.magnitude — raw Unity units/s
_AGENT_OBS_RAY_FRACTION_IDX = 8 # forward ray hit_fraction (0=agent, 1=max range / no hit)
_AGENT_OBS_RAY_TAG_IDX = 9      # forward ray tag encoded as float index (see _TagToIndex in C#)

# TrainingAgent._arenaSize used to normalise world position before sending.
# Animal AI's default arena is 40×40 Unity units. Must match the C# constant.
_ARENA_SIZE_UNITY = 40.0

# Maps the float tag index from _TagToIndex (TrainingAgent.cs) to internal canonical names.
# Index 0 = no hit / unrecognised tag.  Used by the 10-float vector obs path (post-rebuild).
_RAY_TAG_INDEX_MAP: Dict[float, Optional[str]] = {
    0.0: None,
    1.0: "GoodGoal",
    2.0: "GoodGoalMulti",
    3.0: "BadGoal",
    4.0: "wall",   # Immovable
    5.0: "wall",   # OuterWall
}

# ---------------------------------------------------------------------------
# RayPerceptionSensor separate obs array — current pre-rebuild binary
#
# shape (98,) = 7 rays × 14 floats, format [one_hot(13), hit_fraction]
# ML-Agents 1.1.0 uses N+1 per ray (no separate hit_bool).
#
# Tag order matches the 13-tag detectable list compiled into the binary prefab.
# Index mapping (same order as m_DetectableTags, binary predates DecoyGoalBounce):
#   0=arena, 1=Immovable, 2=Movable, 3=goodGoal, 4=goodGoalMulti, 5=badGoal,
#   6=GoalSpawner, 7=DeathZone, 8=HotZone, 9=Ramp, 10=PillarButton,
#   11=SignPoster, 12=DecoyGoal
#
# After rebuild (15 tags + OuterWall): shape becomes (112,) = 7 × (15+1).
# The 10-float vector obs path supersedes this once rebuilt, but this path
# provides working ray detection in the interim.
# ---------------------------------------------------------------------------
_RAY_SENSOR_TAG_REMAP: Dict[int, Optional[str]] = {
    0:  "arena",        # floor/ceiling
    1:  "wall",         # Immovable (obstacle walls)
    2:  None,           # Movable — not relevant for nav
    3:  "GoodGoal",
    4:  "GoodGoalMulti",
    5:  "BadGoal",
    6:  None,           # GoalSpawner
    7:  None,           # DeathZone
    8:  None,           # HotZone
    9:  None,           # Ramp
    10: None,           # PillarButton
    11: None,           # SignPoster
    12: None,           # DecoyGoal
    # After rebuild: 13=DecoyGoalBounce, 14=OuterWall (→ "wall")
    14: "wall",         # OuterWall (post-rebuild binary)
}
_RAY_SENSOR_TAGS_PER_RAY_LEGACY = 13    # pre-rebuild: 13 tags + 1 fraction = 14 per ray
_RAY_SENSOR_TAGS_PER_RAY_NEW    = 15    # post-rebuild: 15 tags + 1 fraction = 16 per ray
_RAY_SENSOR_N_RAYS = 7                  # 2 * raysPerSide(3) + 1 centre
_RAY_SENSOR_LEN_LEGACY = _RAY_SENSOR_N_RAYS * (_RAY_SENSOR_TAGS_PER_RAY_LEGACY + 1)  # 98
_RAY_SENSOR_LEN_NEW    = _RAY_SENSOR_N_RAYS * (_RAY_SENSOR_TAGS_PER_RAY_NEW    + 1)  # 112

# Compact single-ray format from current binary: 1 ray × (6 one-hot tags + 1 hit_fraction).
# Tag order matches the first 6 entries of _RAY_SENSOR_TAG_REMAP:
#   0=arena, 1=Immovable(wall), 2=Movable, 3=GoodGoal, 4=GoodGoalMulti, 5=BadGoal
_RAY_SENSOR_LEN_COMPACT          = 7
_RAY_SENSOR_TAGS_PER_RAY_COMPACT = 6


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

    # Raycast — forward-ray dict parsed from RayPerceptionSensorComponent3D obs
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
            "stuck_steps": info.get("stuck_steps", 0),
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

    Searched in reverse so the proprioceptive obs (last 1D array in obs_list)
    is found before the compact ray obs (also 7 elements, earlier in the list).

    Current binary (7-element prop obs layout):
      [0] health  [1] vel_x  [2] vel_y  [3] vel_z  [4] pos_x/40  [5] pos_y/40  [6] pos_z/40
      arr[3] = vel_z ≈ local-forward velocity (Unity units/s, ~17 u/s at full throttle)

    Post-rebuild (10-element prop obs):
      arr[7] = linearVelocity.magnitude — authoritative speed signal

    NOTE: Motor efficiency is now computed from consecutive position delta in
    env_adapter.py rather than raw speed. This function is used only as a
    fallback for dead-reckoning when world position is unavailable.

    Returns (fwd_speed, 0.0, 0.0).
    """
    for obs in reversed(obs_list):
        arr = np.asarray(obs)
        if arr.ndim != 1:
            continue
        n = len(arr)
        if n == _AGENT_OBS_LEN and n > _AGENT_OBS_SPEED_IDX:
            return float(arr[_AGENT_OBS_SPEED_IDX]), 0.0, 0.0
        if n == 7:
            # 7-element prop obs: vel_z at index 3 approximates forward speed
            return float(arr[3]), 0.0, 0.0
    return 0.0, 0.0, 0.0


def _extract_world_pos(obs_list: List[Any]) -> Tuple[float, float, float]:
    """
    Extract world position from proprioceptive obs indices [4, 5, 6] and un-normalise.

    TrainingAgent.CollectObservations divides transform.position by _arenaSize before
    sending, so multiply by _ARENA_SIZE_UNITY to recover Unity world coordinates.

    Both the 10-element (post-rebuild) and 7-element (current binary) proprioceptive
    obs have normalised position at [4, 5, 6]:
      7-element:  [health, vel_x, vel_y, vel_z, px/40, py/40, pz/40]
      10-element: [health, vel_x, vel_y, vel_z, px/40, py/40, pz/40, speed_mag, ray_frac, ray_tag]

    Searched in REVERSE so the proprioceptive obs (last in obs_list) is found before
    the compact ray obs (also 7 elements, earlier in obs_list). The ray obs has
    one-hot values at [0:6] which would be misread as fractional positions — reversing
    the search eliminates the collision without needing a content-based discriminator.
    """
    for obs in reversed(obs_list):
        arr = np.asarray(obs)
        if arr.ndim == 1 and len(arr) in (_AGENT_OBS_LEN, 7):
            x = float(arr[4]) * _ARENA_SIZE_UNITY
            y = float(arr[5]) * _ARENA_SIZE_UNITY
            z = float(arr[6]) * _ARENA_SIZE_UNITY
            return x, y, z
    return 0.0, 0.0, 0.0


def _extract_unity_health(obs_list: List[Any]) -> Optional[float]:
    """
    Extract Unity's ground-truth health from the proprioceptive obs (index 0).

    Returns None if the proprioceptive obs is not found (caller should not sync).
    Searched in reverse for the same collision-avoidance reason as _extract_world_pos.
    """
    for obs in reversed(obs_list):
        arr = np.asarray(obs)
        if arr.ndim == 1 and len(arr) in (_AGENT_OBS_LEN, 7):
            return float(np.clip(arr[0], 0.0, 1.0))
    return None


def _parse_raycasts(obs_list: List[Any]) -> Dict[str, Any]:
    """
    Extract forward ray data from obs_list, handling three binary states:

    Post-rebuild (preferred): 10-float vector obs [indices 8–9]
        TrainingAgent.CollectObservations encodes the forward ray directly:
          [8] hit_fraction, [9] tag_index via _TagToIndex()
        Matched when len == _AGENT_OBS_LEN (10).

    Pre-rebuild legacy: separate RayPerceptionSensor array (98 or 112 floats)
        ML-Agents 1.1.0 format: [one_hot(N), hit_fraction] per ray, N+1 floats/ray.
        Pre-rebuild binary: N=13, total=98. Post-rebuild (before next rebuild): N=15, total=112.
        Forward ray = index 0 (AlternatingRayOrder=1, center-first).

    Returns the no-hit sentinel {"hit_tag": None, "distance": 1.0} when no matching
    array is found.
    """
    for obs in obs_list:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1:
            continue

        n = len(arr)

        # Post-rebuild: forward ray embedded in 10-float vector obs
        if n == _AGENT_OBS_LEN:
            fraction = float(arr[_AGENT_OBS_RAY_FRACTION_IDX])
            tag = _RAY_TAG_INDEX_MAP.get(round(float(arr[_AGENT_OBS_RAY_TAG_IDX])), None)
            return {"hit_tag": tag, "distance": fraction}

        # Compact single-ray format from current binary: [one_hot(6), hit_fraction]
        # 1 ray × 6 one-hot tags + 1 distance float = 7 elements.
        if n == _RAY_SENSOR_LEN_COMPACT:
            one_hot = arr[:_RAY_SENSOR_TAGS_PER_RAY_COMPACT]
            fraction = float(arr[_RAY_SENSOR_TAGS_PER_RAY_COMPACT])
            best_idx = int(np.argmax(one_hot))
            hit_tag: Optional[str] = (
                _RAY_SENSOR_TAG_REMAP.get(best_idx)
                if float(one_hot[best_idx]) > 0.5 else None
            )
            return {"hit_tag": hit_tag, "distance": fraction}

        # Pre-rebuild: separate RayPerceptionSensor flat array (7-ray full format)
        if n in (_RAY_SENSOR_LEN_LEGACY, _RAY_SENSOR_LEN_NEW):
            n_tags = _RAY_SENSOR_TAGS_PER_RAY_LEGACY if n == _RAY_SENSOR_LEN_LEGACY else _RAY_SENSOR_TAGS_PER_RAY_NEW
            # Forward ray is at offset 0 (AlternatingRayOrder=1, center first)
            one_hot = arr[:n_tags]
            fraction = float(arr[n_tags])
            best_idx = int(np.argmax(one_hot))
            hit_tag = (
                _RAY_SENSOR_TAG_REMAP.get(best_idx)
                if float(one_hot[best_idx]) > 0.5 else None
            )
            return {"hit_tag": hit_tag, "distance": fraction}

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
