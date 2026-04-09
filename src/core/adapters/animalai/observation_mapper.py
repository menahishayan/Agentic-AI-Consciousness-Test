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
# RayPerceptionSensor obs array
#
# ML-Agents format: [one_hot(N_tags), hit_fraction] per ray, N+1 floats per ray.
#
# TWO possible layouts depending on binary build:
#
# 13-tag layout (legacy, raysPerSide=3):   7 × 14 = 98 floats
#   0=goodGoal, 1=goodGoalMulti, 2=badGoal, 3=?, 4=Immovable, 5=OuterWall, ...
#   (confirmed empirically: hot_slot=0 fires at food collection — RAY_DUMP logs)
#
# 13-tag layout (current binary, raysPerSide=4): 9 × 14 = 126 floats
#   Same tag order as above.
#
# 15-tag layout (current prefab m_DetectableTags, raysPerSide=4): 9 × 16 = 144 floats
#   0=arena, 1=OuterWall, 2=Immovable, 3=Movable, 4=goodGoal, 5=goodGoalMulti,
#   6=badGoal, 7=GoalSpawner, 8=DeathZone, 9=HotZone, 10=Ramp, 11=PillarButton,
#   12=SignPoster, 13=DecoyGoal, 14=DecoyGoalBounce
#
# CRITICAL: goodGoal is at index 0 in 13-tag layout, index 4 in 15-tag layout.
# Using the wrong remap silently drops all food detections.
# The actual array size is logged on step 1 — check run logs to confirm.
# ---------------------------------------------------------------------------

# 13-tag remap: used for 98-float (7-ray) and 126-float (9-ray) arrays.
# Indices are the compiled m_DetectableTags slot order, NOT _TagToIndex return values.
# Confirmed empirically: hot_slot=0 fires at GoodGoal food collection (RAY_DUMP logs).
_RAY_SENSOR_TAG_REMAP_13: Dict[int, Optional[str]] = {
    0:  "GoodGoal",       # confirmed empirically — hot_slot=0 at food collection
    1:  "GoodGoalMulti",
    2:  "BadGoal",
    3:  None,
    4:  "wall",           # Immovable
    5:  "wall",           # OuterWall
    6:  None,
    7:  None,
    8:  None,
    9:  None,
    10: None,
    11: None,
    12: None,
}

# 15-tag remap: used for 144-float (9-ray) arrays (current prefab m_DetectableTags)
_RAY_SENSOR_TAG_REMAP_15: Dict[int, Optional[str]] = {
    0:  "arena",
    1:  "wall",           # OuterWall
    2:  "wall",           # Immovable
    3:  None,             # Movable
    4:  "GoodGoal",       # goodGoal  ← index 4 in 15-tag layout
    5:  "GoodGoalMulti",  # goodGoalMulti
    6:  "BadGoal",
    7:  None,             # GoalSpawner
    8:  None,             # DeathZone
    9:  None,             # HotZone
    10: None,             # Ramp
    11: None,             # PillarButton
    12: None,             # SignPoster
    13: None,             # DecoyGoal
    14: None,             # DecoyGoalBounce
}

# Backward-compat alias — code outside this module that imported _RAY_SENSOR_TAG_REMAP
# gets the 15-tag remap (the prefab's declared layout).
_RAY_SENSOR_TAG_REMAP = _RAY_SENSOR_TAG_REMAP_15

_RAY_SENSOR_TAGS_PER_RAY_LEGACY = 13
_RAY_SENSOR_TAGS_PER_RAY_NEW    = 15
_RAY_SENSOR_N_RAYS_LEGACY  = 7   # raysPerSide=3
_RAY_SENSOR_N_RAYS_CURRENT = 9   # raysPerSide=4
_RAY_SENSOR_LEN_LEGACY  = _RAY_SENSOR_N_RAYS_LEGACY  * (_RAY_SENSOR_TAGS_PER_RAY_LEGACY + 1)  # 98
_RAY_SENSOR_LEN_CURRENT = _RAY_SENSOR_N_RAYS_CURRENT * (_RAY_SENSOR_TAGS_PER_RAY_LEGACY + 1)  # 126
_RAY_SENSOR_LEN_NEW     = _RAY_SENSOR_N_RAYS_CURRENT * (_RAY_SENSOR_TAGS_PER_RAY_NEW    + 1)  # 144

# Ray angles per format (AlternatingRayOrder=1, center-first)
_RAY_ANGLES_7 = [0.0, 23.3, -23.3, 46.6, -46.6, 70.0, -70.0]   # raysPerSide=3, maxDeg=70
_RAY_ANGLES_9 = [0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0, 80.0, -80.0]  # raysPerSide=4, maxDeg=80


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

    # Raycasts — list of all rays parsed from RayPerceptionSensorComponent3D obs.
    # _parse_raycasts always returns a list; may be a 1-entry list for compact formats.
    rays: List[Dict[str, Any]] = info.get("raycast_hits") or []

    # Collect all detected tags across all rays for detected_objects
    detected_objects: List[str] = list({
        r["hit_tag"] for r in rays if r.get("hit_tag")
    })

    # Use forward ray (index 0) as the primary signal for resource/threat estimates,
    # falling back to the closest food/threat ray across the full fan.
    forward_ray = rays[0] if rays else {"hit_tag": None, "distance": 1.0}
    food_ray = next(
        (r for r in rays if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")),
        forward_ray,
    )
    threat_ray = next(
        (r for r in rays if r.get("hit_tag") in ("BadGoal", "BadGoalMulti")),
        forward_ray,
    )

    # Resource/threat estimates: raycasts take priority over visual heuristics
    resource_level = _estimate_resource(visual_features, food_ray)
    threat_proximity = _estimate_threat(visual_features, threat_ray)

    # Store all rays in perception; None only if sentinel (no real obs arrived)
    any_real_hit = any(
        r.get("hit_tag") is not None or r.get("distance", 1.0) < 1.0
        for r in rays
    )
    raycast_hits_out = rays if any_real_hit else None

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
            raycast_hits=raycast_hits_out,
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


def _parse_raycasts(obs_list: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract all ray hits from obs_list. Returns a list of dicts:
      [{"hit_tag": str|None, "distance": float, "angle_deg": float}, ...]

    Ray ordering follows AlternatingRayOrder=1 (center-first).
    Pre-rebuild (7 rays, raysPerSide=3, maxDeg=70):
      0=0°, 1=+23.3°, 2=-23.3°, 3=+46.6°, 4=-46.6°, 5=+70°, 6=-70°
    Post-rebuild (9 rays, raysPerSide=4, maxDeg=80):
      0=0°, 1=+20°, 2=-20°, 3=+40°, 4=-40°, 5=+60°, 6=-60°, 7=+80°, 8=-80°

    Handles two obs formats:
      Post-rebuild 10-float vector obs: encodes only the forward ray → 1 entry.
      Full ray array (98 legacy / 144 post-rebuild floats): all rays decoded.

    Returns the no-hit sentinel list when no matching array is found.
    """
    _ANGLES_LEGACY = [0.0, 23.3, -23.3, 46.6, -46.6, 70.0, -70.0]
    _ANGLES_NEW    = [0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0, 80.0, -80.0]

    # Pass 1: prefer the 10-float vector obs (fresh raycast injected by CollectObservations).
    # The 126-float RayPerceptionSensor array can be stale/alternating due to Unity's
    # sensor double-buffering — always use the 10-float path when it's available.
    for obs in obs_list:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1:
            continue
        if len(arr) == _AGENT_OBS_LEN:
            fraction = float(arr[_AGENT_OBS_RAY_FRACTION_IDX])
            tag = _RAY_TAG_INDEX_MAP.get(round(float(arr[_AGENT_OBS_RAY_TAG_IDX])), None)
            return [{"hit_tag": tag, "distance": fraction, "angle_deg": 0.0}]

    # Pass 2: fall back to full ray array (98/126/144 floats) when 10-float obs absent.
    for obs in obs_list:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1:
            continue
        n = len(arr)

        # Full ray array: parse every ray.
        # Select remap and food indices based on tag count (array size determines layout):
        #   98/126 floats → 13-tag layout: goodGoal at index 0 (confirmed empirically)
        #   144 floats    → 15-tag layout: goodGoal at index 4
        if n in (_RAY_SENSOR_LEN_LEGACY, _RAY_SENSOR_LEN_CURRENT, _RAY_SENSOR_LEN_NEW):
            is_legacy   = (n == _RAY_SENSOR_LEN_LEGACY)
            use_13_tags = (n in (_RAY_SENSOR_LEN_LEGACY, _RAY_SENSOR_LEN_CURRENT))

            n_tags = _RAY_SENSOR_TAGS_PER_RAY_LEGACY if use_13_tags else _RAY_SENSOR_TAGS_PER_RAY_NEW
            n_rays = _RAY_SENSOR_N_RAYS_LEGACY if is_legacy else _RAY_SENSOR_N_RAYS_CURRENT
            angles = _RAY_ANGLES_7 if is_legacy else _RAY_ANGLES_9
            remap  = _RAY_SENSOR_TAG_REMAP_13 if use_13_tags else _RAY_SENSOR_TAG_REMAP_15
            food_indices = (0, 1) if use_13_tags else (4, 5)

            floats_per_ray = n_tags + 1
            rays = []
            for i in range(n_rays):
                offset = i * floats_per_ray
                one_hot = arr[offset:offset + n_tags]
                fraction = float(arr[offset + n_tags])

                # Priority-check food indices before falling back to argmax.
                ray_tag: Optional[str] = None
                for fi in food_indices:
                    if fi < len(one_hot) and float(one_hot[fi]) > 0.08:
                        ray_tag = remap.get(fi)
                        break
                if ray_tag is None:
                    best_idx = int(np.argmax(one_hot))
                    ray_tag = (remap.get(best_idx)
                               if float(one_hot[best_idx]) > 0.15 else None)

                rays.append({"hit_tag": ray_tag, "distance": fraction,
                             "angle_deg": angles[i] if i < len(angles) else 0.0})
            return rays

    return [{"hit_tag": None, "distance": 1.0, "angle_deg": 0.0}]


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
