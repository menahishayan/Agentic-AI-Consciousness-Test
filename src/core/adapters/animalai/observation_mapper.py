"""
Observation Mapper — converts raw Animal AI observations into AgentState.

Animal AI provides:
  - Visual observation: numpy array of shape (H, W, 3) or (3, H, W) — RGB camera
  - Velocity/position: 1D float array with 3 elements (forward, right, up) [m/s]
  - Raycasts (if configured): 1D float array of length n_rays × (n_tags + 1)
      Each ray: [one-hot over tags..., normalized_distance]
      distance 0.0 = right next to agent, 1.0 = max range / no hit

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

# Raycast configuration — Animal AI basic food arena default.
# Each ray encodes: [one-hot over _RAYCAST_TAGS..., normalized_distance]
# distance: 0.0 = at agent, 1.0 = max range / no hit
_RAYCAST_TAGS = ["GoodGoal", "BadGoal", "wall"]
_FLOATS_PER_RAY = len(_RAYCAST_TAGS) + 1  # 4

# Angle labels indexed from left to right (Animal AI fan is symmetric around forward)
_RAY_ANGLE_LABELS: Dict[int, List[str]] = {
    1: ["forward"],
    3: ["left-45°", "forward", "right-45°"],
    5: ["left-45°", "left-22°", "forward", "right-22°", "right-45°"],
}


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

    # Raycast hits — primary perception source when available
    raycast_hits: Optional[List[Dict[str, Any]]] = info.get("raycast_hits") or None

    # Resource/threat estimates: raycasts take priority over visual heuristics
    resource_level = _estimate_resource(visual_features, raycast_hits)
    threat_proximity = _estimate_threat(visual_features, raycast_hits)

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
            detected_objects=info.get("detected_objects", []),
            area_id=area_id,
            terrain_novelty=terrain_novelty,
            entity_density=entity_density,
            raycast_hits=raycast_hits,
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
    Extract (forward, right, up) local speeds from Animal AI's vector observation.

    Animal AI's velocity array has exactly 3 elements. We look for exactly-3 first
    to avoid accidentally consuming the raycast array (which is longer).
    """
    # Prefer exact-3 match (velocity)
    for obs in obs_list:
        arr = np.asarray(obs)
        if arr.ndim == 1 and len(arr) == 3:
            return float(arr[0]), float(arr[1]), float(arr[2])
    # Fallback: any short 1D array (handles edge cases)
    for obs in obs_list:
        arr = np.asarray(obs)
        if arr.ndim == 1 and 3 <= len(arr) <= 5:
            return float(arr[0]), float(arr[1]), float(arr[2])
    return 0.0, 0.0, 0.0


def _parse_raycasts(obs_list: List[Any]) -> List[Dict[str, Any]]:
    """
    Parse raycast observations from Animal AI's obs_list.

    Looks for a 1D float array whose length is a multiple of _FLOATS_PER_RAY (4)
    and longer than the velocity array (>3). If found, decodes into a list of dicts:
        [{"angle_idx": i, "angle_label": str, "hit_tag": str|None, "distance": float}]

    Returns an empty list if no raycast array is detected (binary not configured).
    """
    for obs in obs_list:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1 or len(arr) <= 3:
            continue
        if len(arr) % _FLOATS_PER_RAY != 0:
            continue
        n_rays = len(arr) // _FLOATS_PER_RAY
        angle_labels = _RAY_ANGLE_LABELS.get(n_rays, [f"ray_{i}" for i in range(n_rays)])
        results: List[Dict[str, Any]] = []
        for i in range(n_rays):
            base = i * _FLOATS_PER_RAY
            onehot = arr[base:base + len(_RAYCAST_TAGS)]
            distance = float(arr[base + len(_RAYCAST_TAGS)])
            hit_tag: Optional[str] = None
            best_idx = int(np.argmax(onehot))
            if float(onehot[best_idx]) > 0.5:
                hit_tag = _RAYCAST_TAGS[best_idx] if best_idx < len(_RAYCAST_TAGS) else f"tag_{best_idx}"
            results.append({
                "angle_idx": i,
                "angle_label": angle_labels[i] if i < len(angle_labels) else f"ray_{i}",
                "hit_tag": hit_tag,
                "distance": distance,
            })
        return results
    return []


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
    raycast_hits: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Estimate food resource level.
    Primary: closest GoodGoal raycast hit (accurate, directional).
    Fallback: green channel dominance from visual features (heuristic).
    """
    if raycast_hits:
        food_hits = [r for r in raycast_hits if r.get("hit_tag") == "GoodGoal"]
        if food_hits:
            closest = min(food_hits, key=lambda r: r["distance"])
            # 1.0 = food right next to agent, fades linearly with distance
            return float(max(0.0, 1.0 - closest["distance"]))
        return 0.1  # Raycasts available but no food visible

    # Visual fallback
    if len(features) < 6:
        return 0.5
    r_mean, g_mean, b_mean = features[0], features[2], features[4]
    green_dominance = max(0.0, g_mean - (r_mean + b_mean) / 2.0)
    return float(min(green_dominance * 3.0 + 0.1, 1.0))


def _estimate_threat(
    features: List[float],
    raycast_hits: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Estimate threat proximity.
    Primary: BadGoal raycast hits (accurate). Walls are lower-weight threat.
    Fallback: red channel anomaly from visual features (heuristic).
    """
    if raycast_hits:
        threat = 0.0
        for r in raycast_hits:
            tag = r.get("hit_tag")
            dist = r.get("distance", 1.0)
            if tag == "BadGoal":
                threat = max(threat, 1.0 - dist)
            elif tag == "wall":
                # Walls are always nearby in bounded arenas — lower weight
                threat = max(threat, (1.0 - dist) * 0.25)
        return float(threat)

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
