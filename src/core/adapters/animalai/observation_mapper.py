"""
Observation Mapper — converts raw Animal AI observations into AgentState.

Animal AI provides:
  - Visual observation: numpy array of shape (H, W, 3) or (3, H, W) — RGB camera
  - Health scalar: float in Animal AI's observation
  - Velocity/position: sometimes available in step info

We extract a compact feature vector from the visual observation without GPU:
  - Downsample to 21×21
  - Per-channel (R, G, B) mean and variance → 6 floats
  - Edge density via simple gradient magnitude → 1 float
  - Brightness histogram (4 buckets) → 4 floats
  Total: 11-dim visual_features

Area ID is derived from a discretized position grid.
Terrain novelty and entity density are heuristically estimated from visual features.
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

    # Resource level: heuristic from green channel (food/grass proxy)
    resource_level = _estimate_resource(visual_features)
    # Threat proximity: red channel anomaly proxy
    threat_proximity = _estimate_threat(visual_features)

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
        ),
        resources=ResourceState(
            resource_level=resource_level,
            threat_proximity=threat_proximity,
        ),
        step=step,
        raw_metadata={
            "raw_reward": info.get("raw_reward", 0.0),
            "env_done": info.get("env_done", False),
        },
    )


def _extract_visual_features(obs: Optional[np.ndarray]) -> List[float]:
    """Extract compact 11-dim feature vector from RGB frame."""
    if obs is None:
        return [0.0] * 11

    arr = np.asarray(obs, dtype=np.float32)

    # Normalize to [0,1]
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Ensure (H, W, 3) layout
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)

    if arr.ndim != 3 or arr.shape[2] != 3:
        return [0.0] * 11

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

    return channel_features + [edge_density] + hist_features


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


def _estimate_resource(features: List[float]) -> float:
    """Green channel dominance suggests food/vegetation (resources)."""
    if len(features) < 6:
        return 0.5
    r_mean, g_mean, b_mean = features[0], features[2], features[4]
    green_dominance = max(0.0, g_mean - (r_mean + b_mean) / 2.0)
    return float(min(green_dominance * 3.0 + 0.1, 1.0))


def _estimate_threat(features: List[float]) -> float:
    """Red channel anomaly (unexpected red) suggests hazard proximity."""
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
        vx = info.get("velocity_x", 0.0)
        vz = info.get("velocity_z", 0.0)

        self._x = float(x + vx)
        self._z = float(z + vz)

        return PositionState(
            x=float(x),
            y=float(y),
            z=float(z),
            heading=float(heading),
            velocity_x=float(vx),
            velocity_z=float(vz),
        )
