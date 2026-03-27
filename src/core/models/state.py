from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HomeostasisState:
    """
    Internal physiological state — environment-agnostic.
    All values normalized to [0.0, 1.0].
    """
    health: Optional[float] = None       # Overall integrity / vitality
    saturation: Optional[float] = None   # Fed/hunger proxy (1=full, 0=starving)
    energy: Optional[float] = None       # Composite resource level
    oxygen: Optional[float] = None       # Available for environments with suffocation risk
    is_alive: Optional[bool] = None


@dataclass
class PositionState:
    """Spatial location and orientation. Units are environment-relative."""
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    heading: Optional[float] = None     # Radians [0, 2π]
    velocity_x: Optional[float] = None
    velocity_z: Optional[float] = None


@dataclass
class PerceptionState:
    """
    Processed sensory features. visual_features is a low-dim embedding
    extracted by the adapter's observation mapper — the brain never sees raw pixels.
    """
    visual_features: Optional[List[float]] = None   # Adapter-extracted embedding
    detected_objects: Optional[List[str]] = None    # Object class names in FOV
    area_id: Optional[str] = None                   # Spatial hash for FAISS lookup
    terrain_novelty: Optional[float] = None         # [0,1] how novel this area is
    entity_density: Optional[float] = None          # [0,1] nearby entity density


@dataclass
class ResourceState:
    """Derived resource estimates computed by the adapter."""
    resource_level: Optional[float] = None      # [0,1] available resources
    threat_proximity: Optional[float] = None    # [0,1] 1=immediate threat


@dataclass
class AgentState:
    """
    Complete agent state snapshot. Fully environment-agnostic.

    The adapter populates this from raw environment observations.
    The brain consumes this without any knowledge of the source environment.
    raw_metadata is an escape hatch for adapter-internal bookkeeping — brain layers
    must NOT read from it.
    """
    homeostasis: HomeostasisState = field(default_factory=HomeostasisState)
    position: PositionState = field(default_factory=PositionState)
    perception: PerceptionState = field(default_factory=PerceptionState)
    resources: ResourceState = field(default_factory=ResourceState)
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def homeostatic_vector(self) -> List[float]:
        """Compact float vector over homeostatic channels for FAISS indexing."""
        h = self.homeostasis
        r = self.resources
        return [
            h.health or 0.0,
            h.saturation or 0.0,
            h.energy or 0.0,
            h.oxygen or 1.0,
            r.threat_proximity or 0.0,
            r.resource_level or 0.5,
        ]
