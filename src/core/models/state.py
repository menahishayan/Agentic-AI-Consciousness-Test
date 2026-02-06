from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HomeostasisState:
    life: Optional[float] = None
    armor: Optional[float] = None
    food: Optional[float] = None
    saturation: Optional[float] = None
    xp: Optional[float] = None
    air: Optional[float] = None
    is_sleeping: Optional[bool] = None
    is_alive: Optional[bool] = None
    is_dead: Optional[bool] = None


@dataclass
class PositionState:
    xpos: Optional[float] = None
    ypos: Optional[float] = None
    zpos: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None


@dataclass
class BiomeState:
    biome_name: Optional[str] = None
    biome_id: Optional[int] = None
    biome_temperature: Optional[float] = None
    biome_rainfall: Optional[float] = None
    sea_level: Optional[float] = None


@dataclass
class LightingWeatherState:
    light_level: Optional[float] = None
    sky_light_level: Optional[float] = None
    sun_brightness: Optional[float] = None
    is_raining: Optional[bool] = None
    can_see_sky: Optional[bool] = None


@dataclass
class WorldTimeState:
    world_time: Optional[float] = None
    total_time: Optional[float] = None


@dataclass
class NearbyState:
    nearby_furnace: Optional[bool] = None
    nearby_crafting_table: Optional[bool] = None


@dataclass
class InventoryState:
    inventory: Optional[Any] = None
    inventories_available: Optional[Any] = None
    current_item_index: Optional[int] = None


@dataclass
class MiscState:
    distance_travelled_cm: Optional[float] = None
    stat: Optional[Any] = None
    achievement: Optional[Any] = None
    damage_source: Optional[Any] = None
    score: Optional[float] = None
    name: Optional[str] = None


@dataclass
class AgentState:
    homeostasis: HomeostasisState = field(default_factory=HomeostasisState)
    position: PositionState = field(default_factory=PositionState)
    biome: BiomeState = field(default_factory=BiomeState)
    lighting_weather: LightingWeatherState = field(default_factory=LightingWeatherState)
    world_time: WorldTimeState = field(default_factory=WorldTimeState)
    nearby: NearbyState = field(default_factory=NearbyState)
    inventory_state: InventoryState = field(default_factory=InventoryState)
    misc: MiscState = field(default_factory=MiscState)
    voxels: Optional[Any] = None

    @staticmethod
    def from_info(info: Dict[str, Any]) -> "AgentState":
        raise NotImplementedError("Mapping from MineDojo info is not implemented yet.")

    def to_dict(self, include_inventory: bool = False, include_voxels: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_inventory:
            data["inventory_state"]["inventory"] = None
            data["inventory_state"]["inventories_available"] = None
        if not include_voxels:
            data["voxels"] = None
        return data
