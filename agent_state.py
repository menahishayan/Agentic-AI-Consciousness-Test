from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Homeostasis:
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
class Position:
    xpos: Optional[float] = None
    ypos: Optional[float] = None
    zpos: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None


@dataclass
class Biome:
    biome_name: Optional[str] = None
    biome_id: Optional[int] = None
    biome_temperature: Optional[float] = None
    biome_rainfall: Optional[float] = None
    sea_level: Optional[float] = None


@dataclass
class LightingWeather:
    light_level: Optional[float] = None
    sky_light_level: Optional[float] = None
    sun_brightness: Optional[float] = None
    is_raining: Optional[bool] = None
    can_see_sky: Optional[bool] = None


@dataclass
class WorldTime:
    world_time: Optional[float] = None
    total_time: Optional[float] = None


@dataclass
class Nearby:
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
    homeostasis: Homeostasis = field(default_factory=Homeostasis)
    position: Position = field(default_factory=Position)
    biome: Biome = field(default_factory=Biome)
    lighting_weather: LightingWeather = field(default_factory=LightingWeather)
    world_time: WorldTime = field(default_factory=WorldTime)
    nearby: Nearby = field(default_factory=Nearby)
    inventory_state: InventoryState = field(default_factory=InventoryState)
    misc: MiscState = field(default_factory=MiscState)
    voxels: Optional[Any] = None

    @staticmethod
    def from_info(info: Dict[str, Any]) -> "AgentState":
        state = AgentState()
        if not isinstance(info, dict):
            return state

        state.homeostasis.life = info.get("life")
        state.homeostasis.armor = info.get("armor")
        state.homeostasis.food = info.get("food")
        state.homeostasis.saturation = info.get("saturation")
        state.homeostasis.xp = info.get("xp")
        state.homeostasis.air = info.get("air")
        state.homeostasis.is_sleeping = info.get("is_sleeping")
        state.homeostasis.is_alive = info.get("is_alive")
        state.homeostasis.is_dead = info.get("is_dead")

        state.position.xpos = info.get("xpos")
        state.position.ypos = info.get("ypos")
        state.position.zpos = info.get("zpos")
        state.position.pitch = info.get("pitch")
        state.position.yaw = info.get("yaw")

        state.biome.biome_name = info.get("biome_name")
        state.biome.biome_id = info.get("biome_id")
        state.biome.biome_temperature = info.get("biome_temperature")
        state.biome.biome_rainfall = info.get("biome_rainfall")
        state.biome.sea_level = info.get("sea_level")

        state.lighting_weather.light_level = info.get("light_level")
        state.lighting_weather.sky_light_level = info.get("sky_light_level")
        state.lighting_weather.sun_brightness = info.get("sun_brightness")
        state.lighting_weather.is_raining = info.get("is_raining")
        state.lighting_weather.can_see_sky = info.get("can_see_sky")

        state.world_time.world_time = info.get("world_time")
        state.world_time.total_time = info.get("total_time")

        state.nearby.nearby_furnace = info.get("nearby_furnace")
        state.nearby.nearby_crafting_table = info.get("nearby_crafting_table")

        state.inventory_state.inventory = info.get("inventory")
        state.inventory_state.inventories_available = info.get("inventories_available")
        state.inventory_state.current_item_index = info.get("current_item_index")

        state.misc.distance_travelled_cm = info.get("distance_travelled_cm")
        state.misc.stat = info.get("stat")
        state.misc.achievement = info.get("achievement")
        state.misc.damage_source = info.get("damage_source")
        state.misc.score = info.get("score")
        state.misc.name = info.get("name")

        state.voxels = info.get("voxels")
        return state

    def to_dict(
        self,
        include_inventory: bool = False,
        include_voxels: bool = False,
    ) -> Dict[str, Any]:
        data = asdict(self)
        if not include_inventory:
            data["inventory_state"]["inventory"] = None
            data["inventory_state"]["inventories_available"] = None
        if not include_voxels:
            data["voxels"] = None
        return data
