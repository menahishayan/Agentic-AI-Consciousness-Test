from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Mapping, Optional, Tuple

from core.layers.interoceptive import VitalStateMonitor
from core.models.state import AgentState

ObservationMapper = Callable[[Any, Any, Optional[VitalStateMonitor]], AgentState]


def build_adapter(
    adapter_folder: str,
    adapter_config: Mapping[str, Any],
) -> Tuple[Any, ObservationMapper]:
    folder = adapter_folder.strip()
    if not folder:
        raise ValueError("adapter_folder cannot be empty.")

    env_module = import_module(f"core.adapters.{folder}.env_adapter")
    obs_module = import_module(f"core.adapters.{folder}.observation_mapper")

    create_adapter = getattr(env_module, "create_adapter", None)
    if not callable(create_adapter):
        raise AttributeError(
            f"core.adapters.{folder}.env_adapter must define create_adapter(config)."
        )

    map_obs = getattr(obs_module, "map_obs", None)
    if not callable(map_obs):
        raise AttributeError(
            f"core.adapters.{folder}.observation_mapper must define map_obs(raw_obs, info, vital_state_monitor)."
        )

    adapter = create_adapter(dict(adapter_config))
    _validate_adapter(adapter, folder)
    return adapter, map_obs


def _validate_adapter(adapter: Any, folder: str) -> None:
    required = (
        "reset",
        "step",
        "close",
        "get_available_vitals",
        "get_available_policies",
        "estimate_resource_level",
        "estimate_threat_proximity",
        "build_area_id",
        "estimate_entity_density",
        "estimate_terrain_novelty",
    )
    for name in required:
        if not callable(getattr(adapter, name, None)):
            raise AttributeError(
                f"Adapter '{folder}' is missing required method '{name}'."
            )

    vitals = adapter.get_available_vitals()
    if not isinstance(vitals, list):
        raise TypeError(
            f"Adapter '{folder}' get_available_vitals() must return list[str]."
        )
    for vital in vitals:
        if not isinstance(vital, str):
            raise TypeError(
                f"Adapter '{folder}' get_available_vitals() must return list[str]."
            )

    policies = adapter.get_available_policies()
    if not isinstance(policies, list):
        raise TypeError(
            f"Adapter '{folder}' get_available_policies() must return list[dict]."
        )
    for descriptor in policies:
        if not isinstance(descriptor, Mapping):
            raise TypeError(
                f"Adapter '{folder}' policy descriptors must be dict-like objects."
            )
        for required_key in ("policy_id", "callable_name"):
            raw = descriptor.get(required_key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(
                    f"Adapter '{folder}' policy descriptor is missing '{required_key}'."
                )
        tags = descriptor.get("tags")
        if not isinstance(tags, list) or not tags:
            raise ValueError(
                f"Adapter '{folder}' policy descriptor '{descriptor.get('policy_id')}' must define non-empty tags."
            )
        if not all(isinstance(tag, str) and tag.strip() for tag in tags):
            raise ValueError(
                f"Adapter '{folder}' policy descriptor '{descriptor.get('policy_id')}' tags must be list[str]."
            )
        drive_tags = descriptor.get("drive_tags")
        if drive_tags is not None:
            if not isinstance(drive_tags, list) or not all(
                isinstance(tag, str) and tag.strip() for tag in drive_tags
            ):
                raise ValueError(
                    f"Adapter '{folder}' policy descriptor '{descriptor.get('policy_id')}' drive_tags must be list[str] when provided."
                )
