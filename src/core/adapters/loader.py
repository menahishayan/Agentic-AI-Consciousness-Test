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
    required = ("reset", "step", "close", "get_available_vitals")
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
