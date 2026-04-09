from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from core.adapters.base import AbstractEnvironmentAdapter


def build_adapter(
    adapter_folder: str,
    adapter_config: Mapping[str, Any],
) -> AbstractEnvironmentAdapter:
    """
    Dynamically load and validate an adapter by folder name.

    The adapter module at core.adapters.<folder>.env_adapter must define
    create_adapter(config: dict) -> AbstractEnvironmentAdapter.

    Changing adapter_folder in config.json swaps the entire environment
    without touching any brain-layer code.
    """
    folder = adapter_folder.strip()
    if not folder:
        raise ValueError("adapter_folder cannot be empty.")

    module_path = f"core.adapters.{folder}.env_adapter"
    try:
        env_module = import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not import adapter module '{module_path}'. "
            f"Ensure src/core/adapters/{folder}/env_adapter.py exists."
        ) from exc

    create_fn = getattr(env_module, "create_adapter", None)
    if not callable(create_fn):
        raise AttributeError(
            f"'{module_path}' must define create_adapter(config: dict) -> AbstractEnvironmentAdapter."
        )

    adapter = create_fn(dict(adapter_config))
    _validate(adapter, folder)
    return adapter


def _validate(adapter: Any, folder: str) -> None:
    if not isinstance(adapter, AbstractEnvironmentAdapter):
        raise TypeError(
            f"Adapter '{folder}' must be an instance of AbstractEnvironmentAdapter. "
            f"Got: {type(adapter)}"
        )

    policies = adapter.get_available_policies()
    if not isinstance(policies, list):
        raise TypeError(f"Adapter '{folder}'.get_available_policies() must return list.")

    for descriptor in policies:
        if not isinstance(descriptor, Mapping):
            raise TypeError(f"Adapter '{folder}' policy descriptors must be dict-like.")
        for key in ("policy_id", "callable_name"):
            if not isinstance(descriptor.get(key), str) or not descriptor[key].strip():
                raise ValueError(
                    f"Adapter '{folder}' policy descriptor missing required str field '{key}'."
                )
        tags = descriptor.get("tags")
        if not isinstance(tags, list) or not tags:
            raise ValueError(
                f"Adapter '{folder}' policy '{descriptor.get('policy_id')}' must define non-empty tags."
            )

    vitals = adapter.get_available_vitals()
    if not isinstance(vitals, list) or not all(isinstance(v, str) for v in vitals):
        raise TypeError(f"Adapter '{folder}'.get_available_vitals() must return List[str].")

    drive_channels = adapter.get_drive_channels()
    if not isinstance(drive_channels, list) or not drive_channels:
        raise ValueError(f"Adapter '{folder}'.get_drive_channels() must return non-empty list.")

    goal = adapter.get_task_goal()
    for key in ("description", "priority", "task_id"):
        if key not in goal:
            raise ValueError(f"Adapter '{folder}'.get_task_goal() must include key '{key}'.")
