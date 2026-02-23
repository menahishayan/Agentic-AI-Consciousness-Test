from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from core.adapters.loader import build_adapter
from core.memory.manager import MemoryManager
from core.observability import LoggingConfig, RunLogger, install_exception_hooks
from core.runtime.loop import AgentLoop


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_config(path: Path) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "adapter_folder": "minedojo",
        "adapter_config": {
            "task_id": "harvest_milk",
            "image_size": [160, 256],
        },
    }
    if not path.exists():
        return default

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("config.json must contain a top-level JSON object.")

    merged = dict(default)
    merged.update(parsed)

    adapter_config = parsed.get("adapter_config")
    if isinstance(adapter_config, dict):
        merged["adapter_config"] = {**default["adapter_config"], **adapter_config}
    elif "adapter_config" not in merged:
        merged["adapter_config"] = dict(default["adapter_config"])
    return merged


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    app_config = load_config(root / "config.json")

    max_steps = int(os.getenv("MAX_STEPS", "50"))
    include_inventory = _env_flag("INCLUDE_INVENTORY", True)
    include_voxels = _env_flag("INCLUDE_VOXELS", True)
    adapter_folder = str(app_config.get("adapter_folder", "minedojo"))
    adapter_cfg = app_config.get("adapter_config", {})
    if not isinstance(adapter_cfg, dict):
        raise ValueError("'adapter_config' in config.json must be an object.")

    config = LoggingConfig.from_env()
    with RunLogger(config) as logger:
        install_exception_hooks(logger)

        adapter, observation_mapper = build_adapter(adapter_folder, adapter_cfg)

        memory_manager = MemoryManager(logger=logger)

        logger.event(
            "run.config",
            {
                "adapter_folder": adapter_folder,
                "adapter_config": adapter_cfg,
                "max_steps": max_steps,
                "include_inventory": include_inventory,
                "include_voxels": include_voxels,
            },
        )

        loop = AgentLoop(
            adapter=adapter,
            observation_mapper=observation_mapper,
            memory_manager=memory_manager,
            logger=logger,
            include_inventory=include_inventory,
            include_voxels=include_voxels,
        )

        try:
            loop.run(max_steps)
        finally:
            adapter.close()


if __name__ == "__main__":
    main()
