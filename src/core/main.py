from __future__ import annotations

import os
from pathlib import Path

import minedojo

from core.adapters.minedojo.env_adapter import MineDojoAdapter
from core.agents.coordinator import MultiAgentCoordinator
from core.agents.homeostatic_agent import HomeostaticAgent
from core.agents.metacognitive_agent import MetacognitiveAgent
from core.agents.motor_agent import MotorAgent
from core.agents.perceptual_agent import PerceptualAgent
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


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")

    task_id = os.getenv("MINEDOJO_TASK_ID", "harvest_milk")
    max_steps = int(os.getenv("MAX_STEPS", "50"))
    include_inventory = _env_flag("INCLUDE_INVENTORY", True)
    include_voxels = _env_flag("INCLUDE_VOXELS", True)

    config = LoggingConfig.from_env()
    with RunLogger(config) as logger:
        install_exception_hooks(logger)

        env = minedojo.make(task_id=task_id, image_size=(160, 256))
        adapter = MineDojoAdapter(env)

        coordinator = MultiAgentCoordinator(
            HomeostaticAgent(),
            PerceptualAgent(),
            MotorAgent(),
            MetacognitiveAgent(),
        )
        memory_manager = MemoryManager(logger=logger)

        logger.event(
            "run.config",
            {
                "task_id": task_id,
                "max_steps": max_steps,
                "include_inventory": include_inventory,
                "include_voxels": include_voxels,
            },
        )

        loop = AgentLoop(
            adapter=adapter,
            coordinator=coordinator,
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
