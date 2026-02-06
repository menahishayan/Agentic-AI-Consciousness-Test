from __future__ import annotations

from typing import Any, Optional

from core.adapters.minedojo.env_adapter import MineDojoAdapter
from core.agents.coordinator import MultiAgentCoordinator
from core.coordination.workspace import GlobalWorkspace
from core.memory.manager import MemoryManager


class AgentLoop:
    def __init__(
        self,
        adapter: MineDojoAdapter,
        coordinator: MultiAgentCoordinator,
        memory_manager: MemoryManager,
        workspace: Optional[GlobalWorkspace] = None,
    ) -> None:
        self.adapter = adapter
        self.coordinator = coordinator
        self.memory_manager = memory_manager
        self.workspace = workspace or GlobalWorkspace()

    def run_step(self) -> Any:
        raise NotImplementedError("Agent loop step not implemented.")

    def run(self, max_steps: int) -> None:
        raise NotImplementedError("Agent loop run not implemented.")
