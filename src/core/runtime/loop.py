from __future__ import annotations

from typing import Any, Optional

from core.adapters.minedojo.env_adapter import MineDojoAdapter
from core.agents.coordinator import MultiAgentCoordinator
from core.coordination.workspace import GlobalWorkspace
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager
from core.observability.logger import RunLogger


class AgentLoop:
    def __init__(
        self,
        adapter: MineDojoAdapter,
        coordinator: MultiAgentCoordinator,
        memory_manager: MemoryManager,
        workspace: Optional[GlobalWorkspace] = None,
        llm_client: Optional[LLMClient] = None,
        logger: Optional[RunLogger] = None,
    ) -> None:
        self.adapter = adapter
        self.coordinator = coordinator
        self.memory_manager = memory_manager
        self.workspace = workspace or GlobalWorkspace()
        self.llm_client = llm_client
        self.logger = logger

    def run_step(self) -> Any:
        if self.logger is not None:
            self.logger.event("step.start", {"component": "AgentLoop"})
        raise NotImplementedError("Agent loop step not implemented.")

    def run(self, max_steps: int) -> None:
        if self.logger is not None:
            self.logger.event("run.start", {"max_steps": max_steps})
        try:
            for step in range(max_steps):
                if self.logger is not None:
                    self.logger.event("step.begin", {"step": step}, step=step)
                self.run_step()
                if self.logger is not None:
                    self.logger.event("step.end", {"step": step}, step=step)
        except Exception as exc:
            if self.logger is not None:
                self.logger.exception(exc, context={"stage": "run"})
            raise
        finally:
            if self.logger is not None:
                self.logger.event("run.end", {"max_steps": max_steps})
