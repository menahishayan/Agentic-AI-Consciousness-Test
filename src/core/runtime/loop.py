from __future__ import annotations

from typing import Any, Callable, List, Optional

from core.coordination.workspace import GlobalWorkspace
from core.layers.interoceptive import VitalStateMonitor
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager
from core.models.state import AgentState
from core.observability.logger import RunLogger


class AgentLoop:
    def __init__(
        self,
        adapter: Any,
        observation_mapper: Callable[[Any, Any, Optional[VitalStateMonitor]], AgentState],
        memory_manager: MemoryManager,
        workspace: Optional[GlobalWorkspace] = None,
        llm_client: Optional[LLMClient] = None,
        logger: Optional[RunLogger] = None,
        include_inventory: bool = True,
        include_voxels: bool = True,
    ) -> None:
        self.adapter = adapter
        self.observation_mapper = observation_mapper
        self.memory_manager = memory_manager
        self.workspace = workspace or GlobalWorkspace()
        self.llm_client = llm_client
        self.logger = logger
        self.include_inventory = include_inventory
        self.include_voxels = include_voxels
        self.vital_state_monitor = VitalStateMonitor(
            expected_vitals=self._load_available_vitals(adapter),
        )
        self._initialized = False
        self._last_obs: Any = None
        self._last_info: Any = {}

    @staticmethod
    def _load_available_vitals(adapter: Any) -> List[str]:
        getter = getattr(adapter, "get_available_vitals", None)
        if not callable(getter):
            raise AttributeError("Adapter must implement get_available_vitals().")
        vitals = getter()
        if not isinstance(vitals, list):
            raise TypeError("Adapter get_available_vitals() must return list[str].")
        if not all(isinstance(v, str) for v in vitals):
            raise TypeError("Adapter get_available_vitals() must return list[str].")
        return vitals

    def run_step(self, step: int) -> Any:
        if self.logger is not None:
            self.logger.event("step.start", {"component": "AgentLoop"}, step=step)

        if not self._initialized:
            obs, info = self.adapter.reset()
            self._last_obs = obs
            self._last_info = info
            self._initialized = True
            state = self.observation_mapper(
                raw_obs=obs,
                info=info,
                vital_state_monitor=self.vital_state_monitor,
            )
            if self.logger is not None:
                state_dict = state.to_dict(
                    include_inventory=self.include_inventory,
                    include_voxels=self.include_voxels,
                )
                self.logger.state_snapshot(state_dict, step=step)
                info_keys = list(info.keys()) if isinstance(info, dict) else None
                self.logger.event("env.reset", {"info_keys": info_keys}, step=step)

        action = None
        sample_action = getattr(self.adapter, "sample_action", None)
        if callable(sample_action):
            action = sample_action()

        obs, reward, done, info = self.adapter.step(action)
        self._last_obs = obs
        self._last_info = info

        state = self.observation_mapper(
            raw_obs=obs,
            info=info,
            vital_state_monitor=self.vital_state_monitor,
        )
        if self.logger is not None:
            state_dict = state.to_dict(
                include_inventory=self.include_inventory,
                include_voxels=self.include_voxels,
            )
            self.logger.state_snapshot(state_dict, step=step)
            self.logger.event(
                "env.step",
                {"action": action, "reward": reward, "done": done},
                step=step,
            )

        return {"obs": obs, "reward": reward, "done": done, "info": info, "state": state}

    def run(self, max_steps: int) -> None:
        if self.logger is not None:
            self.logger.event("run.start", {"max_steps": max_steps})
        try:
            for step in range(max_steps):
                if self.logger is not None:
                    self.logger.event("step.begin", {"step": step}, step=step)
                result = self.run_step(step)
                if self.logger is not None:
                    self.logger.event("step.end", {"step": step}, step=step)
                if isinstance(result, dict) and result.get("done"):
                    break
        except Exception as exc:
            if self.logger is not None:
                self.logger.exception(exc, context={"stage": "run"})
            raise
        finally:
            if self.logger is not None:
                self.logger.event("run.end", {"max_steps": max_steps})
