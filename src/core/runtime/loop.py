from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.layers.action_selection import PolicyGenerator
from core.layers.interoceptive import VitalStateMonitor
from core.layers.metacognitive import GoalCoherenceChecker
from core.layers.predictive import PredictionErrorCalculator
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager
from core.models.signals import ActionProposal
from core.models.state import AgentState
from core.observability.logger import RunLogger


class AgentLoop:
    def __init__(
        self,
        adapter: Any,
        observation_mapper: Callable[[Any, Any, Optional[VitalStateMonitor]], AgentState],
        memory_manager: MemoryManager,
        adapter_folder: str,
        policy_config: Optional[Mapping[str, Any]] = None,
        workspace: Optional[GlobalWorkspace] = None,
        llm_client: Optional[LLMClient] = None,
        logger: Optional[RunLogger] = None,
        include_inventory: bool = True,
        include_voxels: bool = True,
    ) -> None:
        self.adapter = adapter
        self.observation_mapper = observation_mapper
        self.memory_manager = memory_manager
        self.adapter_folder = adapter_folder
        self.policy_config = dict(policy_config or {})
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
        self.goal_checker = GoalCoherenceChecker()
        self.prediction_error_calculator = PredictionErrorCalculator(
            max_expected_error=float(self.policy_config.get("max_expected_error", 1.0)),
            window_size=int(self.policy_config.get("prediction_error_window", 20)),
        )
        self.policy_generator = PolicyGenerator(
            adapter=self.adapter,
            adapter_folder=self.adapter_folder,
            memory_manager=self.memory_manager,
            goal_checker=self.goal_checker,
            prediction_error_calculator=self.prediction_error_calculator,
            config=self.policy_config,
            logger=self.logger,
        )

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

        current_state = self.observation_mapper(
            raw_obs=self._last_obs,
            info=self._last_info,
            vital_state_monitor=self.vital_state_monitor,
        )
        workspace_messages = self.workspace.broadcast()
        goals = self._extract_goals(workspace_messages)

        policy_context = {
            "state": current_state,
            "obs": self._last_obs,
            "info": self._last_info,
            "workspace_messages": workspace_messages,
            "step": step,
            "memory_manager": self.memory_manager,
        }
        action_proposal = self.policy_generator.propose_action(
            goals=goals,
            context=policy_context,
        )

        action = self._select_env_action(action_proposal)
        selected_policy_id = (
            action_proposal.action_id if isinstance(action_proposal, ActionProposal) else None
        )

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
                {
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "selected_policy_id": selected_policy_id,
                },
                step=step,
            )

        if selected_policy_id:
            self.memory_manager.record_policy_outcome(
                policy_id=selected_policy_id,
                reward=reward,
                done=done,
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

    def _select_env_action(self, action_proposal: Optional[ActionProposal]) -> Any:
        if isinstance(action_proposal, ActionProposal) and action_proposal.action is not None:
            return action_proposal.action

        sample_action = getattr(self.adapter, "sample_action", None)
        if callable(sample_action):
            return sample_action()
        return None

    def _extract_goals(self, messages: List[AgentMessage]) -> List[Any]:
        goals: List[Any] = []
        for message in messages:
            kind = getattr(message, "kind", None)
            payload = getattr(message, "payload", None)
            if kind == "goal":
                if isinstance(payload, list):
                    goals.extend(payload)
                elif payload is not None:
                    goals.append(payload)
                continue
            if isinstance(payload, Mapping):
                payload_goals = payload.get("goals")
                if isinstance(payload_goals, list):
                    goals.extend(payload_goals)
        return goals
