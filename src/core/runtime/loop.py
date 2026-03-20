from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, List, Mapping, Optional

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.layers.action_selection import PolicyGenerator
from core.layers.interoceptive import (
    AllostaticController,
    ArousalValenceSystem,
    HomeostaticState,
    PredictionError,
    VitalStateMonitor,
)
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
        allostatic_cfg = self.policy_config.get("allostatic_controller", {})
        if not isinstance(allostatic_cfg, Mapping):
            raise TypeError("'policy_generator.allostatic_controller' must be an object.")
        arousal_cfg = self.policy_config.get("arousal_valence", {})
        if not isinstance(arousal_cfg, Mapping):
            raise TypeError("'policy_generator.arousal_valence' must be an object.")
        self.vital_state_monitor = VitalStateMonitor(
            expected_vitals=self._load_available_vitals(adapter),
        )
        self.allostatic_controller = AllostaticController(
            llm_client=self.llm_client,
            config=allostatic_cfg,
            logger=self.logger,
        )
        self.arousal_valence_system = ArousalValenceSystem(
            config=arousal_cfg,
            message_bus=self.workspace,
            self_state_tracker=self.memory_manager.self_state,
            memory_manager=self.memory_manager,
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
        self._record_vital_state(step=step, phase="pre_action")
        workspace_messages = self.workspace.broadcast()
        goals = self._extract_goals(workspace_messages)
        allostatic_assessment = self.allostatic_controller.assess(
            step=step,
            state=current_state,
            goals=goals,
            vital_state=self.vital_state_monitor.to_dict(),
            obs=self._last_obs,
            info=self._last_info,
        )
        homeostatic_state = self._build_homeostatic_state(
            step=step,
            state=current_state,
            workspace_messages=workspace_messages,
        )
        latest_prediction_error = self._latest_prediction_error(step=step)
        arousal_valence_state = self.arousal_valence_system.update(
            homeostatic_state=homeostatic_state,
            prediction_error=latest_prediction_error,
        )
        arousal_payload = asdict(arousal_valence_state)
        if isinstance(allostatic_assessment, Mapping):
            allostatic_assessment = dict(allostatic_assessment)
            allostatic_assessment["policy_bias"] = dict(arousal_payload["policy_bias"])
            allostatic_assessment["urgency_signal"] = arousal_payload["urgency_signal"]
            allostatic_assessment["arousal"] = arousal_payload["arousal"]
            allostatic_assessment["valence"] = arousal_payload["valence"]

        self.memory_manager.snapshot_self_state(
            {
                "step": step,
                "phase": "allostatic_pre_action",
                "allostatic_assessment": allostatic_assessment,
            }
        )
        self.memory_manager.snapshot_self_state(
            {
                "step": step,
                "phase": "arousal_valence_pre_action",
                "arousal_valence": arousal_payload,
            }
        )
        if self.logger is not None:
            self.logger.event(
                "allostatic.assessment",
                {
                    "source": allostatic_assessment.get("source"),
                    "risk_level": allostatic_assessment.get("risk_level"),
                    "confidence": allostatic_assessment.get("confidence"),
                    "survival_horizon_steps": allostatic_assessment.get("survival_horizon_steps"),
                    "needs_count": len(allostatic_assessment.get("needs", [])),
                },
                step=step,
            )
            self.logger.event(
                "homeostatic.arousal_valence",
                {
                    "arousal": arousal_valence_state.arousal,
                    "valence": arousal_valence_state.valence,
                    "urgency_signal": arousal_valence_state.urgency_signal,
                    "learning_rate_mod": arousal_valence_state.learning_rate_mod,
                },
                step=step,
            )

        policy_context = {
            "state": current_state,
            "obs": self._last_obs,
            "info": self._last_info,
            "workspace_messages": workspace_messages,
            "step": step,
            "memory_manager": self.memory_manager,
            "allostatic_assessment": allostatic_assessment,
            "arousal_valence_state": arousal_payload,
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
        self._record_vital_state(step=step, phase="post_step")
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

    def _record_vital_state(self, step: int, phase: str) -> None:
        self.memory_manager.snapshot_self_state(
            {
                "step": step,
                "phase": phase,
                "vital_state": self.vital_state_monitor.to_dict(),
            }
        )

    def _build_homeostatic_state(
        self,
        step: int,
        state: AgentState,
        workspace_messages: List[AgentMessage],
    ) -> HomeostaticState:
        homeostasis = self._as_mapping(getattr(state, "homeostasis", None))
        lighting = self._as_mapping(getattr(state, "lighting_weather", None))
        nearby = self._as_mapping(getattr(state, "nearby", None))
        inventory_state = self._as_mapping(getattr(state, "inventory_state", None))
        vitals = self.vital_state_monitor.last_state()

        life = self._as_optional_float(homeostasis.get("life"))
        if life is None:
            life = self._as_optional_float(vitals.get("life"))
        food = self._as_optional_float(homeostasis.get("food"))
        if food is None:
            food = self._as_optional_float(vitals.get("food"))
        air = self._as_optional_float(homeostasis.get("air"))
        if air is None:
            air = self._as_optional_float(vitals.get("air"))

        health = self._clamp01(0.0 if life is None else life / 20.0)
        hunger = self._clamp01(0.0 if food is None else food / 20.0)
        oxygen = self._clamp01(0.0 if air is None else air / 300.0)
        resource_level = self._estimate_resource_level(
            hunger=hunger,
            lighting=lighting,
            nearby=nearby,
            inventory_state=inventory_state,
        )
        threat_proximity = self._estimate_threat_proximity(
            messages=workspace_messages,
            health=health,
            lighting=lighting,
            homeostasis=homeostasis,
        )
        return HomeostaticState(
            health=health,
            hunger=hunger,
            resource_level=resource_level,
            threat_proximity=threat_proximity,
            oxygen=oxygen,
            tick=step,
        )

    def _estimate_resource_level(
        self,
        hunger: float,
        lighting: Mapping[str, Any],
        nearby: Mapping[str, Any],
        inventory_state: Mapping[str, Any],
    ) -> float:
        signals: List[float] = [self._clamp01(hunger)]
        if nearby.get("nearby_crafting_table") is True or nearby.get("nearby_furnace") is True:
            signals.append(0.75)
        if lighting.get("can_see_sky") is False:
            signals.append(0.70)
        if inventory_state.get("current_item_index") is not None:
            signals.append(0.65)
        if len(signals) == 1:
            signals.append(0.50)
        return self._clamp01(sum(signals) / float(len(signals)))

    def _estimate_threat_proximity(
        self,
        messages: List[AgentMessage],
        health: float,
        lighting: Mapping[str, Any],
        homeostasis: Mapping[str, Any],
    ) -> float:
        threat = 0.0
        for message in messages:
            kind = str(getattr(message, "kind", "")).strip().lower()
            payload = getattr(message, "payload", None)
            if kind in {"threat", "threat_signal", "danger"}:
                if isinstance(payload, Mapping) and isinstance(payload.get("severity"), (int, float)):
                    threat = max(threat, self._clamp01(payload.get("severity")))
                else:
                    threat = max(threat, 0.60)
                continue
            if isinstance(payload, Mapping) and isinstance(payload.get("threat_proximity"), (int, float)):
                threat = max(threat, self._clamp01(payload.get("threat_proximity")))

        light_level = self._as_optional_float(lighting.get("light_level"))
        can_see_sky = lighting.get("can_see_sky")
        if light_level is not None and can_see_sky is True and light_level < 7.0:
            threat = max(threat, self._clamp01((7.0 - light_level) / 7.0 * 0.6))
        threat = max(threat, self._clamp01(1.0 - health))

        if homeostasis.get("is_dead") is True or homeostasis.get("is_alive") is False:
            threat = 1.0
        return self._clamp01(threat)

    def _latest_prediction_error(self, step: int) -> Optional[PredictionError]:
        history = self.memory_manager.prediction_errors.query({"limit": 1})
        if not isinstance(history, list) or not history:
            return None
        record = history[-1]

        magnitude = None
        source = "visual"
        tick = step

        if isinstance(record, Mapping):
            magnitude = self._as_optional_float(record.get("magnitude"))
            if magnitude is None:
                nested = record.get("error")
                if isinstance(nested, Mapping):
                    magnitude = self._as_optional_float(nested.get("magnitude"))
            raw_source = record.get("source")
            if raw_source is None and isinstance(record.get("error"), Mapping):
                raw_source = record.get("error", {}).get("source")
            if isinstance(raw_source, str) and raw_source in {"visual", "proprioceptive", "threat"}:
                source = raw_source
            raw_tick = record.get("tick")
            if isinstance(raw_tick, int):
                tick = raw_tick
        elif hasattr(record, "magnitude"):
            magnitude = self._as_optional_float(getattr(record, "magnitude"))
            raw_source = getattr(record, "source", None)
            if isinstance(raw_source, str) and raw_source in {"visual", "proprioceptive", "threat"}:
                source = raw_source
            raw_tick = getattr(record, "tick", None)
            if isinstance(raw_tick, int):
                tick = raw_tick

        if magnitude is None:
            return None
        return PredictionError(
            magnitude=self._clamp01(magnitude),
            source=source,
            tick=tick,
        )

    @staticmethod
    def _as_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if hasattr(value, "__dict__"):
            mapped = vars(value)
            if isinstance(mapped, Mapping):
                return mapped
        return {}

    @staticmethod
    def _as_optional_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(0.0, min(1.0, numeric))
