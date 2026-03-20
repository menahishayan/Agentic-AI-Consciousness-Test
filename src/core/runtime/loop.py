from __future__ import annotations

from collections import deque
from dataclasses import asdict
import re
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional

import numpy as np

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.layers.action_selection import PolicyGenerator
from core.layers.interoceptive import (
    AllostaticConfig,
    AllostaticController,
    ArousalHomeostaticState as HomeostaticState,
    ArousalValenceSystem,
    DriveChannel,
    HomeostaticHistory as DriveHistory,
    HomeostaticState as DriveState,
    PredictionError,
    PrioritisedDriveSignals,
    VitalStateMonitor,
)
from core.layers.metacognitive import GoalCoherenceChecker
from core.layers.predictive import PredictionErrorCalculator as PolicyPredictionErrorCalculator
from core.models.signals import ActionProposal
from core.models.state import AgentState
from core.observability.logger import RunLogger
from core.memory import MemoryManager
from core.perceptual import (
    ObservationSnapshot,
    PEConfig,
    PredictionErrorBatch,
    PredictionErrorCalculator as PerceptualPredictionErrorCalculator,
)


class AgentLoop:
    _POLICY_DRIVE_TAG_RULES: Dict[str, set[str]] = {
        "health": {"heal", "retreat", "defend", "protect", "shield"},
        "hunger": {"eat", "collect", "cook", "food", "harvest"},
        "oxygen": {"surface", "ascend", "air", "breath"},
        "resource_level": {"gather", "mine", "craft", "smelt"},
        "safety": {"retreat", "avoid", "shelter", "escape"},
    }

    def __init__(
        self,
        adapter: Any,
        observation_mapper: Callable[[Any, Any, Optional[VitalStateMonitor]], AgentState],
        memory_manager: MemoryManager,
        adapter_folder: str,
        policy_config: Optional[Mapping[str, Any]] = None,
        llm_config: Optional[Mapping[str, Any]] = None,
        workspace: Optional[GlobalWorkspace] = None,
        llm_client: Optional[Any] = None,
        logger: Optional[RunLogger] = None,
        include_inventory: bool = True,
        include_voxels: bool = True,
    ) -> None:
        self.adapter = adapter
        self.observation_mapper = observation_mapper
        self.memory_manager = memory_manager
        self.adapter_folder = adapter_folder
        self.policy_config = dict(policy_config or {})
        self.llm_config = dict(llm_config or {})
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
        perceptual_cfg = self.policy_config.get("perceptual_prediction_error", {})
        if not isinstance(perceptual_cfg, Mapping):
            raise TypeError("'policy_generator.perceptual_prediction_error' must be an object.")

        self.vital_state_monitor = VitalStateMonitor(
            expected_vitals=self._load_available_vitals(adapter),
        )
        self._drive_channels = self._build_drive_channels(allostatic_cfg)
        self._drive_channel_map = {channel.id: channel for channel in self._drive_channels}
        self._allostatic_config = self._build_allostatic_config(allostatic_cfg)
        self._homeostatic_history: Deque[DriveState] = deque(
            maxlen=max(1, int(self._allostatic_config.history_window))
        )
        self.allostatic_controller = AllostaticController(
            config=self._allostatic_config,
            channels=self._drive_channels,
            memory_manager=self.memory_manager,
            message_bus=self.workspace,
        )
        self.arousal_valence_system = ArousalValenceSystem(
            config=arousal_cfg,
            message_bus=self.workspace,
            self_state_tracker=self.memory_manager.self_state,
            memory_manager=self.memory_manager,
        )
        self.perceptual_prediction_error_calculator = PerceptualPredictionErrorCalculator(
            config=self._build_pe_config(perceptual_cfg),
            memory_manager=self.memory_manager,
            message_bus=self.workspace,
        )

        self._initialized = False
        self._last_obs: Any = None
        self._last_info: Any = {}
        self._last_selected_policy_id = "bootstrap"
        self._last_drive_signals: Optional[PrioritisedDriveSignals] = None
        self._last_perceptual_batch: Optional[PredictionErrorBatch] = None
        self._last_homeostatic_state_for_memory: Optional[HomeostaticState] = None

        self.goal_checker = GoalCoherenceChecker()
        self.policy_prediction_error_calculator = PolicyPredictionErrorCalculator(
            max_expected_error=float(self.policy_config.get("max_expected_error", 1.0)),
            window_size=int(self.policy_config.get("prediction_error_window", 20)),
        )
        policy_generator_config = dict(self.policy_config)
        if "model" not in policy_generator_config and self.llm_config.get("model") is not None:
            policy_generator_config["model"] = self.llm_config.get("model")
        self.policy_generator = PolicyGenerator(
            adapter=self.adapter,
            adapter_folder=self.adapter_folder,
            memory_manager=self.memory_manager,
            goal_checker=self.goal_checker,
            prediction_error_calculator=self.policy_prediction_error_calculator,
            config=policy_generator_config,
            logger=self.logger,
            llm_client=self.llm_client,
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
            current_state = self.observation_mapper(
                raw_obs=obs,
                info=info,
                vital_state_monitor=self.vital_state_monitor,
            )
            if self.logger is not None:
                state_dict = current_state.to_dict(
                    include_inventory=self.include_inventory,
                    include_voxels=self.include_voxels,
                )
                self.logger.state_snapshot(state_dict, step=step)
                info_keys = list(info.keys()) if isinstance(info, dict) else None
                self.logger.event("env.reset", {"info_keys": info_keys}, step=step)
        else:
            current_state = self.observation_mapper(
                raw_obs=self._last_obs,
                info=self._last_info,
                vital_state_monitor=self.vital_state_monitor,
            )

        self._record_vital_state(step=step, phase="pre_action")
        workspace_messages = self.workspace.broadcast()
        goals = self._extract_goals(workspace_messages)

        homeostatic_state = self._build_homeostatic_state(
            step=step,
            state=current_state,
            workspace_messages=workspace_messages,
        )
        area_id = self._build_area_id(
            step=step,
            state=current_state,
            info=self._last_info,
            obs=self._last_obs,
        )
        drive_state = self._build_drive_state(
            homeostatic_state=homeostatic_state,
            step=step,
            area_id=area_id,
        )
        self._homeostatic_history.appendleft(drive_state)

        drive_history = DriveHistory(
            snapshots=list(self._homeostatic_history),
            channels=self._drive_channels,
            tick=step,
        )
        drive_signals = self.allostatic_controller.update(
            history=drive_history,
            area_id=area_id,
        )
        self._last_drive_signals = drive_signals
        allostatic_assessment = self._build_allostatic_assessment(drive_signals)

        self.memory_manager.set_active_policy_for_pe(self._last_selected_policy_id)
        perceptual_snapshot = self._build_perceptual_snapshot(
            step=step,
            state=current_state,
            homeostatic_state=homeostatic_state,
            area_id=area_id,
        )
        prediction_error_batch = self.perceptual_prediction_error_calculator.update(
            perceptual_snapshot
        )
        self._last_perceptual_batch = prediction_error_batch
        latest_prediction_error = self._prediction_error_from_batch(
            prediction_error_batch
        )
        arousal_valence_state = self.arousal_valence_system.update(
            homeostatic_state=homeostatic_state,
            prediction_error=latest_prediction_error,
        )
        arousal_payload = asdict(arousal_valence_state)
        channel_deltas = self._homeostatic_deltas(homeostatic_state)
        self._last_homeostatic_state_for_memory = homeostatic_state
        self.memory_manager.record_state(
            state=homeostatic_state,
            channel_deltas=channel_deltas,
            context_tags=[
                f"area:{area_id}",
                f"tick_bucket:{step // 100}",
            ],
            arousal=float(arousal_valence_state.arousal),
        )

        allostatic_assessment["policy_bias"] = dict(arousal_payload["policy_bias"])
        allostatic_assessment["urgency_signal"] = arousal_payload["urgency_signal"]
        allostatic_assessment["arousal"] = arousal_payload["arousal"]
        allostatic_assessment["valence"] = arousal_payload["valence"]

        self.memory_manager.snapshot_self_state(
            {
                "step": step,
                "phase": "allostatic_pre_action",
                "allostatic_assessment": allostatic_assessment,
                "drive_signals": asdict(drive_signals),
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
                    "highest_urgency": allostatic_assessment.get("highest_urgency"),
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
            "drive_signals": asdict(drive_signals),
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
            self._record_policy_trace(
                selected_policy_id=selected_policy_id,
                drive_signals=drive_signals,
                reward=reward,
                step=step,
            )
            self._last_selected_policy_id = selected_policy_id
        else:
            self._last_selected_policy_id = "bootstrap"

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
            state=state,
            hunger=hunger,
            lighting=lighting,
            nearby=nearby,
            inventory_state=inventory_state,
            info=self._last_info,
            obs=self._last_obs,
        )
        threat_proximity = self._estimate_threat_proximity(
            state=state,
            messages=workspace_messages,
            health=health,
            lighting=lighting,
            homeostasis=homeostasis,
            info=self._last_info,
            obs=self._last_obs,
        )
        return HomeostaticState(
            health=health,
            hunger=hunger,
            resource_level=resource_level,
            threat_proximity=threat_proximity,
            oxygen=oxygen,
            tick=step,
        )

    def _build_drive_state(
        self,
        homeostatic_state: HomeostaticState,
        step: int,
        area_id: str,
    ) -> DriveState:
        values = {
            "health": self._clamp01(homeostatic_state.health),
            "hunger": self._clamp01(homeostatic_state.hunger),
            "oxygen": self._clamp01(homeostatic_state.oxygen),
            "resource_level": self._clamp01(homeostatic_state.resource_level),
            "safety": self._clamp01(1.0 - homeostatic_state.threat_proximity),
        }
        return DriveState(values=values, tick=step, context_hash=area_id)

    def _build_allostatic_assessment(
        self,
        drive_signals: PrioritisedDriveSignals,
    ) -> Dict[str, Any]:
        needs: List[Dict[str, Any]] = []
        tags: List[str] = []
        min_ttc: Optional[float] = None
        for signal in drive_signals.signals:
            need = {
                "need_id": f"stabilize_{signal.channel_id}",
                "urgency": self._clamp01(signal.urgency),
                "time_to_critical_steps": max(0, int(round(signal.ticks_to_critical))),
                "actions": [signal.suggested_action_tag] if signal.suggested_action_tag else [],
                "resources": [signal.channel_id],
                "evidence": [
                    f"current={signal.current_value:.3f}",
                    f"projected={signal.projected_value:.3f}",
                ],
            }
            needs.append(need)
            if signal.suggested_action_tag:
                tags.append(str(signal.suggested_action_tag).strip().lower())
            tags.append(str(signal.channel_id).strip().lower())
            if min_ttc is None:
                min_ttc = signal.ticks_to_critical
            else:
                min_ttc = min(min_ttc, signal.ticks_to_critical)

        horizon = (
            int(round(min_ttc))
            if isinstance(min_ttc, (int, float))
            else int(self._allostatic_config.planning_horizon)
        )
        return {
            "source": "drive_model",
            "risk_level": self._clamp01(drive_signals.highest_urgency),
            "confidence": 1.0,
            "highest_urgency": self._clamp01(drive_signals.highest_urgency),
            "survival_horizon_steps": max(1, horizon),
            "needs": needs,
            "policy_bias_tags": list(dict.fromkeys(tag for tag in tags if tag)),
        }

    def _build_perceptual_snapshot(
        self,
        step: int,
        state: AgentState,
        homeostatic_state: HomeostaticState,
        area_id: str,
    ) -> ObservationSnapshot:
        entity_density = self._estimate_entity_density(
            state=state,
            info=self._last_info,
            obs=self._last_obs,
        )
        terrain_novelty = self._estimate_terrain_novelty(
            state=state,
            info=self._last_info,
            obs=self._last_obs,
        )
        return ObservationSnapshot(
            health=self._clamp01(homeostatic_state.health),
            hunger=self._clamp01(homeostatic_state.hunger),
            resource_level=self._clamp01(homeostatic_state.resource_level),
            oxygen=self._clamp01(homeostatic_state.oxygen),
            threat_proximity=self._clamp01(homeostatic_state.threat_proximity),
            entity_density=self._clamp01(entity_density),
            terrain_novelty=self._clamp01(terrain_novelty),
            area_id=area_id,
            last_action=self._last_selected_policy_id,
            tick=int(step),
        )

    def _prediction_error_from_batch(
        self,
        batch: PredictionErrorBatch,
    ) -> Optional[PredictionError]:
        if not batch.errors:
            return None
        source = str(batch.dominant_source or "visual")
        if source not in {"visual", "proprioceptive", "threat"}:
            source = "visual"
        return PredictionError(
            magnitude=self._clamp01(batch.aggregate_magnitude),
            source=source,
            tick=int(batch.tick),
        )

    def _build_area_id(
        self,
        step: int,
        state: AgentState,
        info: Any,
        obs: Any,
    ) -> str:
        builder = getattr(self.adapter, "build_area_id", None)
        if callable(builder):
            try:
                area_id = builder(state=state, info=info, obs=obs, step=step)
                if isinstance(area_id, str) and area_id.strip():
                    return area_id.strip()
            except Exception:
                pass

        biome = self._as_mapping(getattr(state, "biome", None))
        position = self._as_mapping(getattr(state, "position", None))
        biome_name = str(biome.get("biome_name") or "unknown").strip().lower() or "unknown"
        xpos = self._as_optional_float(position.get("xpos")) or 0.0
        zpos = self._as_optional_float(position.get("zpos")) or 0.0
        return f"{biome_name}:{int(xpos // 32.0)}:{int(zpos // 32.0)}"

    def _estimate_resource_level(
        self,
        state: AgentState,
        hunger: float,
        lighting: Mapping[str, Any],
        nearby: Mapping[str, Any],
        inventory_state: Mapping[str, Any],
        info: Any,
        obs: Any,
    ) -> float:
        estimator = getattr(self.adapter, "estimate_resource_level", None)
        if callable(estimator):
            try:
                value = estimator(
                    state=state,
                    hunger=hunger,
                    lighting=lighting,
                    nearby=nearby,
                    inventory_state=inventory_state,
                    info=info,
                    obs=obs,
                )
                if isinstance(value, (int, float)):
                    return self._clamp01(float(value))
            except Exception:
                pass
        return self._clamp01(hunger)

    def _estimate_threat_proximity(
        self,
        state: AgentState,
        messages: List[AgentMessage],
        health: float,
        lighting: Mapping[str, Any],
        homeostasis: Mapping[str, Any],
        info: Any,
        obs: Any,
    ) -> float:
        estimator = getattr(self.adapter, "estimate_threat_proximity", None)
        if callable(estimator):
            try:
                value = estimator(
                    state=state,
                    messages=messages,
                    health=health,
                    lighting=lighting,
                    homeostasis=homeostasis,
                    info=info,
                    obs=obs,
                )
                if isinstance(value, (int, float)):
                    return self._clamp01(float(value))
            except Exception:
                pass
        return self._clamp01(1.0 - health)

    def _estimate_entity_density(self, state: AgentState, info: Any, obs: Any) -> float:
        estimator = getattr(self.adapter, "estimate_entity_density", None)
        if callable(estimator):
            try:
                value = estimator(state=state, info=info, obs=obs)
                if isinstance(value, (int, float)):
                    return self._clamp01(float(value))
            except Exception:
                pass
        return 0.0

    def _estimate_terrain_novelty(self, state: AgentState, info: Any, obs: Any) -> float:
        estimator = getattr(self.adapter, "estimate_terrain_novelty", None)
        if callable(estimator):
            try:
                value = estimator(state=state, info=info, obs=obs)
                if isinstance(value, (int, float)):
                    return self._clamp01(float(value))
            except Exception:
                pass
        return 0.0

    def _homeostatic_deltas(self, current: HomeostaticState) -> Dict[str, float]:
        previous = self._last_homeostatic_state_for_memory
        if previous is None:
            return {}

        dt = max(1, int(current.tick) - int(previous.tick))
        channels = ("health", "hunger", "resource_level", "threat_proximity", "oxygen")
        out: Dict[str, float] = {}
        for channel in channels:
            prev_value = getattr(previous, channel, None)
            curr_value = getattr(current, channel, None)
            if not isinstance(prev_value, (int, float)):
                continue
            if not isinstance(curr_value, (int, float)):
                continue
            out[channel] = float(curr_value - prev_value) / float(dt)
        return out

    def _record_policy_trace(
        self,
        selected_policy_id: str,
        drive_signals: PrioritisedDriveSignals,
        reward: Any,
        step: int,
    ) -> None:
        signals = list(drive_signals.signals)
        if len(signals) < 2:
            return

        winner, runner_up = self._trace_channels_for_policy(
            selected_policy_id=selected_policy_id,
            signals=signals,
        )
        if winner is None or runner_up is None:
            return
        if winner.channel_id == runner_up.channel_id:
            return

        winner_channel = self._drive_channel_map.get(winner.channel_id)
        runner_up_channel = self._drive_channel_map.get(runner_up.channel_id)
        survival_weight = 1.0 if (
            bool(getattr(winner_channel, "irreversible", False))
            or bool(getattr(runner_up_channel, "irreversible", False))
        ) else 0.5
        tick_norm = self._clamp01(
            float(step % max(1, self._allostatic_config.planning_horizon))
            / float(max(1, self._allostatic_config.planning_horizon))
        )
        context_vector = np.asarray(
            [
                self._clamp01(winner.urgency),
                self._clamp01(runner_up.urgency),
                self._clamp01(max(winner.urgency, runner_up.urgency)),
                self._clamp(
                    winner.projected_value - runner_up.projected_value,
                    -1.0,
                    1.0,
                ),
                self._clamp01(survival_weight),
                tick_norm,
            ],
            dtype=np.float32,
        )
        self.memory_manager.record_trace(
            channel_a_id=winner.channel_id,
            channel_b_id=runner_up.channel_id,
            winner_channel_id=winner.channel_id,
            action_tag=str(selected_policy_id),
            context_vector=context_vector,
            outcome_score=self._reward_to_outcome_score(reward),
            tick=int(step),
        )

    def _trace_channels_for_policy(
        self,
        selected_policy_id: str,
        signals: List[Any],
    ) -> tuple[Optional[Any], Optional[Any]]:
        policy_drive_tags = self._policy_drive_tags(selected_policy_id)
        winner: Optional[Any] = None
        if policy_drive_tags:
            for signal in signals:
                channel_id = str(getattr(signal, "channel_id", "")).strip().lower()
                if channel_id in policy_drive_tags:
                    winner = signal
                    break
        if winner is None and signals:
            winner = signals[0]

        if winner is None:
            return None, None

        winner_channel_id = str(getattr(winner, "channel_id", "")).strip().lower()
        runner_up: Optional[Any] = None
        for signal in signals:
            channel_id = str(getattr(signal, "channel_id", "")).strip().lower()
            if channel_id == winner_channel_id:
                continue
            runner_up = signal
            break
        return winner, runner_up

    def _policy_drive_tags(self, policy_id: str) -> set[str]:
        getter = getattr(self.adapter, "get_available_policies", None)
        if not callable(getter):
            return set()
        try:
            policies = getter()
        except Exception:
            return set()
        if not isinstance(policies, list):
            return set()

        descriptor: Optional[Mapping[str, Any]] = None
        for item in policies:
            if not isinstance(item, Mapping):
                continue
            item_policy_id = item.get("policy_id")
            if isinstance(item_policy_id, str) and item_policy_id == policy_id:
                descriptor = item
                break
        if descriptor is None:
            return set()

        drive_tags = descriptor.get("drive_tags")
        if isinstance(drive_tags, list):
            normalized = {
                str(tag).strip().lower()
                for tag in drive_tags
                if isinstance(tag, str) and tag.strip()
            }
            if normalized:
                return normalized

        tags = descriptor.get("tags")
        tokens: set[str] = set()
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str):
                    tokens.update(self._name_tokens(tag))
        description = descriptor.get("description")
        if isinstance(description, str):
            tokens.update(self._name_tokens(description))
        callable_name = descriptor.get("callable_name")
        if isinstance(callable_name, str):
            tokens.update(self._name_tokens(callable_name))

        inferred: set[str] = set()
        for drive_tag, marker_tokens in self._POLICY_DRIVE_TAG_RULES.items():
            if drive_tag in tokens or marker_tokens.intersection(tokens):
                inferred.add(drive_tag)
        return inferred

    @staticmethod
    def _name_tokens(name: str) -> List[str]:
        text = str(name).strip().lower()
        if not text:
            return []
        tokens = re.split(r"[^a-z0-9]+", text)
        return [token for token in tokens if token]

    @staticmethod
    def _reward_to_outcome_score(reward: Any) -> float:
        try:
            numeric = float(reward)
        except (TypeError, ValueError):
            return 0.0
        # Sparse non-negative reward assumption: 0 is failure/neutral, positives are progress.
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _build_drive_channels(config: Mapping[str, Any]) -> List[DriveChannel]:
        default_channels: List[DriveChannel] = [
            DriveChannel(
                id="health",
                setpoint=0.9,
                critical_threshold=0.25,
                irreversible=True,
                recovery_cost_ticks=25,
                suggested_action_tag="heal",
            ),
            DriveChannel(
                id="hunger",
                setpoint=0.8,
                critical_threshold=0.2,
                irreversible=False,
                recovery_cost_ticks=20,
                suggested_action_tag="eat",
            ),
            DriveChannel(
                id="oxygen",
                setpoint=0.9,
                critical_threshold=0.2,
                irreversible=True,
                recovery_cost_ticks=30,
                suggested_action_tag="surface",
            ),
            DriveChannel(
                id="resource_level",
                setpoint=0.7,
                critical_threshold=0.3,
                irreversible=False,
                recovery_cost_ticks=15,
                suggested_action_tag="gather",
            ),
            DriveChannel(
                id="safety",
                setpoint=0.8,
                critical_threshold=0.35,
                irreversible=True,
                recovery_cost_ticks=20,
                suggested_action_tag="retreat",
            ),
        ]
        raw_channels = config.get("channels")
        if not isinstance(raw_channels, list):
            return default_channels

        out: List[DriveChannel] = []
        for item in raw_channels:
            if not isinstance(item, Mapping):
                continue
            channel_id = item.get("id")
            if not isinstance(channel_id, str) or not channel_id.strip():
                continue
            out.append(
                DriveChannel(
                    id=channel_id.strip(),
                    setpoint=AgentLoop._clamp(item.get("setpoint", 0.8), 0.0, 1.0),
                    critical_threshold=AgentLoop._clamp(
                        item.get("critical_threshold", 0.3), 0.0, 1.0
                    ),
                    irreversible=bool(item.get("irreversible", False)),
                    recovery_cost_ticks=max(0, int(item.get("recovery_cost_ticks", 20))),
                    suggested_action_tag=str(item.get("suggested_action_tag", "")).strip(),
                )
            )
        return out or default_channels

    @staticmethod
    def _build_allostatic_config(config: Mapping[str, Any]) -> AllostaticConfig:
        return AllostaticConfig(
            planning_horizon=max(1, int(config.get("planning_horizon", 50))),
            history_window=max(1, int(config.get("history_window", 20))),
            irreversibility_bonus=AgentLoop._clamp(  # type: ignore[arg-type]
                config.get("irreversibility_bonus", 0.3), 0.0, 1.0
            ),
            recovery_weight_factor=AgentLoop._clamp(  # type: ignore[arg-type]
                config.get("recovery_weight_factor", 0.2), 0.0, 1.0
            ),
            urgency_tie_epsilon=AgentLoop._clamp(  # type: ignore[arg-type]
                config.get("urgency_tie_epsilon", 0.05), 0.0, 1.0
            ),
            threat_prior_weight=AgentLoop._clamp(  # type: ignore[arg-type]
                config.get("threat_prior_weight", 0.3), 0.0, 1.0
            ),
            min_confidence=AgentLoop._clamp(  # type: ignore[arg-type]
                config.get("min_confidence", 0.5), 0.0, 1.0
            ),
        )

    @staticmethod
    def _build_pe_config(config: Mapping[str, Any]) -> PEConfig:
        return PEConfig(
            alpha=AgentLoop._clamp(config.get("alpha", 0.1), 1e-6, 1.0),  # type: ignore[arg-type]
            epsilon=AgentLoop._clamp(config.get("epsilon", 0.01), 1e-6, 1.0),  # type: ignore[arg-type]
            sigma_clip=AgentLoop._clamp(config.get("sigma_clip", 3.0), 1e-6, 10.0),  # type: ignore[arg-type]
            default_precision=AgentLoop._clamp(config.get("default_precision", 0.5), 0.0, 1.0),  # type: ignore[arg-type]
            min_precision=AgentLoop._clamp(config.get("min_precision", 0.3), 0.0, 1.0),  # type: ignore[arg-type]
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

    @staticmethod
    def _clamp(value: Any, lo: float, hi: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = lo
        return max(lo, min(hi, numeric))
