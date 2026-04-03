"""
AgentLoop — main per-step orchestration engine.

Wires all brain layers together and drives the cognitive cycle:
  1. Receive AgentState from environment
  2. Layer 1: VitalStateMonitor → drive signals → arousal/valence
  3. Layer 2: WorldModel predict → PredictionError → publish
  4. Layer 4: MetacognitiveMonitor → context assembly
  5. Layer 3: FreeEnergy → PolicyGenerator → action
  6. MotorControlInterface.execute → next state
  7. WorldModel.update → learn from transition
  8. Memory record → log metrics

The AgentLoop holds references to the adapter (as AbstractEnvironmentAdapter)
and dispatches actions through MotorControlInterface's action_dispatcher closure.
Brain layers access the adapter ONLY through this controlled interface.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from core.adapters.base import AbstractEnvironmentAdapter
from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.layers.action_selection.FreeEnergyMinimizer import FreeEnergyMinimizer
from core.layers.action_selection.MotorControlInterface import MotorControlInterface
from core.layers.action_selection.PolicyGenerator import PolicyGenerator
from core.layers.interoceptive.AllostaticController import AllostaticController
from core.layers.interoceptive.ArousalValenceSystem import ArousalValenceSystem
from core.layers.interoceptive.VitalStateMonitor import VitalStateMonitor
from core.layers.metacognitive.MetacognitiveMonitor import MetacognitiveMonitor
from core.layers.predictive.PredictionErrorCalculator import PredictionErrorCalculator
from core.layers.predictive.WorldModelGenerator import WorldModelGenerator
from core.llm.base import AbstractLLMClient
from core.memory.manager import MemoryManager
from core.models.state import AgentState
from core.observability.logger import RunLogger

log = logging.getLogger(__name__)


class AgentLoop:
    """
    Orchestrates a single agent episode.

    Receives fully initialized components at construction time.
    No imports of concrete adapters, providers, or memory implementations.
    """

    def __init__(
        self,
        adapter: AbstractEnvironmentAdapter,
        memory: MemoryManager,
        llm_client: Optional[AbstractLLMClient],
        logger: RunLogger,
        config: Dict[str, Any],
    ) -> None:
        self._adapter = adapter
        self._memory = memory
        self._llm = llm_client
        self._logger = logger
        self._config = config

        # Workspace — cleared at top of each step
        self._workspace = GlobalWorkspace()

        # Available policies from adapter
        self._policies = adapter.get_available_policies()

        # Layer 1: Interoceptive
        vitals = adapter.get_available_vitals()
        drive_channels = adapter.get_drive_channels()

        self._vital_monitor = VitalStateMonitor(available_vitals=vitals + ["resource_level", "threat_proximity"])
        self._allostatic = AllostaticController(drive_channels=drive_channels, config=config)
        self._arousal_valence = ArousalValenceSystem(config=config)

        # Layer 2: Predictive
        self._world_model = WorldModelGenerator(config=config)
        self._pe_calc = PredictionErrorCalculator(world_model=self._world_model, config=config)

        # Layer 3: Action selection
        self._policy_gen = PolicyGenerator(llm_client=llm_client, config=config)
        self._policy_gen.set_policy_history_callback(memory.get_ltm_success_rate)

        def _llm_log_cb(prompt: str, resp: Any, reason: str, step: int, selected: Optional[str] = None) -> None:
            logger.llm(
                prompt=prompt,
                response=resp.content,
                model=resp.model or "",
                latency_ms=resp.latency_ms or 0.0,
                input_tokens=resp.input_tokens or 0,
                output_tokens=resp.output_tokens or 0,
                trigger_reason=reason,
                selected=selected,
                step=step,
            )

        self._policy_gen.set_llm_log_callback(_llm_log_cb)
        self._free_energy = FreeEnergyMinimizer(config=config)

        # Action dispatcher closure — MotorControlInterface has NO adapter import
        def _dispatch(action_id: str):
            return adapter.step(action_id)

        self._motor = MotorControlInterface(action_dispatcher=_dispatch)

        # Layer 4: Metacognitive
        self._metacognitive = MetacognitiveMonitor(config=config)

        # Runtime state
        self._current_state: Optional[AgentState] = None
        self._last_policy_id: Optional[str] = None
        self._step: int = 0
        # Rolling action history for LLM context (last 10 actions)
        self._recent_actions: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_steps: int = 500) -> Dict[str, Any]:
        """
        Run a full episode for up to max_steps steps.

        Returns summary dict with episode statistics.
        """
        log.info("Episode starting (max_steps=%d, adapter=%s)",
                 max_steps, type(self._adapter).__name__)

        self._current_state = self._adapter.reset()
        self._step = 0

        episode_stats = {
            "steps_run": 0,
            "done_reason": "max_steps",
            "final_health": 0.0,
            "final_saturation": 0.0,
        }

        for step in range(max_steps):
            self._step = step
            try:
                done = self._run_step()
                episode_stats["steps_run"] = step + 1
            except Exception as exc:
                self._logger.traceback(exc, context="AgentLoop.run_step", step=step)
                log.error("Step %d failed: %s", step, exc)
                episode_stats["done_reason"] = "error"
                break

            if done:
                episode_stats["done_reason"] = "episode_done"
                break

        # Final stats
        if self._current_state:
            episode_stats["final_health"] = self._current_state.homeostasis.health or 0.0
            episode_stats["final_saturation"] = self._current_state.homeostasis.saturation or 0.0

        self._memory.log_summary()
        log.info("Episode complete: %s", episode_stats)
        self._logger.event("episode_complete", episode_stats, step=self._step)
        return episode_stats

    def run_step(self) -> bool:
        """Run a single step. Returns True if episode is done."""
        return self._run_step()

    # ------------------------------------------------------------------
    # Core step logic
    # ------------------------------------------------------------------

    def _run_step(self) -> bool:
        step = self._step
        state = self._current_state
        if state is None:
            return True

        t_step_start = time.monotonic()

        self._workspace.clear()

        # --- Layer 1: Interoceptive ---
        vitals = self._vital_monitor.update(state, self._workspace, step)

        # Inject memory depletion rates into allostatic controller
        depletion_rates = self._memory.get_depletion_rates()
        drive_batch = self._allostatic.update(
            vitals, self._workspace, step,
            memory_depletion_rates=depletion_rates,
        )

        # --- Layer 2: Predictive ---
        predicted = self._world_model.predict(state, self._last_policy_id)
        area_id = state.perception.area_id or "unknown"
        familiarity = self._memory.get_area_familiarity(area_id)
        pe_batch = self._pe_calc.update(
            predicted=predicted,
            observed=state,
            last_action=self._last_policy_id,
            workspace=self._workspace,
            step=step,
            area_familiarity=familiarity,
        )

        # --- Layer 1 continued: Arousal/Valence ---
        av = self._arousal_valence.update(
            vitals=vitals,
            drive_batch=drive_batch,
            pe_batch=pe_batch,
            workspace=self._workspace,
            step=step,
        )

        # --- Layer 4: Metacognitive ---
        context = self._metacognitive.update(self._workspace, [], step)
        context["policies"] = self._policies
        context["heading"] = state.position.heading or 0.0
        context["motor_efficiency"] = float(state.raw_metadata.get("motor_efficiency", 1.0))
        # Affect state — used by PolicyGenerator arousal-diversity fallback
        context["arousal"] = av.arousal
        context["valence"] = av.valence
        # Last action — excluded from candidates when arousal is high
        context["last_action"] = self._last_policy_id
        # Rolling history for LLM prompt — lets the model detect repetition
        context["recent_actions"] = list(self._recent_actions)
        # Raycast hits from perception — directional food/threat bearing for LLM
        context["raycast_hits"] = state.perception.raycast_hits

        # --- Layer 3: Free energy scoring ---
        fe_scores = self._free_energy.score(
            policies=self._policies,
            drive_batch=drive_batch,
            pe_batch=pe_batch,
            area_familiarity=familiarity,
        )
        context["free_energy_scores"] = fe_scores

        # --- Layer 3: Policy selection ---
        selected_id = self._policy_gen.propose_action(
            policies=self._policies,
            goals=[],
            context=context,
            workspace=self._workspace,
            step=step,
        )

        if selected_id is None:
            selected_id = "idle"

        # --- Execute action ---
        prev_state = state
        next_state, done = self._motor.execute(selected_id)
        self._last_policy_id = selected_id
        self._current_state = next_state
        # Maintain rolling action history (last 10 steps)
        self._recent_actions.append(selected_id)
        if len(self._recent_actions) > 10:
            self._recent_actions.pop(0)

        # --- Layer 2: World model learning ---
        self._world_model.update(
            prev_state=prev_state,
            action_id=selected_id,
            next_state=next_state,
            workspace=self._workspace,
            step=step,
        )

        # --- Memory recording ---
        self._memory.record_state(next_state, action=selected_id)
        self._memory.update_state_outcome(next_state)
        self._memory.record_prediction_error(next_state, pe_batch, action=selected_id)

        # Outcome score: allostatic error after action (lower urgency = better outcome)
        next_vitals = self._vital_monitor.read(next_state)
        outcome_score = float(max(0.0, min(1.0, 1.0 - self._allostatic.peek_max_urgency(next_vitals))))

        self._memory.record_policy_trace(
            state=prev_state,
            policy_id=selected_id,
            outcome_score=outcome_score,
            drive_signals={s.channel_id: s.urgency for s in drive_batch.signals},
        )
        self._memory.record_episode_outcome(
            selected_id,
            score=outcome_score,
            outcome="success" if outcome_score > 0.6 else "partial" if outcome_score > 0.3 else "failure",
        )

        # --- Structured metrics logging ---
        step_ms = (time.monotonic() - t_step_start) * 1000.0
        self._logger.metrics(
            step=step,
            health=next_state.homeostasis.health or 0.0,
            saturation=next_state.homeostasis.saturation or 0.0,
            energy=next_state.homeostasis.energy or 0.0,
            arousal=av.arousal,
            valence=av.valence,
            pe_mean=pe_batch.mean_magnitude,
            policy_id=selected_id,
            extra={
                "step_ms": round(step_ms, 1),
                "drive_urgency": drive_batch.max_urgency,
                "dominant_channel": drive_batch.dominant_channel,
                "area_id": area_id,
                "outcome_score": outcome_score,
            },
        )

        self._logger.event("step", {
            "policy": selected_id,
            "drive_urgency": drive_batch.max_urgency,
            "pe_mean": pe_batch.mean_magnitude,
            "arousal": av.arousal,
            "valence": av.valence,
            "step_ms": round(step_ms, 1),
        }, step=step)

        if self._config.get("observability", {}).get("log_state", True):
            self._logger.state(next_state.to_dict(), step=step)

        if done:
            log.info("Step %d: episode done (health=%.3f)", step, next_state.homeostasis.health or 0.0)
            # Structured homeostatic collapse event — queryable across runs in events.jsonl
            if (next_state.homeostasis.health is not None
                    and next_state.homeostasis.health <= 0.0):
                self._logger.event("homeostatic_collapse", {
                    "health": next_state.homeostasis.health,
                    "saturation": next_state.homeostasis.saturation,
                    "dominant_drive": drive_batch.dominant_channel,
                    "steps_survived": step,
                }, step=step)

        return done
