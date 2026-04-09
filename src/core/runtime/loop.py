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
import os
import time
from typing import Any, Callable, Dict, List, Optional

# Set AAI_DEBUG=1 to enable per-step signal diagnostics (Gate 5).
DEBUG: bool = os.getenv("AAI_DEBUG", "").lower() in ("1", "true", "yes")

from core.adapters.base import AbstractEnvironmentAdapter
from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.layers.action_selection.FreeEnergyMinimizer import FreeEnergyMinimizer
from core.layers.action_selection.MotorControlInterface import MotorControlInterface
from core.layers.action_selection.PolicyGenerator import PolicyGenerator, _affect_label
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

        # Debug log: written to the run directory so each run gets its own file.
        # Falls back to "debug.log" in cwd if run_dir is unavailable.
        _debug_path = str(logger.run_dir / "debug.log") if hasattr(logger, "run_dir") else "debug.log"
        self._debug_log_path: str = _debug_path

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
        self._policy_gen.set_episodic_memory_callback(memory.query_similar_traces)

        def _llm_log_cb(
            prompt: str,
            resp: Any,
            reason: str,
            step: int,
            selected: Optional[str] = None,
            llm_reason: Optional[str] = None,
        ) -> None:
            logger.llm(
                prompt=prompt,
                response=resp.content,
                model=resp.model or "",
                latency_ms=resp.latency_ms or 0.0,
                input_tokens=resp.input_tokens or 0,
                output_tokens=resp.output_tokens or 0,
                trigger_reason=reason,
                selected=selected,
                reason=llm_reason,
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
        # Rolling action history for LLM context (last 10 actions).
        # _recent_motor_effs is a parallel list: motor_efficiency of next_state
        # after each action, so the prompt can annotate each step as blocked/ok.
        self._recent_actions: List[str] = []
        self._recent_motor_effs: List[float] = []
        # Previous step's arousal — passed to PEC so LC-NE precision scaling uses
        # the arousal that was current when the action was taken, not the one computed
        # from its outcome (which isn't available yet at PEC call time).
        self._last_arousal: float = 0.0

    def _dbg(self, msg: str) -> None:
        """Append a debug line to this run's debug.log (line-buffered)."""
        with open(self._debug_log_path, "a", buffering=1) as _f:
            _f.write(msg + "\n")

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

        # Reset area familiarity for the starting position so area_novelty = 1.0
        # on step 0. This lets the epistemic value of turns dominate the urgency
        # fallback, producing a natural orienting scan at episode start without
        # any hardcoded behaviour.
        start_area = (self._current_state.perception.area_id or "unknown")
        self._memory.reset_area_familiarity(start_area)

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

        self._memory.save_faiss_stores()
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
            arousal=self._last_arousal,
        )

        # --- Layer 1 continued: Arousal/Valence ---
        av = self._arousal_valence.update(
            vitals=vitals,
            drive_batch=drive_batch,
            pe_batch=pe_batch,
            workspace=self._workspace,
            step=step,
            raycast_hits=state.perception.raycast_hits,
        )

        if DEBUG:
            _motor_pe_dbg = pe_batch.motor_pe if pe_batch else 0.0
            self._dbg(f"[DBG STEP {step}] signals →")
            self._dbg(f"  arousal={av.arousal:.4f}  valence={av.valence:.4f}  "
                  f"lr_mod={av.learning_rate_mod:.4f}")
            self._dbg(f"  drive_urgency max={drive_batch.max_urgency:.4f}  "
                  f"dominant={drive_batch.dominant_channel}")
            for _s in drive_batch.signals:
                self._dbg(f"    {_s.channel_id}: val={_s.current_value:.4f}  "
                      f"urgency={_s.urgency:.4f}")
            if pe_batch:
                self._dbg(f"  PE mean(perceptual)={pe_batch.mean_magnitude:.4f}  "
                      f"motor={_motor_pe_dbg:.4f}  max={pe_batch.max_magnitude:.4f}")
                _warn = "  *** always 1.0 = motor_efficiency broken" if _motor_pe_dbg > 0.9 else ""
                self._dbg(f"  lc_ne_motor_pe_contribution≈{min(_motor_pe_dbg, 1.0) * 0.4:.4f}"
                      f"  (should be ~0 in open space, ~0.4 at wall){_warn}")

        # --- Layer 4: Metacognitive ---
        context = self._metacognitive.update(self._workspace, [], step)
        context["policies"] = self._policies
        context["heading"] = state.position.heading or 0.0
        context["motor_efficiency"] = float(state.raw_metadata.get("motor_efficiency", 1.0))
        context["stuck_steps"] = int(state.raw_metadata.get("stuck_steps", 0))
        # Affect state — used by PolicyGenerator arousal-diversity fallback
        context["arousal"] = av.arousal
        context["valence"] = av.valence
        self._last_arousal = av.arousal   # carried into next step's PEC precision
        # Last action — excluded from candidates when arousal is high
        context["last_action"] = self._last_policy_id
        # Rolling history for LLM prompt — lets the model detect repetition.
        # recent_motor_effs is parallel: outcome efficiency of each past action.
        context["recent_actions"] = list(self._recent_actions)
        context["recent_motor_effs"] = list(self._recent_motor_effs)
        # Raycast hits from perception — directional food/threat bearing for LLM.
        # Use `or []` so context always holds a list; downstream consumers (FEA,
        # PolicyGenerator) iterate directly without their own None-guards.
        context["raycast_hits"] = state.perception.raycast_hits or []
        # Current AgentState — used by PolicyGenerator to query episodic memory
        context["current_state"] = state

        if DEBUG:
            _rc_ctx = context["raycast_hits"]
            _food_rc = [r for r in _rc_ctx if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")]
            self._dbg(
                f"CONTEXT_RC step={step}  n_rays={len(_rc_ctx)}  food_rays={_food_rc}"
                f"  state_rc_none={state.perception.raycast_hits is None}"
            )

        # --- Layer 3: Free energy scoring ---
        fe_scores = self._free_energy.score(
            policies=self._policies,
            drive_batch=drive_batch,
            pe_batch=pe_batch,
            area_familiarity=familiarity,
            context=context,
        )
        context["free_energy_scores"] = fe_scores
        context["decisive_signal"] = self._free_energy.last_decisive_signal

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
        # Maintain rolling action history (last 10 steps).
        # Parallel list records the motor outcome so prompts can show blocked/ok.
        self._recent_actions.append(selected_id)
        if len(self._recent_actions) > 10:
            self._recent_actions.pop(0)
        self._recent_motor_effs.append(float(next_state.raw_metadata.get("motor_efficiency", 1.0)))
        if len(self._recent_motor_effs) > 10:
            self._recent_motor_effs.pop(0)

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

        # Outcome score: causal action effectiveness.
        #
        # Four components measuring orthogonal consequence types so the LTM
        # receives a gradient that differentiates actions with identical
        # homeostatic outcomes but different strategic values.
        next_vitals = self._vital_monitor.read(next_state)
        prev_health = float(prev_state.homeostasis.health or 0.0)
        prev_sat = float(prev_state.homeostasis.saturation or 0.0)
        next_health = float(next_state.homeostasis.health or 0.0)
        next_sat = float(next_state.homeostasis.saturation or 0.0)
        prev_rc = prev_state.perception.raycast_hits or []
        next_rc = next_state.perception.raycast_hits or []
        prev_threat_v = float(vitals.get("threat_proximity") or 0.0)
        next_threat_v = float(next_vitals.get("threat_proximity") or 0.0)

        # 1. Homeostatic RPE: actual delta minus expected passive loss, centred at 0.5.
        #    Single signal replacing the correlated (relief_score, homeo_delta) pair.
        #    ~0 on a passive depletion tick; ~0.9 on food collection.
        _homeo_cfg = self._config.get("adapter_config", {}).get("homeostatic", {})
        _expected_delta = -(
            float(_homeo_cfg.get("health_depletion_rate", 0.002))
            + float(_homeo_cfg.get("saturation_depletion_rate", 0.002))
        )
        _actual_delta = (next_health - prev_health) + (next_sat - prev_sat)
        homeo_score = max(0.0, min(1.0, 0.5 + (_actual_delta - _expected_delta) * 3.0))

        # 2. Motor alignment: perceptual consequence matches action's causal expectation.
        #    Extends the former move_forward-only extero_match to all action types so
        #    "turned toward food" and "turned away from food" receive different scores
        #    even when their homeostatic outcomes are identical.
        #    Idle → neutral (no expected perceptual change).
        motor_align = 0.5
        if selected_id == "move_forward":
            p_food = next((r for r in prev_rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None)
            n_food = next((r for r in next_rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None)
            if p_food:
                prev_dist = float(p_food.get("distance", 1.0))
                next_dist = float(n_food.get("distance", prev_dist)) if n_food else prev_dist
                motor_align = max(0.0, min(1.0, 0.5 + (prev_dist - next_dist) * 5.0))
        elif selected_id in ("turn_left", "turn_right"):
            # |angle_deg| decreased → food moved toward center → good turn.
            # Scaled by sensor range (80°): 10° improvement → motor_align ≈ 0.625.
            p_food = next((r for r in prev_rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None)
            n_food = next((r for r in next_rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None)
            if p_food and n_food:
                p_angle = abs(float(p_food.get("angle_deg", 0.0)))
                n_angle = abs(float(n_food.get("angle_deg", 0.0)))
                motor_align = max(0.0, min(1.0, 0.5 + (p_angle - n_angle) / 80.0))
        elif selected_id == "move_backward":
            # Backing away from threat → threat distance increased.
            motor_align = max(0.0, min(1.0, 0.5 + (prev_threat_v - next_threat_v) * 5.0))

        # 3. Threat outcome: did threat recede this step, regardless of homeostasis?
        #    Applies to all actions; escaping danger is a success on its own axis.
        threat_score = max(0.0, min(1.0, 0.5 + (prev_threat_v - next_threat_v) * 5.0))

        # 4. Epistemic yield: novelty gain during low-urgency exploratory steps.
        #    Active only when no food visible and urgency < 0.4; neutral (0.5) otherwise.
        #    Moving to a less-familiar area is epistemically valuable even if it
        #    produces no immediate homeostatic benefit.
        epistemic_score = 0.5
        _food_visible_prev = any(
            r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti") for r in prev_rc
        )
        if not _food_visible_prev and drive_batch.max_urgency < 0.4:
            _next_area = next_state.perception.area_id or "unknown"
            _next_familiarity = self._memory.get_area_familiarity(_next_area)
            # familiarity = pre-action familiarity of current area (computed above).
            # Moving to a less-familiar area → positive delta → score > 0.5.
            _novelty_delta = familiarity - _next_familiarity
            epistemic_score = max(0.0, min(1.0, 0.5 + _novelty_delta * 5.0))

        outcome_score = float(max(0.0, min(1.0,
            homeo_score     * 0.35
            + motor_align   * 0.30
            + threat_score  * 0.20
            + epistemic_score * 0.15
        )))

        # Situation note for episodic memory — encodes perceptual context + causal
        # consequence so the LLM can read "GoodGoal at 1.32, health_delta=+0.300" and
        # distinguish "approached food" from "blocked facing food" in retrieved traces.
        _rc = prev_state.perception.raycast_hits
        if _rc:
            # Use the closest food ray for the situation note; fall back to forward ray.
            _food_r = next(
                (r for r in _rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), _rc[0]
            )
            _tag = _food_r.get("hit_tag")
            _dist = _food_r.get("distance", 1.0)
            _situation = f"{_tag} at {_dist:.2f}" if _tag else "open"
        else:
            _situation = "no raycast"
        _motor = float(prev_state.raw_metadata.get("motor_efficiency", 1.0))
        if _motor < 0.3:
            _situation += ", blocked"
        _health_before = float(prev_state.homeostasis.health or 0.0)
        _health_after = float(next_state.homeostasis.health or 0.0)
        _health_delta = _health_after - _health_before
        _situation += f", health_delta={_health_delta:+.3f}"
        situation_note = _situation

        self._memory.record_policy_trace(
            state=prev_state,
            policy_id=selected_id,
            outcome_score=outcome_score,
            drive_signals={s.channel_id: s.urgency for s in drive_batch.signals},
            notes=situation_note,
        )
        self._memory.record_episode_outcome(
            selected_id,
            score=outcome_score,
            outcome="success" if outcome_score > 0.6 else "partial" if outcome_score > 0.3 else "failure",
        )

        # --- Structured metrics logging ---
        step_ms = (time.monotonic() - t_step_start) * 1000.0
        _rc = next_state.perception.raycast_hits
        _food_vis = bool(_rc and any(r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti") for r in _rc))
        affect_state = _affect_label(
            av.arousal, av.valence, drive_batch.dominant_channel, _food_vis
        )
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
                "affect_state": affect_state,
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
