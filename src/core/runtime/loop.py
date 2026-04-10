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
from core.models.ablation import AblationMode
from core.models.signals import ArousalValence, DriveSignal, DriveSignalBatch
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
        # Rolling action history for LLM context (last 10 actions)
        self._recent_actions: List[str] = []
        # Previous step's arousal — passed to PEC so LC-NE precision scaling uses
        # the arousal that was current when the action was taken, not the one computed
        # from its outcome (which isn't available yet at PEC call time).
        self._last_arousal: float = 0.0

        # Ablation study configuration
        ablation_cfg = config.get("ablation", {})
        self._ablation_mode: AblationMode = AblationMode(
            ablation_cfg.get("mode", AblationMode.FULL)
        )
        # Drive injection state for Probe 3 (Satiation Reorientation Latency)
        self._injection_active: bool = False
        self._injection_channel: str = ablation_cfg.get("drive_injection_channel", "saturation")
        self._injection_value: float = float(ablation_cfg.get("drive_injection_value", 0.9))
        self._injection_trigger_step: Optional[int] = (
            int(ablation_cfg["drive_injection_step"])
            if ablation_cfg.get("drive_injection_step") is not None
            else None
        )

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

        cancelled = False
        try:
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
        except KeyboardInterrupt:
            cancelled = True
            episode_stats["done_reason"] = "cancelled"

        # Final stats
        if self._current_state:
            episode_stats["final_health"] = self._current_state.homeostasis.health or 0.0
            episode_stats["final_saturation"] = self._current_state.homeostasis.saturation or 0.0

        self._memory.save_faiss_stores()
        self._memory.log_summary()

        if cancelled:
            log.info("Episode cancelled at step %d: %s", self._step, episode_stats)
            self._logger.event("episode_cancelled", episode_stats, step=self._step)
            raise KeyboardInterrupt
        else:
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

        # --- Drive injection (Probe 3 — Satiation Reorientation Latency) ---
        # Activates at the configured step and persists: keeps the injected channel
        # at or above the injection value for as long as the adapter's natural
        # depletion hasn't caught up (simulates the agent eating a large meal).
        if self._injection_trigger_step is not None and step == self._injection_trigger_step:
            self._injection_active = True
            self._logger.event("drive_injection", {
                "channel": self._injection_channel,
                "value": self._injection_value,
                "step": step,
            }, step=step)

        if self._injection_active and state.homeostasis:
            chan = self._injection_channel
            if chan == "saturation":
                current = state.homeostasis.saturation or 0.0
                if current < self._injection_value:
                    state.homeostasis.saturation = self._injection_value
                    # Update composite energy too
                    health = state.homeostasis.health or 0.0
                    state.homeostasis.energy = self._injection_value * 0.7 + health * 0.3
            elif chan == "health":
                current = state.homeostasis.health or 0.0
                if current < self._injection_value:
                    state.homeostasis.health = self._injection_value

        self._workspace.clear()

        # --- Layer 1: Interoceptive ---
        vitals = self._vital_monitor.update(state, self._workspace, step)

        # Inject memory depletion rates into allostatic controller
        depletion_rates = self._memory.get_depletion_rates()
        drive_batch = self._allostatic.update(
            vitals, self._workspace, step,
            memory_depletion_rates=depletion_rates,
        )

        # Ablation: no_interoceptive — drop AllostaticController output.
        # Replace with a zero-urgency batch so FreeEnergyMinimizer and PolicyGenerator
        # receive no pragmatic signal; raw homeostatic values still reach the LLM prompt.
        if self._ablation_mode == AblationMode.NO_INTEROCEPTIVE:
            zero_signals = [
                DriveSignal(
                    channel_id=s.channel_id,
                    current_value=s.current_value,
                    setpoint=s.setpoint,
                    urgency=0.0,
                    ticks_to_critical=None,
                    suggested_action_tags=s.suggested_action_tags,
                )
                for s in drive_batch.signals
            ]
            drive_batch = DriveSignalBatch(signals=zero_signals)

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

        # Ablation: no_arousal — zero the arousal/valence signal.
        # FreeEnergyMinimizer will use arousal=0 (no LC-NE epistemic gain scaling);
        # PolicyGenerator will not show arousal/valence in the prompt and will not
        # promote depth to "full" due to the high-arousal trigger.
        if self._ablation_mode == AblationMode.NO_AROUSAL:
            av = ArousalValence(arousal=0.0, valence=0.0, learning_rate_mod=1.0)

        if DEBUG:
            _motor_pe_dbg = 0.0
            if pe_batch:
                for _e in pe_batch.errors:
                    if _e.channel == "motor":
                        _motor_pe_dbg = float(_e.magnitude)
                        break
            self._dbg(f"[DBG STEP {step}] signals →")
            self._dbg(f"  arousal={av.arousal:.4f}  valence={av.valence:.4f}  "
                  f"lr_mod={av.learning_rate_mod:.4f}")
            self._dbg(f"  drive_urgency max={drive_batch.max_urgency:.4f}  "
                  f"dominant={drive_batch.dominant_channel}")
            for _s in drive_batch.signals:
                self._dbg(f"    {_s.channel_id}: val={_s.current_value:.4f}  "
                      f"urgency={_s.urgency:.4f}")
            if pe_batch:
                self._dbg(f"  PE mean={pe_batch.mean_magnitude:.4f}  "
                      f"max={pe_batch.max_magnitude:.4f}")
                _me = next((e for e in pe_batch.errors if e.channel == "motor"), None)
                if _me:
                    _warn = "  *** always 1.0 = motor_efficiency broken" if _me.magnitude > 0.9 else ""
                    self._dbg(f"  PE motor: expected={_me.expected:.3f}  "
                          f"observed={_me.observed:.3f}  "
                          f"magnitude={_me.magnitude:.4f}{_warn}")
            self._dbg(f"  lc_ne_motor_pe_contribution≈{min(_motor_pe_dbg, 1.0) * 0.4:.4f}  "
                  f"(should be ~0 in open space, ~0.4 at wall)")

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
        # Rolling history for LLM prompt — lets the model detect repetition
        context["recent_actions"] = list(self._recent_actions)
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

        # Ablation: no_efe — zero all EFE scores so LLM picks from uniform distribution.
        # Tests whether EFE is doing real work vs. the LLM picking correctly anyway.
        if self._ablation_mode == AblationMode.NO_EFE:
            fe_scores = {pid: 0.0 for pid in fe_scores}

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

        # Fix 1: log every positive reward from Animal AI.
        # Fires even when the contact happens mid-turn (physics frames between decisions).
        _raw_reward = float(next_state.raw_metadata.get("raw_reward", 0.0))
        if _raw_reward > 0.0:
            _prev_rc = prev_state.perception.raycast_hits or []
            _food_rc = [r for r in _prev_rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")]
            self._logger.event("food_collected", {
                "raw_reward": _raw_reward,
                "action": selected_id,
                "food_ray_dist_before": _food_rc[0].get("distance") if _food_rc else None,
                "saturation_before": round(float(prev_state.homeostasis.saturation or 0.0), 4),
                "saturation_after": round(float(next_state.homeostasis.saturation or 0.0), 4),
                "health_before": round(float(prev_state.homeostasis.health or 0.0), 4),
                "health_after": round(float(next_state.homeostasis.health or 0.0), 4),
            }, step=step)

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
        self._logger.memory_op("pe_recorded", {
            "pe_mean": round(pe_batch.mean_magnitude, 4),
            "area_id": next_state.perception.area_id or "unknown",
            "action": selected_id,
        }, step=step)

        # Outcome score: drive-relief based.
        #
        # Measures whether this action reduced allostatic urgency, not whether
        # urgency is currently low. An agent at low urgency scoring high for
        # every action (including wall hits) would give the LTM no useful signal.
        #
        # relief > 0 → action reduced urgency (good)
        # relief < 0 → action increased urgency (bad, e.g. depletion tick with no progress)
        #
        # Per-step urgency change is tiny (~0.001–0.002); scale=5 spreads the
        # signal so ±0.1 relief maps to ±0.5 around the neutral midpoint of 0.5.
        # A food collection (~0.18 urgency relief) saturates to 1.0.
        #
        # Motor efficiency is a necessary second component: without it, move_forward
        # into a wall and turn_right produce the same drive urgency delta in most
        # steps, so the LTM can't differentiate them. motor_eff=0 (wall-blocked)
        # pulls the score down regardless of urgency change.
        next_vitals = self._vital_monitor.read(next_state)
        prev_urgency = drive_batch.max_urgency
        next_urgency = self._allostatic.peek_max_urgency(next_vitals)
        relief = prev_urgency - next_urgency
        relief_score = max(0.0, min(1.0, 0.5 + relief * 5.0))

        # Homeostatic delta: direct physiological improvement this step.
        # Replaces motor_eff * 0.3, which was always 0.0 when the proprioceptive
        # obs wasn't available and added no signal. A positive health+saturation
        # delta is the ground-truth consequence of successful foraging.
        prev_health = float(prev_state.homeostasis.health or 0.0)
        prev_sat = float(prev_state.homeostasis.saturation or 0.0)
        next_health = float(next_state.homeostasis.health or 0.0)
        next_sat = float(next_state.homeostasis.saturation or 0.0)
        homeo_delta = max(0.0, (next_health - prev_health) + (next_sat - prev_sat))

        # Exteroceptive component: did food distance change as the world model expected?
        # Only computed when move_forward was executed with food visible.
        # Differentiates "moving toward food" from "blocked facing food" — two outcomes
        # that produce identical drive relief but opposite exteroceptive consequences.
        extero_match = 1.0
        if selected_id == "move_forward":
            prev_rc = prev_state.perception.raycast_hits
            next_rc = next_state.perception.raycast_hits
            if prev_rc and next_rc:
                p_food = next(
                    (r for r in prev_rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None
                )
                n_food = next(
                    (r for r in next_rc if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")), None
                )
                if p_food:
                    prev_dist = float(p_food.get("distance", 1.0))
                    next_dist = float(n_food.get("distance", prev_dist)) if n_food else prev_dist
                    actual_delta = next_dist - prev_dist   # negative = approached
                    heading_deg = float(prev_state.position.heading or 0.0)
                    expected_delta = self._world_model.get_expected_delta(
                        "move_forward", "food_distance", heading_deg
                    )
                    extero_match = max(0.0, min(1.0, 1.0 - abs(expected_delta - actual_delta)))

        outcome_score = float(max(0.0, min(1.0, relief_score * 0.5 + homeo_delta * 0.3 + extero_match * 0.2)))

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
        self._logger.memory_op("trace_stored", {
            "policy_id": selected_id,
            "outcome_score": round(outcome_score, 4),
            "notes": situation_note,
        }, step=step)
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
                "ablation_mode": self._ablation_mode.value,
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
