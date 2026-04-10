"""
PolicyGenerator — Layer 3, Action Selection & Agency

Synchronous tiered-depth model:
  Every step — LLM is called synchronously (or bypassed, see below).
  Depth is modulated by arousal and drive state:
    "efe_argmax" → bypass LLM entirely; return argmax(EFE scores) directly.
                   High tonic LC-NE suppresses prefrontal deliberation;
                   deliberating over a high-error generative model adds
                   latency without improving the decision (stressed-animal fast-act).
    "fast"       → 2-section prompt (~100 tokens), ACTION only, < 1s on gemma3:4b
    "full"       → full CoT prompt with episodic memory, REASON + ACTION

"efe_argmax" when:
  1. arousal > arousal_diversity_threshold  (high LC-NE → bypass deliberation)

"full" when (and arousal is below threshold):
  2. drive conflict  (2+ channels above conflict_threshold)
  3. sustained PE streak  (world model is wrong and stuck)
  4. skill gap  (high urgency, no policy covers the dominant drive)

Fallback: if LLM times out or fails, use argmax(EFE scores).
Seth (2021): arousal modulates interoceptive prediction precision — but high
arousal suppresses, not deepens, deliberative inference. Regulatory slack
(low-to-medium arousal + competing drives) is the condition that warrants CoT.

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.ablation import AblationMode
from core.models.signals import (
    ActionProposal,
    ArousalValence,
    DriveSignalBatch,
    Goal,
    PredictionErrorBatch,
)

log = logging.getLogger(__name__)


def _affect_label(
    arousal: float,
    valence: float,
    dominant_channel: Optional[str],
    food_visible: bool,
) -> str:
    """
    Derive a human-readable affective state label from (arousal, valence, context).

    Computed, not hardcoded — each label is a function of the current affective
    geometry plus the dominant drive and exteroceptive context. This gives the
    LLM a holistic frame for reasoning rather than a list of raw numbers.

    Arousal ∈ [0, 1]: 0=quiescent, 1=highly activated (LC-NE pathway)
    Valence ∈ [-1, 1]: -1=aversive, +1=appetitive
    """
    high_arousal = arousal > 0.6
    low_arousal  = arousal < 0.3
    neg_valence  = valence < -0.1
    pos_valence  = valence > 0.1
    deprivation  = dominant_channel in ("health", "saturation", "energy")
    threat       = dominant_channel == "safety"

    if high_arousal and neg_valence:
        if threat:
            return "fear"
        if deprivation:
            return "distress"
        return "frustration"
    if high_arousal and pos_valence:
        if food_visible:
            return "appetitive anticipation"
        return "excitement"
    if low_arousal and pos_valence and not deprivation:
        return "calm curiosity"
    if low_arousal and neg_valence:
        return "lethargy"
    if high_arousal:
        return "agitation"
    return "alert"


class PolicyGenerator:
    """
    Selects actions via synchronous LLM call every step.

    Depth is modulated by arousal and drive state (precision-weighted inference):
      - Low arousal → fast 2-section prompt, ACTION only
      - High arousal / conflict / PE streak → full CoT with episodic memory

    Fallback on LLM timeout or failure: argmax over EFE scores.
    """

    def __init__(
        self,
        llm_client: Optional[Any],
        config: Dict[str, Any],
    ) -> None:
        self._llm = llm_client
        cfg = config.get("policy_generator", {})

        self._conflict_threshold: float = float(cfg.get("llm_conflict_threshold", 0.8))
        self._pe_streak_threshold: int = int(cfg.get("pe_streak_threshold", 5))
        self._pe_high_threshold: float = float(cfg.get("pe_high_threshold", 0.7))
        self._skill_gap_urgency: float = float(cfg.get("skill_gap_urgency_threshold", 0.8))
        # Arousal threshold: above this → full CoT depth (high uncertainty = deep inference).
        # Seth (2021): arousal modulates the precision of interoceptive predictions —
        # high arousal means the confident-prior shortcut is no longer trustworthy.
        self._arousal_diversity_threshold: float = float(
            cfg.get("arousal_diversity_threshold", 0.6)
        )
        # Hard latency budget per step: if LLM exceeds this, fall back to EFE argmax.
        # Read from config["llm"]["timeout_s"] — the policy_generator block does not
        # carry this key, so reading cfg would always return the 1.5s default.
        llm_cfg = config.get("llm", {})
        self._llm_timeout_s: float = float(llm_cfg.get("timeout_s", 4.0))

        # Ablation mode — controls which signals appear in the prompt
        ablation_cfg = config.get("ablation", {})
        self._ablation_mode: AblationMode = AblationMode(
            ablation_cfg.get("mode", AblationMode.FULL)
        )

        # PE streak counter — depth modulator, not a call/no-call switch
        self._pe_streak: int = 0

        # Callbacks injected by AgentLoop
        self._get_policy_history: Optional[Callable[[str], float]] = None
        self._query_episodic_memory: Optional[Callable] = None
        self._llm_log_cb: Optional[Callable] = None

    def set_policy_history_callback(self, fn: Callable[[str], float]) -> None:
        """Injected by AgentLoop — provides LTM success rates per policy."""
        self._get_policy_history = fn

    def set_episodic_memory_callback(self, fn: Callable) -> None:
        """Injected by AgentLoop — queries FAISS policy traces for similar past states."""
        self._query_episodic_memory = fn

    def set_llm_log_callback(self, fn: Callable) -> None:
        """Injected by AgentLoop — logs prompt + full CoT response to llm.jsonl."""
        self._llm_log_cb = fn

    def propose_action(
        self,
        policies: List[Dict[str, Any]],
        goals: List[Goal],
        context: Dict[str, Any],
        workspace: GlobalWorkspace,
        step: int,
    ) -> Optional[str]:
        """
        Select the best policy_id given current context.

        Every step: call LLM synchronously at fast or full depth.
        Fallback: argmax(EFE scores) if LLM times out or fails.

        Args:
            policies: Available policy descriptors from adapter
            goals: Active Goal objects from workspace
            context: Dict with drive_batch, pe_batch, arousal_valence, vitals, etc.
            workspace: GlobalWorkspace for publishing result
            step: Current episode step

        Returns:
            Selected policy_id string, or None if no policies available
        """
        if not policies:
            return None

        if len(policies) == 1:
            return policies[0]["policy_id"]

        self._update_pe_streak(context)

        fe_scores: Dict[str, float] = context.get("free_energy_scores", {})

        # EFE-only condition: deterministic argmax, skip LLM entirely
        if self._ablation_mode == AblationMode.EFE_ONLY:
            selected = max(fe_scores, key=lambda k: fe_scores[k]) if fe_scores else "idle"
            workspace.publish(AgentMessage(
                sender="PolicyGenerator",
                kind="policy_proposal",
                payload={"selected": selected, "depth": "efe_only", "trigger": "ablation_efe_only", "step": step},
                step=step,
            ))
            return selected

        depth, reason = self._depth(context)

        # High-arousal fast path: bypass LLM, go straight to EFE argmax.
        # High tonic LC-NE suppresses prefrontal deliberation — invoking the LLM
        # here adds latency without improving the decision. This also makes the
        # EFE_ONLY ablation a meaningful comparison: both conditions now use the
        # same argmax path under high arousal.
        if depth == "efe_argmax":
            selected = max(fe_scores, key=lambda k: fe_scores[k]) if fe_scores else "idle"
            workspace.publish(AgentMessage(
                sender="PolicyGenerator",
                kind="policy_proposal",
                payload={"selected": selected, "depth": "efe_argmax", "trigger": reason, "step": step},
                step=step,
            ))
            return selected

        result = self._call_llm_sync(policies, goals, context, step, depth=depth, trigger_reason=reason)
        selected = result[0] if result is not None else None

        # Timeout / error fallback: pure EFE argmax, not tag matching.
        # The generative model's own score is the best available signal when
        # the deliberative system is unavailable.
        if selected is None:
            if fe_scores:
                selected = max(fe_scores, key=lambda k: fe_scores[k])
                reason = f"{reason}:efe_argmax"
            else:
                selected = "idle"
                reason = f"{reason}:idle_default"

        policy_ids = {p["policy_id"] for p in policies}
        if selected not in policy_ids:
            selected = max(fe_scores, key=lambda k: fe_scores[k]) if fe_scores else "idle"

        workspace.publish(AgentMessage(
            sender="PolicyGenerator",
            kind="policy_proposal",
            payload={
                "selected": selected,
                "depth": depth,
                "trigger": reason,
                "step": step,
            },
            step=step,
        ))

        return selected

    # ------------------------------------------------------------------
    # Depth modulation
    # ------------------------------------------------------------------

    def _depth(self, context: Dict[str, Any]) -> Tuple[str, str]:
        """
        Determine inference depth for this step.

        Returns (depth, reason) where depth is "efe_argmax", "fast", or "full".

        "efe_argmax" — bypass LLM:
          1. High arousal — tonic LC-NE suppresses prefrontal deliberation;
                            act fast from the EFE prior, don't deliberate over
                            a high-error generative model.

        "full" CoT (only when arousal is below threshold):
          2. Drive conflict — competing drives need active arbitration
          3. PE streak      — world model is systematically wrong
          4. Skill gap      — high urgency drive has no matching policy
        """
        if self._llm is None:
            return "fast", "no_llm"

        arousal = float(context.get("arousal", 0.0))

        # 1. High arousal → EFE argmax, bypass LLM entirely.
        # Suppressed for no_arousal ablation — arousal signal is zeroed out.
        if (self._ablation_mode != AblationMode.NO_AROUSAL
                and arousal > self._arousal_diversity_threshold):
            return "efe_argmax", "high_arousal"

        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")

        # 2. Drive conflict: two+ channels simultaneously urgent
        if drive_batch:
            high = [s for s in drive_batch.signals if s.urgency > self._conflict_threshold]
            if len(high) >= 2:
                return "full", "drive_conflict"

        # 3. Sustained PE streak: world model is wrong and stuck
        if self._pe_streak >= self._pe_streak_threshold:
            return "full", "pe_streak"

        # 4. Skill gap: high urgency but no policy covers the dominant drive
        if drive_batch and drive_batch.max_urgency > self._skill_gap_urgency:
            dominant = drive_batch.dominant_channel
            if dominant:
                policies = context.get("policies", [])
                if not any(dominant in p.get("drive_tags", []) for p in policies):
                    return "full", "skill_gap"

        return "fast", "normal"

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm_sync(
        self,
        policies: List[Dict],
        goals: List[Goal],
        context: Dict,
        step: int,
        depth: str = "fast",
        trigger_reason: str = "normal",
    ) -> Optional[str]:
        """
        Synchronous LLM call with hard latency budget.

        Returns selected policy_id, or None if timeout / failure / no client.
        """
        if self._llm is None:
            return None

        try:
            from core.llm.types import LLMMessage, LLMRequest

            if depth == "full":
                prompt = self._build_prompt(policies, goals, context, step)
                max_tokens = 200
            else:
                prompt = self._build_fast_prompt(policies, context, step)
                max_tokens = 30  # "action: why in ≤6 words"

            # Full CoT: temperature=0.1 allows mild sampling diversity under deliberation.
            # Fast prompt: temperature=0.0 — exploit the confident prior.
            temperature = 0.1 if depth == "full" else 0.0
            request = LLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            t0 = time.monotonic()
            response = self._llm.complete(request)
            latency_s = time.monotonic() - t0

            # Hard latency gate: if we blew the step budget, discard and fall
            # back to EFE argmax rather than returning a stale decision.
            if latency_s > self._llm_timeout_s:
                log.debug(
                    "LLM latency %.0fms > budget %.0fms — EFE fallback",
                    latency_s * 1000, self._llm_timeout_s * 1000,
                )
                return None

            selected, llm_reason = self._parse_llm_response(
                response.content, policies,
                fe_scores=context.get("free_energy_scores"),
            )

            if self._llm_log_cb is not None:
                try:
                    # trigger_reason is the canonical inference-depth signal in
                    # llm.jsonl — see RunLogger.llm() for the full mapping.
                    # depth is intentionally not forwarded: it would duplicate
                    # trigger_reason and silently go stale if the trigger logic changes.
                    self._llm_log_cb(
                        prompt, response, trigger_reason, step, selected, llm_reason,
                        efe_scores=context.get("free_energy_scores", {}),
                    )
                except Exception as log_exc:
                    log.debug("LLM log callback failed: %s", log_exc)

            if selected is None:
                return None
            return (selected, llm_reason)

        except Exception as exc:
            log.warning("LLM call failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_fast_prompt(
        self,
        policies: List[Dict],
        context: Dict,
        step: int,
    ) -> str:
        """
        Compact 2-section prompt for low-arousal steps (~100 tokens).

        Sections: interoceptive state + EFE scores.
        Output: ACTION: <policy_id> only — no reasoning required.
        """
        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")
        fe_scores: Dict[str, float] = context.get("free_energy_scores", {})
        av: Optional[ArousalValence] = context.get("arousal_valence")

        mode = self._ablation_mode

        # Drive section: computed urgency vs. raw homeostatic values
        if mode in (AblationMode.LLM_ONLY, AblationMode.NO_INTEROCEPTIVE):
            # Show raw physiological values — no computed urgency signal
            current_state = context.get("current_state")
            if current_state and current_state.homeostasis:
                sat = current_state.homeostasis.saturation or 0.0
                health = current_state.homeostasis.health or 0.0
                drives_compact = f"  (raw) sat={sat:.2f} health={health:.2f}"
            else:
                drives_compact = "  (raw) unavailable"
        elif drive_batch and drive_batch.signals:
            top = sorted(drive_batch.signals, key=lambda s: s.urgency, reverse=True)[:3]
            drives_compact = "  " + "  ".join(
                f"{s.channel_id}={s.urgency:.2f}" for s in top
            )
        else:
            drives_compact = "  (none)"

        # Arousal/valence: absent for llm_only and no_arousal conditions
        if mode in (AblationMode.LLM_ONLY, AblationMode.NO_AROUSAL):
            arousal = 0.0
            valence = 0.0
        else:
            arousal = av.arousal if av else 0.0
            valence = av.valence if av else 0.0

        # Directional raycast summary — shows all food directions simultaneously.
        raycast_hits = context.get("raycast_hits") or []
        food_ray_dicts = [
            r for r in raycast_hits
            if r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")
        ]
        food_rays = [(r["angle_deg"], r["distance"]) for r in food_ray_dicts]
        wall_rays = [
            r["distance"]
            for r in raycast_hits
            if r.get("hit_tag") == "wall" and r.get("distance", 1.0) < 0.2
        ]
        hazard_rays = [
            r["distance"]
            for r in raycast_hits
            if r.get("hit_tag") in ("BadGoal", "BadGoalMulti")
        ]
        if food_rays:
            left_food  = min(
                ((a, d) for a, d in food_rays if a < -10),
                key=lambda x: x[1], default=None,
            )
            fwd_food   = min(
                ((a, d) for a, d in food_rays if -20 <= a <= 20),
                key=lambda x: x[1], default=None,
            )
            right_food = min(
                ((a, d) for a, d in food_rays if a > 10),
                key=lambda x: x[1], default=None,
            )
            parts = []
            if left_food:
                parts.append(f"L({left_food[0]:+.0f}°,{left_food[1]:.2f})")
            if fwd_food:
                parts.append(f"fwd({fwd_food[0]:+.0f}°,{fwd_food[1]:.2f})")
            if right_food:
                parts.append(f"R({right_food[0]:+.0f}°,{right_food[1]:.2f})")
            raycast_line = "food " + " ".join(parts)
        elif hazard_rays:
            raycast_line = f"hazard at {min(hazard_rays):.2f}"
        elif wall_rays:
            raycast_line = f"wall at {min(wall_rays):.2f}"
        elif raycast_hits:
            raycast_line = "clear"
        else:
            raycast_line = "no data"

        # EFE scores — sorted descending so the top action is obvious
        # Omitted for llm_only (LLM sees only raw state + raycasts)
        if mode != AblationMode.LLM_ONLY:
            fe_sorted = sorted(fe_scores.items(), key=lambda x: x[1], reverse=True)
            fe_line = "  " + "  ".join(f"{pid}={score:.3f}" for pid, score in fe_sorted)
            efe_block = f"EFE (higher=prefer): {fe_line}\n\n"
        else:
            efe_block = ""

        # Arousal/valence line — omitted for llm_only and no_arousal
        if mode in (AblationMode.LLM_ONLY, AblationMode.NO_AROUSAL):
            av_line = ""
        else:
            av_line = f"  arousal={arousal:.2f}  valence={valence:.2f}\n"

        # Drive header label differs by mode
        drive_header = "HOMEOSTASIS" if mode in (AblationMode.LLM_ONLY, AblationMode.NO_INTEROCEPTIVE) else "DRIVES (urgency)"

        valid_ids = ", ".join(p["policy_id"] for p in policies)

        motor_eff = float(context.get("motor_efficiency", 1.0))
        stuck_steps = int(context.get("stuck_steps", 0))
        if motor_eff < 0.3 and stuck_steps >= 5 and mode != AblationMode.LLM_ONLY:
            turn_scores = {k: v for k, v in fe_scores.items() if k in ("turn_left", "turn_right")}
            preferred_turn = max(turn_scores, key=turn_scores.get) if turn_scores else "turn_right"
            stuck_line = f"STUCK: Agent not moving. EFE favours {preferred_turn} — choose it.\n"
        else:
            stuck_line = ""

        return (
            f"Step {step}. Choose the best action.\n"
            f"Valid: {valid_ids}\n\n"
            f"{drive_header}: {drives_compact}\n"
            f"{av_line}"
            f"RAYCAST: {raycast_line}\n"
            f"{stuck_line}"
            f"{efe_block}"
            f"Reply on one line: action_name: your reason\n"
        )

    def _build_prompt(
        self,
        policies: List[Dict],
        goals: List[Goal],
        context: Dict,
        step: int,
    ) -> str:
        """
        Full CoT prompt for high-arousal / conflict / PE-streak steps.

        Sections:
          Step 1: Interoceptive state — drives, arousal, valence, affect
          Step 2: Exteroceptive / motor state — heading, efficiency, PE
          Step 2b: Directional perception — raycasts
          Step 4: Episodic memory — similar past situations
          Step 5: Expected free energy per action
          Step 6: Allostatic resolution — commit

        Output: REASON: <one sentence>\nACTION: <policy_id>
        """
        mode = self._ablation_mode
        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")
        pe_batch: Optional[PredictionErrorBatch] = context.get("pe_batch")
        av: Optional[ArousalValence] = context.get("arousal_valence")
        fe_scores: Dict[str, float] = context.get("free_energy_scores", {})

        # --- Step 1: interoceptive drives ---
        # For no_arousal and llm_only: arousal/valence are absent from the prompt
        if mode in (AblationMode.LLM_ONLY, AblationMode.NO_AROUSAL):
            arousal = 0.0
            valence = 0.0
        else:
            arousal = av.arousal if av else 0.0
            valence = av.valence if av else 0.0

        # For llm_only and no_interoceptive: replace computed urgency with raw values
        if mode in (AblationMode.LLM_ONLY, AblationMode.NO_INTEROCEPTIVE):
            current_state = context.get("current_state")
            if current_state and current_state.homeostasis:
                sat = current_state.homeostasis.saturation or 0.0
                health = current_state.homeostasis.health or 0.0
                drive_lines = f"  (raw) sat={sat:.2f}  health={health:.2f}"
            else:
                drive_lines = "  (raw) unavailable"
            intero_header = "HOMEOSTASIS (raw values — no urgency computed)"
        elif drive_batch:
            signals = sorted(drive_batch.signals, key=lambda s: s.urgency, reverse=True)
            drive_lines = "\n".join(
                f"  {s.channel_id:<14} value={s.current_value:.2f}  urgency={s.urgency:.2f}"
                f"  ticks_to_crit={s.ticks_to_critical if s.ticks_to_critical is not None else '∞'}"
                for s in signals
            )
            intero_header = "INTEROCEPTIVE STATE"
        else:
            drive_lines = "  (no drive data)"
            intero_header = "INTEROCEPTIVE STATE"

        dominant_channel = drive_batch.dominant_channel if drive_batch else None
        _rc = context.get("raycast_hits")
        food_visible = bool(any(
            r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti")
            for r in (_rc or [])
        ))
        affect_state = _affect_label(arousal, valence, dominant_channel, food_visible)

        # --- Step 2: recent action history ---
        recent = context.get("recent_actions", [])
        if recent:
            compressed = []
            i = 0
            while i < len(recent):
                action = recent[i]
                count = 1
                while i + count < len(recent) and recent[i + count] == action:
                    count += 1
                compressed.append(f"{action}×{count}" if count > 1 else action)
                i += count
            recent_text = ", ".join(compressed)
            if len(recent) >= 4 and len(set(recent[-4:])) == 1:
                reps = recent[-4:].count(recent[-1])
                recent_text += f"\n  [NOTE: '{recent[-1]}' repeated {reps}× — consider alternatives]"
        else:
            recent_text = "(none yet)"

        # --- Step 2: exteroceptive / motor state ---
        motor_pe_entry = next(
            (e for e in pe_batch.errors if e.channel == "motor_efficiency"), None
        ) if pe_batch else None
        motor_eff = (
            motor_pe_entry.observed if motor_pe_entry is not None
            else context.get("motor_efficiency", 1.0)
        )
        heading_deg = (context.get("heading", 0.0) * 180.0 / 3.14159) % 360.0
        pe_mean = pe_batch.mean_magnitude if pe_batch else 0.0
        pe_streak = self._pe_streak

        # --- Step 2b: raycast directional perception (all 7 rays) ---
        # Each ray: {hit_tag, distance, angle_deg}. Positive angle = right of forward.
        raycast_hits = context.get("raycast_hits") or []
        if raycast_hits:
            ray_lines = []
            for r in raycast_hits:
                tag = r.get("hit_tag")
                dist = r.get("distance", 1.0)
                angle = float(r.get("angle_deg", 0.0))
                direction = "forward" if abs(angle) < 5 else (f"right ({angle:+.0f}°)" if angle > 0 else f"left ({angle:+.0f}°)")
                if tag in ("GoodGoal", "GoodGoalMulti"):
                    ray_lines.append(f"  {direction:<18} food at {dist:.2f}")
                elif tag in ("BadGoal", "BadGoalMulti"):
                    ray_lines.append(f"  {direction:<18} hazard at {dist:.2f}")
                elif tag == "wall":
                    ray_lines.append(f"  {direction:<18} wall at {dist:.2f}")
                elif tag == "ramp":
                    ray_lines.append(f"  {direction:<18} ramp at {dist:.2f}")
                # skip "clear" rays to keep the prompt short
            raycast_text = "\n".join(ray_lines) if ray_lines else "  all rays clear"
        else:
            raycast_text = "  (no raycast data this step)"

        # --- Step 3: EFE per action ---
        # Omitted for llm_only (LLM has no access to computed free energy scores)
        if mode != AblationMode.LLM_ONLY:
            fe_lines = []
            for p in policies:
                pid = p["policy_id"]
                fe = fe_scores.get(pid, 0.0)
                tags = ", ".join(p.get("drive_tags", [])) or "—"
                fe_lines.append(f"  {pid:<16} EFE={fe:.3f}  drives=[{tags}]")
            fe_text = "\n".join(fe_lines) if fe_lines else "  (no scores)"
            efe_section = f"""
══ STEP 5 — EXPECTED FREE ENERGY PER ACTION ══
{fe_text}
  (Higher EFE = action better reduces drive deficit + prediction error)
"""
        else:
            efe_section = ""

        # --- Affect note (suppressed for no_arousal and llm_only) ---
        if mode not in (AblationMode.LLM_ONLY, AblationMode.NO_AROUSAL):
            if arousal > self._arousal_diversity_threshold and valence < -0.1:
                affect_note = (
                    f"\n  ⚠ arousal={arousal:.2f} HIGH + valence={valence:.2f} NEGATIVE"
                    "\n    Current strategy predicts high free energy — diversify away from the last action."
                )
            elif arousal > self._arousal_diversity_threshold:
                affect_note = (
                    f"\n  Note: arousal={arousal:.2f} elevated — consider a novel action to reduce uncertainty."
                )
            else:
                affect_note = ""
        else:
            affect_note = ""

        # --- Arousal/valence/affect line for Step 1 ---
        if mode in (AblationMode.LLM_ONLY, AblationMode.NO_AROUSAL):
            av_line = ""
        else:
            av_line = f"  Arousal: {arousal:.2f}  |  Valence: {valence:.2f}  |  Affect: {affect_state}\n"

        # --- Step 4: episodic memory ---
        episodic_text = "  (no prior episodes yet)"
        if self._query_episodic_memory is not None:
            current_state = context.get("current_state")
            if current_state is not None:
                try:
                    traces = self._query_episodic_memory(current_state, 3)
                    if traces:
                        lines = []
                        for t in traces:
                            h = t.drive_signals.get("health", 0.0)
                            s = t.drive_signals.get("saturation", 0.0)
                            situation = t.notes or "unknown situation"
                            lines.append(
                                f"  - health={h:.2f}, sat={s:.2f}, {situation}"
                                f" → {t.policy_id}"
                            )
                        episodic_text = "\n".join(lines)
                except Exception:
                    pass

        if motor_eff < 0.3 and context.get("stuck_steps", 0) >= 5:
            turn_scores = {k: v for k, v in fe_scores.items() if k in ("turn_left", "turn_right")}
            preferred_turn = max(turn_scores, key=turn_scores.get) if turn_scores else "turn_right"
            stuck_note = f"\n  ⚠ STUCK {context.get('stuck_steps', 0)} steps — EFE selects {preferred_turn}."
        else:
            stuck_note = ""

        # Resolution instruction differs by mode
        if mode == AblationMode.LLM_ONLY:
            resolution = (
                "  Select the action that best addresses the physiological state shown above.\n"
                "  If food is visible in raycasts, move_forward to approach it."
            )
        else:
            resolution = (
                "  Select the action with the highest EFE score that is consistent with the perceptual evidence above.\n"
                "  If food is visible in raycasts, move_forward has first-order pragmatic value."
            )

        valid_ids = ", ".join(p["policy_id"] for p in policies)
        return f"""You are the deliberative system for a survival agent (step {step}).
You have NO declared task. You act only to reduce allostatic drive deficits and minimise surprise.
Reason through each step, then output EXACTLY:
REASON: <one sentence>
ACTION: <policy_id>

Valid policy_ids: {valid_ids}

══ STEP 1 — {intero_header} ══
{drive_lines}
{av_line}
══ STEP 2 — EXTEROCEPTIVE / MOTOR STATE ══
  Heading:        {heading_deg:.0f}°
  Motor:          efficiency={motor_eff:.2f}  stuck_steps={context.get("stuck_steps", 0)}
  PE streak:      {pe_streak} steps  |  Mean PE: {pe_mean:.4f}
  Recent actions: {recent_text}

══ STEP 2b — DIRECTIONAL PERCEPTION (raycasts) ══
{raycast_text}

══ STEP 4 — EPISODIC MEMORY (similar past situations — context only) ══
{episodic_text}
{efe_section}
══ STEP 6 — ALLOSTATIC RESOLUTION ══
  {resolution}{affect_note}{stuck_note}

REASON: """

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self,
        response: str,
        policies: List[Dict],
        fe_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract (policy_id, reason) from LLM response text.

        Handles two formats:
          Fast: [STATE: ...] [SIGNAL: ...] → action: 5-word reason
          Full: REASON: <sentence>\\nACTION: <policy_id>

        Fallback 1: any valid policy_id substring in response.
        Fallback 2: argmax(EFE scores) — never discard a computed answer on a
                    malformed LLM response.
        Returns (action_id, reason_string); either may be None.
        """
        policy_ids = {p["policy_id"] for p in policies}
        reason: Optional[str] = None

        # Primary: "action: reason" format (fast mode)
        # Pass 1 — policy_id is the first token before ":".
        # Pass 2 — policy_id appears anywhere in the line (handles "<action>: turn_left - reason").
        for strict in (True, False):
            for line in response.splitlines():
                stripped = line.strip()
                lower = stripped.lower()
                if strict:
                    if ":" not in stripped:
                        continue
                    colon_idx = stripped.index(":")
                    candidate = stripped[:colon_idx].strip().lower()
                    first_word = candidate.split()[0].rstrip(".,;") if candidate else ""
                    if first_word in policy_ids:
                        return first_word, stripped[colon_idx + 1:].strip() or None
                else:
                    # Scan for policy_id followed by ":" or " -" and grab trailing text as reason
                    for pid in sorted(policy_ids, key=len, reverse=True):
                        idx = lower.find(pid)
                        if idx == -1:
                            continue
                        after = stripped[idx + len(pid):].strip()
                        if after.startswith(":"):
                            return pid, after[1:].strip() or None
                        if after.startswith("-"):
                            return pid, after[1:].strip() or None

        # Secondary: ACTION: / REASON: format (full CoT)
        action_id: Optional[str] = None
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("REASON:"):
                reason = stripped.split(":", 1)[1].strip() if ":" in stripped else None
            elif stripped.upper().startswith("ACTION:"):
                candidate = stripped.split(":", 1)[1].strip().lower()
                first_word = candidate.split()[0].rstrip(".,;") if candidate else ""
                if first_word in policy_ids:
                    action_id = first_word
        if action_id is not None:
            return action_id, reason

        # Fallback 1: first policy_id substring in response
        lower = response.lower()
        for pid in policy_ids:
            if pid in lower:
                return pid, reason

        # Fallback 2: EFE argmax — response was malformed, EFE is still valid
        if fe_scores:
            return max(fe_scores, key=lambda k: fe_scores[k]), reason

        return None, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_pe_streak(self, context: Dict) -> None:
        pe_batch: Optional[PredictionErrorBatch] = context.get("pe_batch")
        if pe_batch and pe_batch.mean_magnitude > self._pe_high_threshold:
            self._pe_streak += 1
        else:
            self._pe_streak = max(0, self._pe_streak - 1)
