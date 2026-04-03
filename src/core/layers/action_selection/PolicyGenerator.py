"""
PolicyGenerator — Layer 3, Action Selection & Agency

Implements selective LLM calling with 5 trigger conditions:
  1. Drive conflict — two or more channels above urgency threshold
  2. Sustained high PE streak — model is wrong and stuck
  3. No skill satisfies current drive — skill gap
  4. Periodic re-evaluation — every N steps
  5. Goal change — new goal message in workspace

LLM calls are async for non-urgent triggers (periodic, goal change).
Urgent triggers (conflict, PE streak, skill gap) block synchronously.
Urgency fallback handles all routine steps.

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.signals import (
    ActionProposal,
    ArousalValence,
    DriveSignalBatch,
    Goal,
    PredictionErrorBatch,
)

log = logging.getLogger(__name__)


class PolicyGenerator:
    """
    Arbitrates between competing action proposals using a hierarchy:
      1. Urgency fallback (fast, no LLM, handles ~80% of steps)
      2. Async LLM (non-urgent but novel situations, fires in background)
      3. Sync LLM (urgent conflicts, blocks until resolved)

    This maps onto Seth's System 1 / System 2 distinction:
      - Urgency fallback = beast machine reactive reflex
      - LLM arbitration = metacognitive global workspace deliberation
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
        self._reeval_interval: int = int(cfg.get("llm_reeval_interval", 10))
        self._weights: Dict[str, float] = cfg.get("weights", {
            "goal_coherence": 0.6,
            "prediction_error": 0.4,
            "allostatic_survival_fit": 0.2,
            "allostatic_urgency_alignment": 0.2,
        })
        self._fallback_scores: Dict[str, float] = cfg.get("fallback_scores", {
            "goal_coherence": 0.5,
            "prediction_error": 0.5,
            "allostatic_survival_fit": 0.5,
            "allostatic_urgency_alignment": 0.5,
        })

        # State tracking
        self._pe_streak: int = 0
        self._last_goals_fingerprint: str = ""
        self._last_llm_step: int = -999

        # Async LLM result slot
        self._pending_llm_result: Optional[str] = None
        self._llm_thread: Optional[threading.Thread] = None

        # Long-term memory callback (injected by AgentLoop)
        self._get_policy_history: Optional[Callable[[str], float]] = None

        # LLM logging callback (injected by AgentLoop)
        # Signature: (prompt: str, response: Any, trigger_reason: str, step: int) -> None
        self._llm_log_cb: Optional[Callable] = None

        # Arousal-driven diversity threshold:
        # When arousal exceeds this value the fallback excludes the last-taken
        # action from candidates, reducing precision on the failing policy
        # rather than committing to a hard-coded escape sequence.
        self._arousal_diversity_threshold: float = float(
            cfg.get("arousal_diversity_threshold", 0.6)
        )

    def set_policy_history_callback(self, fn: Callable[[str], float]) -> None:
        """Injected by AgentLoop — provides LTM success rates per policy."""
        self._get_policy_history = fn

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

        # Update PE streak tracker
        self._update_pe_streak(context)

        # Determine whether to invoke LLM
        trigger, reason = self._should_call_llm(goals, context, step)

        selected_id: Optional[str] = None
        rationale = "reactive_gate"

        if trigger == "urgent":
            # Block synchronously — conflict or critical PE streak
            selected_id = self._call_llm_sync(policies, goals, context, step, reason)
            rationale = f"llm_sync:{reason}"
            self._pe_streak = 0
            self._last_goals_fingerprint = self._fingerprint(goals)
            self._last_llm_step = step

        elif trigger == "async":
            # Fire LLM in background; use last result or fallback this step
            self._fire_async_llm(policies, goals, context, step, reason)
            selected_id = self._pending_llm_result
            # Reject a stale "idle" when the agent is blocked or arousal is high
            arousal = context.get("arousal", 0.0)
            motor_blocked = context.get("motor_stuck", False) or arousal > self._arousal_diversity_threshold
            if motor_blocked and selected_id == "idle":
                selected_id = None
            if selected_id is None or selected_id not in {p["policy_id"] for p in policies}:
                selected_id = self._urgency_fallback(policies, context)
                rationale = f"async_pending:{reason}"
            else:
                rationale = f"async_result:{reason}"

        else:
            # Pure reactive
            selected_id = self._urgency_fallback(policies, context)

        # Ensure we have a valid selection
        policy_ids = {p["policy_id"] for p in policies}
        if selected_id not in policy_ids:
            selected_id = self._urgency_fallback(policies, context)
            rationale = "fallback_invalid"

        workspace.publish(AgentMessage(
            sender="PolicyGenerator",
            kind="policy_proposal",
            payload={
                "selected": selected_id,
                "rationale": rationale,
                "trigger": trigger or "reactive",
                "step": step,
            },
            step=step,
        ))

        return selected_id

    # ------------------------------------------------------------------
    # LLM trigger gate
    # ------------------------------------------------------------------

    def _should_call_llm(
        self,
        goals: List[Goal],
        context: Dict[str, Any],
        step: int,
    ):
        """
        Evaluate 5 trigger conditions. Returns (trigger_type, reason).
        trigger_type: "urgent" | "async" | None
        """
        if self._llm is None:
            return None, "no_llm"

        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")
        pe_batch: Optional[PredictionErrorBatch] = context.get("pe_batch")

        # 1. Drive conflict — two+ channels above threshold (URGENT)
        if drive_batch:
            high = [s for s in drive_batch.signals if s.urgency > self._conflict_threshold]
            if len(high) >= 2:
                return "urgent", "drive_conflict"

        # 2. Sustained high PE streak (URGENT)
        if self._pe_streak >= self._pe_streak_threshold:
            return "urgent", "pe_streak"

        # 3. Skill gap — high urgency but no matching policy tags (URGENT)
        if drive_batch and drive_batch.max_urgency > self._skill_gap_urgency:
            dominant = drive_batch.dominant_channel
            if dominant:
                policies = context.get("policies", [])
                matching = [
                    p for p in policies
                    if dominant in p.get("drive_tags", [])
                ]
                if not matching:
                    return "urgent", "skill_gap"

        # 4. Periodic re-evaluation (ASYNC)
        if step - self._last_llm_step >= self._reeval_interval:
            return "async", "periodic"

        return None, "reactive"

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llm_sync(
        self,
        policies: List[Dict],
        goals: List[Goal],
        context: Dict,
        step: int,
        trigger_reason: str = "urgent",
    ) -> Optional[str]:
        """Blocking LLM call for urgent decisions."""
        try:
            from core.llm.types import LLMMessage, LLMRequest
            prompt = self._build_prompt(policies, goals, context, step)
            request = LLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                max_tokens=200,
                temperature=0.0,
            )
            t0 = time.monotonic()
            response = self._llm.complete(request)
            latency_ms = (time.monotonic() - t0) * 1000.0
            selected = self._parse_llm_response(response.content, policies)
            self._pending_llm_result = selected
            if self._llm_log_cb is not None:
                try:
                    self._llm_log_cb(prompt, response, trigger_reason, step, selected)
                except Exception as log_exc:
                    log.debug("LLM log callback failed: %s", log_exc)
            return selected
        except Exception as exc:
            log.warning("LLM sync call failed: %s", exc)
            return self._urgency_fallback(policies, context)

    def _fire_async_llm(
        self,
        policies: List[Dict],
        goals: List[Goal],
        context: Dict,
        step: int,
        trigger_reason: str = "async",
    ) -> None:
        """Non-blocking LLM call — result available next step."""
        if self._llm_thread is not None and self._llm_thread.is_alive():
            return  # Already running

        # Capture prompt on the main thread so context is not mutated by the time
        # the thread reads it (context dict is recreated each step).
        prompt = self._build_prompt(policies, goals, context, step)

        def _run() -> None:
            try:
                from core.llm.types import LLMMessage, LLMRequest
                request = LLMRequest(
                    messages=[LLMMessage(role="user", content=prompt)],
                    max_tokens=200,
                    temperature=0.1,
                )
                response = self._llm.complete(request)
                selected = self._parse_llm_response(response.content, policies)
                self._pending_llm_result = selected
                self._last_llm_step = step
                self._last_goals_fingerprint = self._fingerprint(goals)
                if self._llm_log_cb is not None:
                    try:
                        self._llm_log_cb(prompt, response, trigger_reason, step, selected)
                    except Exception as log_exc:
                        log.debug("LLM log callback failed: %s", log_exc)
            except Exception as exc:
                log.warning("LLM async call failed: %s", exc)
                self._pending_llm_result = None

        self._llm_thread = threading.Thread(target=_run, daemon=True)
        self._llm_thread.start()

    def _build_prompt(
        self,
        policies: List[Dict],
        goals: List[Goal],
        context: Dict,
        step: int,
    ) -> str:
        """
        Drive-arbitration prompt: LLM reasons as a planner-as-inference over
        expected free energy across the drive space (Seth/Friston hierarchy).

        No task-goal framing — the agent has no declared objective.
        It acts solely to reduce allostatic error and prediction error.

          Step 1: Interoceptive state — which drives are urgent and trending?
          Step 2: Exteroceptive / motor state — what is the agent doing?
          Step 3: EFE scores per action — what does the generative model predict?
          Step 4: Allostatic resolution — commit to the action that most relieves
                  the highest-urgency drive while minimising surprise.
        """
        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")
        pe_batch: Optional[PredictionErrorBatch] = context.get("pe_batch")
        av: Optional[ArousalValence] = context.get("arousal_valence")
        fe_scores: Dict[str, float] = context.get("free_energy_scores", {})

        # --- Step 1: interoceptive drives ---
        arousal = av.arousal if av else 0.0
        valence = av.valence if av else 0.0
        if drive_batch:
            signals = sorted(drive_batch.signals, key=lambda s: s.urgency, reverse=True)
            drive_lines = "\n".join(
                f"  {s.channel_id:<14} value={s.current_value:.2f}  urgency={s.urgency:.2f}"
                f"  ticks_to_crit={s.ticks_to_critical if s.ticks_to_critical is not None else '∞'}"
                for s in signals
            )
        else:
            drive_lines = "  (no drive data)"

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
            # Degenerate sequence note — surfaces stuck loops to the model
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

        # --- Step 2b: raycast directional perception ---
        raycast_hits = context.get("raycast_hits")
        if raycast_hits:
            raycast_lines = []
            for r in raycast_hits:
                label = r.get("angle_label", f"ray_{r.get('angle_idx', '?')}")
                tag = r.get("hit_tag")
                dist = r.get("distance", 1.0)
                if tag == "GoodGoal":
                    raycast_lines.append(f"  {label:<12}  food at {dist:.2f} (→ approach)")
                elif tag == "BadGoal":
                    raycast_lines.append(f"  {label:<12}  hazard at {dist:.2f} (→ avoid)")
                elif tag == "wall":
                    raycast_lines.append(f"  {label:<12}  wall at {dist:.2f}")
                else:
                    raycast_lines.append(f"  {label:<12}  clear")
            raycast_text = "\n".join(raycast_lines)
        else:
            raycast_text = "  (no raycast sensor — use visual heuristics only)"

        # --- Step 3: EFE per action — show drive_tags so LLM sees the connection ---
        fe_lines = []
        for p in policies:
            pid = p["policy_id"]
            fe = fe_scores.get(pid, 0.0)
            tags = ", ".join(p.get("drive_tags", [])) or "—"
            fe_lines.append(f"  {pid:<16} EFE={fe:.3f}  drives=[{tags}]")
        fe_text = "\n".join(fe_lines) if fe_lines else "  (no scores)"

        # --- Affect interpretation for Step 4 ---
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

        valid_ids = ", ".join(p["policy_id"] for p in policies)
        return f"""You are the deliberative system for a survival agent (step {step}).
You have NO declared task. You act only to reduce allostatic drive deficits and minimise surprise.
Reason through each step, then output EXACTLY:
REASON: <one sentence>
ACTION: <policy_id>

Valid policy_ids: {valid_ids}

══ STEP 1 — INTEROCEPTIVE STATE ══
{drive_lines}
  Arousal: {arousal:.2f}  |  Valence: {valence:.2f}

══ STEP 2 — EXTEROCEPTIVE / MOTOR STATE ══
  Heading:        {heading_deg:.0f}°
  Motor:          efficiency={motor_eff:.2f}
  PE streak:      {pe_streak} steps  |  Mean PE: {pe_mean:.4f}
  Recent actions: {recent_text}

══ STEP 2b — DIRECTIONAL PERCEPTION (raycasts) ══
{raycast_text}

══ STEP 3 — EXPECTED FREE ENERGY PER ACTION ══
{fe_text}
  (Higher EFE = action better reduces drive deficit + prediction error)

══ STEP 4 — ALLOSTATIC RESOLUTION ══
  Which action most reduces the highest-urgency drive deficit and minimises surprise?{affect_note}
  If food is visible in raycasts, prioritise turning toward it then moving forward.

REASON: """

    def _parse_llm_response(
        self,
        response: str,
        policies: List[Dict],
    ) -> Optional[str]:
        """Extract policy_id from LLM response text.

        Primary: look for the explicit ACTION: line from the structured prompt.
        Fallback: scan full response text for any valid policy_id substring.
        """
        policy_ids = {p["policy_id"] for p in policies}

        # The prompt primes "REASON: ...\nACTION: ..." — scan all lines
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("ACTION:"):
                candidate = stripped.split(":", 1)[1].strip().lower()
                first_word = candidate.split()[0].rstrip(".,;") if candidate else ""
                if first_word in policy_ids:
                    return first_word

        # Fallback: find first policy_id substring in response
        lower = response.lower()
        for pid in policy_ids:
            if pid in lower:
                return pid

        return None

    # ------------------------------------------------------------------
    # Urgency fallback
    # ------------------------------------------------------------------

    def _urgency_fallback(
        self,
        policies: List[Dict],
        context: Dict,
    ) -> str:
        """
        Fast reactive selection — no LLM, purely drive-tag matching.

        Arousal-sensitive diversity:
          When arousal exceeds the threshold (motor PE high = agent is blocked,
          or drives are critically low), the last-taken action is excluded from
          the candidate set. This reduces the precision weight on the failing
          policy, allowing other actions to compete — Seth-consistent affect
          modulation without a hard-coded escape sequence.
        """
        arousal: float = context.get("arousal", 0.0)
        active_policies = list(policies)

        if arousal > self._arousal_diversity_threshold:
            last_action = context.get("last_action")
            if last_action is not None:
                eligible = [p for p in active_policies if p["policy_id"] != last_action]
                if eligible:
                    active_policies = eligible

        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")
        urgency_by_tag: Dict[str, float] = {}

        if drive_batch:
            for signal in drive_batch.signals:
                for tag in signal.suggested_action_tags:
                    urgency_by_tag[tag] = max(urgency_by_tag.get(tag, 0.0), signal.urgency)

        best_id = None
        best_score = -1.0

        for policy in active_policies:
            score = 0.0
            for tag in policy.get("drive_tags", []) + policy.get("tags", []):
                score = max(score, urgency_by_tag.get(tag, 0.0))

            if self._get_policy_history is not None:
                ltm_rate = self._get_policy_history(policy["policy_id"])
                score += ltm_rate * 0.1

            if score > best_score:
                best_score = score
                best_id = policy["policy_id"]

        # Default to idle — a satisfied, unsurprised agent does nothing
        return best_id or "idle"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_pe_streak(self, context: Dict) -> None:
        pe_batch: Optional[PredictionErrorBatch] = context.get("pe_batch")
        if pe_batch and pe_batch.mean_magnitude > self._pe_high_threshold:
            self._pe_streak += 1
        else:
            self._pe_streak = max(0, self._pe_streak - 1)

    def _fingerprint(self, goals: List[Goal]) -> str:
        return "|".join(sorted(g.goal_id for g in goals))
