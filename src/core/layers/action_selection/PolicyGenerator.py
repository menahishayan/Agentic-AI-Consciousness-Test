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

    def set_policy_history_callback(self, fn: Callable[[str], float]) -> None:
        """Injected by AgentLoop — provides LTM success rates per policy."""
        self._get_policy_history = fn

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
            selected_id = self._call_llm_sync(policies, goals, context, step)
            rationale = f"llm_sync:{reason}"
            self._pe_streak = 0
            self._last_goals_fingerprint = self._fingerprint(goals)
            self._last_llm_step = step

        elif trigger == "async":
            # Fire LLM in background; use last result or fallback this step
            self._fire_async_llm(policies, goals, context, step)
            selected_id = self._pending_llm_result
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

        # 4. Goal changed (ASYNC)
        current_fp = self._fingerprint(goals)
        if current_fp != self._last_goals_fingerprint:
            return "async", "goal_changed"

        # 5. Periodic re-evaluation (ASYNC)
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
            response = self._llm.complete(request)
            selected = self._parse_llm_response(response.content, policies)
            self._pending_llm_result = selected
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
    ) -> None:
        """Non-blocking LLM call — result available next step."""
        if self._llm_thread is not None and self._llm_thread.is_alive():
            return  # Already running

        def _run() -> None:
            try:
                from core.llm.types import LLMMessage, LLMRequest
                prompt = self._build_prompt(policies, goals, context, step)
                request = LLMRequest(
                    messages=[LLMMessage(role="user", content=prompt)],
                    max_tokens=200,
                    temperature=0.1,
                )
                response = self._llm.complete(request)
                self._pending_llm_result = self._parse_llm_response(response.content, policies)
                self._last_llm_step = step
                self._last_goals_fingerprint = self._fingerprint(goals)
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
        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")
        pe_batch: Optional[PredictionErrorBatch] = context.get("pe_batch")
        av: Optional[ArousalValence] = context.get("arousal_valence")

        goal_text = "; ".join(g.description for g in goals) if goals else "no active goal"
        drive_text = ""
        if drive_batch:
            signals = sorted(drive_batch.signals, key=lambda s: s.urgency, reverse=True)[:3]
            drive_text = "\n".join(
                f"  - {s.channel_id}: value={s.current_value:.2f}, urgency={s.urgency:.2f}"
                for s in signals
            )
        pe_text = f"mean={pe_batch.mean_magnitude:.3f}" if pe_batch else "unknown"
        arousal_text = f"arousal={av.arousal:.2f}, valence={av.valence:.2f}" if av else "unknown"

        policy_list = "\n".join(
            f"  - {p['policy_id']}: {p.get('description', '')} [drive_tags: {p.get('drive_tags', [])}]"
            for p in policies
        )

        return f"""You are the deliberative reasoning system for a survival agent (step {step}).

TASK GOAL: {goal_text}

CURRENT BODY STATE:
{drive_text if drive_text else "  (no drive data)"}

COGNITIVE STATE: {arousal_text}
PREDICTION ERROR: {pe_text}

AVAILABLE ACTIONS:
{policy_list}

Select the single best action to take RIGHT NOW to survive and achieve the goal.
Respond with ONLY the policy_id, nothing else.
Example: move_forward"""

    def _parse_llm_response(
        self,
        response: str,
        policies: List[Dict],
    ) -> Optional[str]:
        """Extract policy_id from LLM response text."""
        policy_ids = {p["policy_id"] for p in policies}
        # Try exact match first
        stripped = response.strip().lower().split()[0] if response.strip() else ""
        if stripped in policy_ids:
            return stripped
        # Scan for any policy_id in the response
        for pid in policy_ids:
            if pid in response.lower():
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
        Scores each policy by urgency of matching drive channels.
        """
        drive_batch: Optional[DriveSignalBatch] = context.get("drive_batch")
        urgency_by_tag: Dict[str, float] = {}

        if drive_batch:
            for signal in drive_batch.signals:
                for tag in signal.suggested_action_tags:
                    urgency_by_tag[tag] = max(urgency_by_tag.get(tag, 0.0), signal.urgency)

        best_id = None
        best_score = -1.0

        for policy in policies:
            score = 0.0
            for tag in policy.get("drive_tags", []) + policy.get("tags", []):
                score = max(score, urgency_by_tag.get(tag, 0.0))

            # Add LTM success rate bonus
            if self._get_policy_history is not None:
                ltm_rate = self._get_policy_history(policy["policy_id"])
                score += ltm_rate * 0.1

            if score > best_score:
                best_score = score
                best_id = policy["policy_id"]

        return best_id or policies[0]["policy_id"]

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
