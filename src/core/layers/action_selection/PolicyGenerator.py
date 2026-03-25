from __future__ import annotations

import copy
from datetime import datetime
import inspect
import json
import re
import threading
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.llm.client import LLMClient
from core.llm.types import LLMMessage, LLMRequest
from core.memory import WorkingMemoryEntry
from core.models.signals import ActionProposal
from core.observability.logger import RunLogger


def _utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


class PolicyGenerator:
    def __init__(
        self,
        adapter: Any,
        adapter_folder: str,
        memory_manager: Any,
        goal_checker: Any,
        prediction_error_calculator: Any,
        config: Optional[Mapping[str, Any]] = None,
        logger: Optional[RunLogger] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.adapter = adapter
        self.adapter_folder = adapter_folder
        self.memory_manager = memory_manager
        self.goal_checker = goal_checker
        self.prediction_error_calculator = prediction_error_calculator
        self.config = dict(config or {})
        self.logger = logger
        self.llm_client = llm_client

        self.temperature = self._as_float(self.config.get("temperature"), 0.3)
        self.horizon = max(1, int(self._as_float(self.config.get("horizon"), 5.0)))
        self.skill_plan_bias = max(0.0, self._as_float(self.config.get("skill_plan_bias"), 0.25))
        self.skill_plan_emergency_threshold = self._clamp01(
            self._as_float(
                self.config.get("skill_plan_emergency_urgency_threshold"),
                0.85,
            )
        )
        self.conflict_threshold = self._clamp01(
            self._as_float(self.config.get("llm_conflict_threshold"), 0.8)
        )
        self.skill_gap_threshold = self._clamp01(
            self._as_float(self.config.get("skill_gap_urgency_threshold"), self.conflict_threshold)
        )
        self.pe_high_threshold = self._clamp01(
            self._as_float(self.config.get("pe_high_threshold"), 0.7)
        )
        self.pe_streak_threshold = max(1, int(self._as_float(self.config.get("pe_streak_threshold"), 5.0)))
        self.llm_reeval_interval = max(1, int(self._as_float(self.config.get("llm_reeval_interval"), 10.0)))
        self.learning_context_window = max(
            1,
            int(self._as_float(self.config.get("llm_learning_context_window"), 10.0)),
        )
        self.plan_default_window_size = max(
            1,
            int(self._as_float(self.config.get("plan_default_window_size"), 20.0)),
        )
        self.plan_min_window_size = max(
            1,
            int(self._as_float(self.config.get("plan_min_window_size"), 5.0)),
        )
        self.plan_max_window_size = max(
            self.plan_min_window_size,
            int(self._as_float(self.config.get("plan_max_window_size"), 100.0)),
        )
        self.movement_phase_min_window = max(
            1,
            int(self._as_float(self.config.get("movement_phase_min_window"), 20.0)),
        )
        self.plan_max_phases = max(
            1,
            int(self._as_float(self.config.get("plan_max_phases"), 5.0)),
        )
        self.plan_window_escalation_streak = max(
            1,
            int(self._as_float(self.config.get("plan_window_escalation_streak"), 3.0)),
        )
        self.plan_stall_step_threshold = max(
            1,
            int(self._as_float(self.config.get("plan_stall_step_threshold"), 20.0)),
        )
        self.plan_stall_x_threshold = max(
            0.0,
            self._as_float(self.config.get("plan_stall_x_threshold"), 0.5),
        )
        model_value = self.config.get("model", "claude-sonnet-4-20250514")
        self.model = str(model_value).strip() or "claude-sonnet-4-20250514"
        self._pe_streak = 0
        self._last_goals_fingerprint = ""
        self._pending_llm_result: Optional[Dict[str, Any]] = None
        self._llm_thread: Optional[threading.Thread] = None
        self._llm_state_lock = threading.Lock()
        self._active_plan_cache: Optional[Dict[str, Any]] = None
        self._plan_window_elapsed_nonurgent_streak = 0

    def propose_action(self, goals: Any, context: Mapping[str, Any]) -> Optional[ActionProposal]:
        policies = self.discover_policies()
        if not policies:
            return None

        selected_policy = self._arbitrate(policies=policies, goals=goals, context=context)
        if selected_policy is None:
            return None

        try:
            action_payload = self._invoke_policy(selected_policy, goals, context)
        except Exception as exc:
            self._record_policy_invoke_error(selected_policy, context, exc)
            return None

        coherence_result = self.goal_checker.check(goals, selected_policy, context)
        prediction_result = self.prediction_error_calculator.compute(
            selected_policy["policy_id"],
            context=context,
            memory_manager=self.memory_manager,
        )
        coherence_score = self._score_from_result(
            coherence_result,
            "coherence_score",
            0.5,
        )
        prediction_error_score = self._score_from_result(
            prediction_result,
            "prediction_error_score",
            0.5,
        )
        combined_score = self._clamp01((coherence_score + (1.0 - prediction_error_score)) / 2.0)

        components = {
            "coherence_score": coherence_score,
            "prediction_error_score": prediction_error_score,
            "coherence_result": coherence_result,
            "prediction_result": prediction_result,
            "selected_at": _utc_ts(),
            "selection_mode": "llm_arbitration",
        }
        self.memory_manager.record_policy_selection(
            policy_id=selected_policy["policy_id"],
            score=combined_score,
            components=components,
            step=context.get("step"),
        )

        return ActionProposal(
            action_id=selected_policy["policy_id"],
            action=action_payload,
            expected_outcome=selected_policy.get("description"),
            cost=max(0.0, 1.0 - float(combined_score)),
        )

    def discover_policies(self) -> List[Dict[str, Any]]:
        policies = self._discover_from_adapter_api()
        self.memory_manager.register_policies(policies)
        return policies

    def _discover_from_adapter_api(self) -> List[Dict[str, Any]]:
        getter = getattr(self.adapter, "get_available_policies", None)
        if not callable(getter):
            return []
        raw = getter()
        if not isinstance(raw, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            descriptor = self._normalize_policy_descriptor(item, source="adapter_api")
            if descriptor is not None:
                normalized.append(descriptor)
        return normalized

    def _normalize_policy_descriptor(
        self,
        policy: Mapping[str, Any],
        source: str,
    ) -> Optional[Dict[str, Any]]:
        callable_name = (
            policy.get("callable_name")
            or policy.get("name")
            or policy.get("method")
            or policy.get("id")
        )
        if callable_name is None:
            return None
        callable_name = str(callable_name).strip()
        if not callable_name:
            return None

        member = getattr(self.adapter, callable_name, None)
        if not callable(member):
            return None

        description = policy.get("description")
        if description is not None:
            description = str(description)
        tags = self._normalize_tags(policy.get("tags"))
        if not tags:
            return None
        drive_tags = self._normalize_tags(policy.get("drive_tags"))
        if not drive_tags:
            descriptor_texts = [callable_name]
            if isinstance(description, str):
                descriptor_texts.append(description)
            drive_tags = self._infer_drive_tags(tags=tags, descriptor_texts=descriptor_texts)
        signature = str(inspect.signature(member))
        provided_policy_id = policy.get("policy_id")
        if isinstance(provided_policy_id, str) and provided_policy_id.strip():
            policy_id = provided_policy_id.strip()
        else:
            policy_id = f"{self.adapter_folder}:{callable_name}"

        return {
            "policy_id": policy_id,
            "adapter_folder": self.adapter_folder,
            "callable_name": callable_name,
            "source": source,
            "signature": signature,
            "tags": tags,
            "drive_tags": drive_tags,
            "description": description,
        }

    def _invoke_policy(
        self,
        policy: Mapping[str, Any],
        goals: Any,
        context: Mapping[str, Any],
    ) -> Any:
        callable_name = str(policy["callable_name"])
        policy_callable = getattr(self.adapter, callable_name)
        if not callable(policy_callable):
            raise TypeError(f"Policy '{callable_name}' is not callable.")

        invocation_context = dict(context)
        invocation_context["goals"] = goals
        invocation_context["policy_descriptor"] = dict(policy)

        signature = inspect.signature(policy_callable)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if accepts_kwargs:
            return policy_callable(**invocation_context)

        kwargs: Dict[str, Any] = {}
        for name, param in signature.parameters.items():
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                if name in invocation_context:
                    kwargs[name] = invocation_context[name]
                elif param.default is inspect.Parameter.empty:
                    raise TypeError(
                        f"Policy '{callable_name}' requires missing argument '{name}'."
                    )
            elif param.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError(
                    f"Policy '{callable_name}' has positional-only parameter '{name}'."
                )
        return policy_callable(**kwargs)

    def _record_policy_invoke_error(
        self,
        policy: Mapping[str, Any],
        context: Mapping[str, Any],
        exc: Exception,
    ) -> None:
        trace = {
            "policy_id": policy.get("policy_id"),
            "operation": "invoke",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "step": context.get("step"),
            "ts": _utc_ts(),
        }
        self.memory_manager.record_policy_trace(trace)
        if self.logger is not None:
            self.logger.event(
                "policy.invoke_error",
                {
                    "policy_id": policy.get("policy_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                step=context.get("step"),
            )

    def _arbitrate(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not policies:
            return None

        if self.llm_client is None or len(policies) == 1:
            selected = self._urgency_fallback(policies, context, goals=goals)
            self._write_arbitration_fallback_trace(
                selected_policy=selected,
                rationale="urgency_fallback",
                context=context,
                status="fallback",
            )
            return selected

        self._update_pe_streak(context)
        gate = self._should_call_llm(policies=policies, goals=goals, context=context)
        reasons = list(gate.get("reasons", []))
        if self.logger is not None:
            self.logger.event(
                "policy.llm_gate",
                {
                    "should_call_llm": bool(gate.get("should_call")),
                    "urgent": bool(gate.get("urgent")),
                    "reasons": reasons,
                    "pe_streak": self._pe_streak,
                    "plan_window_elapsed_nonurgent_streak": self._plan_window_elapsed_nonurgent_streak,
                },
                step=context.get("step"),
            )

        if bool(gate.get("plan_window_elapsed")) and not bool(gate.get("urgent")):
            self._plan_window_elapsed_nonurgent_streak += 1
        else:
            self._plan_window_elapsed_nonurgent_streak = 0

        if not bool(gate.get("should_call")):
            selected = self._urgency_fallback(policies, context, goals=goals)
            self._write_arbitration_fallback_trace(
                selected_policy=selected,
                rationale="reactive_gate:no_trigger",
                context=context,
                status="reactive",
            )
            return selected

        if not bool(gate.get("urgent")):
            selected = self._pending_llm_policy(policies, context=context)
            if selected is None:
                selected = self._urgency_fallback(policies, context, goals=goals)
            self._write_arbitration_fallback_trace(
                selected_policy=selected,
                rationale=f"reactive_gate:{','.join(reasons)}",
                context=context,
                status="reactive",
            )
            self._start_async_llm_arbitration(
                policies=policies,
                goals=goals,
                context=context,
                gate_reasons=reasons,
                goals_fingerprint=str(gate.get("goals_fingerprint", "")),
            )
            return selected

        selected = self._run_llm_arbitration(
            policies=policies,
            goals=goals,
            context=context,
            gate_reasons=reasons,
            record_trace=True,
        )
        self._plan_window_elapsed_nonurgent_streak = 0
        self._pe_streak = 0
        self._last_goals_fingerprint = str(gate.get("goals_fingerprint", ""))
        if isinstance(selected, Mapping):
            with self._llm_state_lock:
                self._pending_llm_result = dict(selected)
        return selected

    def _run_llm_arbitration(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Mapping[str, Any],
        gate_reasons: Optional[List[str]] = None,
        record_trace: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if self.llm_client is None:
            return self._urgency_fallback(policies, context, goals=goals)

        system_prompt, user_prompt = self._build_arbitration_prompt(
            policies=policies,
            goals=goals,
            context=context,
        )

        step = context.get("step")
        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        allostatic_assessment = context.get("allostatic_assessment")
        skill_plan = self._resolve_skill_plan(context.get("skill_plan"))
        reasons = list(gate_reasons or [])

        if self.logger is not None:
            self.logger.llm_request(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": 512,
                    "candidate_policy_ids": [p.get("policy_id") for p in policies],
                    "drive_signals": drive_signals,
                    "allostatic_assessment": allostatic_assessment,
                    "skill_plan": skill_plan,
                    "gate_reasons": reasons,
                    "async": not record_trace,
                },
                step=step,
            )

        response_text = ""
        try:
            response = self.llm_client.generate(
                LLMRequest(
                    messages=[
                        LLMMessage("system", system_prompt),
                        LLMMessage("user", user_prompt),
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=512,
                )
            )
            response_text = str(getattr(response, "text", "") or "")
        except Exception as exc:
            selected = self._urgency_fallback(policies, context, goals=goals)
            if record_trace:
                self._write_arbitration_fallback_trace(
                    selected_policy=selected,
                    rationale=f"llm_error:{type(exc).__name__}:{exc}",
                    context=context,
                    status="fallback_error",
                )
            if self.logger is not None:
                self.logger.event(
                    "policy.arbitration_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fallback_policy_id": selected.get("policy_id") if selected else None,
                        "gate_reasons": reasons,
                        "async": not record_trace,
                    },
                    step=step,
                )
            return selected

        if self.logger is not None:
            self.logger.llm_response(
                {"raw": response_text, "async": not record_trace},
                step=step,
            )

        parsed_response = self._parse_full_response(response_text)
        selected, generated_plan = self._select_policy_and_plan_from_response(
            parsed_response=parsed_response,
            raw_response=response_text,
            policies=policies,
            context=context,
        )
        if selected is None:
            fallback = self._urgency_fallback(policies, context, goals=goals)
            if record_trace:
                self._write_arbitration_fallback_trace(
                    selected_policy=fallback,
                    rationale=response_text,
                    context=context,
                    status="fallback_parse_error",
                )
            if self.logger is not None:
                self.logger.event(
                    "policy.arbitration_parse_error",
                    {
                        "raw_response": response_text,
                        "fallback_policy_id": fallback.get("policy_id") if fallback else None,
                        "gate_reasons": reasons,
                        "async": not record_trace,
                    },
                    step=step,
                )
            return fallback

        rationale = parsed_response.get("rationale", self._extract_rationale(response_text))
        reasoning = parsed_response.get("reasoning", "")
        drive_conflict = parsed_response.get("drive_conflict_detected", False)
        confidence = parsed_response.get("confidence")
        if generated_plan is not None:
            self._store_active_plan(generated_plan, step=step)
            selected = dict(selected)
            selected["_llm_plan"] = generated_plan

        if record_trace:
            self._write_arbitration_trace(
                selected_policy=selected,
                rationale=rationale,
                context=context,
            )

        if self.logger is not None:
            self.logger.event(
                "policy.decision",
                {
                    "selected_policy_id": selected.get("policy_id"),
                    "reasoning": reasoning,
                    "rationale": rationale,
                    "drive_conflict_detected": drive_conflict,
                    "confidence": confidence,
                    "drive_signals": [
                        {
                            "channel_id": s.get("channel_id"),
                            "urgency": s.get("urgency"),
                            "projected_value": s.get("projected_value"),
                            "ticks_to_critical": s.get("ticks_to_critical"),
                        }
                        for s in drive_signals
                    ],
                    "allostatic_needs": (
                        allostatic_assessment.get("needs")
                        if isinstance(allostatic_assessment, Mapping)
                        else []
                    ),
                    "skill_plan": skill_plan,
                    "llm_plan": generated_plan,
                    "candidate_policy_ids": [p.get("policy_id") for p in policies],
                    "gate_reasons": reasons,
                    "async": not record_trace,
                },
                step=step,
            )

        return selected

    def _start_async_llm_arbitration(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Mapping[str, Any],
        gate_reasons: List[str],
        goals_fingerprint: str,
    ) -> None:
        with self._llm_state_lock:
            if self._llm_thread is not None and self._llm_thread.is_alive():
                return
            self._last_goals_fingerprint = goals_fingerprint
            policies_copy = [dict(policy) for policy in policies]
            context_copy = dict(context)
            try:
                goals_copy = copy.deepcopy(goals)
            except Exception:
                goals_copy = goals
            thread = threading.Thread(
                target=self._async_llm_call,
                args=(policies_copy, goals_copy, context_copy, list(gate_reasons)),
                daemon=True,
            )
            self._llm_thread = thread
            thread.start()

    def _async_llm_call(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Mapping[str, Any],
        gate_reasons: List[str],
    ) -> None:
        try:
            selected = self._run_llm_arbitration(
                policies=policies,
                goals=goals,
                context=context,
                gate_reasons=gate_reasons,
                record_trace=False,
            )
            if isinstance(selected, Mapping):
                with self._llm_state_lock:
                    self._pending_llm_result = dict(selected)
                if self.logger is not None:
                    self.logger.event(
                        "policy.arbitration_async_ready",
                        {
                            "policy_id": selected.get("policy_id"),
                            "gate_reasons": gate_reasons,
                        },
                        step=context.get("step"),
                    )
        except Exception as exc:
            if self.logger is not None:
                self.logger.event(
                    "policy.arbitration_async_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    step=context.get("step"),
                )

    def _pending_llm_policy(
        self,
        policies: List[Dict[str, Any]],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._llm_state_lock:
            pending = dict(self._pending_llm_result) if isinstance(self._pending_llm_result, Mapping) else None
        if pending is None:
            return None
        policy_id = pending.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id.strip():
            return None
        selected = self._find_policy_by_id(policies, policy_id.strip())
        if selected is None:
            return None
        pending_plan = pending.get("_llm_plan") or pending.get("plan")
        if isinstance(pending_plan, Mapping):
            step = context.get("step") if isinstance(context, Mapping) else None
            self._store_active_plan(dict(pending_plan), step=step)
            selected = dict(selected)
            selected["_llm_plan"] = dict(pending_plan)
        return selected

    def _should_call_llm(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        drive_conflict = self._drive_conflict_detected(drive_signals)
        pe_high = self._pe_streak >= self.pe_streak_threshold
        skill_gap = self._has_skill_gap(policies=policies, drive_signals=drive_signals, context=context)

        step_raw = context.get("step", 0)
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            step = 0
        active_plan = self._get_active_plan()
        plan_window_elapsed = self._active_plan_window_elapsed(
            active_plan=active_plan,
            context=context,
            step=step,
        )
        plan_stalled = self._active_plan_stalled(
            active_plan=active_plan,
            context=context,
            step=step,
        )
        triggered_interrupts = self._triggered_interrupt_conditions(
            active_plan=active_plan,
            context=context,
            step=step,
        )
        periodic = (step % self.llm_reeval_interval) == 0
        if active_plan is not None and not plan_window_elapsed and not triggered_interrupts:
            periodic = False

        goals_fingerprint = self._goals_fingerprint(goals)
        goal_changed = goals_fingerprint != self._last_goals_fingerprint

        urgent_reasons: List[str] = []
        non_urgent_reasons: List[str] = []
        if self._has_irreversible_emergency(context.get("allostatic_assessment")):
            urgent_reasons.append("irreversible_plan_preemption")
        if drive_conflict:
            urgent_reasons.append("drive_conflict")
        if pe_high:
            urgent_reasons.append("sustained_high_prediction_error")
        if skill_gap:
            urgent_reasons.append("skill_gap")
        if plan_stalled:
            urgent_reasons.append("plan_stall_zero_displacement")
        if plan_window_elapsed:
            if self._plan_window_elapsed_nonurgent_streak >= self.plan_window_escalation_streak:
                urgent_reasons.append("plan_window_elapsed_escalated")
            else:
                non_urgent_reasons.append("plan_window_elapsed")
        for reason in triggered_interrupts:
            if self._interrupt_reason_is_urgent(reason):
                urgent_reasons.append(f"plan_interrupt:{reason}")
            else:
                non_urgent_reasons.append(f"plan_interrupt:{reason}")
        if active_plan is not None:
            if plan_stalled:
                self._set_active_plan_status(
                    active_plan=active_plan,
                    status="suspended",
                    step=step,
                    reason="zero_displacement_stall",
                )
            elif triggered_interrupts:
                self._set_active_plan_status(
                    active_plan=active_plan,
                    status="suspended",
                    step=step,
                    reason=";".join(triggered_interrupts),
                )
        if periodic:
            # Periodic reeval is urgent (synchronous) when the skill plan is
            # exhausted (remaining_count == 0 or head is None).  An exhausted
            # plan means the agent has consumed its prior without reaching the
            # goal — exactly the high-precision prediction-error condition Seth
            # describes that demands a full generative-model update, not a
            # background thread that will likely outlive the remaining episode.
            skill_plan_raw = context.get("skill_plan")
            plan_exhausted = False
            if isinstance(skill_plan_raw, Mapping):
                remaining = skill_plan_raw.get("remaining_count", 1)
                head = skill_plan_raw.get("head_policy_id")
                plan_exhausted = (
                    (isinstance(remaining, int) and remaining == 0)
                    or head is None
                )
            if plan_exhausted:
                urgent_reasons.append("periodic_reeval_plan_exhausted")
            else:
                non_urgent_reasons.append("periodic_reeval")
        if goal_changed:
            # A new goal with no prior LLM deliberation is treated as urgent:
            # encountering a new goal generates a high-precision prediction error
            # that requires an immediate (synchronous) generative-model update.
            # Seth's beast machine does not defer its initial response to a new
            # prior — it resolves prediction error before acting.
            # Once a result already exists the update can be deferred (async).
            with self._llm_state_lock:
                has_pending = self._pending_llm_result is not None
            if not has_pending:
                urgent_reasons.append("goal_changed")
            else:
                non_urgent_reasons.append("goal_changed")

        reasons = urgent_reasons + non_urgent_reasons
        return {
            "should_call": bool(reasons),
            "urgent": bool(urgent_reasons),
            "reasons": reasons,
            "goals_fingerprint": goals_fingerprint,
            "drive_conflict": drive_conflict,
            "pe_high": pe_high,
            "skill_gap": skill_gap,
            "active_plan": active_plan,
            "plan_window_elapsed": plan_window_elapsed,
            "plan_stalled": plan_stalled,
            "plan_interrupts": triggered_interrupts,
            "periodic": periodic,
            "goal_changed": goal_changed,
        }

    def _drive_conflict_detected(self, drive_signals: List[Dict[str, Any]]) -> bool:
        high_urgency: List[Dict[str, Any]] = []
        for signal in drive_signals:
            urgency_raw = signal.get("urgency")
            if not isinstance(urgency_raw, (int, float)):
                continue
            if self._clamp01(float(urgency_raw)) > self.conflict_threshold:
                high_urgency.append(signal)
        if len(high_urgency) < 2:
            return False
        competing_tags: set[str] = set()
        for signal in high_urgency:
            candidate = signal.get("suggested_action_tag")
            if not isinstance(candidate, str) or not candidate.strip():
                candidate = signal.get("channel_id")
            if not isinstance(candidate, str):
                continue
            normalized = candidate.strip().lower()
            if normalized:
                competing_tags.add(normalized)
        return len(competing_tags) >= 2

    def _has_skill_gap(
        self,
        policies: List[Dict[str, Any]],
        drive_signals: List[Dict[str, Any]],
        context: Mapping[str, Any],
    ) -> bool:
        active_channels: set[str] = set()
        for signal in drive_signals:
            channel_id = signal.get("channel_id")
            urgency = signal.get("urgency")
            if not isinstance(channel_id, str) or not isinstance(urgency, (int, float)):
                continue
            if self._clamp01(float(urgency)) > self.skill_gap_threshold:
                normalized = channel_id.strip().lower()
                if normalized:
                    active_channels.add(normalized)
        if not active_channels:
            return False

        matching_skills: List[Dict[str, Any]] = []
        for policy in policies:
            drive_tags = set(self._normalize_tags(policy.get("drive_tags")))
            if drive_tags.intersection(active_channels):
                matching_skills.append(policy)
        if not matching_skills:
            return True
        return not any(self._skill_preconditions_met(skill, context) for skill in matching_skills)

    def _skill_preconditions_met(self, skill: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        hardcoded = skill.get("preconditions_met")
        if isinstance(hardcoded, bool):
            return hardcoded

        checkers = (
            "skill_preconditions_met",
            "are_skill_preconditions_met",
            "check_skill_preconditions",
        )
        for name in checkers:
            checker = getattr(self.adapter, name, None)
            if not callable(checker):
                continue
            result: Any = None
            try:
                result = checker(skill=skill, context=context)
            except TypeError:
                try:
                    result = checker(skill, context)
                except TypeError:
                    try:
                        result = checker(skill)
                    except Exception:
                        result = None
                except Exception:
                    result = None
            except Exception:
                result = None

            if isinstance(result, bool):
                return result
            if isinstance(result, Mapping):
                for key in ("met", "ok", "is_met"):
                    value = result.get(key)
                    if isinstance(value, bool):
                        return value
        return True

    def _update_pe_streak(self, context: Mapping[str, Any]) -> int:
        magnitude = self._extract_prediction_error_magnitude(context)
        if magnitude is None:
            self._pe_streak = 0
            return self._pe_streak
        if self._clamp01(magnitude) > self.pe_high_threshold:
            self._pe_streak += 1
        else:
            self._pe_streak = 0
        return self._pe_streak

    def _extract_prediction_error_magnitude(self, context: Mapping[str, Any]) -> Optional[float]:
        candidates = (
            context.get("perceptual_prediction_error"),
            context.get("latest_prediction_error"),
            context.get("prediction_error"),
        )
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                for key in ("aggregate_magnitude", "magnitude"):
                    value = candidate.get(key)
                    if isinstance(value, (int, float)):
                        return self._clamp01(float(value))
            if hasattr(candidate, "aggregate_magnitude"):
                value = getattr(candidate, "aggregate_magnitude")
                if isinstance(value, (int, float)):
                    return self._clamp01(float(value))
            if hasattr(candidate, "magnitude"):
                value = getattr(candidate, "magnitude")
                if isinstance(value, (int, float)):
                    return self._clamp01(float(value))
        return None

    def _goals_fingerprint(self, goals: Any) -> str:
        try:
            return json.dumps(goals, ensure_ascii=True, sort_keys=True, default=str)
        except Exception:
            return str(goals)

    def _build_arbitration_prompt(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Mapping[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are an active inference arbitration system under the free energy principle.\n"
            "Select the action that minimizes expected free energy by reducing homeostatic deficits while managing epistemic uncertainty.\n"
            "Reason only from the provided drive states, observation context, learning context, and policy descriptors.\n"
            "Do not use prior knowledge about what any action physically does in the world.\n"
            "When a skill plan is provided, prefer the plan head unless an irreversible drive is in emergency.\n"
            "Window guidance: movement/exploration phases should usually use window_size >= 20 (often 20-60).\n"
            "Use window_size < 20 only for high-confidence goal-completion or safety-critical phases.\n"
            "Output only valid JSON with this schema:\n"
            "{"
            '"reasoning": "<brief reasoning>", '
            '"rationale": "<one-sentence summary>", '
            '"confidence": <float 0-1>, '
            '"window_size": <int recommended max plan window>, '
            '"interrupt_conditions": ["<condition string>", "..."], '
            '"phases": ['
            '{"action": "<policy_id or policy index>", "until": "<termination condition>", "max_steps": <optional int>}'
            "]"
            "}\n"
            "Use at most 5 phases. Each phase should specify an action and a termination condition.\n"
            "Backward compatibility: include selected_index if you cannot produce phases."
        )

        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        allostatic_assessment = context.get("allostatic_assessment")
        skill_plan = self._resolve_skill_plan(context.get("skill_plan"))

        drive_lines: List[str] = [
            "DRIVE STATE",
            f"goals: {self._serialize_for_prompt(goals)}",
            f"signal_count: {len(drive_signals)}",
        ]
        for index, signal in enumerate(drive_signals):
            channel_id = signal.get("channel_id")
            urgency = signal.get("urgency")
            projected_value = signal.get("projected_value")
            ticks_to_critical = signal.get("ticks_to_critical")
            drive_lines.append(f"signal[{index}].channel_id: {channel_id}")
            drive_lines.append(f"signal[{index}].urgency: {urgency}")
            drive_lines.append(f"signal[{index}].projected_value: {projected_value}")
            drive_lines.append(f"signal[{index}].ticks_to_critical: {ticks_to_critical}")

        if isinstance(allostatic_assessment, Mapping):
            needs = allostatic_assessment.get("needs")
            drive_lines.append("allostatic_assessment.present: true")
            drive_lines.append("allostatic_assessment.needs:")
            if isinstance(needs, list) and needs:
                for index, need in enumerate(needs):
                    if not isinstance(need, Mapping):
                        continue
                    drive_lines.append(f"need[{index}].need_id: {need.get('need_id')}")
                    drive_lines.append(f"need[{index}].urgency: {need.get('urgency')}")
            else:
                drive_lines.append("none")

            policy_bias = allostatic_assessment.get("policy_bias")
            drive_lines.append("allostatic_assessment.policy_bias:")
            if isinstance(policy_bias, Mapping):
                drive_lines.append(
                    f"survival_weight: {policy_bias.get('survival_weight')}"
                )
                drive_lines.append(
                    f"exploration_weight: {policy_bias.get('exploration_weight')}"
                )
            else:
                drive_lines.append("survival_weight: null")
                drive_lines.append("exploration_weight: null")
        else:
            drive_lines.append("allostatic_assessment.present: false")

        drive_lines.append(f"skill_plan.present: {bool(skill_plan)}")
        if skill_plan:
            drive_lines.append(
                f"skill_plan.head_policy_id: {skill_plan.get('head_policy_id')}"
            )
            drive_lines.append(
                f"skill_plan.remaining_count: {skill_plan.get('remaining_count')}"
            )
            drive_lines.append(
                f"skill_plan.remaining_policy_ids: {self._serialize_for_prompt(skill_plan.get('remaining_policy_ids'))}"
            )
            drive_lines.append(
                f"skill_plan.metadata: {self._serialize_for_prompt(skill_plan.get('metadata'))}"
            )

        policy_lines: List[str] = ["CANDIDATE POLICIES"]
        for index, policy in enumerate(policies):
            policy_lines.append(f"[{index}] policy_id: {policy.get('policy_id')}")
            policy_lines.append(f"[{index}] description: {policy.get('description')}")
            policy_lines.append(
                f"[{index}] tags: {self._serialize_for_prompt(self._normalize_tags(policy.get('tags')))}"
            )
            policy_lines.append(
                "[{}] drive_tags: {}".format(
                    index,
                    self._serialize_for_prompt(self._normalize_tags(policy.get("drive_tags"))),
                )
            )

        observation_lines = self._format_observation_context(context.get("world_facts"))
        learning_lines = self._format_learning_context(window=self.learning_context_window)

        instruction = (
            "INSTRUCTION\n"
            f"Reason over expected free energy across the next {self.horizon} steps. "
            "Return JSON only."
        )

        user_prompt = "\n\n".join(
            [
                "\n".join(drive_lines),
                "\n".join(observation_lines),
                "\n".join(policy_lines),
                "\n".join(learning_lines),
                instruction,
            ]
        )
        return system_prompt, user_prompt

    def _format_observation_context(self, world_facts: Any) -> List[str]:
        lines: List[str] = ["OBSERVATION CONTEXT"]
        if not isinstance(world_facts, Mapping):
            lines.append("world_facts.present: false")
            return lines

        facts = dict(world_facts)
        position = facts.get("position")
        position_map = dict(position) if isinstance(position, Mapping) else {}
        lines.append("world_facts.present: true")
        lines.append(f"biome: {facts.get('biome')}")
        lines.append(f"position.x: {position_map.get('x')}")
        lines.append(f"position.y: {position_map.get('y')}")
        lines.append(f"position.z: {position_map.get('z')}")
        lines.append(f"nearby_crafting_table: {facts.get('nearby_crafting_table')}")
        lines.append(f"nearby_cow: {facts.get('nearby_cow')}")
        lines.append(f"has_bucket: {facts.get('has_bucket')}")
        lines.append(f"bucket_count: {facts.get('bucket_count')}")
        lines.append(f"inventory_non_air_slots: {facts.get('inventory_non_air_slots')}")
        lines.append(f"inventory_total_quantity: {facts.get('inventory_total_quantity')}")
        lines.append(f"inventory_fullness: {facts.get('inventory_fullness')}")
        return lines

    def _format_learning_context(self, *, window: int) -> List[str]:
        transitions = self._recent_transition_payloads(limit=window)
        lines: List[str] = [
            "LEARNING CONTEXT",
            f"transition_window: {window}",
            f"transitions_found: {len(transitions)}",
        ]

        if transitions:
            attempts_by_policy: Dict[str, int] = {}
            zero_reward_by_policy: Dict[str, int] = {}
            no_progress_by_policy: Dict[str, int] = {}
            inventory_progress_by_policy: Dict[str, int] = {}
            bucket_progress_by_policy: Dict[str, int] = {}

            zero_reward_streak = 0
            no_progress_streak = 0

            for transition in transitions:
                policy_id = str(transition.get("policy_id") or "unknown")
                attempts_by_policy[policy_id] = attempts_by_policy.get(policy_id, 0) + 1

                reward = self._as_float(transition.get("reward"), 0.0)
                inventory_progress = transition.get("inventory_progress")
                bucket_progress = transition.get("bucket_progress")
                inventory_changed = self._progress_changed(inventory_progress)
                bucket_changed = self._progress_changed(bucket_progress)
                has_progress = inventory_changed or bucket_changed

                if reward <= 0.0:
                    zero_reward_by_policy[policy_id] = zero_reward_by_policy.get(policy_id, 0) + 1
                    if not has_progress:
                        no_progress_by_policy[policy_id] = no_progress_by_policy.get(policy_id, 0) + 1
                if inventory_changed:
                    inventory_progress_by_policy[policy_id] = (
                        inventory_progress_by_policy.get(policy_id, 0) + 1
                    )
                if bucket_changed:
                    bucket_progress_by_policy[policy_id] = (
                        bucket_progress_by_policy.get(policy_id, 0) + 1
                    )

            for transition in transitions:
                reward = self._as_float(transition.get("reward"), 0.0)
                if reward <= 0.0:
                    zero_reward_streak += 1
                else:
                    break
            for transition in transitions:
                reward = self._as_float(transition.get("reward"), 0.0)
                has_progress = self._progress_changed(transition.get("inventory_progress")) or self._progress_changed(
                    transition.get("bucket_progress")
                )
                if reward <= 0.0 and not has_progress:
                    no_progress_streak += 1
                else:
                    break

            lines.append(f"zero_reward_streak: {zero_reward_streak}")
            lines.append(f"no_progress_streak: {no_progress_streak}")
            for policy_id in sorted(attempts_by_policy.keys()):
                lines.append(f"policy[{policy_id}].attempts: {attempts_by_policy[policy_id]}")
                lines.append(
                    f"policy[{policy_id}].zero_reward_attempts: {zero_reward_by_policy.get(policy_id, 0)}"
                )
                lines.append(
                    f"policy[{policy_id}].no_progress_attempts: {no_progress_by_policy.get(policy_id, 0)}"
                )
                lines.append(
                    f"policy[{policy_id}].inventory_progress_steps: {inventory_progress_by_policy.get(policy_id, 0)}"
                )
                lines.append(
                    f"policy[{policy_id}].bucket_progress_steps: {bucket_progress_by_policy.get(policy_id, 0)}"
                )

            if no_progress_by_policy:
                stuck_policy = sorted(
                    no_progress_by_policy.items(),
                    key=lambda item: (-item[1], item[0]),
                )[0]
                lines.append(
                    "no_progress_summary: policy={} failed {} times with 0 reward and no inventory/bucket change".format(
                        stuck_policy[0],
                        stuck_policy[1],
                    )
                )
        else:
            lines.append("zero_reward_streak: 0")
            lines.append("no_progress_streak: 0")

        position_delta = self._transition_position_delta(transitions)
        if position_delta is not None:
            lines.append(f"position_delta_blocks: {position_delta:.2f}")
            lines.append(f"position_changed_meaningfully: {position_delta >= 3.0}")
        else:
            lines.append("position_delta_blocks: null")
            lines.append("position_changed_meaningfully: false")

        plan_snapshot = self._latest_plan_payload()
        if isinstance(plan_snapshot, Mapping):
            lines.append("plan_summary.present: true")
            lines.append(f"plan_summary.status: {plan_snapshot.get('status')}")
            lines.append(f"plan_summary.window_size: {plan_snapshot.get('window_size')}")
            lines.append(f"plan_summary.current_phase_index: {plan_snapshot.get('current_phase_index')}")
            history = plan_snapshot.get("history")
            if isinstance(history, list) and history:
                lines.append(f"plan_summary.history_events: {len(history)}")
                recent_events = history[-3:]
                for index, event in enumerate(recent_events):
                    if not isinstance(event, Mapping):
                        continue
                    lines.append(
                        "plan_event[{}]: event={}, phase_index={}, reason={}, step={}, world_facts={}".format(
                            index,
                            event.get("event"),
                            event.get("phase_index"),
                            event.get("reason"),
                            event.get("step"),
                            self._serialize_for_prompt(event.get("world_facts")),
                        )
                    )
            else:
                lines.append("plan_summary.history_events: 0")
        else:
            lines.append("plan_summary.present: false")

        pe_records = self._recent_prediction_error_records(limit=window)
        lines.append(f"prediction_error_window: {window}")
        lines.append(f"prediction_errors_found: {len(pe_records)}")
        if not pe_records:
            lines.append("prediction_error_summary: none")
            return lines

        aggregates: Dict[tuple[str, str], Dict[str, float]] = {}
        for record in pe_records:
            policy_id = str(record.get("policy_id") or "bootstrap")
            channel = str(record.get("channel") or "unknown").strip().lower() or "unknown"
            magnitude = self._extract_prediction_error_magnitude_from_record(record)
            if magnitude is None:
                continue
            key = (policy_id, channel)
            bucket = aggregates.setdefault(key, {"sum": 0.0, "count": 0.0})
            bucket["sum"] += float(magnitude)
            bucket["count"] += 1.0

        if not aggregates:
            lines.append("prediction_error_summary: none")
            return lines

        for policy_id, channel in sorted(aggregates.keys()):
            stats = aggregates[(policy_id, channel)]
            count = int(stats["count"])
            mean = float(stats["sum"] / float(max(1, count)))
            lines.append(
                f"pe_summary[policy={policy_id}|channel={channel}]: mean={mean:.3f}, samples={count}"
            )
        return lines

    def _recent_transition_payloads(self, *, limit: int) -> List[Dict[str, Any]]:
        getter = getattr(self.memory_manager, "get_recent", None)
        if not callable(getter):
            return []
        try:
            entries = getter(limit, entry_type="transition")
        except TypeError:
            try:
                entries = getter(limit)
            except Exception:
                return []
        except Exception:
            return []

        if not isinstance(entries, list):
            return []
        out: List[Dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, Mapping):
                payload = entry.get("payload")
                entry_type = entry.get("entry_type")
                tick = entry.get("tick")
            else:
                payload = getattr(entry, "payload", None)
                entry_type = getattr(entry, "entry_type", None)
                tick = getattr(entry, "tick", None)
            if entry_type != "transition":
                continue
            if not isinstance(payload, Mapping):
                continue
            row = dict(payload)
            row.setdefault("tick", tick)
            out.append(row)
        return out

    def _latest_plan_payload(self) -> Optional[Dict[str, Any]]:
        getter = getattr(self.memory_manager, "get_recent", None)
        if not callable(getter):
            return None
        try:
            entries = getter(1, entry_type="active_plan")
        except TypeError:
            try:
                entries = getter(1)
            except Exception:
                return None
        except Exception:
            return None
        if not isinstance(entries, list) or not entries:
            return None
        entry = entries[0]
        payload = getattr(entry, "payload", None)
        if isinstance(entry, Mapping):
            payload = entry.get("payload", payload)
        if not isinstance(payload, Mapping):
            return None
        return dict(payload)

    def _transition_position_delta(self, transitions: List[Mapping[str, Any]]) -> Optional[float]:
        if not transitions:
            return None
        earliest = transitions[-1]
        latest = transitions[0]
        start_position = self._extract_position(earliest.get("prev_facts"))
        end_position = self._extract_position(latest.get("next_facts"))
        if start_position is None or end_position is None:
            return None
        dx = float(end_position[0] - start_position[0])
        dz = float(end_position[1] - start_position[1])
        return (dx**2 + dz**2) ** 0.5

    def _extract_position(self, facts: Any) -> Optional[tuple[float, float]]:
        if not isinstance(facts, Mapping):
            return None
        position = facts.get("position")
        if not isinstance(position, Mapping):
            return None
        x = position.get("x")
        z = position.get("z")
        if not isinstance(x, (int, float)) or not isinstance(z, (int, float)):
            return None
        return float(x), float(z)

    def _compact_world_facts(self, world_facts: Any) -> Dict[str, Any]:
        if not isinstance(world_facts, Mapping):
            return {}
        position = world_facts.get("position")
        position_map = dict(position) if isinstance(position, Mapping) else {}
        return {
            "biome": world_facts.get("biome"),
            "position": {
                "x": position_map.get("x"),
                "y": position_map.get("y"),
                "z": position_map.get("z"),
            },
            "nearby_crafting_table": world_facts.get("nearby_crafting_table"),
            "nearby_cow": world_facts.get("nearby_cow"),
            "has_bucket": world_facts.get("has_bucket"),
            "bucket_count": world_facts.get("bucket_count"),
        }

    def _recent_prediction_error_records(self, *, limit: int) -> List[Dict[str, Any]]:
        query_prediction_errors = getattr(self.memory_manager, "query_prediction_errors", None)
        if callable(query_prediction_errors):
            try:
                records = query_prediction_errors(limit=limit)
                if isinstance(records, list):
                    return [dict(record) for record in records if isinstance(record, Mapping)]
            except Exception:
                pass

        query = getattr(self.memory_manager, "query", None)
        if not callable(query):
            return []
        try:
            records = query({"target": "prediction_errors", "limit": limit})
        except Exception:
            return []
        if not isinstance(records, list):
            return []
        return [dict(record) for record in records if isinstance(record, Mapping)]

    def _extract_prediction_error_magnitude_from_record(
        self,
        record: Mapping[str, Any],
    ) -> Optional[float]:
        magnitude = record.get("magnitude")
        if isinstance(magnitude, (int, float)):
            return self._clamp01(abs(float(magnitude)))
        nested = record.get("error")
        if isinstance(nested, Mapping):
            nested_magnitude = nested.get("magnitude")
            if isinstance(nested_magnitude, (int, float)):
                return self._clamp01(abs(float(nested_magnitude)))
        return None

    def _progress_changed(self, progress: Any) -> bool:
        if isinstance(progress, Mapping):
            changed = progress.get("changed")
            if isinstance(changed, bool):
                return changed
            for key in ("slot_delta", "quantity_delta", "fullness_delta", "bucket_count_delta"):
                value = progress.get(key)
                if isinstance(value, (int, float)) and abs(float(value)) > 0.0:
                    return True
            for key in ("acquired_bucket",):
                value = progress.get(key)
                if isinstance(value, bool) and value:
                    return True
        return False

    def _select_policy_and_plan_from_response(
        self,
        *,
        parsed_response: Mapping[str, Any],
        raw_response: str,
        policies: List[Dict[str, Any]],
        context: Mapping[str, Any],
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        plan = self._normalize_plan_from_response(
            payload=parsed_response,
            policies=policies,
            context=context,
        )
        if plan is not None:
            phases = plan.get("phases")
            if isinstance(phases, list) and phases:
                first_phase = phases[0]
                if isinstance(first_phase, Mapping):
                    policy_id = first_phase.get("policy_id")
                    if isinstance(policy_id, str) and policy_id.strip():
                        selected = self._find_policy_by_id(policies, policy_id.strip())
                        if selected is not None:
                            return selected, plan

        selected = self._parse_arbitration_response(raw_response, policies)
        if selected is None:
            return None, None

        fallback_plan = self._build_default_plan(
            selected_policy_id=str(selected.get("policy_id") or ""),
            payload=parsed_response,
            context=context,
        )
        return selected, fallback_plan

    def _normalize_plan_from_response(
        self,
        *,
        payload: Mapping[str, Any],
        policies: List[Dict[str, Any]],
        context: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        raw_phases = payload.get("phases")
        if not isinstance(raw_phases, list) or not raw_phases:
            return None

        phases: List[Dict[str, Any]] = []
        for index, raw_phase in enumerate(raw_phases[: self.plan_max_phases]):
            if not isinstance(raw_phase, Mapping):
                continue
            policy_id = self._resolve_phase_policy_id(raw_phase.get("action"), policies)
            if policy_id is None:
                policy_id = self._resolve_phase_policy_id(raw_phase.get("policy_id"), policies)
            if policy_id is None:
                policy_id = self._resolve_phase_policy_id(raw_phase.get("selected_index"), policies)
            if policy_id is None:
                continue
            until_raw = raw_phase.get("until")
            until = str(until_raw).strip() if until_raw is not None else ""
            max_steps = self._coerce_positive_int(raw_phase.get("max_steps"))
            phases.append(
                {
                    "phase_index": index,
                    "policy_id": policy_id,
                    "until": until,
                    "max_steps": max_steps,
                }
            )

        if not phases:
            return None

        return self._build_plan_payload(
            phases=phases,
            payload=payload,
            context=context,
        )

    def _build_default_plan(
        self,
        *,
        selected_policy_id: str,
        payload: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not selected_policy_id:
            return None
        phases = [
            {
                "phase_index": 0,
                "policy_id": selected_policy_id,
                "until": "",
                "max_steps": None,
            }
        ]
        return self._build_plan_payload(
            phases=phases,
            payload=payload,
            context=context,
        )

    def _build_plan_payload(
        self,
        *,
        phases: List[Dict[str, Any]],
        payload: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        step_raw = context.get("step", 0)
        step = self._coerce_non_negative_int(step_raw) or 0
        requested_window = self._coerce_positive_int(payload.get("window_size"))
        window_size = requested_window if requested_window is not None else self.plan_default_window_size
        window_floor = self._minimum_window_for_phases(phases)
        window_size = max(window_floor, min(self.plan_max_window_size, int(window_size)))
        initial_plan_x = self._current_world_x(context)
        initial_plan_z = self._current_world_z(context)

        interrupt_conditions: List[str] = []
        raw_interrupts = payload.get("interrupt_conditions")
        if isinstance(raw_interrupts, list):
            for item in raw_interrupts:
                if isinstance(item, str) and item.strip():
                    interrupt_conditions.append(item.strip())
                elif isinstance(item, Mapping):
                    condition = item.get("condition")
                    if isinstance(condition, str) and condition.strip():
                        interrupt_conditions.append(condition.strip())

        return {
            "plan_id": f"plan_{uuid.uuid4().hex[:8]}",
            "status": "active",
            "created_step": step,
            "window_size": window_size,
            "window_expires_step": step + window_size,
            "window_minimum_applied": window_floor,
            "interrupt_conditions": interrupt_conditions,
            "phases": phases,
            "current_phase_index": 0,
            "phase_started_step": step,
            "initial_plan_x": initial_plan_x,
            "initial_plan_z": initial_plan_z,
            "confidence": payload.get("confidence"),
            "rationale": payload.get("rationale"),
            "history": [],
        }

    def _resolve_phase_policy_id(
        self,
        value: Any,
        policies: List[Dict[str, Any]],
    ) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            for policy in policies:
                policy_id = str(policy.get("policy_id") or "").strip()
                if not policy_id:
                    continue
                if text == policy_id:
                    return policy_id
            index = self._coerce_index(text)
            if isinstance(index, int) and 0 <= index < len(policies):
                return str(policies[index].get("policy_id") or "").strip() or None
            return None
        index = self._coerce_index(value)
        if isinstance(index, int) and 0 <= index < len(policies):
            return str(policies[index].get("policy_id") or "").strip() or None
        return None

    def _store_active_plan(self, active_plan: Mapping[str, Any], step: Any) -> None:
        if not isinstance(active_plan, Mapping):
            return
        payload = dict(active_plan)
        normalized_step = self._coerce_non_negative_int(step)
        if normalized_step is None:
            normalized_step = self._coerce_non_negative_int(payload.get("created_step")) or 0
        payload.setdefault("status", "active")
        payload.setdefault("created_step", normalized_step)
        payload.setdefault("phase_started_step", normalized_step)
        payload.setdefault("current_phase_index", 0)
        payload.setdefault("history", [])
        self._active_plan_cache = dict(payload) if payload.get("status") == "active" else None
        self._record_working_entry(
            step=normalized_step,
            entry_type="active_plan",
            payload=payload,
            priority=0.85,
        )

    def _set_active_plan_status(
        self,
        *,
        active_plan: Mapping[str, Any],
        status: str,
        step: int,
        reason: str,
    ) -> None:
        if not isinstance(active_plan, Mapping):
            return
        payload = dict(active_plan)
        payload["status"] = str(status)
        payload["status_reason"] = str(reason)
        history = payload.get("history")
        if isinstance(history, list):
            events = list(history)
        else:
            events = []
        events.append(
            {
                "event": "status_change",
                "step": int(step),
                "status": str(status),
                "reason": str(reason),
            }
        )
        payload["history"] = events[-50:]
        self._active_plan_cache = dict(payload) if payload.get("status") == "active" else None
        self._record_working_entry(
            step=int(step),
            entry_type="active_plan",
            payload=payload,
            priority=0.85,
        )

    def _get_active_plan(self) -> Optional[Dict[str, Any]]:
        if isinstance(self._active_plan_cache, Mapping):
            return dict(self._active_plan_cache)
        getter = getattr(self.memory_manager, "get_recent", None)
        if not callable(getter):
            return None
        try:
            recent = getter(1, entry_type="active_plan")
        except TypeError:
            try:
                recent = getter(1)
            except Exception:
                return None
        except Exception:
            return None
        if not isinstance(recent, list) or not recent:
            return None
        entry = recent[0]
        payload = getattr(entry, "payload", None)
        if isinstance(entry, Mapping):
            payload = entry.get("payload", payload)
        if not isinstance(payload, Mapping):
            return None
        plan = dict(payload)
        if str(plan.get("status") or "active") != "active":
            return None
        self._active_plan_cache = dict(plan)
        return plan

    def _record_working_entry(
        self,
        *,
        step: int,
        entry_type: str,
        payload: Mapping[str, Any],
        priority: float,
    ) -> None:
        recorder = getattr(self.memory_manager, "record_working", None)
        if not callable(recorder):
            return
        try:
            recorder(
                WorkingMemoryEntry(
                    tick=int(step),
                    entry_type=str(entry_type),
                    payload=dict(payload),
                    priority=float(priority),
                )
            )
        except Exception:
            return

    def _select_policy_from_active_plan(
        self,
        *,
        policies: List[Dict[str, Any]],
        context: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        active_plan = self._get_active_plan()
        if active_plan is None:
            return None
        step = context.get("step", 0)
        step_i = int(step) if isinstance(step, int) else 0

        if self._active_plan_window_elapsed(active_plan=active_plan, context=context, step=step_i):
            self._set_active_plan_status(
                active_plan=active_plan,
                status="expired",
                step=step_i,
                reason="window_elapsed",
            )
            return None

        interrupts = self._triggered_interrupt_conditions(
            active_plan=active_plan,
            context=context,
            step=step_i,
        )
        if interrupts:
            self._set_active_plan_status(
                active_plan=active_plan,
                status="suspended",
                step=step_i,
                reason=";".join(interrupts),
            )
            return None

        phases = active_plan.get("phases")
        if not isinstance(phases, list) or not phases:
            self._set_active_plan_status(
                active_plan=active_plan,
                status="suspended",
                step=step_i,
                reason="no_phases",
            )
            return None

        current_index = self._coerce_non_negative_int(active_plan.get("current_phase_index")) or 0
        current_index = max(0, min(len(phases) - 1, current_index))
        phase = phases[current_index]
        if not isinstance(phase, Mapping):
            return None
        phase_started = self._coerce_non_negative_int(active_plan.get("phase_started_step"))
        if phase_started is None:
            phase_started = self._coerce_non_negative_int(active_plan.get("created_step")) or step_i
        steps_elapsed = max(0, step_i - phase_started)

        should_terminate, reason = self._phase_should_terminate(
            phase=phase,
            context=context,
            steps_elapsed=steps_elapsed,
        )
        if should_terminate:
            if current_index + 1 < len(phases):
                active_plan = dict(active_plan)
                active_plan["current_phase_index"] = current_index + 1
                active_plan["phase_started_step"] = step_i
                history = list(active_plan.get("history") or [])
                history.append(
                    {
                        "event": "phase_terminated",
                        "phase_index": current_index,
                        "reason": reason,
                        "step": step_i,
                        "world_facts": self._compact_world_facts(context.get("world_facts")),
                    }
                )
                active_plan["history"] = history[-50:]
                self._store_active_plan(active_plan, step=step_i)
                phase = phases[current_index + 1]
            else:
                self._set_active_plan_status(
                    active_plan=active_plan,
                    status="completed",
                    step=step_i,
                    reason=reason,
                )
                return None

        if not isinstance(phase, Mapping):
            return None
        policy_id = phase.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id.strip():
            return None
        return self._find_policy_by_id(policies, policy_id.strip())

    def _phase_should_terminate(
        self,
        *,
        phase: Mapping[str, Any],
        context: Mapping[str, Any],
        steps_elapsed: int,
    ) -> tuple[bool, str]:
        max_steps = self._coerce_positive_int(phase.get("max_steps"))
        if isinstance(max_steps, int) and steps_elapsed >= max_steps:
            return True, "max_steps"
        condition = phase.get("until")
        if isinstance(condition, str) and condition.strip():
            if self._evaluate_condition_expression(condition.strip(), context=context, steps_elapsed=steps_elapsed):
                return True, f"condition:{condition.strip()}"
        return False, ""

    def _active_plan_window_elapsed(
        self,
        *,
        active_plan: Optional[Mapping[str, Any]],
        context: Mapping[str, Any],
        step: int,
    ) -> bool:
        if not isinstance(active_plan, Mapping):
            return False
        created_step = self._coerce_non_negative_int(active_plan.get("created_step"))
        if created_step is None:
            created_step = step
        window_size = self._coerce_positive_int(active_plan.get("window_size"))
        if window_size is None:
            window_size = self.plan_default_window_size
        effective_window = self._effective_window_size(window_size=window_size, context=context)
        return (step - created_step) >= effective_window

    def _active_plan_stalled(
        self,
        *,
        active_plan: Optional[Mapping[str, Any]],
        context: Mapping[str, Any],
        step: int,
    ) -> bool:
        if not isinstance(active_plan, Mapping):
            return False
        created_step = self._coerce_non_negative_int(active_plan.get("created_step"))
        if created_step is None:
            created_step = step
        steps_in_plan = max(0, int(step) - int(created_step))
        if steps_in_plan <= self.plan_stall_step_threshold:
            return False

        initial_x = active_plan.get("initial_plan_x")
        current_x = self._current_world_x(context)
        if not isinstance(initial_x, (int, float)) or not isinstance(current_x, (int, float)):
            return False
        return abs(float(current_x) - float(initial_x)) < float(self.plan_stall_x_threshold)

    def _triggered_interrupt_conditions(
        self,
        *,
        active_plan: Optional[Mapping[str, Any]],
        context: Mapping[str, Any],
        step: int,
    ) -> List[str]:
        if not isinstance(active_plan, Mapping):
            return []
        created_step = self._coerce_non_negative_int(active_plan.get("created_step"))
        if created_step is None:
            created_step = step
        steps_elapsed = max(0, int(step) - created_step)

        raw_conditions = active_plan.get("interrupt_conditions")
        if not isinstance(raw_conditions, list):
            return []
        triggered: List[str] = []
        for item in raw_conditions:
            if not isinstance(item, str) or not item.strip():
                continue
            condition = item.strip()
            if self._evaluate_condition_expression(condition, context=context, steps_elapsed=steps_elapsed):
                triggered.append(condition)
        return triggered

    def _effective_window_size(self, *, window_size: int, context: Mapping[str, Any]) -> int:
        base = max(self.plan_min_window_size, min(self.plan_max_window_size, int(window_size)))
        arousal = 0.0
        arousal_state = context.get("arousal_valence_state")
        if isinstance(arousal_state, Mapping):
            arousal = self._clamp01(self._as_float(arousal_state.get("arousal"), 0.0))
        if arousal >= 0.8:
            base = int(round(base * 0.25))
        elif arousal >= 0.6:
            base = int(round(base * 0.5))
        elif arousal <= 0.2:
            base = int(round(base * 1.5))
        return max(self.plan_min_window_size, min(self.plan_max_window_size, base))

    def _current_world_x(self, context: Mapping[str, Any]) -> Optional[float]:
        world_facts = context.get("world_facts")
        if not isinstance(world_facts, Mapping):
            return None
        position = world_facts.get("position")
        if not isinstance(position, Mapping):
            return None
        x = position.get("x")
        if not isinstance(x, (int, float)):
            return None
        return float(x)

    def _current_world_z(self, context: Mapping[str, Any]) -> Optional[float]:
        world_facts = context.get("world_facts")
        if not isinstance(world_facts, Mapping):
            return None
        position = world_facts.get("position")
        if not isinstance(position, Mapping):
            return None
        z = position.get("z")
        if not isinstance(z, (int, float)):
            return None
        return float(z)

    def _minimum_window_for_phases(self, phases: List[Mapping[str, Any]]) -> int:
        if phases and all(
            self._is_movement_policy_id(str(phase.get("policy_id") or ""))
            for phase in phases
            if isinstance(phase, Mapping)
        ):
            return max(self.plan_min_window_size, self.movement_phase_min_window)
        return self.plan_min_window_size

    def _is_movement_policy_id(self, policy_id: str) -> bool:
        tokens = set(self._name_tokens(str(policy_id)))
        if not tokens:
            return False
        movement_tokens = {"move", "forward", "back", "left", "right", "strafe", "sprint", "explore"}
        return bool(tokens.intersection(movement_tokens))

    def _interrupt_reason_is_urgent(self, condition: str) -> bool:
        lowered = str(condition or "").strip().lower()
        return any(token in lowered for token in ("health", "oxygen", "threat", "critical"))

    def _evaluate_condition_expression(
        self,
        expression: str,
        *,
        context: Mapping[str, Any],
        steps_elapsed: int,
    ) -> bool:
        normalized = str(expression or "").strip()
        if not normalized:
            return False
        variables = self._condition_variables(context=context, steps_elapsed=steps_elapsed)
        or_parts = re.split(r"\s+\bOR\b\s+|\s*\|\|\s*", normalized, flags=re.IGNORECASE)
        for or_part in or_parts:
            and_parts = re.split(r"\s+\bAND\b\s+|\s*&&\s*", or_part, flags=re.IGNORECASE)
            if and_parts and all(self._evaluate_condition_atom(atom, variables) for atom in and_parts):
                return True
        return False

    def _evaluate_condition_atom(self, atom: str, variables: Mapping[str, Any]) -> bool:
        text = str(atom or "").strip()
        if not text:
            return False
        if text.upper().startswith("NOT "):
            return not self._evaluate_condition_atom(text[4:], variables)
        if text.startswith("!"):
            return not self._evaluate_condition_atom(text[1:], variables)

        match = re.match(r"^([A-Za-z0-9_.]+)\s*(==|!=|<=|>=|<|>)\s*(.+)$", text)
        if match is None:
            value = variables.get(text)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value) != 0.0
            return bool(value)

        key, operator, raw_rhs = match.group(1), match.group(2), match.group(3)
        lhs = variables.get(key)
        rhs = self._parse_condition_value(raw_rhs, variables)
        return self._compare_condition_values(lhs, rhs, operator)

    def _parse_condition_value(self, raw: str, variables: Mapping[str, Any]) -> Any:
        text = str(raw or "").strip()
        if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
            return text[1:-1]
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return float(text)
        except (TypeError, ValueError):
            pass
        if text in variables:
            return variables[text]
        return text

    def _compare_condition_values(self, lhs: Any, rhs: Any, operator: str) -> bool:
        if isinstance(lhs, bool) or isinstance(rhs, bool):
            lhs_bool = bool(lhs)
            rhs_bool = bool(rhs)
            if operator == "==":
                return lhs_bool == rhs_bool
            if operator == "!=":
                return lhs_bool != rhs_bool
            return False

        if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
            left = float(lhs)
            right = float(rhs)
            if operator == "==":
                return left == right
            if operator == "!=":
                return left != right
            if operator == "<":
                return left < right
            if operator == "<=":
                return left <= right
            if operator == ">":
                return left > right
            if operator == ">=":
                return left >= right
            return False

        left_text = str(lhs)
        right_text = str(rhs)
        if operator == "==":
            return left_text == right_text
        if operator == "!=":
            return left_text != right_text
        return False

    def _condition_variables(
        self,
        *,
        context: Mapping[str, Any],
        steps_elapsed: int,
    ) -> Dict[str, Any]:
        variables: Dict[str, Any] = {"steps_elapsed": int(max(0, steps_elapsed))}

        world_facts = context.get("world_facts")
        if isinstance(world_facts, Mapping):
            for key, value in world_facts.items():
                if isinstance(value, Mapping):
                    for sub_key, sub_value in value.items():
                        variables[f"{key}.{sub_key}"] = sub_value
                        variables[f"{key}_{sub_key}"] = sub_value
                else:
                    variables[str(key)] = value
            if "nearby_cow" in world_facts:
                variables["cow_visible"] = bool(world_facts.get("nearby_cow"))
            if "has_bucket" in world_facts:
                variables["bucket_acquired"] = bool(world_facts.get("has_bucket"))

        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        for signal in drive_signals:
            channel_id = signal.get("channel_id")
            if not isinstance(channel_id, str) or not channel_id.strip():
                continue
            key = channel_id.strip().lower()
            current_value = signal.get("current_value")
            urgency = signal.get("urgency")
            if isinstance(current_value, (int, float)):
                variables[key] = float(current_value)
            if isinstance(urgency, (int, float)):
                variables[f"{key}_urgency"] = float(urgency)

        state = context.get("state")
        homeostasis = getattr(state, "homeostasis", None)
        if homeostasis is not None:
            life = getattr(homeostasis, "life", None)
            food = getattr(homeostasis, "food", None)
            air = getattr(homeostasis, "air", None)
            if isinstance(life, (int, float)):
                variables.setdefault("health", self._clamp01(float(life) / 20.0))
            if isinstance(food, (int, float)):
                variables.setdefault("hunger", self._clamp01(float(food) / 20.0))
            if isinstance(air, (int, float)):
                variables.setdefault("oxygen", self._clamp01(float(air) / 300.0))
        return variables

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            out = int(float(value))
        except (TypeError, ValueError):
            return None
        if out <= 0:
            return None
        return out

    @staticmethod
    def _coerce_non_negative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            out = int(float(value))
        except (TypeError, ValueError):
            return None
        if out < 0:
            return None
        return out

    def _parse_arbitration_response(
        self,
        text: str,
        policies: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        def _log_failure(reason: str) -> None:
            if self.logger is not None:
                self.logger.event(
                    "policy.arbitration_parse_error",
                    {
                        "reason": reason,
                    },
                )

        try:
            cleaned = self._strip_markdown_fences(text)
            payload = json.loads(cleaned)
        except Exception as exc:
            _log_failure(f"{type(exc).__name__}:{exc}")
            return None

        if not isinstance(payload, Mapping):
            _log_failure("response_not_json_object")
            return None

        selected_index_raw = payload.get("selected_index")
        selected_index = self._coerce_index(selected_index_raw)
        if selected_index is None:
            _log_failure("selected_index_invalid")
            return None
        if selected_index < 0 or selected_index >= len(policies):
            _log_failure("selected_index_out_of_range")
            return None
        return policies[selected_index]

    def _urgency_fallback(
        self,
        policies: List[Dict[str, Any]],
        context: Mapping[str, Any],
        goals: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if not policies:
            return None

        skill_plan = self._resolve_skill_plan(context.get("skill_plan"))
        emergency_override = self._has_irreversible_emergency(
            context.get("allostatic_assessment")
        )
        if not emergency_override:
            active_plan_selected = self._select_policy_from_active_plan(
                policies=policies,
                context=context,
            )
            if active_plan_selected is not None:
                return active_plan_selected
        planned_policy_id: Optional[str] = None
        if skill_plan and not emergency_override:
            raw_head = skill_plan.get("head_policy_id")
            if isinstance(raw_head, str) and raw_head.strip():
                planned_policy_id = raw_head.strip()

            # When the skill plan is exhausted (head=None, remaining=0) but
            # metadata is available, cycle through the curated phases round-robin.
            # This keeps the agent attempting the goal-relevant action sequence
            # rather than drifting to a random or cached policy while waiting
            # for LLM deliberation — consistent with Seth's continuous active
            # inference under uncertainty.
            if planned_policy_id is None:
                meta = skill_plan.get("metadata")
                if isinstance(meta, list) and meta:
                    step = context.get("step", 0) or 0
                    phase = meta[int(step) % len(meta)]
                    phase_pid = phase.get("selected_policy_id")
                    if isinstance(phase_pid, str) and phase_pid.strip():
                        planned_policy_id = phase_pid.strip()

        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        if not drive_signals:
            if planned_policy_id:
                planned = self._find_policy_by_id(policies, planned_policy_id)
                if planned is not None:
                    return planned
            # No drive urgency and no skill plan: attempt goal-directed selection
            # before falling back to the first policy in the list.
            if goals:
                goal_selected = self._goal_directed_fallback(policies, goals, context)
                if goal_selected is not None:
                    return goal_selected
            return policies[0]

        urgency_by_channel: Dict[str, float] = {}
        for signal in drive_signals:
            channel_id = signal.get("channel_id")
            urgency_raw = signal.get("urgency")
            if not isinstance(channel_id, str):
                continue
            if not isinstance(urgency_raw, (int, float)):
                continue
            channel_key = channel_id.strip().lower()
            if not channel_key:
                continue
            urgency = self._clamp01(float(urgency_raw))
            urgency_by_channel[channel_key] = max(urgency, urgency_by_channel.get(channel_key, 0.0))

        if not urgency_by_channel:
            if planned_policy_id:
                planned = self._find_policy_by_id(policies, planned_policy_id)
                if planned is not None:
                    return planned
            if goals:
                goal_selected = self._goal_directed_fallback(policies, goals, context)
                if goal_selected is not None:
                    return goal_selected
            return policies[0]

        best_index = 0
        best_score = -1.0
        for index, policy in enumerate(policies):
            policy_id = str(policy.get("policy_id") or "").strip()
            drive_tags = self._normalize_tags(policy.get("drive_tags"))
            score = 0.0
            for tag in drive_tags:
                score = max(score, urgency_by_channel.get(tag, 0.0))
            if planned_policy_id and policy_id == planned_policy_id:
                score += self.skill_plan_bias
            if score > best_score:
                best_score = score
                best_index = index

        # When no drive signal matches any policy tag (all scores stayed at 0),
        # attempt goal-directed selection rather than silently picking no_op.
        # This is the homeostatic equilibrium → goal pursuit transition in Seth's
        # framework: a calm body is free to pursue allostatic goals.
        if best_score <= 0.0 and goals:
            goal_selected = self._goal_directed_fallback(policies, goals, context)
            if goal_selected is not None:
                return goal_selected

        return policies[best_index]

    # Common English stop-words and generic terms that appear in almost every
    # policy description and goal string — matches on these are meaningless.
    _GOAL_STOP_WORDS: frozenset = frozenset({
        "a", "an", "the", "and", "or", "of", "in", "to", "by", "with",
        "for", "on", "at", "is", "it", "as", "be", "are", "was", "were",
        "has", "have", "had", "will", "can", "do", "not", "this", "that",
        "from", "up", "out", "into", "its", "their", "you", "your",
        "action", "space", "primitive", "minedojo",
        "s", "t", "d",
    })

    def _goal_directed_fallback(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select a policy via goal-keyword overlap when drive urgency is zero.

        Token sources in descending quality:
        1. Skill plan intent_tokens per phase (adapter-curated, no noise).
        2. Goal ``goal`` / ``task_id`` fields (compact, low noise).
        3. Goal ``description`` (verbose, filtered through stop-words).
        """
        goal_tokens: set = set()

        # Priority 1: skill plan intent tokens (curated by adapter)
        if isinstance(context, Mapping):
            skill_plan_raw = context.get("skill_plan")
            if isinstance(skill_plan_raw, Mapping):
                for phase in skill_plan_raw.get("metadata", []):
                    if not isinstance(phase, Mapping):
                        continue
                    for tok in phase.get("intent_tokens", []):
                        if isinstance(tok, str) and tok.strip():
                            goal_tokens.add(tok.strip().lower())

        # Priority 2: goal struct compact fields
        goals_list = goals if isinstance(goals, list) else [goals]
        for goal in goals_list:
            if isinstance(goal, Mapping):
                for field in ("goal", "task_id"):
                    text = goal.get(field)
                    if isinstance(text, str):
                        goal_tokens.update(self._name_tokens(text))
                if not goal_tokens:
                    desc = goal.get("description")
                    if isinstance(desc, str):
                        goal_tokens.update(self._name_tokens(desc))
            elif isinstance(goal, str):
                goal_tokens.update(self._name_tokens(goal))

        goal_tokens -= self._GOAL_STOP_WORDS
        if not goal_tokens:
            return None

        best_policy: Optional[Dict[str, Any]] = None
        best_overlap = 0
        for policy in policies:
            pid = str(policy.get("policy_id") or "").strip()
            # Never select no_op as a goal-directed action.
            if pid.endswith(":no_op") or pid == "no_op":
                continue
            policy_tokens: set = set()
            pid_tokens = set(self._name_tokens(pid)) - self._GOAL_STOP_WORDS
            policy_tokens.update(pid_tokens)
            drive_tags = self._normalize_tags(policy.get("drive_tags"))
            policy_tokens.update(t for t in drive_tags if t not in self._GOAL_STOP_WORDS)
            if not policy_tokens:
                desc = policy.get("description")
                if isinstance(desc, str):
                    policy_tokens.update(
                        t for t in self._name_tokens(desc) if t not in self._GOAL_STOP_WORDS
                    )
            overlap = len(goal_tokens & policy_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_policy = policy

        return best_policy if best_overlap > 0 else None

    def _write_arbitration_trace(
        self,
        selected_policy: Dict[str, Any],
        rationale: str,
        context: Mapping[str, Any],
    ) -> None:
        self.memory_manager.record_policy_trace(
            {
                "policy_id": selected_policy["policy_id"],
                "operation": "arbitrate",
                "status": "selected",
                "rationale": rationale,
                "step": context.get("step"),
                "ts": _utc_ts(),
            }
        )

    def _write_arbitration_fallback_trace(
        self,
        selected_policy: Optional[Mapping[str, Any]],
        rationale: str,
        context: Mapping[str, Any],
        status: str,
    ) -> None:
        policy_id: Optional[str] = None
        if isinstance(selected_policy, Mapping):
            raw_policy_id = selected_policy.get("policy_id")
            if raw_policy_id is not None:
                policy_id = str(raw_policy_id)
        self.memory_manager.record_policy_trace(
            {
                "policy_id": policy_id,
                "operation": "arbitrate",
                "status": str(status),
                "rationale": rationale,
                "step": context.get("step"),
                "ts": _utc_ts(),
            }
        )

    @staticmethod
    def _score_from_result(
        result: Any,
        key: str,
        fallback: float,
    ) -> float:
        if isinstance(result, Mapping):
            raw = result.get(key)
            if isinstance(raw, (int, float)):
                return max(0.0, min(1.0, float(raw)))
        return fallback

    def _policy_tokens(self, policy: Mapping[str, Any]) -> set[str]:
        sources: List[str] = []
        for key in ("policy_id", "callable_name", "description"):
            value = policy.get(key)
            if value is not None:
                sources.append(str(value))

        for key in ("tags", "drive_tags", "survival_domains"):
            values = policy.get(key)
            if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
                for value in values:
                    sources.append(str(value))

        tokens: set[str] = set()
        for source in sources:
            tokens.update(self._name_tokens(source))
        return tokens

    def _allostatic_tokens(self, allostatic_assessment: Mapping[str, Any]) -> set[str]:
        tokens: set[str] = set()

        policy_tags = allostatic_assessment.get("policy_bias_tags")
        if isinstance(policy_tags, Iterable) and not isinstance(policy_tags, (str, bytes)):
            for tag in policy_tags:
                tokens.update(self._name_tokens(str(tag)))

        needs = allostatic_assessment.get("needs")
        if isinstance(needs, list):
            for need in needs:
                if not isinstance(need, Mapping):
                    continue
                tokens.update(self._need_tokens(need))

        return tokens

    def _need_tokens(self, need: Mapping[str, Any]) -> set[str]:
        sources: List[str] = [str(need.get("need_id") or "")]
        for key in ("actions", "resources"):
            values = need.get(key)
            if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
                for value in values:
                    sources.append(str(value))

        tokens: set[str] = set()
        for source in sources:
            tokens.update(self._name_tokens(source))
        return tokens

    def _resolve_drive_signals(self, drive_signals: Any) -> List[Dict[str, Any]]:
        if drive_signals is None:
            return []

        if isinstance(drive_signals, Mapping):
            raw_signals = drive_signals.get("signals")
        elif hasattr(drive_signals, "signals"):
            raw_signals = getattr(drive_signals, "signals")
        else:
            raw_signals = None

        if not isinstance(raw_signals, list):
            return []

        out: List[Dict[str, Any]] = []
        for item in raw_signals:
            if isinstance(item, Mapping):
                out.append(dict(item))
                continue
            if hasattr(item, "__dict__"):
                out.append(dict(vars(item)))
        return out

    @staticmethod
    def _resolve_skill_plan(skill_plan: Any) -> Dict[str, Any]:
        if isinstance(skill_plan, Mapping):
            return dict(skill_plan)
        if hasattr(skill_plan, "__dict__"):
            mapped = vars(skill_plan)
            if isinstance(mapped, Mapping):
                return dict(mapped)
        return {}

    @staticmethod
    def _find_policy_by_id(
        policies: List[Dict[str, Any]],
        policy_id: str,
    ) -> Optional[Dict[str, Any]]:
        for policy in policies:
            candidate = str(policy.get("policy_id") or "").strip()
            if candidate and candidate == policy_id:
                return policy
        return None

    def _has_irreversible_emergency(self, allostatic_assessment: Any) -> bool:
        if not isinstance(allostatic_assessment, Mapping):
            return False
        needs = allostatic_assessment.get("needs")
        if not isinstance(needs, list):
            return False
        for need in needs:
            if not isinstance(need, Mapping):
                continue
            if need.get("irreversible") is not True:
                continue
            urgency = need.get("urgency")
            if isinstance(urgency, (int, float)) and self._clamp01(float(urgency)) >= self.skill_plan_emergency_threshold:
                return True
        return False

    def _infer_drive_tags(self, tags: List[str], descriptor_texts: List[str]) -> List[str]:
        candidates: List[str] = []
        seen = set()
        for token in tags:
            normalized = str(token).strip().lower()
            if normalized and normalized not in seen:
                candidates.append(normalized)
                seen.add(normalized)
        for text in descriptor_texts:
            for token in self._name_tokens(text):
                if token and token not in seen:
                    candidates.append(token)
                    seen.add(token)
        return candidates

    @staticmethod
    def _normalize_tags(raw: Any) -> List[str]:
        if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
            return []
        tags: List[str] = []
        seen = set()
        for value in raw:
            tag = str(value).strip().lower()
            if not tag or tag in seen:
                continue
            tags.append(tag)
            seen.add(tag)
        return tags

    @staticmethod
    def _name_tokens(name: str) -> List[str]:
        tokens = [token.lower() for token in re.split(r"[_\W]+", name) if token]
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _serialize_for_prompt(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        except Exception:
            return str(value)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _coerce_index(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return None
        return None

    def _extract_rationale(self, text: str) -> str:
        cleaned = self._strip_markdown_fences(text)
        try:
            payload = json.loads(cleaned)
            rationale = payload.get("rationale") if isinstance(payload, Mapping) else None
            if isinstance(rationale, str):
                return rationale
        except Exception:
            pass
        return cleaned

    def _parse_full_response(self, text: str) -> Dict[str, Any]:
        """Parse all fields from the LLM JSON response without raising."""
        try:
            payload = json.loads(self._strip_markdown_fences(text))
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:
            pass
        return {}
