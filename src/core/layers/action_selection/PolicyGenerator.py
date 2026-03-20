from __future__ import annotations

from datetime import datetime
import inspect
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.llm.client import LLMClient
from core.llm.types import LLMMessage, LLMRequest
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
        model_value = self.config.get("model", "claude-sonnet-4-20250514")
        self.model = str(model_value).strip() or "claude-sonnet-4-20250514"

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
            selected = self._urgency_fallback(policies, context)
            self._write_arbitration_fallback_trace(
                selected_policy=selected,
                rationale="urgency_fallback",
                context=context,
                status="fallback",
            )
            return selected

        system_prompt, user_prompt = self._build_arbitration_prompt(
            policies=policies,
            goals=goals,
            context=context,
        )

        step = context.get("step")
        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        allostatic_assessment = context.get("allostatic_assessment")

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
            selected = self._urgency_fallback(policies, context)
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
                    },
                    step=step,
                )
            return selected

        if self.logger is not None:
            self.logger.llm_response(
                {"raw": response_text},
                step=step,
            )

        selected = self._parse_arbitration_response(response_text, policies)
        if selected is None:
            fallback = self._urgency_fallback(policies, context)
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
                    },
                    step=step,
                )
            return fallback

        parsed_response = self._parse_full_response(response_text)
        rationale = parsed_response.get("rationale", self._extract_rationale(response_text))
        reasoning = parsed_response.get("reasoning", "")
        drive_conflict = parsed_response.get("drive_conflict_detected", False)
        confidence = parsed_response.get("confidence")

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
                    "candidate_policy_ids": [p.get("policy_id") for p in policies],
                },
                step=step,
            )

        return selected

    def _build_arbitration_prompt(
        self,
        policies: List[Dict[str, Any]],
        goals: Any,
        context: Mapping[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are an active inference arbitration system under the free energy principle.\n"
            "Select the action that minimizes expected free energy by reducing homeostatic deficits while managing epistemic uncertainty.\n"
            "Reason only from the provided drive states and policy descriptors.\n"
            "Do not use prior knowledge about what any action physically does in the world.\n"
            "Output only valid JSON with this schema:\n"
            '{"reasoning": "<step-by-step chain of thought: identify which drives are most urgent, '
            "explain any conflicts between drives, evaluate each candidate policy against the drive state, "
            'and justify your final selection>", '
            '"selected_index": <int>, '
            '"rationale": "<one-sentence summary of the decision>", '
            '"drive_conflict_detected": <bool>, '
            '"confidence": <float 0-1>}'
        )

        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        allostatic_assessment = context.get("allostatic_assessment")

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

        recent_traces: List[Any] = []
        recent_self_state: List[Any] = []
        query = getattr(self.memory_manager, "query", None)
        if callable(query):
            try:
                traces = query({"target": "policy_traces", "limit": 5})
                if isinstance(traces, list):
                    recent_traces = traces[-5:]
            except Exception:
                recent_traces = []
            try:
                self_state = query({"target": "self_state", "limit": 3})
                if isinstance(self_state, list):
                    recent_self_state = self_state[-3:]
            except Exception:
                recent_self_state = []

        memory_lines: List[str] = [
            "MEMORY CONTEXT",
            "Recent arbitration history:",
        ]
        if recent_traces:
            for index, trace in enumerate(recent_traces):
                memory_lines.append(f"trace[{index}]: {self._serialize_for_prompt(trace)}")
        else:
            memory_lines.append("none")

        memory_lines.append("Known agent capabilities:")
        if recent_self_state:
            for index, snapshot in enumerate(recent_self_state):
                memory_lines.append(
                    f"self_state[{index}]: {self._serialize_for_prompt(snapshot)}"
                )
        else:
            memory_lines.append("none")

        instruction = (
            "INSTRUCTION\n"
            f"Reason over expected free energy across the next {self.horizon} steps. "
            "Return JSON only."
        )

        user_prompt = "\n\n".join(
            [
                "\n".join(drive_lines),
                "\n".join(policy_lines),
                "\n".join(memory_lines),
                instruction,
            ]
        )
        return system_prompt, user_prompt

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
    ) -> Optional[Dict[str, Any]]:
        if not policies:
            return None

        drive_signals = self._resolve_drive_signals(context.get("drive_signals"))
        if not drive_signals:
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
            return policies[0]

        best_index = 0
        best_score = -1.0
        for index, policy in enumerate(policies):
            drive_tags = self._normalize_tags(policy.get("drive_tags"))
            score = 0.0
            for tag in drive_tags:
                score = max(score, urgency_by_channel.get(tag, 0.0))
            if score > best_score:
                best_score = score
                best_index = index
        return policies[best_index]

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
