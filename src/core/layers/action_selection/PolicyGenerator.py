from __future__ import annotations

from datetime import datetime
import inspect
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
    ) -> None:
        self.adapter = adapter
        self.adapter_folder = adapter_folder
        self.memory_manager = memory_manager
        self.goal_checker = goal_checker
        self.prediction_error_calculator = prediction_error_calculator
        self.config = dict(config or {})
        self.logger = logger

        weights = self.config.get("weights", {})
        self.weight_goal = self._as_float(weights.get("goal_coherence"), 0.6)
        self.weight_prediction_error = self._as_float(weights.get("prediction_error"), 0.4)

        fallback_scores = self.config.get("fallback_scores", {})
        self.fallback_goal_score = self._as_float(fallback_scores.get("goal_coherence"), 0.5)
        self.fallback_prediction_error_score = self._as_float(
            fallback_scores.get("prediction_error"), 0.5
        )

        discovery = self.config.get("discovery", {})
        reserved = discovery.get("reserved_methods", ())
        self.reserved_methods = set(self._normalize_reserved_methods(reserved))

    def propose_action(self, goals: Any, context: Mapping[str, Any]) -> Optional[ActionProposal]:
        policies = self.discover_policies()
        if not policies:
            return None

        policy_records = self._policy_record_map()
        scored: List[Dict[str, Any]] = []
        for policy in policies:
            coherence_result = self.goal_checker.check(goals, policy, context)
            prediction_result = self.prediction_error_calculator.compute(
                policy["policy_id"],
                context=context,
                memory_manager=self.memory_manager,
            )
            coherence_score = self._score_from_result(
                coherence_result,
                "coherence_score",
                self.fallback_goal_score,
            )
            prediction_error_score = self._score_from_result(
                prediction_result,
                "prediction_error_score",
                self.fallback_prediction_error_score,
            )
            combined_score = (
                self.weight_goal * coherence_score
                + self.weight_prediction_error * (1.0 - prediction_error_score)
            )
            record = policy_records.get(policy["policy_id"], {})
            last_selected_at = str(record.get("last_selected_at") or "")
            scored.append(
                {
                    "policy": policy,
                    "combined_score": combined_score,
                    "last_selected_at": last_selected_at,
                    "coherence_result": coherence_result,
                    "prediction_result": prediction_result,
                    "coherence_score": coherence_score,
                    "prediction_error_score": prediction_error_score,
                }
            )

        scored.sort(
            key=lambda item: (
                -float(item["combined_score"]),
                item["last_selected_at"],
                item["policy"]["policy_id"],
            )
        )

        for candidate in scored:
            policy = candidate["policy"]
            try:
                action_payload = self._invoke_policy(policy, goals, context)
            except Exception as exc:
                self._record_policy_invoke_error(policy, context, exc)
                continue

            components = {
                "coherence_score": candidate["coherence_score"],
                "prediction_error_score": candidate["prediction_error_score"],
                "coherence_result": candidate["coherence_result"],
                "prediction_result": candidate["prediction_result"],
                "selected_at": _utc_ts(),
            }
            self.memory_manager.record_policy_selection(
                policy_id=policy["policy_id"],
                score=candidate["combined_score"],
                components=components,
                step=context.get("step"),
            )

            return ActionProposal(
                action_id=policy["policy_id"],
                action=action_payload,
                expected_outcome=policy.get("description"),
                cost=max(0.0, 1.0 - float(candidate["combined_score"])),
            )

        return None

    def discover_policies(self) -> List[Dict[str, Any]]:
        policies = self._discover_from_adapter_api()
        if not policies:
            policies = self._discover_from_public_callables()
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

    def _discover_from_public_callables(self) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for name in dir(self.adapter):
            if name.startswith("_"):
                continue
            if name in self.reserved_methods:
                continue
            try:
                member = getattr(self.adapter, name)
            except Exception:
                continue
            if not callable(member):
                continue
            descriptor = self._normalize_policy_descriptor(
                {"callable_name": name, "tags": self._name_tokens(name)},
                source="introspection",
            )
            if descriptor is not None:
                found.append(descriptor)
        return found

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

    def _policy_record_map(self) -> Dict[str, Dict[str, Any]]:
        records = self.memory_manager.get_policies(adapter_folder=self.adapter_folder)
        out: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                continue
            policy_id = record.get("policy_id")
            if not isinstance(policy_id, str):
                continue
            out[policy_id] = dict(record)
        return out

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

    @staticmethod
    def _normalize_reserved_methods(values: Any) -> Sequence[str]:
        default = (
            "reset",
            "step",
            "close",
            "sample_action",
            "get_available_vitals",
            "get_available_policies",
            "get_action_space_actions",
            "get_raw_observation",
        )
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return default

        out: List[str] = []
        seen = set()
        for value in values:
            name = str(value).strip()
            if not name or name in seen:
                continue
            out.append(name)
            seen.add(name)
        return out or list(default)

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
