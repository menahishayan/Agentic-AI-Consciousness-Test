from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.llm.types import LLMMessage, LLMRequest


class AllostaticController:
    def __init__(
        self,
        llm_client: Any = None,
        config: Optional[Mapping[str, Any]] = None,
        logger: Any = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = dict(config or {})
        self.logger = logger

        self.enabled = self._as_bool(self.config.get("enabled"), True)
        self.llm_interval_steps = max(1, self._as_int(self.config.get("llm_interval_steps"), 5))
        self.llm_timeout_s = max(0.1, self._as_float(self.config.get("llm_timeout_s"), 1.0))
        self.max_voxel_chars = max(128, self._as_int(self.config.get("max_voxel_chars"), 2000))

        self.life_drop_threshold = max(1.0, self._as_float(self.config.get("life_drop_threshold"), 2.0))
        self.food_drop_threshold = max(1.0, self._as_float(self.config.get("food_drop_threshold"), 2.0))
        self.air_drop_threshold = max(1.0, self._as_float(self.config.get("air_drop_threshold"), 20.0))
        self.high_risk_trigger = self._clamp01(
            self._as_float(self.config.get("high_risk_trigger"), 0.7)
        )

        self.default_goal_horizon_steps = max(
            1,
            self._as_int(self.config.get("default_goal_horizon_steps"), 160),
        )
        self.heuristic_confidence = self._clamp01(
            self._as_float(self.config.get("heuristic_confidence"), 0.65)
        )

        self.request_model = self._as_optional_str(self.config.get("model"))
        self.request_temperature = self._as_float(self.config.get("temperature"), 0.1)
        self.request_max_tokens = max(128, self._as_int(self.config.get("max_tokens"), 500))

        self._last_assessment: Optional[Dict[str, Any]] = None
        self._last_refresh_step: Optional[int] = None
        self._last_refresh_vitals: Dict[str, Any] = {}

    def assess(
        self,
        step: Any,
        state: Any,
        goals: Any,
        vital_state: Any,
        obs: Any = None,
        info: Any = None,
    ) -> Dict[str, Any]:
        step_int = self._as_int(step, 0)
        state_data = self._extract_state_data(state)
        vital_payload = self._normalize_vital_payload(vital_state)
        vitals = vital_payload.get("state", {})
        prompt_payload = self._build_prompt_payload(
            step=step_int,
            state_data=state_data,
            goals=goals,
            vital_payload=vital_payload,
            obs=obs,
            info=info,
        )
        input_digest = self._build_input_digest(
            step=step_int,
            state_data=state_data,
            goals=goals,
            vital_payload=vital_payload,
            prompt_payload=prompt_payload,
        )

        if not self.enabled:
            disabled = self._heuristic_assessment(
                step=step_int,
                goals=goals,
                state_data=state_data,
                vitals=vitals,
                input_digest=input_digest,
                rationale="Allostatic controller disabled; using heuristic baseline.",
            )
            self._cache_assessment(step_int, vitals, disabled)
            return disabled

        if not self._needs_refresh(step=step_int, vitals=vitals):
            cached = deepcopy(self._last_assessment) if self._last_assessment is not None else {}
            cached["step"] = step_int
            cached["source"] = "cache"
            cached["input_digest"] = input_digest
            return self._ensure_assessment_shape(cached, fallback=None)

        if self.llm_client is None:
            heuristic = self._heuristic_assessment(
                step=step_int,
                goals=goals,
                state_data=state_data,
                vitals=vitals,
                input_digest=input_digest,
                rationale="LLM unavailable; using heuristic allostatic estimate.",
            )
            self._cache_assessment(step_int, vitals, heuristic)
            return heuristic

        try:
            llm_assessment = self._assess_with_llm(
                step=step_int,
                goals=goals,
                state_data=state_data,
                prompt_payload=prompt_payload,
                vitals=vitals,
                input_digest=input_digest,
            )
            self._cache_assessment(step_int, vitals, llm_assessment)
            return llm_assessment
        except Exception as exc:
            if self.logger is not None:
                self.logger.event(
                    "allostatic.llm_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "step": step_int,
                    },
                    step=step_int,
                )
            heuristic = self._heuristic_assessment(
                step=step_int,
                goals=goals,
                state_data=state_data,
                vitals=vitals,
                input_digest=input_digest,
                rationale=f"LLM failed ({type(exc).__name__}); using heuristic fallback.",
            )
            self._cache_assessment(step_int, vitals, heuristic)
            return heuristic

    def _assess_with_llm(
        self,
        step: int,
        goals: Any,
        state_data: Mapping[str, Any],
        prompt_payload: Mapping[str, Any],
        vitals: Mapping[str, Any],
        input_digest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        prompt_json = json.dumps(prompt_payload, ensure_ascii=True, separators=(",", ":"))
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are an allostatic controller for a MineDojo agent. "
                    "Return exactly one JSON object and no surrounding text. "
                    "Do not include chain-of-thought, hidden reasoning traces, or step-by-step reasoning. "
                    "Return keys: survival_horizon_steps, risk_level, confidence, rationale_summary, "
                    "needs (list), policy_bias_tags (list). "
                    "Each need object must contain: need_id, urgency, time_to_critical_steps, actions, resources, evidence."
                ),
            ),
            LLMMessage(role="user", content=prompt_json),
        ]
        request = LLMRequest(
            messages=messages,
            model=self.request_model,
            temperature=self.request_temperature,
            max_tokens=self.request_max_tokens,
            metadata={
                "purpose": "allostatic_assessment",
                "timeout_s": self.llm_timeout_s,
            },
        )
        response = self.llm_client.generate(request)
        parsed = self._extract_first_json_object(getattr(response, "text", ""))
        if parsed is None:
            raise ValueError("LLM allostatic response did not contain a valid JSON object.")
        payload = json.loads(parsed)
        if not isinstance(payload, Mapping):
            raise TypeError("LLM allostatic response must be a JSON object.")

        baseline = self._heuristic_assessment(
            step=step,
            goals=goals,
            state_data=state_data,
            vitals=vitals,
            input_digest=input_digest,
            rationale="LLM response incomplete; baseline values applied where needed.",
        )
        merged = self._normalize_llm_payload(payload=payload, step=step, baseline=baseline)
        merged["source"] = "llm"
        merged["input_digest"] = dict(input_digest)
        return merged

    def _normalize_llm_payload(
        self,
        payload: Mapping[str, Any],
        step: int,
        baseline: Mapping[str, Any],
    ) -> Dict[str, Any]:
        out = dict(baseline)
        out["step"] = step
        out["survival_horizon_steps"] = max(
            1,
            self._as_int(payload.get("survival_horizon_steps"), self._as_int(baseline.get("survival_horizon_steps"), 1)),
        )
        out["risk_level"] = self._clamp01(
            self._as_float(payload.get("risk_level"), self._as_float(baseline.get("risk_level"), 0.0))
        )
        out["confidence"] = self._clamp01(
            self._as_float(payload.get("confidence"), self._as_float(baseline.get("confidence"), 0.0))
        )
        rationale = payload.get("rationale_summary")
        if isinstance(rationale, str) and rationale.strip():
            out["rationale_summary"] = rationale.strip()

        needs = self._normalize_needs(payload.get("needs"), fallback=baseline.get("needs"))
        out["needs"] = needs
        tags = self._normalize_tags(payload.get("policy_bias_tags"))
        if not tags:
            tags = self._derive_policy_bias_tags(needs)
        out["policy_bias_tags"] = tags
        return self._ensure_assessment_shape(out, fallback=baseline)

    def _heuristic_assessment(
        self,
        step: int,
        goals: Any,
        state_data: Mapping[str, Any],
        vitals: Mapping[str, Any],
        input_digest: Mapping[str, Any],
        rationale: str,
    ) -> Dict[str, Any]:
        life = self._as_optional_float(vitals.get("life"))
        food = self._as_optional_float(vitals.get("food"))
        air = self._as_optional_float(vitals.get("air"))
        is_alive = vitals.get("is_alive")
        is_dead = vitals.get("is_dead")

        needs: List[Dict[str, Any]] = []

        life_urgency = 0.0
        life_ttc: Optional[int] = None
        if life is not None:
            life_urgency = self._clamp01((20.0 - life) / 20.0)
            life_ttc = max(1, int(max(0.0, life) * 20.0))
            if life_urgency >= 0.20:
                needs.append(
                    {
                        "need_id": "preserve_health",
                        "urgency": life_urgency,
                        "time_to_critical_steps": life_ttc,
                        "actions": ["retreat", "eat", "use"],
                        "resources": ["food", "cover"],
                        "evidence": [f"life={life:.2f}"],
                    }
                )

        food_urgency = 0.0
        food_ttc: Optional[int] = None
        if food is not None:
            food_urgency = self._clamp01((20.0 - food) / 20.0)
            food_ttc = max(1, int(max(0.0, food) * 60.0))
            if food_urgency >= 0.20:
                needs.append(
                    {
                        "need_id": "restore_calories",
                        "urgency": food_urgency,
                        "time_to_critical_steps": food_ttc,
                        "actions": ["collect", "use", "eat"],
                        "resources": ["food", "animals", "crops"],
                        "evidence": [f"food={food:.2f}"],
                    }
                )

        air_urgency = 0.0
        air_ttc: Optional[int] = None
        if air is not None:
            max_air = max(1.0, self._as_float(self.config.get("air_capacity"), 300.0))
            air_urgency = self._clamp01((max_air - air) / max_air)
            air_ttc = max(1, int(max(0.0, air)))
            if air_urgency >= 0.15:
                needs.append(
                    {
                        "need_id": "restore_oxygen",
                        "urgency": air_urgency,
                        "time_to_critical_steps": air_ttc,
                        "actions": ["ascend", "move", "jump"],
                        "resources": ["air", "surface", "air_pocket"],
                        "evidence": [f"air={air:.2f}"],
                    }
                )

        light_level = self._as_optional_float(
            self._field(state_data.get("lighting_weather"), "light_level")
        )
        can_see_sky = self._field(state_data.get("lighting_weather"), "can_see_sky")
        dark_exposure = 0.0
        if isinstance(light_level, float) and light_level < 7.0:
            dark_exposure = self._clamp01((7.0 - light_level) / 7.0)
        if dark_exposure > 0.0 and can_see_sky:
            needs.append(
                {
                    "need_id": "seek_shelter",
                    "urgency": min(0.75, 0.35 + dark_exposure * 0.4),
                    "time_to_critical_steps": None,
                    "actions": ["move", "place", "craft"],
                    "resources": ["shelter", "light_source"],
                    "evidence": [f"light_level={light_level:.2f}" if light_level is not None else "low_light"],
                }
            )

        dead_risk = 0.0
        if is_dead is True or is_alive is False:
            dead_risk = 1.0
            needs.append(
                {
                    "need_id": "critical_recovery",
                    "urgency": 1.0,
                    "time_to_critical_steps": 1,
                    "actions": ["stop_risk", "retreat"],
                    "resources": ["safety"],
                    "evidence": [f"is_dead={bool(is_dead)}", f"is_alive={bool(is_alive)}"],
                }
            )

        goal_count = len(self._normalize_goal_items(goals))
        goal_horizon = max(self.default_goal_horizon_steps, goal_count * 80)
        critical_values = [value for value in (life_ttc, food_ttc, air_ttc) if isinstance(value, int)]
        critical_ttc = min(critical_values) if critical_values else None

        horizon_risk = 0.0
        if critical_ttc is not None and goal_horizon > 0:
            horizon_risk = self._clamp01((float(goal_horizon) - float(critical_ttc)) / float(goal_horizon))

        risk_level = self._clamp01(max(life_urgency, food_urgency, air_urgency, horizon_risk, dead_risk))
        policy_tags = self._derive_policy_bias_tags(needs)

        return self._ensure_assessment_shape(
            {
                "step": step,
                "source": "heuristic",
                "survival_horizon_steps": max(1, goal_horizon),
                "risk_level": risk_level,
                "confidence": self.heuristic_confidence,
                "rationale_summary": rationale,
                "needs": needs,
                "policy_bias_tags": policy_tags,
                "input_digest": dict(input_digest),
            },
            fallback=None,
        )

    def _build_prompt_payload(
        self,
        step: int,
        state_data: Mapping[str, Any],
        goals: Any,
        vital_payload: Mapping[str, Any],
        obs: Any,
        info: Any,
    ) -> Dict[str, Any]:
        voxels = self._resolve_voxels(state_data=state_data, obs=obs, info=info)
        voxels_truncated = self._truncate_voxels(voxels)
        position = self._as_mapping(state_data.get("position"))
        return {
            "step": step,
            "vital_state_monitor": {
                "expected_vitals": list(vital_payload.get("expected_vitals", [])),
                "state": dict(self._as_mapping(vital_payload.get("state"))),
                "missing": list(vital_payload.get("missing", [])),
            },
            "position": position,
            "coordinates": {
                "xpos": position.get("xpos"),
                "ypos": position.get("ypos"),
                "zpos": position.get("zpos"),
                "pitch": position.get("pitch"),
                "yaw": position.get("yaw"),
            },
            "world_time": dict(self._as_mapping(state_data.get("world_time"))),
            "lighting_weather": dict(self._as_mapping(state_data.get("lighting_weather"))),
            "biome": dict(self._as_mapping(state_data.get("biome"))),
            "goals": self._normalize_goal_items(goals),
            "voxels_truncated": voxels_truncated,
        }

    def _build_input_digest(
        self,
        step: int,
        state_data: Mapping[str, Any],
        goals: Any,
        vital_payload: Mapping[str, Any],
        prompt_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        position = self._as_mapping(state_data.get("position"))
        voxels_payload = self._as_mapping(prompt_payload.get("voxels_truncated"))
        missing = vital_payload.get("missing")
        if not isinstance(missing, list):
            missing = []
        return {
            "step": step,
            "goal_count": len(self._normalize_goal_items(goals)),
            "missing_vitals": [str(item) for item in missing[:20]],
            "position_known": any(
                position.get(key) is not None for key in ("xpos", "ypos", "zpos")
            ),
            "has_voxels": bool(voxels_payload.get("text")),
            "voxel_chars": self._as_int(voxels_payload.get("chars"), 0),
        }

    def _extract_state_data(self, state: Any) -> Dict[str, Any]:
        if hasattr(state, "to_dict") and callable(getattr(state, "to_dict")):
            try:
                state_payload = state.to_dict(include_inventory=False, include_voxels=True)
            except Exception:
                state_payload = {}
        elif isinstance(state, Mapping):
            state_payload = dict(state)
        elif hasattr(state, "__dict__"):
            state_payload = vars(state)
        else:
            state_payload = {}
        return {
            "position": dict(self._as_mapping(state_payload.get("position"))),
            "world_time": dict(self._as_mapping(state_payload.get("world_time"))),
            "lighting_weather": dict(self._as_mapping(state_payload.get("lighting_weather"))),
            "biome": dict(self._as_mapping(state_payload.get("biome"))),
            "voxels": state_payload.get("voxels"),
        }

    def _resolve_voxels(self, state_data: Mapping[str, Any], obs: Any, info: Any) -> Any:
        if "voxels" in state_data and state_data.get("voxels") is not None:
            return state_data.get("voxels")
        if isinstance(info, Mapping) and info.get("voxels") is not None:
            return info.get("voxels")
        if isinstance(obs, Mapping) and obs.get("voxels") is not None:
            return obs.get("voxels")
        return None

    def _truncate_voxels(self, voxels: Any) -> Dict[str, Any]:
        if voxels is None:
            return {"shape": None, "chars": 0, "truncated": False, "text": ""}

        shape = None
        if hasattr(voxels, "shape"):
            try:
                shape = [int(item) for item in list(getattr(voxels, "shape"))]
            except Exception:
                shape = None

        text: str
        if isinstance(voxels, str):
            text = voxels
        else:
            try:
                text = json.dumps(voxels, ensure_ascii=True, default=self._json_default)
            except Exception:
                text = repr(voxels)

        truncated = len(text) > self.max_voxel_chars
        if truncated:
            text = text[: self.max_voxel_chars]
        return {
            "shape": shape,
            "chars": len(text),
            "truncated": truncated,
            "text": text,
        }

    def _needs_refresh(self, step: int, vitals: Mapping[str, Any]) -> bool:
        if self._last_assessment is None or self._last_refresh_step is None:
            return True

        if (step - self._last_refresh_step) >= self.llm_interval_steps:
            return True

        last_risk = self._as_float(self._last_assessment.get("risk_level"), 0.0)
        if last_risk >= self.high_risk_trigger:
            return True

        if vitals.get("is_dead") is True or vitals.get("is_alive") is False:
            return True

        if self._drop_exceeds(vitals, "life", self.life_drop_threshold):
            return True
        if self._drop_exceeds(vitals, "food", self.food_drop_threshold):
            return True
        if self._drop_exceeds(vitals, "air", self.air_drop_threshold):
            return True

        return False

    def _drop_exceeds(self, vitals: Mapping[str, Any], key: str, threshold: float) -> bool:
        if threshold <= 0:
            return False
        current = self._as_optional_float(vitals.get(key))
        previous = self._as_optional_float(self._last_refresh_vitals.get(key))
        if current is None or previous is None:
            return False
        return (previous - current) >= threshold

    def _cache_assessment(self, step: int, vitals: Mapping[str, Any], assessment: Mapping[str, Any]) -> None:
        self._last_assessment = deepcopy(dict(assessment))
        self._last_refresh_step = step
        self._last_refresh_vitals = dict(vitals)

    @staticmethod
    def _normalize_vital_payload(vital_state: Any) -> Dict[str, Any]:
        if isinstance(vital_state, Mapping):
            state = vital_state.get("state")
            if isinstance(state, Mapping):
                expected = vital_state.get("expected_vitals")
                missing = vital_state.get("missing")
                return {
                    "expected_vitals": list(expected) if isinstance(expected, list) else list(state.keys()),
                    "state": dict(state),
                    "missing": list(missing) if isinstance(missing, list) else [],
                }
            return {
                "expected_vitals": list(vital_state.keys()),
                "state": dict(vital_state),
                "missing": [],
            }
        return {"expected_vitals": [], "state": {}, "missing": []}

    @staticmethod
    def _normalize_goal_items(goals: Any) -> List[Dict[str, Any]]:
        if goals is None:
            return []
        if not isinstance(goals, list):
            goals = [goals]

        out: List[Dict[str, Any]] = []
        for item in goals:
            if isinstance(item, Mapping):
                description = item.get("description") or item.get("goal")
                priority = item.get("priority")
                if description is None:
                    continue
                out.append(
                    {
                        "description": str(description),
                        "priority": float(priority) if isinstance(priority, (int, float)) else 1.0,
                    }
                )
            elif hasattr(item, "description"):
                description = getattr(item, "description")
                priority = getattr(item, "priority", 1.0)
                if description is None:
                    continue
                out.append(
                    {
                        "description": str(description),
                        "priority": float(priority) if isinstance(priority, (int, float)) else 1.0,
                    }
                )
            else:
                out.append({"description": str(item), "priority": 1.0})
        return out

    def _normalize_needs(self, payload: Any, fallback: Any = None) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            payload = fallback if isinstance(fallback, list) else []

        out: List[Dict[str, Any]] = []
        for raw in payload:
            if not isinstance(raw, Mapping):
                continue
            need_id = str(raw.get("need_id") or "").strip()
            if not need_id:
                continue
            urgency = self._clamp01(self._as_float(raw.get("urgency"), 0.0))
            raw_ttc = raw.get("time_to_critical_steps")
            ttc: Optional[int]
            if raw_ttc is None:
                ttc = None
            else:
                ttc_value = self._as_int(raw_ttc, -1)
                ttc = None if ttc_value < 0 else ttc_value
            actions = self._normalize_tags(raw.get("actions"))
            resources = self._normalize_tags(raw.get("resources"))
            evidence = self._normalize_free_text_list(raw.get("evidence"))
            out.append(
                {
                    "need_id": need_id,
                    "urgency": urgency,
                    "time_to_critical_steps": ttc,
                    "actions": actions,
                    "resources": resources,
                    "evidence": evidence,
                }
            )
        return out

    def _derive_policy_bias_tags(self, needs: Iterable[Mapping[str, Any]]) -> List[str]:
        tags: List[str] = ["survival", "allostasis"]
        seen = set(tags)
        for need in needs:
            for token in self._tokenize(str(need.get("need_id") or "")):
                if token in seen:
                    continue
                tags.append(token)
                seen.add(token)
            for key in ("actions", "resources"):
                values = need.get(key)
                if not isinstance(values, list):
                    continue
                for value in values:
                    for token in self._tokenize(str(value)):
                        if token in seen:
                            continue
                        tags.append(token)
                        seen.add(token)
        return tags

    def _ensure_assessment_shape(
        self,
        payload: Mapping[str, Any],
        fallback: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        baseline = dict(fallback or {})
        out = dict(payload)

        out["step"] = self._as_int(out.get("step"), self._as_int(baseline.get("step"), 0))
        source = out.get("source")
        out["source"] = source if source in {"llm", "heuristic", "cache"} else "heuristic"
        out["survival_horizon_steps"] = max(
            1,
            self._as_int(
                out.get("survival_horizon_steps"),
                self._as_int(baseline.get("survival_horizon_steps"), self.default_goal_horizon_steps),
            ),
        )
        out["risk_level"] = self._clamp01(
            self._as_float(out.get("risk_level"), self._as_float(baseline.get("risk_level"), 0.0))
        )
        out["confidence"] = self._clamp01(
            self._as_float(out.get("confidence"), self._as_float(baseline.get("confidence"), self.heuristic_confidence))
        )
        rationale = out.get("rationale_summary")
        if not isinstance(rationale, str) or not rationale.strip():
            fallback_rationale = baseline.get("rationale_summary")
            if isinstance(fallback_rationale, str) and fallback_rationale.strip():
                rationale = fallback_rationale.strip()
            else:
                rationale = "No rationale summary available."
        out["rationale_summary"] = rationale

        out["needs"] = self._normalize_needs(out.get("needs"), fallback=baseline.get("needs"))
        out["policy_bias_tags"] = self._normalize_tags(out.get("policy_bias_tags"))
        if not out["policy_bias_tags"]:
            out["policy_bias_tags"] = self._derive_policy_bias_tags(out["needs"])

        input_digest = out.get("input_digest")
        if not isinstance(input_digest, Mapping):
            fallback_digest = baseline.get("input_digest")
            input_digest = fallback_digest if isinstance(fallback_digest, Mapping) else {}
        out["input_digest"] = dict(input_digest)
        return out

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None

        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        object_start = -1

        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char == "{":
                if depth == 0:
                    object_start = idx
                depth += 1
                continue
            if char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and object_start >= 0:
                    return text[object_start : idx + 1]
        return None

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return repr(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if hasattr(value, "__dict__"):
            return vars(value)
        return repr(value)

    @staticmethod
    def _tokenize(value: str) -> List[str]:
        return [token for token in re.split(r"[^a-zA-Z0-9_]+", value.lower()) if token]

    @staticmethod
    def _normalize_tags(values: Any) -> List[str]:
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return []
        tags: List[str] = []
        seen = set()
        for value in values:
            token = str(value).strip().lower()
            if not token or token in seen:
                continue
            tags.append(token)
            seen.add(token)
        return tags

    @staticmethod
    def _normalize_free_text_list(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for value in values:
            text = str(value).strip()
            if text:
                out.append(text)
        return out

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
    def _field(value: Any, key: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(key)
        if hasattr(value, key):
            return getattr(value, key)
        return None

    @staticmethod
    def _as_optional_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _as_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
