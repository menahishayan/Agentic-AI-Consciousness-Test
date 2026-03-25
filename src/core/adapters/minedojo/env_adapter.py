from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.adapters.minedojo.task_profiles import (
    build_skill_plan_queue,
    drive_channels_for_task,
    estimate_inventory_score,
    normalize_task_id,
    task_goal_payload,
)

_ADAPTER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class MineDojoAdapter:
    _VITALS: Tuple[str, ...] = (
        "life",
        "armor",
        "food",
        "saturation",
        "xp",
        "air",
        "is_sleeping",
        "is_alive",
        "is_dead",
    )

    _ACTION_DIMENSION_NAMES: Dict[int, str] = {
        0: "move_forward_back",
        1: "move_left_right",
        2: "posture",
        3: "camera_pitch_bin",
        4: "camera_yaw_bin",
        5: "functional_action",
        6: "functional_arg",
        7: "inventory_slot",
    }

    _ACTION_VALUE_LABELS: Dict[int, Dict[int, str]] = {
        0: {0: "noop", 1: "forward", 2: "back"},
        1: {0: "noop", 1: "left", 2: "right"},
        2: {0: "noop", 1: "jump", 2: "sneak", 3: "sprint"},
        5: {
            0: "noop",
            1: "use",
            2: "drop",
            3: "attack",
            4: "craft_or_smelt",
            5: "equip",
            6: "place",
            7: "destroy",
        },
    }

    _DRIVE_TAG_RULES: Dict[str, set[str]] = {
        "health": {"heal", "retreat", "defend", "protect", "shield", "escape"},
        "hunger": {"eat", "collect", "cook", "food", "harvest", "fish", "drink"},
        "oxygen": {"surface", "ascend", "air", "breath"},
        "resource_level": {
            "gather",
            "mine",
            "craft",
            "smelt",
            "chop",
            "dig",
            "harvest",
            "interact",
            "bucket",
            "milk",
        },
        "safety": {"retreat", "avoid", "shelter", "defend", "escape"},
    }

    def __init__(self, env: Any, config: Optional[Mapping[str, Any]] = None) -> None:
        self.env = env
        self._config = dict(config or {})
        self._task_id = normalize_task_id(self._config.get("task_id", "harvest_milk"))
        self._task_goal = task_goal_payload(self._task_id)
        self._max_inventory_slots = max(
            1,
            self._as_int(self._config.get("max_inventory_slots"), 36),
        )
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}
        self._last_area_signature: Optional[str] = None
        self._nvec = self._extract_action_nvec()
        self._noop_action = self._build_noop_action()
        self._action_space_actions = self._build_action_space_actions()

        self._policies: List[Dict[str, Any]] = []
        self._voyager_skills: Dict[str, Dict[str, Any]] = {}
        self._skill_plan_queue: List[str] = []
        self._skill_plan_metadata: List[Dict[str, Any]] = []
        self._last_skill_plan_signature: Optional[str] = None
        self._skill_plan_exhausted = False
        self._policy_method_names = set(dir(self))
        self._register_action_space_policies()
        self._register_voyager_skill_policies()
        self._apply_task_policy_overrides()

    def reset(self) -> Tuple[Any, Any]:
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}
        self._skill_plan_queue = []
        self._skill_plan_metadata = []
        self._last_skill_plan_signature = None
        self._skill_plan_exhausted = False
        return obs, self._last_info

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any]:
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}
        return obs, reward, done, self._last_info

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def get_raw_observation(self) -> Any:
        return self._last_obs

    def sample_action(self) -> Any:
        action_space = getattr(self.env, "action_space", None)
        if action_space is None or not hasattr(action_space, "sample"):
            return None
        return action_space.sample()

    def get_available_vitals(self) -> List[str]:
        return list(self._VITALS)

    def get_action_space_actions(self) -> List[Dict[str, Any]]:
        return deepcopy(self._action_space_actions)

    def get_available_policies(self) -> List[Dict[str, Any]]:
        return deepcopy(self._policies)

    def get_task_goals(self) -> List[Dict[str, Any]]:
        return [deepcopy(self._task_goal)]

    def get_drive_channels(self) -> List[Any]:
        return drive_channels_for_task(self._task_id)

    def get_skill_plan(
        self,
        goals: Any = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = context
        signature = self._skill_plan_signature(goals)
        should_refresh = False
        if signature != self._last_skill_plan_signature:
            should_refresh = True
        elif self._last_skill_plan_signature is None:
            should_refresh = True
        elif not self._skill_plan_queue and not self._skill_plan_exhausted:
            should_refresh = True

        if should_refresh:
            queue, metadata = build_skill_plan_queue(
                task_id=self._task_id,
                policies=self._policies,
            )
            self._skill_plan_queue = list(queue)
            self._skill_plan_metadata = list(metadata)
            self._last_skill_plan_signature = signature
            self._skill_plan_exhausted = not bool(self._skill_plan_queue)

        return {
            "task_id": self._task_id,
            "head_policy_id": self._skill_plan_queue[0] if self._skill_plan_queue else None,
            "queue": list(self._skill_plan_queue),
            "remaining_policy_ids": list(self._skill_plan_queue),
            "remaining_count": len(self._skill_plan_queue),
            "metadata": deepcopy(self._skill_plan_metadata),
        }

    def notify_policy_selected(
        self,
        policy_id: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        _ = context
        selected = str(policy_id or "").strip()
        if not selected or not self._skill_plan_queue:
            return
        if selected != self._skill_plan_queue[0]:
            return
        self._skill_plan_queue.pop(0)
        if not self._skill_plan_queue:
            self._skill_plan_exhausted = True

    def estimate_resource_level(
        self,
        *,
        hunger: float,
        lighting: Mapping[str, Any],
        nearby: Mapping[str, Any],
        inventory_state: Mapping[str, Any],
        state: Any = None,
        info: Any = None,
        obs: Any = None,
    ) -> float:
        _ = state
        _ = info
        _ = obs
        _ = lighting
        _ = nearby
        hunger_score = self._clip01(hunger)
        inventory_score = estimate_inventory_score(
            inventory_state,
            max_slots=self._max_inventory_slots,
        )
        if inventory_score is None:
            return hunger_score
        return self._clip01(0.6 * float(inventory_score) + 0.4 * hunger_score)

    def estimate_threat_proximity(
        self,
        *,
        messages: List[Any],
        health: float,
        lighting: Mapping[str, Any],
        homeostasis: Mapping[str, Any],
        state: Any = None,
        info: Any = None,
        obs: Any = None,
    ) -> float:
        _ = state
        _ = info
        _ = obs
        threat = 0.0
        for message in messages:
            kind = str(getattr(message, "kind", "")).strip().lower()
            payload = getattr(message, "payload", None)
            if kind in {"threat", "threat_signal", "danger"}:
                if isinstance(payload, Mapping) and isinstance(payload.get("severity"), (int, float)):
                    threat = max(threat, self._clip01(float(payload.get("severity"))))
                else:
                    threat = max(threat, 0.60)
                continue
            if isinstance(payload, Mapping) and isinstance(payload.get("threat_proximity"), (int, float)):
                threat = max(threat, self._clip01(float(payload.get("threat_proximity"))))

        light_level = self._as_optional_float(lighting.get("light_level"))
        can_see_sky = lighting.get("can_see_sky")
        if light_level is not None and can_see_sky is True and light_level < 7.0:
            threat = max(threat, self._clip01((7.0 - light_level) / 7.0 * 0.6))
        threat = max(threat, self._clip01(1.0 - health))

        if homeostasis.get("is_dead") is True or homeostasis.get("is_alive") is False:
            threat = 1.0
        return self._clip01(threat)

    def build_area_id(
        self,
        *,
        state: Any = None,
        info: Any = None,
        obs: Any = None,
        step: Optional[int] = None,
    ) -> str:
        _ = obs
        _ = step
        info_map = info if isinstance(info, Mapping) else {}
        biome_name = str(info_map.get("biome_name") or "unknown").strip().lower() or "unknown"
        xpos = self._as_optional_float(info_map.get("xpos"))
        zpos = self._as_optional_float(info_map.get("zpos"))
        ypos = self._as_optional_float(info_map.get("ypos"))
        if xpos is None and state is not None and hasattr(state, "position"):
            xpos = self._as_optional_float(getattr(state.position, "xpos", None))
            zpos = self._as_optional_float(getattr(state.position, "zpos", None))
            ypos = self._as_optional_float(getattr(state.position, "ypos", None))
        chunk_x = int((xpos or 0.0) // 32.0)
        chunk_z = int((zpos or 0.0) // 32.0)
        height = int((ypos or 0.0) // 16.0)
        return f"{biome_name}:{chunk_x}:{chunk_z}:{height}"

    def estimate_entity_density(
        self,
        *,
        state: Any = None,
        info: Any = None,
        obs: Any = None,
    ) -> float:
        _ = state
        _ = obs
        info_map = info if isinstance(info, Mapping) else {}
        explicit = self._as_optional_float(info_map.get("entity_density"))
        if explicit is not None:
            return self._clip01(explicit)
        if info_map.get("damage_source") not in (None, "", []):
            return 0.8
        if info_map.get("nearby_furnace") is True or info_map.get("nearby_crafting_table") is True:
            return 0.2
        return 0.05

    def estimate_terrain_novelty(
        self,
        *,
        state: Any = None,
        info: Any = None,
        obs: Any = None,
    ) -> float:
        _ = state
        _ = obs
        info_map = info if isinstance(info, Mapping) else {}
        biome = str(info_map.get("biome_name") or "unknown").strip().lower() or "unknown"
        light = int(self._as_optional_float(info_map.get("light_level")) or 0.0)
        sky = "sky" if info_map.get("can_see_sky") is True else "nosky"
        signature = f"{biome}:{light}:{sky}"
        if self._last_area_signature is None:
            self._last_area_signature = signature
            return 0.25
        novelty = 0.1 if signature == self._last_area_signature else 0.75
        self._last_area_signature = signature
        return self._clip01(novelty)

    def _register_action_space_policies(self) -> None:
        if not self._nvec:
            return
        if not self._as_bool(self._config.get("include_action_space_policies"), True):
            return

        primitive_specs: List[Dict[str, Any]] = [
            {
                "name": "no_op",
                "overrides": {},
                "description": "No-op action from MineDojo action space.",
                "tags": ["action_space", "primitive", "noop"],
            }
        ]

        if len(self._nvec) > 0 and self._nvec[0] >= 3:
            primitive_specs.extend(
                [
                    {
                        "name": "move_forward",
                        "overrides": {0: 1},
                        "description": "Move forward.",
                        "tags": ["action_space", "primitive", "move", "forward"],
                    },
                    {
                        "name": "move_back",
                        "overrides": {0: 2},
                        "description": "Move backward.",
                        "tags": ["action_space", "primitive", "move", "back"],
                    },
                ]
            )

        if len(self._nvec) > 1 and self._nvec[1] >= 3:
            primitive_specs.extend(
                [
                    {
                        "name": "strafe_left",
                        "overrides": {1: 1},
                        "description": "Strafe left.",
                        "tags": ["action_space", "primitive", "move", "left"],
                    },
                    {
                        "name": "strafe_right",
                        "overrides": {1: 2},
                        "description": "Strafe right.",
                        "tags": ["action_space", "primitive", "move", "right"],
                    },
                ]
            )

        if len(self._nvec) > 2 and self._nvec[2] >= 4:
            primitive_specs.extend(
                [
                    {
                        "name": "jump",
                        "overrides": {2: 1},
                        "description": "Jump in place.",
                        "tags": ["action_space", "primitive", "jump"],
                    },
                    {
                        "name": "sneak",
                        "overrides": {2: 2},
                        "description": "Sneak posture.",
                        "tags": ["action_space", "primitive", "sneak"],
                    },
                    {
                        "name": "sprint",
                        "overrides": {2: 3},
                        "description": "Sprint posture.",
                        "tags": ["action_space", "primitive", "sprint"],
                    },
                ]
            )

        if len(self._nvec) > 3 and len(self._nvec) > 4:
            pitch = self._centered_camera_value(3)
            yaw = self._centered_camera_value(4)
            pitch_step = max(1, self._nvec[3] // 6) if self._nvec[3] > 1 else 0
            yaw_step = max(1, self._nvec[4] // 6) if self._nvec[4] > 1 else 0
            primitive_specs.extend(
                [
                    {
                        "name": "look_up",
                        "overrides": {3: max(0, pitch - pitch_step)},
                        "description": "Tilt camera up.",
                        "tags": ["action_space", "primitive", "camera", "up"],
                    },
                    {
                        "name": "look_down",
                        "overrides": {3: min(self._nvec[3] - 1, pitch + pitch_step)},
                        "description": "Tilt camera down.",
                        "tags": ["action_space", "primitive", "camera", "down"],
                    },
                    {
                        "name": "look_left",
                        "overrides": {4: max(0, yaw - yaw_step)},
                        "description": "Turn camera left.",
                        "tags": ["action_space", "primitive", "camera", "left"],
                    },
                    {
                        "name": "look_right",
                        "overrides": {4: min(self._nvec[4] - 1, yaw + yaw_step)},
                        "description": "Turn camera right.",
                        "tags": ["action_space", "primitive", "camera", "right"],
                    },
                ]
            )

        if len(self._nvec) > 5 and self._nvec[5] >= 8:
            primitive_specs.extend(
                [
                    {
                        "name": "use",
                        "overrides": {5: 1},
                        "description": "Use/interact functional action.",
                        "tags": ["action_space", "primitive", "use"],
                    },
                    {
                        "name": "drop",
                        "overrides": {5: 2},
                        "description": "Drop currently held item.",
                        "tags": ["action_space", "primitive", "drop"],
                    },
                    {
                        "name": "attack",
                        "overrides": {5: 3},
                        "description": "Attack functional action.",
                        "tags": ["action_space", "primitive", "attack"],
                    },
                    {
                        "name": "craft_or_smelt",
                        "overrides": {5: 4, 6: 0},
                        "description": "Craft/smelt action with default argument slot.",
                        "tags": ["action_space", "primitive", "craft", "smelt"],
                    },
                    {
                        "name": "equip_slot_zero",
                        "overrides": {5: 5, 7: 0},
                        "description": "Equip action targeting slot 0.",
                        "tags": ["action_space", "primitive", "equip"],
                    },
                    {
                        "name": "place_slot_zero",
                        "overrides": {5: 6, 7: 0},
                        "description": "Place action targeting slot 0.",
                        "tags": ["action_space", "primitive", "place"],
                    },
                    {
                        "name": "destroy_slot_zero",
                        "overrides": {5: 7, 7: 0},
                        "description": "Destroy action targeting slot 0.",
                        "tags": ["action_space", "primitive", "destroy"],
                    },
                ]
            )

        for spec in primitive_specs:
            overrides = dict(spec.get("overrides", {}))
            action_vector = self._vector_from_overrides(overrides)
            name = str(spec["name"])
            self._register_policy_method(
                method_base_name=f"policy_{name}",
                description=str(spec["description"]),
                tags=spec.get("tags", []),
                source="action_space",
                action_builder=lambda _this, _ctx, action_vector=action_vector: list(action_vector),
                metadata={
                    "policy_id": f"minedojo:action_space:{name}",
                    "kind": "primitive",
                    "action_overrides": overrides,
                },
            )

    def _register_voyager_skill_policies(self) -> None:
        if not self._as_bool(self._config.get("include_voyager_policies"), True):
            return
        skill_specs = self._load_voyager_skills()
        for spec in skill_specs:
            skill_id = str(spec["skill_id"])
            self._voyager_skills[skill_id] = spec
            skill_name = str(spec["skill_name"])
            description = str(spec.get("description") or f"Voyager skill '{skill_name}'.")
            tags = list(spec.get("tags", [])) + ["voyager", "skill"]
            trial = str(spec.get("trial") or "")
            self._register_policy_method(
                method_base_name=f"voyager_skill_{skill_name}",
                description=description,
                tags=tags,
                source="voyager_skill_library",
                action_builder=lambda this, ctx, skill_id=skill_id: this._execute_voyager_skill(
                    skill_id,
                    ctx,
                ),
                metadata={
                    "policy_id": f"minedojo:voyager:{skill_name}",
                    "kind": "voyager_skill",
                    "trial": trial,
                    "skill_name": skill_name,
                },
            )

    def _apply_task_policy_overrides(self) -> None:
        if self._task_id != "harvest_milk":
            return

        interaction_policy_found = False
        for descriptor in self._policies:
            if not isinstance(descriptor, dict):
                continue
            policy_id = str(descriptor.get("policy_id") or "").strip().lower()
            tags = self._normalize_tags(descriptor.get("tags"))
            drive_tags = self._normalize_tags(descriptor.get("drive_tags"))
            drive_tag_set = set(drive_tags)

            if policy_id.endswith(":attack"):
                drive_tag_set.discard("resource_level")

            if "use" in tags or "interact" in tags:
                drive_tag_set.add("resource_level")
                interaction_policy_found = True
            elif {"milk", "cow", "bucket", "harvest"}.intersection(tags):
                drive_tag_set.add("resource_level")
                interaction_policy_found = True

            descriptor["drive_tags"] = sorted(drive_tag_set)

        if interaction_policy_found:
            return

        for descriptor in self._policies:
            if not isinstance(descriptor, dict):
                continue
            policy_id = str(descriptor.get("policy_id") or "").strip().lower()
            if policy_id.endswith(":use"):
                drive_tags = set(self._normalize_tags(descriptor.get("drive_tags")))
                drive_tags.add("resource_level")
                descriptor["drive_tags"] = sorted(drive_tags)
                break

    def _execute_voyager_skill(self, skill_id: str, context: Optional[Mapping[str, Any]] = None) -> List[int]:
        _ = context
        spec = self._voyager_skills.get(skill_id, {})
        skill_name = str(spec.get("skill_name") or skill_id)
        tags = set(self._normalize_tags(spec.get("tags")))
        overrides: Dict[int, int] = {}

        if "explore" in tags and len(self._nvec) > 0 and self._nvec[0] > 1:
            overrides[0] = 1
            if len(self._nvec) > 2 and self._nvec[2] > 3:
                overrides[2] = 3

        if tags.intersection({"attack", "kill", "mine", "chop", "dig"}):
            if len(self._nvec) > 5 and self._nvec[5] > 3:
                overrides[5] = 3
            if len(self._nvec) > 0 and self._nvec[0] > 1 and 0 not in overrides:
                overrides[0] = 1
        elif tags.intersection({"craft", "smelt"}):
            if len(self._nvec) > 5 and self._nvec[5] > 4:
                overrides[5] = 4
            if len(self._nvec) > 6 and self._nvec[6] > 0:
                overrides[6] = self._stable_index(skill_name, self._nvec[6])
        elif "equip" in tags:
            if len(self._nvec) > 5 and self._nvec[5] > 5:
                overrides[5] = 5
            if len(self._nvec) > 7 and self._nvec[7] > 0:
                overrides[7] = self._stable_index(skill_name, self._nvec[7])
        elif "place" in tags:
            if len(self._nvec) > 5 and self._nvec[5] > 6:
                overrides[5] = 6
            if len(self._nvec) > 7 and self._nvec[7] > 0:
                overrides[7] = self._stable_index(skill_name, self._nvec[7])
        elif "destroy" in tags:
            if len(self._nvec) > 5 and self._nvec[5] > 7:
                overrides[5] = 7
            if len(self._nvec) > 7 and self._nvec[7] > 0:
                overrides[7] = self._stable_index(skill_name, self._nvec[7])
        elif "drop" in tags:
            if len(self._nvec) > 5 and self._nvec[5] > 2:
                overrides[5] = 2
        elif tags.intersection({"eat", "drink", "cook", "collect", "fill", "fish", "use"}):
            if len(self._nvec) > 5 and self._nvec[5] > 1:
                overrides[5] = 1
            if len(self._nvec) > 0 and self._nvec[0] > 1:
                overrides[0] = 1

        if not overrides:
            fallback = self._stable_index(skill_name, 4)
            if fallback == 0 and len(self._nvec) > 0 and self._nvec[0] > 1:
                overrides[0] = 1
            elif fallback == 1 and len(self._nvec) > 5 and self._nvec[5] > 1:
                overrides[5] = 1
            elif fallback == 2 and len(self._nvec) > 5 and self._nvec[5] > 3:
                overrides[5] = 3
            elif fallback == 3 and len(self._nvec) > 2 and self._nvec[2] > 1:
                overrides[2] = 1

        return self._vector_from_overrides(overrides)

    def _load_voyager_skills(self) -> List[Dict[str, Any]]:
        root_path = self._resolve_skill_library_root()
        if root_path is None:
            return []

        trials = self._resolve_trial_paths(root_path, self._config.get("skill_trials"))
        max_skills = self._as_int(self._config.get("max_voyager_skills"), 0)
        specs: Dict[str, Dict[str, Any]] = {}

        for trial_path in trials:
            skill_json_path = trial_path / "skill" / "skills.json"
            if not skill_json_path.exists():
                continue
            try:
                payload = json.loads(skill_json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue

            for skill_name, value in payload.items():
                key = str(skill_name).strip()
                if not key or key in specs:
                    continue
                record = value if isinstance(value, Mapping) else {}
                description = self._resolve_skill_description(
                    trial_path=trial_path,
                    skill_name=key,
                    skill_record=record,
                )
                tags = self._normalize_tags(
                    list(self._name_tokens(key)) + list(self._name_tokens(description))
                )
                specs[key] = {
                    "skill_id": f"{trial_path.name}:{key}",
                    "skill_name": key,
                    "trial": trial_path.name,
                    "description": description,
                    "tags": tags,
                }
                if max_skills > 0 and len(specs) >= max_skills:
                    break
            if max_skills > 0 and len(specs) >= max_skills:
                break

        return [specs[name] for name in sorted(specs.keys())]

    def _resolve_skill_library_root(self) -> Optional[Path]:
        configured = self._config.get("skill_library_dir")
        candidate_values: List[Path] = []

        if configured is not None:
            cfg_path = Path(str(configured))
            if cfg_path.is_absolute():
                candidate_values.append(cfg_path)
            else:
                candidate_values.append(_ADAPTER_DIR / cfg_path)
                candidate_values.append(_PROJECT_ROOT / cfg_path)
        else:
            candidate_values.extend(
                [
                    _ADAPTER_DIR / "skill_library",
                    _PROJECT_ROOT / "skill_library",
                    _PROJECT_ROOT / "Voyager" / "skill_library",
                ]
            )

        seen = set()
        for raw_path in candidate_values:
            key = str(raw_path)
            if key in seen:
                continue
            seen.add(key)
            try:
                resolved = raw_path.resolve(strict=False)
            except Exception:
                resolved = raw_path
            if resolved.exists() and resolved.is_dir() and resolved.name == "skill_library":
                return resolved

        return None

    def _resolve_skill_description(
        self,
        trial_path: Path,
        skill_name: str,
        skill_record: Mapping[str, Any],
    ) -> str:
        description_path = trial_path / "skill" / "description" / f"{skill_name}.txt"
        if description_path.exists():
            text = description_path.read_text(encoding="utf-8").strip()
            if text:
                return text

        raw_description = skill_record.get("description")
        if isinstance(raw_description, str):
            text = raw_description.strip()
            if text:
                if text.startswith("async function"):
                    first_comment = self._extract_comment_line(text)
                    if first_comment:
                        return first_comment
                    return f"Voyager skill '{skill_name}' from {trial_path.name}."
                return text

        raw_code = skill_record.get("code")
        if isinstance(raw_code, str):
            first_comment = self._extract_comment_line(raw_code)
            if first_comment:
                return first_comment

        return f"Voyager skill '{skill_name}' from {trial_path.name}."

    def _resolve_trial_paths(self, root_path: Path, selected_trials: Any) -> List[Path]:
        if isinstance(selected_trials, Sequence) and not isinstance(selected_trials, (str, bytes)):
            paths: List[Path] = []
            seen = set()
            for value in selected_trials:
                name = str(value).strip()
                if not name or name in seen:
                    continue
                path = root_path / name
                if path.exists() and path.is_dir():
                    paths.append(path)
                    seen.add(name)
            return paths

        trial_paths = [
            path
            for path in root_path.iterdir()
            if path.is_dir() and path.name.startswith("trial")
        ]
        return sorted(trial_paths, key=lambda path: path.name, reverse=True)

    def _register_policy_method(
        self,
        method_base_name: str,
        description: str,
        tags: Iterable[str],
        source: str,
        action_builder: Callable[[MineDojoAdapter, Mapping[str, Any]], Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        method_name = self._unique_method_name(method_base_name)

        def _policy_method(this: MineDojoAdapter, **kwargs: Any) -> Any:
            return action_builder(this, kwargs)

        _policy_method.__name__ = method_name
        setattr(self, method_name, MethodType(_policy_method, self))

        descriptor: Dict[str, Any] = {
            "callable_name": method_name,
            "source": source,
            "description": description,
            "tags": self._normalize_tags(tags),
        }
        descriptor["policy_id"] = str(metadata.get("policy_id")) if isinstance(metadata, Mapping) and metadata.get("policy_id") is not None else f"{source}:{method_name}"
        descriptor["drive_tags"] = self._infer_drive_tags(
            tags=descriptor["tags"],
            descriptor_texts=[method_name, description],
        )
        if metadata is not None:
            for key, value in metadata.items():
                if value is not None:
                    descriptor[str(key)] = value
        self._policies.append(descriptor)

    def _unique_method_name(self, base_name: str) -> str:
        candidate = self._sanitize_name(base_name)
        if not candidate:
            candidate = "policy"
        if candidate[0].isdigit():
            candidate = f"p_{candidate}"

        unique_name = candidate
        suffix = 2
        while unique_name in self._policy_method_names or hasattr(self, unique_name):
            unique_name = f"{candidate}_{suffix}"
            suffix += 1
        self._policy_method_names.add(unique_name)
        return unique_name

    def _vector_from_overrides(self, overrides: Mapping[int, int]) -> List[int]:
        vector = list(self._noop_action)
        for dim, value in overrides.items():
            idx = int(dim)
            if idx < 0 or idx >= len(vector):
                continue
            size = self._nvec[idx]
            if size <= 0:
                continue
            clipped = max(0, min(int(value), size - 1))
            vector[idx] = clipped
        return vector

    def _build_action_space_actions(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for idx, size in enumerate(self._nvec):
            labels = self._ACTION_VALUE_LABELS.get(idx, {})
            values: List[Dict[str, Any]] = []
            for value in range(size):
                label = labels.get(value)
                if label is None:
                    if idx in {3, 4}:
                        label = f"bin_{value}"
                    elif idx == 6:
                        label = f"craft_arg_{value}"
                    elif idx == 7:
                        label = f"slot_{value}"
                    else:
                        label = f"value_{value}"
                values.append({"value": value, "label": label})

            summary.append(
                {
                    "dimension": idx,
                    "name": self._ACTION_DIMENSION_NAMES.get(idx, f"dimension_{idx}"),
                    "size": size,
                    "values": values,
                }
            )
        return summary

    def _extract_action_nvec(self) -> List[int]:
        action_space = getattr(self.env, "action_space", None)
        nvec = getattr(action_space, "nvec", None)
        if nvec is None:
            return []
        try:
            return [int(value) for value in list(nvec)]
        except Exception:
            return []

    def _build_noop_action(self) -> List[int]:
        if self._nvec:
            action_space = getattr(self.env, "action_space", None)
            if action_space is not None and callable(getattr(action_space, "no_op", None)):
                try:
                    noop = action_space.no_op()
                    return [int(value) for value in list(noop)]
                except Exception:
                    pass
            return [0 for _ in self._nvec]
        return []

    def _centered_camera_value(self, dim: int) -> int:
        if dim < 0 or dim >= len(self._nvec):
            return 0
        if dim < len(self._noop_action):
            return int(self._noop_action[dim])
        return max(0, min(self._nvec[dim] - 1, self._nvec[dim] // 2))

    @staticmethod
    def _extract_comment_line(text: str) -> Optional[str]:
        match = re.search(r"//\s*(.+)", text)
        if match is None:
            return None
        value = match.group(1).strip()
        return value or None

    @staticmethod
    def _sanitize_name(value: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
        return sanitized.strip("_")

    @staticmethod
    def _name_tokens(text: str) -> List[str]:
        split = re.split(r"[^a-zA-Z0-9_]+", text)
        return [token.lower() for token in split if token]

    @staticmethod
    def _normalize_tags(values: Any) -> List[str]:
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return []
        tags: List[str] = []
        seen = set()
        for value in values:
            tag = str(value).strip().lower()
            if not tag or tag in seen:
                continue
            tags.append(tag)
            seen.add(tag)
        return tags

    def _infer_drive_tags(self, tags: List[str], descriptor_texts: List[str]) -> List[str]:
        tokens = set(tags)
        for text in descriptor_texts:
            tokens.update(self._name_tokens(text))
        inferred: List[str] = []
        for drive_tag, marker_tags in self._DRIVE_TAG_RULES.items():
            if drive_tag in tokens or marker_tags.intersection(tokens):
                inferred.append(drive_tag)
        return inferred

    @staticmethod
    def _stable_index(seed: str, modulo: int) -> int:
        if modulo <= 0:
            return 0
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % modulo

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
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _as_optional_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _skill_plan_signature(self, goals: Any) -> str:
        goal_tokens: List[str] = []
        if isinstance(goals, list):
            for goal in goals:
                if isinstance(goal, Mapping):
                    raw = goal.get("goal_id") or goal.get("description") or goal.get("goal")
                else:
                    raw = goal
                token = str(raw or "").strip().lower()
                if token:
                    goal_tokens.append(token)
        elif goals is not None:
            goal_tokens.append(str(goals).strip().lower())
        deduped = list(dict.fromkeys(sorted(goal_tokens)))
        return f"{self._task_id}|{'|'.join(deduped)}"


def create_adapter(config: Optional[Mapping[str, Any]] = None) -> Any:
    cfg = dict(config or {})

    # If use_remote=true, connect to a running persistent_server instead of
    # spawning a new Minecraft process.
    if cfg.get("use_remote", False):
        from core.adapters.minedojo.remote_adapter import RemoteMineDojoAdapter
        return RemoteMineDojoAdapter(cfg)

    try:
        import minedojo
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("minedojo package is required for the minedojo adapter") from exc

    task_id = cfg.get("task_id", "harvest_milk")
    image_size_value = cfg.get("image_size", (160, 256))
    if isinstance(image_size_value, (list, tuple)) and len(image_size_value) == 2:
        image_size = (int(image_size_value[0]), int(image_size_value[1]))
    else:
        image_size = (160, 256)
    env = minedojo.make(task_id=task_id, image_size=image_size)
    return MineDojoAdapter(env, config=cfg)
