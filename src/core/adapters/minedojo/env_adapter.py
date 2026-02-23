from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

    def __init__(self, env: Any, config: Optional[Mapping[str, Any]] = None) -> None:
        self.env = env
        self._config = dict(config or {})
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}
        self._nvec = self._extract_action_nvec()
        self._noop_action = self._build_noop_action()
        self._action_space_actions = self._build_action_space_actions()

        self._policies: List[Dict[str, Any]] = []
        self._voyager_skills: Dict[str, Dict[str, Any]] = {}
        self._policy_method_names = set(dir(self))
        self._register_action_space_policies()
        self._register_voyager_skill_policies()

    def reset(self) -> Tuple[Any, Any]:
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}
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


def create_adapter(config: Optional[Mapping[str, Any]] = None) -> MineDojoAdapter:
    try:
        import minedojo
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("minedojo package is required for the minedojo adapter") from exc

    cfg = dict(config or {})
    task_id = cfg.get("task_id", "harvest_milk")
    image_size_value = cfg.get("image_size", (160, 256))
    if isinstance(image_size_value, (list, tuple)) and len(image_size_value) == 2:
        image_size = (int(image_size_value[0]), int(image_size_value[1]))
    else:
        image_size = (160, 256)
    env = minedojo.make(task_id=task_id, image_size=image_size)
    return MineDojoAdapter(env, config=cfg)
