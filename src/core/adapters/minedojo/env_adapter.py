from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


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

    def __init__(self, env: Any) -> None:
        self.env = env
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}

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
    return MineDojoAdapter(env)
