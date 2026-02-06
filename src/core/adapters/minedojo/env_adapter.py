from __future__ import annotations

from typing import Any, Dict, Tuple


class MineDojoAdapter:
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
