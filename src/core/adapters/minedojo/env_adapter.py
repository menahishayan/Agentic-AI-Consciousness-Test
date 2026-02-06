from __future__ import annotations

from typing import Any, Tuple


class MineDojoAdapter:
    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self) -> Tuple[Any, Any]:
        raise NotImplementedError("MineDojo reset not implemented.")

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any]:
        raise NotImplementedError("MineDojo step not implemented.")

    def close(self) -> None:
        raise NotImplementedError("MineDojo close not implemented.")

    def get_raw_observation(self) -> Any:
        raise NotImplementedError("MineDojo raw observation not implemented.")
