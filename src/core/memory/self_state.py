from __future__ import annotations

from typing import Any


class SelfStateMemory:
    def __init__(self) -> None:
        pass

    def record(self, snapshot: Any) -> None:
        raise NotImplementedError("Self-state recording not implemented.")

    def query(self, query: Any) -> Any:
        raise NotImplementedError("Self-state query not implemented.")
