from __future__ import annotations

from typing import Any, List


class WorkingMemoryBuffer:
    def __init__(self) -> None:
        self.items: List[Any] = []

    def push(self, item: Any) -> None:
        raise NotImplementedError("Working memory push not implemented.")

    def clear(self) -> None:
        raise NotImplementedError("Working memory clear not implemented.")

    def snapshot(self) -> List[Any]:
        raise NotImplementedError("Working memory snapshot not implemented.")
