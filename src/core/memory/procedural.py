from __future__ import annotations

from typing import Any


class ProceduralMemory:
    def __init__(self) -> None:
        pass

    def store(self, skill: Any) -> None:
        raise NotImplementedError("Procedural memory storage not implemented.")

    def query(self, query: Any) -> Any:
        raise NotImplementedError("Procedural memory query not implemented.")
