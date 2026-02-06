from __future__ import annotations

from typing import Any


class SemanticMemory:
    def __init__(self) -> None:
        pass

    def store(self, entry: Any) -> None:
        raise NotImplementedError("Semantic memory storage not implemented.")

    def query(self, query: Any) -> Any:
        raise NotImplementedError("Semantic memory query not implemented.")
