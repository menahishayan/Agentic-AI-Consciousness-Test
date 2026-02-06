from __future__ import annotations

from typing import Any


class EpisodicMemory:
    def __init__(self) -> None:
        pass

    def store(self, episode: Any) -> None:
        raise NotImplementedError("Episodic memory storage not implemented.")

    def query(self, query: Any) -> Any:
        raise NotImplementedError("Episodic memory query not implemented.")
