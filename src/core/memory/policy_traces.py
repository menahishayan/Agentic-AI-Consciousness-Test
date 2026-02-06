from __future__ import annotations

from typing import Any


class PolicyTraces:
    def __init__(self) -> None:
        pass

    def record(self, trace: Any) -> None:
        raise NotImplementedError("Policy trace recording not implemented.")

    def query(self, query: Any) -> Any:
        raise NotImplementedError("Policy trace query not implemented.")
