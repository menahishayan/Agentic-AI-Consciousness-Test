from __future__ import annotations

from typing import Any


class PredictionErrorHistory:
    def __init__(self) -> None:
        pass

    def record(self, error: Any) -> None:
        raise NotImplementedError("Prediction error recording not implemented.")

    def query(self, query: Any) -> Any:
        raise NotImplementedError("Prediction error query not implemented.")
