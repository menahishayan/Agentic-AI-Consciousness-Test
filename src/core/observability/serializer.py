"""Safe JSON serialization — handles non-serializable objects gracefully."""
from __future__ import annotations

import dataclasses
import json
from typing import Any


class SafeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def safe_dumps(obj: Any) -> str:
    return json.dumps(obj, cls=SafeEncoder, ensure_ascii=False, default=str)
