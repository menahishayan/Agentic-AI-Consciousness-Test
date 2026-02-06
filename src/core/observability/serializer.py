from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore


def _summarize_numpy(array: Any) -> Mapping[str, Any]:
    summary: dict[str, Any] = {
        "type": "ndarray",
        "shape": list(getattr(array, "shape", ()) or ()),
        "dtype": str(getattr(array, "dtype", "")),
    }
    if np is not None and isinstance(array, np.ndarray) and array.size:
        if np.issubdtype(array.dtype, np.number):
            summary["min"] = float(np.nanmin(array))
            summary["max"] = float(np.nanmax(array))
    return summary


def _truncate_string(value: str, max_bytes: int) -> str:
    if len(value) <= max_bytes:
        return value
    return value[:max_bytes] + "...(truncated)"


def _truncate_json(value: Any, max_bytes: int, obj_type: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=True)
    except Exception:
        return _truncate_string(repr(value), max_bytes)
    if len(encoded) <= max_bytes:
        return value
    return {"type": obj_type, "truncated": True, "bytes": len(encoded)}


def safe_serialize(
    obj: Any,
    max_bytes: int = 20000,
    redact_keys: Optional[Sequence[str]] = None,
    _depth: int = 0,
) -> Any:
    if redact_keys is None:
        redact_keys = ()
    redact_set = {key.lower() for key in redact_keys}

    if _depth > 6:
        return "<max_depth>"

    if obj is None or isinstance(obj, (bool, int, float)):
        return obj

    if isinstance(obj, str):
        return _truncate_string(obj, max_bytes)

    if isinstance(obj, bytes):
        return {"type": "bytes", "len": len(obj)}

    if np is not None and isinstance(obj, np.ndarray):
        return _summarize_numpy(obj)

    if is_dataclass(obj):
        return safe_serialize(asdict(obj), max_bytes, redact_keys, _depth + 1)

    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            key_str = str(key)
            if key_str.lower() in redact_set:
                out[key_str] = "<redacted>"
            else:
                out[key_str] = safe_serialize(value, max_bytes, redact_keys, _depth + 1)
        return _truncate_json(out, max_bytes, type(obj).__name__)

    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        if len(seq) > 50:
            sample = [
                safe_serialize(item, max_bytes, redact_keys, _depth + 1)
                for item in seq[:10]
            ]
            return {
                "type": type(obj).__name__,
                "len": len(seq),
                "sample": sample,
            }
        return [
            safe_serialize(item, max_bytes, redact_keys, _depth + 1) for item in seq
        ]

    if hasattr(obj, "__dict__"):
        return safe_serialize(vars(obj), max_bytes, redact_keys, _depth + 1)

    return _truncate_json(repr(obj), max_bytes, type(obj).__name__)
