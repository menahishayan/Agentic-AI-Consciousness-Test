from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.observability.config import LoggingConfig
from core.observability.paths import RunPaths, create_run_dir
from core.observability.serializer import safe_serialize


def _utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


class RunLogger:
    def __init__(self, config: LoggingConfig, paths: Optional[RunPaths] = None) -> None:
        self.config = config
        self.paths = paths or create_run_dir(config)
        self._files = {
            "events": self.paths.events_jsonl.open("a", encoding="utf-8"),
            "llm": self.paths.llm_jsonl.open("a", encoding="utf-8"),
            "memory": self.paths.memory_jsonl.open("a", encoding="utf-8"),
            "state": self.paths.state_jsonl.open("a", encoding="utf-8"),
            "tracebacks": self.paths.tracebacks_log.open("a", encoding="utf-8"),
            "tracebacks_jsonl": self.paths.tracebacks_jsonl.open("a", encoding="utf-8"),
        }
        self._faulthandler_file = None
        self._write_run_metadata()

    def _write_run_metadata(self) -> None:
        metadata = {
            "run_id": self.config.run_id,
            "run_name": self.config.run_name,
            "start_time": _utc_ts(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "config": {
                "log_root": str(self.config.log_root),
                "log_prompts": self.config.log_prompts,
                "log_memory": self.config.log_memory,
                "log_state": self.config.log_state,
                "max_field_bytes": self.config.max_field_bytes,
                "include_raw_llm": self.config.include_raw_llm,
                "redact_keys": list(self.config.redact_keys),
            },
        }
        self.paths.run_json.write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _emit(self, stream_key: str, event: str, payload: Any, step: Optional[int]) -> None:
        record = {
            "ts": _utc_ts(),
            "run_id": self.config.run_id,
            "step": step,
            "event": event,
            "payload": safe_serialize(
                payload,
                max_bytes=self.config.max_field_bytes,
                redact_keys=self.config.redact_keys,
            ),
        }
        fh = self._files[stream_key]
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        fh.flush()

    def event(self, name: str, payload: Any, step: Optional[int] = None) -> None:
        self._emit("events", name, payload, step)

    def llm_request(self, payload: Any, step: Optional[int] = None) -> None:
        if not self.config.log_prompts:
            return
        self._emit("llm", "llm.request", payload, step)

    def llm_response(self, payload: Any, step: Optional[int] = None) -> None:
        if (
            not self.config.include_raw_llm
            and isinstance(payload, dict)
            and "raw" in payload
        ):
            payload = {**payload}
            payload.pop("raw", None)
        self._emit("llm", "llm.response", payload, step)

    def memory_event(self, payload: Any, step: Optional[int] = None) -> None:
        if not self.config.log_memory:
            return
        self._emit("memory", "memory.event", payload, step)

    def state_snapshot(self, agent_state: Any, step: Optional[int] = None) -> None:
        if not self.config.log_state:
            return
        data = agent_state
        if hasattr(agent_state, "to_dict"):
            try:
                data = agent_state.to_dict()
            except Exception:
                data = agent_state
        self._emit("state", "state.snapshot", {"state": data}, step)

    def exception(self, exc: BaseException, context: Any = None, step: Optional[int] = None) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
            "context": context,
        }
        self._emit("tracebacks_jsonl", "exception", payload, step)
        self._files["tracebacks"].write(tb + "\n")
        self._files["tracebacks"].flush()

    def close(self) -> None:
        for fh in self._files.values():
            try:
                fh.close()
            except Exception:
                pass
        if self._faulthandler_file is not None:
            try:
                self._faulthandler_file.close()
            except Exception:
                pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.exception(exc)
        self.close()
