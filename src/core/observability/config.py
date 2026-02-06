from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Sequence, Optional
import uuid


def _env_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _env_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class LoggingConfig:
    log_root: Path = Path("logs/runs")
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    run_name: Optional[str] = None
    log_prompts: bool = True
    log_memory: bool = True
    log_state: bool = True
    max_field_bytes: int = 20000
    redact_keys: Sequence[str] = field(
        default_factory=lambda: (
            "api_key",
            "authorization",
            "openai_api_key",
            "anthropic_api_key",
            "gemini_api_key",
        )
    )
    include_raw_llm: bool = True

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        log_root = Path(os.getenv("LOG_ROOT", "logs/runs"))
        run_id = os.getenv("RUN_ID")
        run_name = os.getenv("RUN_NAME")
        log_prompts = _env_bool(os.getenv("LOG_PROMPTS"), True)
        log_memory = _env_bool(os.getenv("LOG_MEMORY"), True)
        log_state = _env_bool(os.getenv("LOG_STATE"), True)
        max_field_bytes = _env_int(os.getenv("LOG_MAX_FIELD_BYTES"), 20000)
        redact_keys_raw = os.getenv("LOG_REDACT_KEYS")
        redact_keys = None
        if redact_keys_raw:
            redact_keys = tuple(
                key.strip().lower() for key in redact_keys_raw.split(",") if key.strip()
            )
        include_raw_llm = _env_bool(os.getenv("LOG_INCLUDE_RAW_LLM"), True)

        return cls(
            log_root=log_root,
            run_id=run_id or uuid.uuid4().hex[:8],
            run_name=run_name,
            log_prompts=log_prompts,
            log_memory=log_memory,
            log_state=log_state,
            max_field_bytes=max_field_bytes,
            redact_keys=redact_keys or cls().redact_keys,
            include_raw_llm=include_raw_llm,
        )
