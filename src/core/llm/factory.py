from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from core.llm.client import LLMClient
from core.llm.providers import OpenAIClient
from core.observability.logger import RunLogger


def build_llm_client(
    llm_config: Optional[Mapping[str, Any]],
    logger: Optional[RunLogger] = None,
) -> Optional[LLMClient]:
    cfg = dict(llm_config or {})
    if not _as_bool(cfg.get("enabled"), False):
        return None

    provider = str(cfg.get("provider", "openai")).strip().lower()
    if provider != "openai":
        raise NotImplementedError(
            f"LLM provider '{provider}' is not implemented. "
            "Supported providers: openai."
        )

    api_key = _as_optional_str(cfg.get("api_key")) or _as_optional_str(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return None

    organization = _as_optional_str(cfg.get("organization"))
    project = _as_optional_str(cfg.get("project"))
    model = _as_optional_str(cfg.get("model"))
    timeout = _as_optional_float(cfg.get("timeout_s"))
    provider_params = cfg.get("provider_params")
    default_params = dict(provider_params) if isinstance(provider_params, Mapping) else None

    return OpenAIClient(
        api_key=api_key,
        organization=organization,
        project=project,
        default_model=model,
        default_params=default_params,
        logger=logger,
        timeout=timeout,
    )


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
