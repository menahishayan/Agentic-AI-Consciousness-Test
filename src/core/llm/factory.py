"""
LLM Factory — builds the correct LLM client from config.

Reads provider from config["llm"]["provider"] and API key from environment.
Default provider is "anthropic" (Claude).

Supported providers:
  "anthropic" → AnthropicClient (full implementation)
  "openai"    → OpenAIClient (full implementation)
  "gemini"    → GeminiClient (stub — raises NotImplementedError on use)
  "ollama"    → OllamaClient (local Ollama instance, no API key required)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from core.llm.base import AbstractLLMClient

log = logging.getLogger(__name__)


def build_llm_client(
    llm_config: Optional[Mapping[str, Any]],
    logger: Optional[Any] = None,
) -> Optional[AbstractLLMClient]:
    """
    Build and return an LLM client from config, or None if disabled/no key.

    Args:
        llm_config: The "llm" section of config.json
        logger: Optional RunLogger for LLM call logging

    Returns:
        Concrete AbstractLLMClient, or None if LLM is disabled
    """
    cfg = dict(llm_config or {})

    if not _as_bool(cfg.get("enabled"), default=True):
        log.info("LLM disabled in config.")
        return None

    provider = str(cfg.get("provider", "anthropic")).strip().lower()
    model = cfg.get("model")
    max_tokens = int(cfg.get("max_tokens", 500))
    temperature = float(cfg.get("temperature", 0.1))
    timeout = float(cfg.get("timeout_s", 15.0))
    base_url = cfg.get("base_url")

    if provider == "anthropic":
        api_key = _get_key(cfg, "ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set — LLM disabled.")
            return None
        from core.llm.providers.anthropic import AnthropicClient
        log.info("Using Anthropic provider (model=%s)", model or "claude-sonnet-4-6")
        return AnthropicClient(
            api_key=api_key,
            default_model=model,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            timeout=timeout,
            logger=logger,
        )

    elif provider == "openai":
        api_key = _get_key(cfg, "OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set — LLM disabled.")
            return None
        from core.llm.providers.openai import OpenAIClient
        log.info("Using OpenAI provider (model=%s)", model or "gpt-4o-mini")
        return OpenAIClient(
            api_key=api_key,
            default_model=model,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            timeout=timeout,
            logger=logger,
        )

    elif provider == "gemini":
        api_key = _get_key(cfg, "GEMINI_API_KEY")
        if not api_key:
            log.warning("GEMINI_API_KEY not set — LLM disabled.")
            return None
        from core.llm.providers.gemini import GeminiClient
        log.info("Using Gemini provider (stub — calls will raise NotImplementedError)")
        return GeminiClient(api_key=api_key)

    elif provider == "ollama":
        from core.llm.providers.ollama import OllamaClient
        kwargs: dict = dict(
            default_model=model,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            timeout=timeout,
            logger=logger,
        )
        if base_url:
            kwargs["base_url"] = base_url
        log.info("Using Ollama provider (model=%s)", model or "gemma3:1b")
        return OllamaClient(**kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Supported: 'anthropic', 'openai', 'gemini', 'ollama'."
        )


def _get_key(cfg: dict, env_name: str) -> Optional[str]:
    """Get API key from config or environment."""
    return (
        _as_optional_str(cfg.get("api_key"))
        or _as_optional_str(os.getenv(env_name))
    )


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None
