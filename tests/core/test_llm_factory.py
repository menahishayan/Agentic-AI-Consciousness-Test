from __future__ import annotations

from typing import Any

import pytest

from core.llm import factory


def test_build_llm_client_returns_none_when_disabled() -> None:
    client = factory.build_llm_client({"enabled": False}, logger=None)
    assert client is None


def test_build_llm_client_returns_none_when_api_key_missing(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = factory.build_llm_client(
        {"enabled": True, "provider": "openai", "model": "gpt-4.1-mini"},
        logger=None,
    )
    assert client is None


def test_build_llm_client_openai_success(monkeypatch: Any) -> None:
    class _FakeOpenAIClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(factory, "OpenAIClient", _FakeOpenAIClient)
    client = factory.build_llm_client(
        {
            "enabled": True,
            "provider": "openai",
            "api_key": "test-key",
            "model": "gpt-test",
            "timeout_s": 1.0,
            "provider_params": {"reasoning": {"effort": "low"}},
        },
        logger=None,
    )

    assert isinstance(client, _FakeOpenAIClient)
    assert client.kwargs["api_key"] == "test-key"
    assert client.kwargs["default_model"] == "gpt-test"
    assert client.kwargs["timeout"] == 1.0
    assert client.kwargs["default_params"] == {"reasoning": {"effort": "low"}}


def test_build_llm_client_unsupported_provider_raises() -> None:
    with pytest.raises(NotImplementedError):
        factory.build_llm_client({"enabled": True, "provider": "anthropic"}, logger=None)
