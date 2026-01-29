"""End-to-end integration tests for the multiplexer backend."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe.core.config import (
    ModelConfig,
    ModelPoolEntry,
    MultiplexerConfig,
    MultiplexerMode,
    ProviderConfig,
)
from vibe.core.llm.backend.multiplexer import MultiplexerBackend
from vibe.core.types import (
    AvailableFunction,
    AvailableTool,
    LLMMessage,
    MultiplexerStats,
    Role,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def primary_provider() -> ProviderConfig:
    return ProviderConfig(
        name="primary-provider",
        api_base="https://api.primary.com/v1",
        api_key_env_var="PRIMARY_API_KEY",
    )


@pytest.fixture
def fallback_provider() -> ProviderConfig:
    return ProviderConfig(
        name="fallback-provider",
        api_base="https://api.fallback.com/v1",
        api_key_env_var="FALLBACK_API_KEY",
    )


@pytest.fixture
def primary_model() -> ModelConfig:
    return ModelConfig(
        name="primary-model",
        provider="primary-provider",
        alias="primary",
    )


@pytest.fixture
def fallback_model() -> ModelConfig:
    return ModelConfig(
        name="fallback-model",
        provider="fallback-provider",
        alias="fallback",
    )


def create_mock_completion(content: str = "Response") -> MagicMock:
    """Create a mock completion object."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message = MagicMock()
    completion.choices[0].message.content = content
    completion.choices[0].message.tool_calls = None
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 10
    completion.usage.completion_tokens = 5
    return completion


# -----------------------------------------------------------------------------
# Single Model Tests
# -----------------------------------------------------------------------------


class TestSingleModelMode:
    """Tests for single model mode (backward compatibility)."""

    @pytest.mark.asyncio
    async def test_single_model_completion(
        self, primary_provider: ProviderConfig, primary_model: ModelConfig
    ) -> None:
        """Single model mode completes successfully."""
        mock_completion = create_mock_completion("Hello!")

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(return_value=mock_completion)

        backend = MultiplexerBackend(
            provider=primary_provider, multiplexer=mock_mux
        )
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Hi")]
        result = await backend.complete(model=primary_model, messages=messages)

        assert result.message.content == "Hello!"
        mock_mux.chat.completions.create.assert_called_once()


# -----------------------------------------------------------------------------
# Failover Mode Tests
# -----------------------------------------------------------------------------


class TestFailoverMode:
    """Tests for failover mode behavior."""

    @pytest.mark.asyncio
    async def test_failover_on_rate_limit(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multiplexer should handle rate limit errors gracefully."""
        from multiplexer_llm.exceptions import RateLimitError

        monkeypatch.setenv("PRIMARY_API_KEY", "test-key")

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("Rate limited")
        )

        backend = MultiplexerBackend(
            provider=primary_provider,
            multiplexer=mock_mux,
            multiplexer_config=MultiplexerConfig(
                enabled=True, mode=MultiplexerMode.FAILOVER
            ),
        )
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Hi")]

        from vibe.core.llm.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            await backend.complete(model=primary_model, messages=messages)

        assert exc_info.value.status == 429

    @pytest.mark.asyncio
    async def test_failover_on_service_error(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multiplexer should handle service errors gracefully."""
        from multiplexer_llm.exceptions import ServiceUnavailableError

        monkeypatch.setenv("PRIMARY_API_KEY", "test-key")

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(
            side_effect=ServiceUnavailableError("Service down")
        )

        backend = MultiplexerBackend(
            provider=primary_provider,
            multiplexer=mock_mux,
        )
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Hi")]

        from vibe.core.llm.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            await backend.complete(model=primary_model, messages=messages)

        assert exc_info.value.status == 503


# -----------------------------------------------------------------------------
# Load Balance Mode Tests
# -----------------------------------------------------------------------------


class TestLoadBalanceMode:
    """Tests for load balance mode behavior."""

    @pytest.mark.asyncio
    async def test_load_balance_uses_weights(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Load balance mode should use model weights."""
        monkeypatch.setenv("PRIMARY_API_KEY", "test-key")

        pool_entry = ModelPoolEntry(model="primary", weight=10)
        model_configs = [(primary_model, primary_provider, pool_entry)]

        mock_mux_class = MagicMock()
        mock_mux_instance = MagicMock()
        mock_mux_class.return_value = mock_mux_instance

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            with patch("vibe.core.llm.backend.multiplexer.AsyncOpenAI"):
                backend = MultiplexerBackend(
                    model_configs=model_configs,
                    multiplexer_config=MultiplexerConfig(
                        enabled=True, mode=MultiplexerMode.LOAD_BALANCE
                    ),
                )
                await backend.__aenter__()

                # Check that model was added with correct weight
                mock_mux_instance.add_model.assert_called_once()
                call_kwargs = mock_mux_instance.add_model.call_args[1]
                assert call_kwargs["weight"] == 10


# -----------------------------------------------------------------------------
# Statistics Tests
# -----------------------------------------------------------------------------


class TestMultiplexerStatistics:
    """Tests for statistics collection."""

    @pytest.mark.asyncio
    async def test_get_stats_when_disabled(
        self, primary_provider: ProviderConfig
    ) -> None:
        """get_stats returns disabled state when multiplexer not initialized."""
        backend = MultiplexerBackend(provider=primary_provider)

        stats = backend.get_stats()
        assert stats.enabled is False

    @pytest.mark.asyncio
    async def test_get_stats_returns_multiplexer_stats(
        self,
        primary_provider: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_stats returns statistics from multiplexer."""
        monkeypatch.setenv("PRIMARY_API_KEY", "test-key")

        mock_mux = MagicMock()
        mock_mux.get_stats.return_value = {
            "model1": {"success": 10, "rateLimited": 2, "failed": 1}
        }
        mock_mux._weighted_models = []
        mock_mux._fallback_models = []

        backend = MultiplexerBackend(
            provider=primary_provider,
            multiplexer=mock_mux,
            multiplexer_config=MultiplexerConfig(
                enabled=True, mode=MultiplexerMode.FAILOVER
            ),
        )
        await backend.__aenter__()

        stats = backend.get_stats()

        assert stats.enabled is True
        assert stats.mode == "failover"
        assert "model1" in stats.per_model
        assert stats.per_model["model1"].success_count == 10


# -----------------------------------------------------------------------------
# Streaming Tests
# -----------------------------------------------------------------------------


class TestMultiplexerStreaming:
    """Tests for streaming behavior."""

    @pytest.mark.asyncio
    async def test_streaming_single_model_uses_client(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Single model streaming uses direct client."""
        monkeypatch.setenv("PRIMARY_API_KEY", "test-key")

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock()
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.prompt_tokens = 5
        mock_chunk.usage.completion_tokens = 1

        async def mock_stream():
            yield mock_chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        mock_mux = MagicMock()
        mock_mux_class = MagicMock(return_value=mock_mux)

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            with patch(
                "vibe.core.llm.backend.multiplexer.AsyncOpenAI",
                return_value=mock_client,
            ):
                backend = MultiplexerBackend(provider=primary_provider)
                await backend.__aenter__()

                messages = [LLMMessage(role=Role.user, content="Hi")]
                chunks = []

                async for chunk in backend.complete_streaming(
                    model=primary_model, messages=messages
                ):
                    chunks.append(chunk)

                assert len(chunks) >= 1


# -----------------------------------------------------------------------------
# Token Counting Tests
# -----------------------------------------------------------------------------


class TestTokenCounting:
    """Tests for token counting functionality."""

    @pytest.mark.asyncio
    async def test_count_tokens(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
    ) -> None:
        """count_tokens returns prompt tokens from completion."""
        mock_completion = create_mock_completion()
        mock_completion.usage.prompt_tokens = 42

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(return_value=mock_completion)

        backend = MultiplexerBackend(
            provider=primary_provider, multiplexer=mock_mux
        )
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Count my tokens")]
        token_count = await backend.count_tokens(model=primary_model, messages=messages)

        assert token_count == 42


# -----------------------------------------------------------------------------
# Error Mapping Tests
# -----------------------------------------------------------------------------


class TestErrorMapping:
    """Tests for exception mapping to BackendError."""

    @pytest.mark.asyncio
    async def test_authentication_error_mapping(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
    ) -> None:
        """AuthenticationError maps to 401 status."""
        from multiplexer_llm.exceptions import AuthenticationError

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError("Invalid API key")
        )

        backend = MultiplexerBackend(
            provider=primary_provider, multiplexer=mock_mux
        )
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Hi")]

        from vibe.core.llm.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            await backend.complete(model=primary_model, messages=messages)

        assert exc_info.value.status == 401

    @pytest.mark.asyncio
    async def test_model_not_found_error_mapping(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
    ) -> None:
        """ModelNotFoundError maps to 404 status."""
        from multiplexer_llm.exceptions import ModelNotFoundError

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(
            side_effect=ModelNotFoundError("Model not found")
        )

        backend = MultiplexerBackend(
            provider=primary_provider, multiplexer=mock_mux
        )
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Hi")]

        from vibe.core.llm.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            await backend.complete(model=primary_model, messages=messages)

        assert exc_info.value.status == 404

    @pytest.mark.asyncio
    async def test_model_selection_error_mapping(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
    ) -> None:
        """ModelSelectionError maps to 503 status."""
        from multiplexer_llm.exceptions import ModelSelectionError

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(
            side_effect=ModelSelectionError("No models available")
        )

        backend = MultiplexerBackend(
            provider=primary_provider, multiplexer=mock_mux
        )
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Hi")]

        from vibe.core.llm.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            await backend.complete(model=primary_model, messages=messages)

        assert exc_info.value.status == 503


# -----------------------------------------------------------------------------
# Tool Calling Tests
# -----------------------------------------------------------------------------


class TestToolCalling:
    """Tests for tool calling with multiplexer."""

    @pytest.mark.asyncio
    async def test_complete_with_tools(
        self,
        primary_provider: ProviderConfig,
        primary_model: ModelConfig,
    ) -> None:
        """Completion with tools passes tool definitions correctly."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = None
        mock_completion.choices[0].message.tool_calls = [MagicMock()]
        mock_completion.choices[0].message.tool_calls[0].id = "call_123"
        mock_completion.choices[0].message.tool_calls[0].type = "function"
        mock_completion.choices[0].message.tool_calls[0].function = MagicMock()
        mock_completion.choices[0].message.tool_calls[0].function.name = "read_file"
        mock_completion.choices[0].message.tool_calls[
            0
        ].function.arguments = '{"path": "test.txt"}'
        mock_completion.usage = MagicMock()
        mock_completion.usage.prompt_tokens = 50
        mock_completion.usage.completion_tokens = 10

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(return_value=mock_completion)

        backend = MultiplexerBackend(
            provider=primary_provider, multiplexer=mock_mux
        )
        await backend.__aenter__()

        tools = [
            AvailableTool(
                function=AvailableFunction(
                    name="read_file",
                    description="Read a file",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            )
        ]
        messages = [LLMMessage(role=Role.user, content="Read the file")]

        result = await backend.complete(
            model=primary_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        assert result.message.tool_calls is not None
        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].function.name == "read_file"

        call_kwargs = mock_mux.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"


# -----------------------------------------------------------------------------
# Resource Cleanup Tests
# -----------------------------------------------------------------------------


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_resets_multiplexer(
        self,
        primary_provider: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """close() resets the multiplexer instance."""
        monkeypatch.setenv("PRIMARY_API_KEY", "test-key")

        mock_mux_class = MagicMock()
        mock_mux_instance = MagicMock()
        mock_mux_instance.async_reset = AsyncMock()
        mock_mux_class.return_value = mock_mux_instance

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            backend = MultiplexerBackend(provider=primary_provider)
            await backend.__aenter__()

            assert backend._multiplexer is not None

            await backend.close()

            mock_mux_instance.async_reset.assert_called_once()
            assert backend._multiplexer is None

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(
        self,
        primary_provider: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Context manager properly cleans up resources."""
        monkeypatch.setenv("PRIMARY_API_KEY", "test-key")

        mock_mux_class = MagicMock()
        mock_mux_instance = MagicMock()
        mock_mux_instance.async_reset = AsyncMock()
        mock_mux_class.return_value = mock_mux_instance

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            async with MultiplexerBackend(provider=primary_provider) as backend:
                assert backend._multiplexer is not None

            mock_mux_instance.async_reset.assert_called_once()
