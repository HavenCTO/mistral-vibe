"""Tests for MultiplexerBackend implementation."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe.core.config import ModelConfig, ModelPoolEntry, MultiplexerConfig, ProviderConfig
from vibe.core.llm.backend.multiplexer import (
    MultiplexerBackend,
    _completion_to_chunk,
    _llm_message_to_openai,
    _openai_message_to_llm,
    _tool_choice_to_openai,
    _tools_to_openai,
)
from vibe.core.types import (
    AvailableFunction,
    AvailableTool,
    FunctionCall,
    LLMMessage,
    LLMUsage,
    MultiplexerStats,
    Role,
    ToolCall,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def provider_config() -> ProviderConfig:
    return ProviderConfig(
        name="test-provider",
        api_base="https://api.test.com/v1",
        api_key_env_var="TEST_API_KEY",
        reasoning_field_name="reasoning_content",
    )


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        provider="test-provider",
        alias="test",
        model_id="test-model-id",
    )


@pytest.fixture
def pool_entry() -> ModelPoolEntry:
    return ModelPoolEntry(
        model="test",
        weight=1,
        max_concurrent=None,
        is_fallback=False,
    )


@pytest.fixture
def fallback_entry() -> ModelPoolEntry:
    return ModelPoolEntry(
        model="fallback",
        weight=1,
        max_concurrent=None,
        is_fallback=True,
    )


@pytest.fixture
def multiplexer_config() -> MultiplexerConfig:
    return MultiplexerConfig(
        enabled=True,
        mode="failover",
        pool=[
            ModelPoolEntry(model="test", weight=2),
            ModelPoolEntry(model="fallback", is_fallback=True),
        ],
    )


# -----------------------------------------------------------------------------
# Message Conversion Tests
# -----------------------------------------------------------------------------


class TestLLMMessageToOpenAI:
    """Tests for _llm_message_to_openai conversion."""

    def test_simple_user_message(self) -> None:
        msg = LLMMessage(role=Role.user, content="Hello")
        result = _llm_message_to_openai(msg)

        assert result["role"] == "user"
        assert result["content"] == "Hello"
        assert "tool_calls" not in result

    def test_assistant_message_with_content(self) -> None:
        msg = LLMMessage(role=Role.assistant, content="Hi there!")
        result = _llm_message_to_openai(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Hi there!"

    def test_system_message(self) -> None:
        msg = LLMMessage(role=Role.system, content="You are a helpful assistant.")
        result = _llm_message_to_openai(msg)

        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_message_with_reasoning_content(self) -> None:
        msg = LLMMessage(
            role=Role.assistant,
            content="The answer is 42.",
            reasoning_content="Let me think...",
        )
        result = _llm_message_to_openai(msg)

        assert result["content"] == "The answer is 42."
        assert result["reasoning_content"] == "Let me think..."

    def test_message_with_custom_reasoning_field(self) -> None:
        msg = LLMMessage(
            role=Role.assistant,
            content="Answer",
            reasoning_content="Thinking...",
        )
        result = _llm_message_to_openai(msg, reasoning_field_name="thinking")

        assert result["thinking"] == "Thinking..."
        assert "reasoning_content" not in result

    def test_message_with_tool_calls(self) -> None:
        msg = LLMMessage(
            role=Role.assistant,
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_123",
                    index=0,
                    function=FunctionCall(
                        name="read_file",
                        arguments='{"path": "/tmp/test.txt"}',
                    ),
                    type="function",
                )
            ],
        )
        result = _llm_message_to_openai(msg)

        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == '{"path": "/tmp/test.txt"}'

    def test_tool_message(self) -> None:
        msg = LLMMessage(
            role=Role.tool,
            content="File contents here",
            tool_call_id="call_123",
            name="read_file",
        )
        result = _llm_message_to_openai(msg)

        assert result["role"] == "tool"
        assert result["content"] == "File contents here"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "read_file"

    def test_tool_call_with_empty_arguments(self) -> None:
        msg = LLMMessage(
            role=Role.assistant,
            tool_calls=[
                ToolCall(
                    id="call_456",
                    function=FunctionCall(name="list_files", arguments=None),
                )
            ],
        )
        result = _llm_message_to_openai(msg)

        assert result["tool_calls"][0]["function"]["arguments"] == ""

    def test_skips_tool_calls_without_id(self) -> None:
        msg = LLMMessage(
            role=Role.assistant,
            tool_calls=[
                ToolCall(
                    id=None,
                    function=FunctionCall(name="test"),
                ),
                ToolCall(
                    id="call_valid",
                    function=FunctionCall(name="valid_call"),
                ),
            ],
        )
        result = _llm_message_to_openai(msg)

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_valid"


class TestOpenAIMessageToLLM:
    """Tests for _openai_message_to_llm conversion."""

    def test_simple_assistant_message(self) -> None:
        msg = {"role": "assistant", "content": "Hello!"}
        result = _openai_message_to_llm(msg)

        assert result.role == Role.assistant
        assert result.content == "Hello!"
        assert result.tool_calls is None

    def test_user_message(self) -> None:
        msg = {"role": "user", "content": "Hi"}
        result = _openai_message_to_llm(msg)

        assert result.role == Role.user
        assert result.content == "Hi"

    def test_system_message(self) -> None:
        msg = {"role": "system", "content": "You are helpful."}
        result = _openai_message_to_llm(msg)

        assert result.role == Role.system
        assert result.content == "You are helpful."

    def test_message_with_reasoning_content(self) -> None:
        msg = {
            "role": "assistant",
            "content": "42",
            "reasoning_content": "Let me calculate...",
        }
        result = _openai_message_to_llm(msg)

        assert result.content == "42"
        assert result.reasoning_content == "Let me calculate..."

    def test_message_with_custom_reasoning_field(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Result",
            "thinking": "Processing...",
        }
        result = _openai_message_to_llm(msg, reasoning_field_name="thinking")

        assert result.reasoning_content == "Processing..."

    def test_message_with_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "ls"}',
                    },
                }
            ],
        }
        result = _openai_message_to_llm(msg)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].function.name == "bash"
        assert result.tool_calls[0].function.arguments == '{"command": "ls"}'

    def test_tool_message(self) -> None:
        msg = {
            "role": "tool",
            "content": "Success",
            "tool_call_id": "call_xyz",
            "name": "bash",
        }
        result = _openai_message_to_llm(msg)

        assert result.role == Role.tool
        assert result.content == "Success"
        assert result.tool_call_id == "call_xyz"
        assert result.name == "bash"

    def test_defaults_to_assistant_role(self) -> None:
        msg = {"content": "test"}
        result = _openai_message_to_llm(msg)

        assert result.role == Role.assistant

    def test_unknown_role_defaults_to_assistant(self) -> None:
        msg = {"role": "unknown_role", "content": "test"}
        result = _openai_message_to_llm(msg)

        assert result.role == Role.assistant


class TestToolsToOpenAI:
    """Tests for _tools_to_openai conversion."""

    def test_converts_single_tool(self) -> None:
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
        result = _tools_to_openai(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["description"] == "Read a file"

    def test_converts_multiple_tools(self) -> None:
        tools = [
            AvailableTool(
                function=AvailableFunction(
                    name="tool1",
                    description="First tool",
                    parameters={},
                )
            ),
            AvailableTool(
                function=AvailableFunction(
                    name="tool2",
                    description="Second tool",
                    parameters={},
                )
            ),
        ]
        result = _tools_to_openai(tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"

    def test_empty_tools_list(self) -> None:
        result = _tools_to_openai([])
        assert result == []


class TestToolChoiceToOpenAI:
    """Tests for _tool_choice_to_openai conversion."""

    def test_none_returns_none(self) -> None:
        result = _tool_choice_to_openai(None)
        assert result is None

    def test_string_tool_choice_auto(self) -> None:
        result = _tool_choice_to_openai("auto")
        assert result == "auto"

    def test_string_tool_choice_none(self) -> None:
        result = _tool_choice_to_openai("none")
        assert result == "none"

    def test_string_tool_choice_required(self) -> None:
        result = _tool_choice_to_openai("required")
        assert result == "required"

    def test_tool_object_choice(self) -> None:
        tool = AvailableTool(
            function=AvailableFunction(
                name="specific_tool",
                description="A specific tool",
                parameters={},
            )
        )
        result = _tool_choice_to_openai(tool)

        assert isinstance(result, dict)
        assert result["type"] == "function"
        assert result["function"]["name"] == "specific_tool"


# -----------------------------------------------------------------------------
# Completion to Chunk Tests
# -----------------------------------------------------------------------------


class TestCompletionToChunk:
    """Tests for _completion_to_chunk conversion."""

    def test_simple_completion(self) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message = MagicMock()
        completion.choices[0].message.content = "Hello!"
        completion.choices[0].message.tool_calls = None
        completion.usage = MagicMock()
        completion.usage.prompt_tokens = 10
        completion.usage.completion_tokens = 5

        result = _completion_to_chunk(completion)

        assert result.message.role == Role.assistant
        assert result.message.content == "Hello!"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_completion_with_tool_calls(self) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message = MagicMock()
        completion.choices[0].message.content = None
        completion.choices[0].message.tool_calls = [MagicMock()]
        completion.choices[0].message.tool_calls[0].id = "call_123"
        completion.choices[0].message.tool_calls[0].type = "function"
        completion.choices[0].message.tool_calls[0].function = MagicMock()
        completion.choices[0].message.tool_calls[0].function.name = "read_file"
        completion.choices[0].message.tool_calls[0].function.arguments = '{"path": "test.txt"}'
        completion.usage = None

        result = _completion_to_chunk(completion)

        assert result.message.tool_calls is not None
        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].id == "call_123"
        assert result.message.tool_calls[0].function.name == "read_file"

    def test_completion_with_reasoning_content(self) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message = MagicMock()
        completion.choices[0].message.content = "Answer"
        completion.choices[0].message.reasoning_content = "Thinking..."
        completion.choices[0].message.tool_calls = None
        completion.usage = None

        result = _completion_to_chunk(completion)

        assert result.message.content == "Answer"
        assert result.message.reasoning_content == "Thinking..."

    def test_completion_no_choices(self) -> None:
        completion = MagicMock()
        completion.choices = []
        completion.usage = None

        result = _completion_to_chunk(completion)

        assert result.message.role == Role.assistant
        assert result.message.content == ""


# -----------------------------------------------------------------------------
# MultiplexerBackend Tests
# -----------------------------------------------------------------------------


class TestMultiplexerBackendInit:
    """Tests for MultiplexerBackend initialization."""

    def test_init_with_provider(self, provider_config: ProviderConfig) -> None:
        backend = MultiplexerBackend(provider=provider_config)
        assert backend._provider == provider_config
        assert backend._multiplexer is None
        assert backend._owns_multiplexer is True

    def test_init_with_simple_model_configs(
        self, provider_config: ProviderConfig, model_config: ModelConfig
    ) -> None:
        model_configs = [(model_config, provider_config)]
        backend = MultiplexerBackend(model_configs=model_configs)

        assert backend._simple_model_configs == model_configs
        assert backend._pool_model_configs is None
        assert backend._provider == provider_config

    def test_init_with_pool_model_configs(
        self,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
        pool_entry: ModelPoolEntry,
    ) -> None:
        model_configs = [(model_config, provider_config, pool_entry)]
        backend = MultiplexerBackend(model_configs=model_configs)

        assert backend._pool_model_configs == model_configs
        assert backend._simple_model_configs is None

    def test_init_with_multiplexer_config(
        self, provider_config: ProviderConfig, multiplexer_config: MultiplexerConfig
    ) -> None:
        backend = MultiplexerBackend(
            provider=provider_config, multiplexer_config=multiplexer_config
        )
        assert backend._multiplexer_config == multiplexer_config

    def test_init_with_pre_configured_multiplexer(
        self, provider_config: ProviderConfig
    ) -> None:
        mock_mux = MagicMock()
        backend = MultiplexerBackend(provider=provider_config, multiplexer=mock_mux)

        assert backend._multiplexer == mock_mux
        assert backend._owns_multiplexer is False

    def test_init_requires_provider_or_model_configs(self) -> None:
        with pytest.raises(ValueError, match="requires either 'provider' or 'model_configs'"):
            MultiplexerBackend()


class TestMultiplexerBackendContextManager:
    """Tests for MultiplexerBackend async context manager."""

    @pytest.mark.asyncio
    async def test_aenter_creates_multiplexer(
        self, provider_config: ProviderConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "test-key")

        mock_mux_class = MagicMock()
        mock_mux_instance = MagicMock()
        mock_mux_class.return_value = mock_mux_instance

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            backend = MultiplexerBackend(provider=provider_config)
            result = await backend.__aenter__()

            assert result == backend
            assert backend._multiplexer == mock_mux_instance
            mock_mux_instance.add_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_aenter_with_pool_model_configs(
        self,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
        pool_entry: ModelPoolEntry,
        fallback_entry: ModelPoolEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "test-key")

        model_configs = [
            (model_config, provider_config, pool_entry),
        ]

        mock_mux_class = MagicMock()
        mock_mux_instance = MagicMock()
        mock_mux_class.return_value = mock_mux_instance

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            backend = MultiplexerBackend(model_configs=model_configs)
            await backend.__aenter__()

            mock_mux_instance.add_model.assert_called_once()
            call_kwargs = mock_mux_instance.add_model.call_args[1]
            assert call_kwargs["weight"] == pool_entry.weight
            assert call_kwargs["model_name"] == model_config.model_id

    @pytest.mark.asyncio
    async def test_aenter_adds_fallback_model(
        self,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "test-key")

        fallback_model = ModelConfig(
            name="fallback-model", provider="test-provider", alias="fallback"
        )
        fallback_entry = ModelPoolEntry(model="fallback", is_fallback=True)

        model_configs = [
            (
                model_config,
                provider_config,
                ModelPoolEntry(model="test", is_fallback=False),
            ),
            (fallback_model, provider_config, fallback_entry),
        ]

        mock_mux_class = MagicMock()
        mock_mux_instance = MagicMock()
        mock_mux_class.return_value = mock_mux_instance

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            backend = MultiplexerBackend(model_configs=model_configs)
            await backend.__aenter__()

            mock_mux_instance.add_model.assert_called_once()
            mock_mux_instance.add_fallback_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_closes_multiplexer(
        self, provider_config: ProviderConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "test-key")

        mock_mux_class = MagicMock()
        mock_mux_instance = MagicMock()
        mock_mux_instance.async_reset = AsyncMock()
        mock_mux_class.return_value = mock_mux_instance

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            backend = MultiplexerBackend(provider=provider_config)
            await backend.__aenter__()
            await backend.__aexit__(None, None, None)

            mock_mux_instance.async_reset.assert_called_once()
            assert backend._multiplexer is None

    @pytest.mark.asyncio
    async def test_aexit_does_not_close_external_multiplexer(
        self, provider_config: ProviderConfig
    ) -> None:
        mock_mux = MagicMock()
        mock_mux.async_reset = AsyncMock()

        backend = MultiplexerBackend(provider=provider_config, multiplexer=mock_mux)
        await backend.__aenter__()
        await backend.__aexit__(None, None, None)

        mock_mux.async_reset.assert_not_called()


class TestMultiplexerBackendComplete:
    """Tests for MultiplexerBackend.complete method."""

    @pytest.mark.asyncio
    async def test_complete_calls_multiplexer(
        self,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
    ) -> None:
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = "Hello!"
        mock_completion.choices[0].message.tool_calls = None
        mock_completion.usage = MagicMock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(return_value=mock_completion)

        backend = MultiplexerBackend(provider=provider_config, multiplexer=mock_mux)
        await backend.__aenter__()

        messages = [LLMMessage(role=Role.user, content="Hi")]
        result = await backend.complete(
            model=model_config,
            messages=messages,
            temperature=0.5,
        )

        assert result.message.content == "Hello!"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10

        mock_mux.chat.completions.create.assert_called_once()
        call_kwargs = mock_mux.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == model_config.model_id
        assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_complete_with_tools(
        self,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
    ) -> None:
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = None
        mock_completion.choices[0].message.tool_calls = []
        mock_completion.usage = None

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(return_value=mock_completion)

        backend = MultiplexerBackend(provider=provider_config, multiplexer=mock_mux)
        await backend.__aenter__()

        tools = [
            AvailableTool(
                function=AvailableFunction(
                    name="test_tool",
                    description="A test tool",
                    parameters={},
                )
            )
        ]
        messages = [LLMMessage(role=Role.user, content="Use the tool")]

        await backend.complete(
            model=model_config,
            messages=messages,
            temperature=0.2,
            tools=tools,
            tool_choice="auto",
        )

        call_kwargs = mock_mux.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_complete_without_context_manager_raises(
        self, provider_config: ProviderConfig, model_config: ModelConfig
    ) -> None:
        backend = MultiplexerBackend(provider=provider_config)
        messages = [LLMMessage(role=Role.user, content="Hi")]

        with pytest.raises(RuntimeError, match="must be used as async context manager"):
            await backend.complete(model=model_config, messages=messages)


class TestMultiplexerBackendStreaming:
    """Tests for MultiplexerBackend.complete_streaming method."""

    @pytest.mark.asyncio
    async def test_streaming_with_single_model(
        self,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "test-key")

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
            with patch("vibe.core.llm.backend.multiplexer.AsyncOpenAI", return_value=mock_client):
                backend = MultiplexerBackend(provider=provider_config)
                await backend.__aenter__()

                messages = [LLMMessage(role=Role.user, content="Hi")]
                chunks = []

                async for chunk in backend.complete_streaming(
                    model=model_config, messages=messages
                ):
                    chunks.append(chunk)

                assert len(chunks) == 1
                assert chunks[0].message.content == "Hello"

    @pytest.mark.asyncio
    async def test_streaming_falls_back_to_non_streaming(
        self,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
        pool_entry: ModelPoolEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When multiple models configured, streaming falls back to non-streaming."""
        monkeypatch.setenv("TEST_API_KEY", "test-key")

        model2 = ModelConfig(name="model2", provider="test-provider", alias="test2")
        entry2 = ModelPoolEntry(model="test2", is_fallback=False)

        model_configs = [
            (model_config, provider_config, pool_entry),
            (model2, provider_config, entry2),
        ]

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = "Response"
        mock_completion.choices[0].message.tool_calls = None
        mock_completion.usage = MagicMock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5

        mock_mux = MagicMock()
        mock_mux.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_mux_class = MagicMock(return_value=mock_mux)

        with patch("multiplexer_llm.Multiplexer", mock_mux_class):
            with patch("vibe.core.llm.backend.multiplexer.AsyncOpenAI"):
                backend = MultiplexerBackend(model_configs=model_configs)
                await backend.__aenter__()

                messages = [LLMMessage(role=Role.user, content="Hi")]
                chunks = []

                async for chunk in backend.complete_streaming(
                    model=model_config, messages=messages
                ):
                    chunks.append(chunk)

                assert len(chunks) == 1
