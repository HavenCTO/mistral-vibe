"""Multiplexer backend for load balancing across multiple LLM providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
import time
import types
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from vibe.core.llm.exceptions import BackendError, BackendErrorBuilder, PayloadSummary
from vibe.core.types import (
    AvailableTool,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    MultiplexerStats,
    PerModelStats,
    Role,
    StrToolChoice,
    ToolCall,
)

if TYPE_CHECKING:
    from multiplexer_llm import Multiplexer

    from vibe.core.config import (
        ModelConfig,
        ModelPoolEntry,
        MultiplexerConfig,
        ProviderConfig,
    )


def _llm_message_to_openai(
    msg: LLMMessage, reasoning_field_name: str = "reasoning_content"
) -> dict[str, Any]:
    """Convert vibe LLMMessage to OpenAI message dict format."""
    result: dict[str, Any] = {"role": msg.role.value}

    if msg.content is not None:
        result["content"] = msg.content

    if msg.reasoning_content is not None:
        result[reasoning_field_name] = msg.reasoning_content

    if msg.tool_calls is not None:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                },
            }
            for tc in msg.tool_calls
            if tc.id is not None and tc.function.name is not None
        ]

    if msg.tool_call_id is not None:
        result["tool_call_id"] = msg.tool_call_id

    if msg.name is not None:
        result["name"] = msg.name

    return result


def _openai_message_to_llm(
    msg: dict[str, Any], reasoning_field_name: str = "reasoning_content"
) -> LLMMessage:
    """Convert OpenAI message dict to vibe LLMMessage."""
    role_str = msg.get("role", "assistant")
    role = Role(role_str) if role_str in Role.__members__ else Role.assistant

    content = msg.get("content")
    reasoning_content = msg.get(reasoning_field_name)

    tool_calls: list[ToolCall] | None = None
    if raw_tool_calls := msg.get("tool_calls"):
        tool_calls = []
        for i, tc in enumerate(raw_tool_calls):
            func_data = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    id=tc.get("id"),
                    index=tc.get("index", i),
                    function=FunctionCall(
                        name=func_data.get("name"),
                        arguments=func_data.get("arguments"),
                    ),
                    type=tc.get("type", "function"),
                )
            )

    return LLMMessage(
        role=role,
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or None,
        name=msg.get("name"),
        tool_call_id=msg.get("tool_call_id"),
    )


def _completion_to_chunk(
    completion: ChatCompletion, reasoning_field_name: str = "reasoning_content"
) -> LLMChunk:
    """Convert OpenAI ChatCompletion to vibe LLMChunk."""
    message = LLMMessage(role=Role.assistant, content="")

    if completion.choices:
        choice = completion.choices[0]
        if choice.message:
            msg_obj = choice.message
            content = msg_obj.content
            reasoning = getattr(msg_obj, reasoning_field_name, None)

            tool_calls: list[ToolCall] | None = None
            if msg_obj.tool_calls:
                tool_calls = []
                for i, tc in enumerate(msg_obj.tool_calls):
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            index=i,
                            function=FunctionCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                            type=tc.type,
                        )
                    )

            message = LLMMessage(
                role=Role.assistant,
                content=content,
                reasoning_content=reasoning,
                tool_calls=tool_calls or None,
            )

    usage: LLMUsage | None = None
    if completion.usage:
        usage = LLMUsage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )

    return LLMChunk(message=message, usage=usage)


def _tools_to_openai(tools: list[AvailableTool]) -> list[dict[str, Any]]:
    """Convert vibe tools to OpenAI tool format."""
    return [tool.model_dump(exclude_none=True) for tool in tools]


def _tool_choice_to_openai(
    tool_choice: StrToolChoice | AvailableTool | None,
) -> str | dict[str, Any] | None:
    """Convert vibe tool_choice to OpenAI format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    return tool_choice.model_dump(exclude_none=True)


class MultiplexerBackend:
    """Backend that uses multiplexer-llm for request routing and failover."""

    def __init__(
        self,
        *,
        multiplexer: Multiplexer | None = None,
        provider: ProviderConfig | None = None,
        model_configs: (
            list[tuple[ModelConfig, ProviderConfig]]
            | list[tuple[ModelConfig, ProviderConfig, ModelPoolEntry]]
            | None
        ) = None,
        multiplexer_config: MultiplexerConfig | None = None,
        timeout: float = 720.0,
    ) -> None:
        """Initialize the multiplexer backend.

        Args:
            multiplexer: Pre-configured Multiplexer instance. If provided, use it directly.
            provider: Primary provider for configuration (used for reasoning_field_name, etc.)
                Required if multiplexer_config is not provided.
            model_configs: List of model config tuples for multi-model setup.
                Can be (ModelConfig, ProviderConfig) for Phase 1 compatibility, or
                (ModelConfig, ProviderConfig, ModelPoolEntry) for Phase 2 pool config.
            multiplexer_config: MultiplexerConfig from vibe config (Phase 2).
            timeout: Request timeout in seconds.
        """
        self._multiplexer = multiplexer
        self._owns_multiplexer = multiplexer is None
        self._multiplexer_config = multiplexer_config
        self._timeout = timeout
        self._single_client: AsyncOpenAI | None = None

        # Store model configs with pool entries (Phase 2 format)
        self._pool_model_configs: (
            list[tuple[ModelConfig, ProviderConfig, ModelPoolEntry]] | None
        ) = None
        # Store model configs without pool entries (Phase 1 format)
        self._simple_model_configs: (
            list[tuple[ModelConfig, ProviderConfig]] | None
        ) = None

        # Determine which format we received
        if model_configs:
            first = model_configs[0]
            if len(first) == 3:
                # Phase 2 format: (ModelConfig, ProviderConfig, ModelPoolEntry)
                self._pool_model_configs = model_configs  # type: ignore[assignment]
            else:
                # Phase 1 format: (ModelConfig, ProviderConfig)
                self._simple_model_configs = model_configs  # type: ignore[assignment]

        # Determine provider for reasoning_field_name
        if provider is not None:
            self._provider = provider
        elif self._pool_model_configs:
            # Use first provider from pool
            self._provider = self._pool_model_configs[0][1]
        elif self._simple_model_configs:
            # Use first provider from simple configs
            self._provider = self._simple_model_configs[0][1]
        else:
            raise ValueError(
                "MultiplexerBackend requires either 'provider' or 'model_configs'"
            )

    async def __aenter__(self) -> MultiplexerBackend:
        if self._multiplexer is None:
            from multiplexer_llm import Multiplexer

            self._multiplexer = Multiplexer()

            if self._pool_model_configs:
                # Phase 2: Use pool entries with weight and fallback info
                for model_config, prov_config, entry in self._pool_model_configs:
                    client = self._create_client(prov_config)

                    if entry.is_fallback:
                        self._multiplexer.add_fallback_model(
                            model=client,
                            weight=entry.weight,
                            model_name=model_config.model_id,
                            base_url=prov_config.api_base,
                            max_concurrent=entry.max_concurrent,
                        )
                    else:
                        self._multiplexer.add_model(
                            model=client,
                            weight=entry.weight,
                            model_name=model_config.model_id,
                            base_url=prov_config.api_base,
                            max_concurrent=entry.max_concurrent,
                        )

                # Track if we have a single non-fallback model for streaming
                non_fallback_count = sum(
                    1 for _, _, e in self._pool_model_configs if not e.is_fallback
                )
                if non_fallback_count == 1:
                    # Find the single non-fallback model
                    for model_config, prov_config, entry in self._pool_model_configs:
                        if not entry.is_fallback:
                            self._single_client = self._create_client(prov_config)
                            break

            elif self._simple_model_configs:
                # Phase 1: Simple model configs without pool entries
                for model_config, prov_config in self._simple_model_configs:
                    client = self._create_client(prov_config)
                    self._multiplexer.add_model(
                        model=client,
                        weight=1,
                        model_name=model_config.model_id,
                        base_url=prov_config.api_base,
                    )

                if len(self._simple_model_configs) == 1:
                    self._single_client = self._create_client(
                        self._simple_model_configs[0][1]
                    )

            else:
                # Fallback: single provider mode
                client = self._create_client(self._provider)
                self._single_client = client
                self._multiplexer.add_model(
                    model=client,
                    weight=1,
                    model_name="default",
                    base_url=self._provider.api_base,
                )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.close()

    def _create_client(self, provider: ProviderConfig) -> AsyncOpenAI:
        """Create an AsyncOpenAI client for the given provider."""
        api_key = (
            os.getenv(provider.api_key_env_var) if provider.api_key_env_var else None
        )
        return AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=provider.api_base,
            timeout=self._timeout,
        )

    def _get_multiplexer(self) -> Multiplexer:
        """Get the multiplexer instance, creating if needed."""
        if self._multiplexer is None:
            raise RuntimeError(
                "MultiplexerBackend must be used as async context manager"
            )
        return self._multiplexer

    def get_first_model_config(self) -> ModelConfig:
        """Get the first model config from the pool for use in chat methods.

        Returns:
            The first ModelConfig from the pool.

        Raises:
            RuntimeError: If no model configs are available.
        """
        if self._pool_model_configs:
            return self._pool_model_configs[0][0]
        if self._simple_model_configs:
            return self._simple_model_configs[0][0]
        raise RuntimeError("No model configs available in MultiplexerBackend")

    def get_loading_status(self) -> str | None:
        """Get a status message describing the current model pool.

        Returns:
            A formatted string describing the primary model and provider,
            or None if using a single model configuration.
        """
        configs = self._pool_model_configs or self._simple_model_configs
        if not configs:
            return None

        # Get primary (non-fallback) models
        primary_configs: list[tuple[ModelConfig, ProviderConfig]] = []
        if self._pool_model_configs:
            primary_configs = [
                (model, provider)
                for model, provider, entry in self._pool_model_configs
                if not entry.is_fallback
            ]
        elif self._simple_model_configs:
            # Phase 1: All simple configs are considered primary
            primary_configs = [
                (model, provider) for model, provider in self._simple_model_configs
            ]

        if not primary_configs:
            return None

        # Show info for the first primary model (highest priority)
        model, provider = primary_configs[0]

        # If there's only one primary model, show specific info
        if len(primary_configs) == 1:
            return f"Generating using model {model.name} from provider {provider.name}..."

        # Multiple primary models - show the first one with a hint
        return f"Generating using model {model.name} from provider {provider.name}..."

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMChunk:
        """Execute completion via multiplexer."""
        multiplexer = self._get_multiplexer()
        reasoning_field = self._provider.reasoning_field_name

        openai_messages = [
            _llm_message_to_openai(msg, reasoning_field) for msg in messages
        ]

        kwargs: dict[str, Any] = {
            "messages": openai_messages,
            "model": model.model_id,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = _tools_to_openai(tools)

        if tool_choice is not None:
            kwargs["tool_choice"] = _tool_choice_to_openai(tool_choice)

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        try:
            completion = await multiplexer.chat.completions.create(**kwargs)
            return _completion_to_chunk(completion, reasoning_field)

        except Exception as e:
            raise self._map_exception(
                e,
                model=model.model_id,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Stream completion responses.

        Strategy for Phase 1:
        - If single model configured: use AsyncOpenAI client directly with stream=True
        - If multiple models: fall back to non-streaming, yield result as single chunk
        """
        if self._single_client is not None:
            async for chunk in self._stream_with_client(
                client=self._single_client,
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                extra_headers=extra_headers,
            ):
                yield chunk
        else:
            result = await self.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                extra_headers=extra_headers,
            )
            yield result

    async def _stream_with_client(
        self,
        *,
        client: AsyncOpenAI,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Stream using a single AsyncOpenAI client."""
        reasoning_field = self._provider.reasoning_field_name

        openai_messages = [
            _llm_message_to_openai(msg, reasoning_field) for msg in messages
        ]

        kwargs: dict[str, Any] = {
            "messages": openai_messages,
            "model": model.model_id,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            kwargs["tools"] = _tools_to_openai(tools)

        if tool_choice is not None:
            kwargs["tool_choice"] = _tool_choice_to_openai(tool_choice)

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                llm_chunk = self._parse_stream_chunk(chunk, reasoning_field)
                yield llm_chunk

        except Exception as e:
            raise self._map_exception(
                e,
                model=model.model_id,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    def _parse_stream_chunk(
        self, chunk: Any, reasoning_field_name: str
    ) -> LLMChunk:
        """Parse a streaming chunk into LLMChunk."""
        message = LLMMessage(role=Role.assistant, content="")

        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta:
                content = delta.content
                reasoning = getattr(delta, reasoning_field_name, None)

                tool_calls: list[ToolCall] | None = None
                if delta.tool_calls:
                    tool_calls = []
                    for tc in delta.tool_calls:
                        func = tc.function
                        tool_calls.append(
                            ToolCall(
                                id=tc.id,
                                index=tc.index,
                                function=FunctionCall(
                                    name=func.name if func else None,
                                    arguments=func.arguments if func else None,
                                ),
                                type=tc.type or "function",
                            )
                        )

                message = LLMMessage(
                    role=Role.assistant,
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls or None,
                )

        usage: LLMUsage | None = None
        if chunk.usage:
            usage = LLMUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
            )

        return LLMChunk(message=message, usage=usage)

    async def count_tokens(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        tools: list[AvailableTool] | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> int:
        """Count tokens by making a minimal completion request."""
        probe_messages = list(messages)
        if not probe_messages or probe_messages[-1].role != Role.user:
            probe_messages.append(LLMMessage(role=Role.user, content=""))

        result = await self.complete(
            model=model,
            messages=probe_messages,
            temperature=temperature,
            tools=tools,
            max_tokens=16,
            tool_choice=tool_choice,
            extra_headers=extra_headers,
        )

        if result.usage is None:
            raise ValueError("Missing usage in completion response")

        return result.usage.prompt_tokens

    async def close(self) -> None:
        """Close the backend and release resources."""
        if self._owns_multiplexer and self._multiplexer is not None:
            await self._multiplexer.async_reset()
            self._multiplexer = None
        self._single_client = None

    def get_stats(self) -> MultiplexerStats:
        """Get current multiplexer statistics."""
        # Determine mode from configuration
        mode = "single"
        if self._multiplexer_config is not None:
            mode = self._multiplexer_config.mode.value

        # If multiplexer not yet initialized, return config-based stats
        if self._multiplexer is None:
            # Determine pool size from stored model configs
            pool_size = 0
            if self._pool_model_configs:
                pool_size = len(self._pool_model_configs)
            elif self._simple_model_configs:
                pool_size = len(self._simple_model_configs)

            return MultiplexerStats(
                enabled=self._multiplexer_config.enabled if self._multiplexer_config else False,
                mode=mode,
                models_in_pool=pool_size,
                models_available=pool_size,
                models_disabled=0,
                per_model={},
            )

        raw_stats = self._multiplexer.get_stats()

        per_model: dict[str, PerModelStats] = {}
        models_disabled = 0
        now = time.time()

        # Get weighted models for disabled state
        all_models = (
            self._multiplexer._weighted_models + self._multiplexer._fallback_models
        )
        disabled_until_map = {wm.model_name: wm.disabled_until for wm in all_models}

        for model_name, stats in raw_stats.items():
            disabled_until = disabled_until_map.get(model_name)
            is_disabled = disabled_until is not None and disabled_until > now

            if is_disabled:
                models_disabled += 1

            per_model[model_name] = PerModelStats(
                success_count=stats["success"],
                rate_limit_count=stats["rateLimited"],
                fail_count=stats["failed"],
                is_disabled=is_disabled,
                disabled_until_timestamp=disabled_until if is_disabled else None,
            )

        return MultiplexerStats(
            enabled=True,
            mode=mode,
            models_in_pool=len(raw_stats),
            models_available=len(raw_stats) - models_disabled,
            models_disabled=models_disabled,
            per_model=per_model,
        )

    def _map_exception(
        self,
        error: Exception,
        *,
        model: str,
        messages: list[LLMMessage],
        temperature: float,
        has_tools: bool,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> BackendError:
        """Map multiplexer exceptions to vibe BackendError."""
        from multiplexer_llm.exceptions import (
            APIError,
            AuthenticationError,
            ModelNotFoundError,
            ModelSelectionError,
            MultiplexerError,
            RateLimitError,
            ServiceUnavailableError,
        )

        status: int | None = None
        reason: str | None = str(error)
        endpoint = self._provider.api_base
        raw_response: str | None = None

        match error:
            case RateLimitError():
                status = 429
                reason = "Rate limit exceeded"
            case AuthenticationError():
                status = 401
                reason = "Authentication failed"
            case ModelNotFoundError():
                status = 404
                reason = f"Model not found: {model}"
            case ServiceUnavailableError():
                status = 503
                reason = "Service unavailable"
            case ModelSelectionError():
                status = 503
                reason = "No models available"
            case APIError() as api_err:
                status = api_err.status_code
                reason = api_err.message
                # Try to get raw response body from APIError
                if hasattr(api_err, "body") and api_err.body:
                    raw_response = str(api_err.body)
                elif hasattr(api_err, "response") and api_err.response:
                    raw_response = str(api_err.response)
            case MultiplexerError() as mux_err:
                reason = mux_err.message
            case _:
                reason = str(error)

        payload_summary = PayloadSummary(
            model=model,
            message_count=len(messages),
            approx_chars=sum(len(m.content or "") for m in messages),
            temperature=temperature,
            has_tools=has_tools,
            tool_choice=tool_choice,
        )

        return BackendError(
            provider=self._provider.name,
            endpoint=endpoint,
            status=status,
            reason=reason,
            headers={},
            body_text=raw_response,
            parsed_error=reason,
            model=model,
            payload_summary=payload_summary,
        )
