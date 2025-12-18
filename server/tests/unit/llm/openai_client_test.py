"""Unit tests for the OpenAI client module.

This module tests the OpenAIClient class and its various methods,
including client initialization, chat completion, streaming, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletion, ChatCompletionChunk,
                               ChatCompletionMessage)

from src.llm.openai_client import (OpenAIClient, close_default_client,
                                   get_client)


@pytest.fixture
def mock_openai_client():
    """Create a mock AsyncOpenAI client instance."""
    client = MagicMock(spec=AsyncOpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_chat_completion():
    """Create a mock ChatCompletion response."""
    response = MagicMock(spec=ChatCompletion)
    response.id = "chatcmpl-123"
    response.object = "chat.completion"
    response.created = 1234567890
    response.model = "gpt-4"

    message = MagicMock(spec=ChatCompletionMessage)
    message.role = "assistant"
    message.content = "Hello! How can I help you?"
    message.tool_calls = None

    choice = MagicMock()
    choice.index = 0
    choice.message = message
    choice.finish_reason = "stop"

    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15

    return response


@pytest.fixture
def mock_chat_completion_chunk():
    """Create a mock ChatCompletionChunk for streaming."""
    chunk = MagicMock(spec=ChatCompletionChunk)
    chunk.id = "chatcmpl-123"
    chunk.object = "chat.completion.chunk"
    chunk.created = 1234567890
    chunk.model = "gpt-4"

    delta = MagicMock()
    delta.role = None
    delta.content = "Hello"
    delta.tool_calls = None

    choice = MagicMock()
    choice.index = 0
    choice.delta = delta
    choice.finish_reason = None

    chunk.choices = [choice]

    return chunk


class TestOpenAIClientInitialization:
    """Test OpenAIClient initialization."""

    def test_init_with_default_settings(self, monkeypatch):
        """Test initialization with default settings."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-api-key"
            client = OpenAIClient()
            assert client._api_key == "test-api-key"
            assert client._base_url is None
            assert client._client is None

    def test_init_with_custom_api_key(self, monkeypatch):
        """Test initialization with custom API key."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "default-key"
            client = OpenAIClient(api_key="custom-key")
            assert client._api_key == "custom-key"
            assert client._base_url is None

    def test_init_with_custom_base_url(self, monkeypatch):
        """Test initialization with custom base URL."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            client = OpenAIClient(base_url="https://api.custom.com")
            assert client._api_key == "test-key"
            assert client._base_url == "https://api.custom.com"

    def test_init_with_both_custom(self, monkeypatch):
        """Test initialization with both custom API key and base URL."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "default-key"
            client = OpenAIClient(
                api_key="custom-key", base_url="https://api.custom.com"
            )
            assert client._api_key == "custom-key"
            assert client._base_url == "https://api.custom.com"

    def test_init_client_not_initialized(self, monkeypatch):
        """Test that client is None after initialization."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            client = OpenAIClient()
            assert client._client is None
            assert not client.is_initialized()


class TestOpenAIClientInitialize:
    """Test OpenAIClient.initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_openai_client, monkeypatch):
        """Test successful client initialization."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-api-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient()
                await client.initialize()

                mock_openai_class.assert_called_once_with(api_key="test-api-key")
                assert client._client == mock_openai_client

    @pytest.mark.asyncio
    async def test_initialize_with_custom_api_key(
        self, mock_openai_client, monkeypatch
    ):
        """Test initialization with custom API key."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "default-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient(api_key="custom-key")
                await client.initialize()

                mock_openai_class.assert_called_once_with(api_key="custom-key")

    @pytest.mark.asyncio
    async def test_initialize_with_base_url(self, mock_openai_client, monkeypatch):
        """Test initialization with custom base URL."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient(base_url="https://api.custom.com")
                await client.initialize()

                mock_openai_class.assert_called_once_with(
                    api_key="test-key", base_url="https://api.custom.com"
                )

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(
        self, mock_openai_client, monkeypatch
    ):
        """Test that re-initialization is skipped with warning."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient()
                await client.initialize()
                await client.initialize()  # Second call

                # Should only be called once
                assert mock_openai_class.call_count == 1
                assert client._client == mock_openai_client

    @pytest.mark.asyncio
    async def test_initialize_no_api_key(self, monkeypatch):
        """Test initialization failure when API key is missing."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = None

            client = OpenAIClient()
            with pytest.raises(ValueError) as exc_info:
                await client.initialize()
            assert "API key is required" in str(exc_info.value)
            assert client._client is None

    @pytest.mark.asyncio
    async def test_initialize_empty_api_key(self, monkeypatch):
        """Test initialization failure when API key is empty string."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = ""

            client = OpenAIClient()
            with pytest.raises(ValueError) as exc_info:
                await client.initialize()
            assert "API key is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, monkeypatch):
        """Test initialization failure raises exception."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.side_effect = Exception("Connection failed")
                client = OpenAIClient()

                with pytest.raises(Exception) as exc_info:
                    await client.initialize()
                assert "Connection failed" in str(exc_info.value)
                assert client._client is None


class TestOpenAIClientClose:
    """Test OpenAIClient.close() method."""

    @pytest.mark.asyncio
    async def test_close_success(self, mock_openai_client, monkeypatch):
        """Test successful client closure."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient()
                await client.initialize()
                await client.close()

                assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self, monkeypatch):
        """Test closing when client is not initialized."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            client = OpenAIClient()
            # Should not raise an error
            await client.close()
            assert client._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_openai_client, monkeypatch):
        """Test that close can be called multiple times safely."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient()
                await client.initialize()
                await client.close()
                await client.close()  # Second call

                assert client._client is None


class TestOpenAIClientContextManager:
    """Test OpenAIClient as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self, mock_openai_client, monkeypatch):
        """Test using client as async context manager."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                async with OpenAIClient() as client:
                    assert client._client == mock_openai_client
                    assert client.is_initialized()

                # Client should be closed after exiting context
                assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self, mock_openai_client, monkeypatch):
        """Test that context manager returns the client instance."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                async with OpenAIClient() as client:
                    assert isinstance(client, OpenAIClient)

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(
        self, mock_openai_client, monkeypatch
    ):
        """Test context manager properly closes client even on exception."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                try:
                    async with OpenAIClient() as client:
                        raise ValueError("Test exception")
                except ValueError:
                    pass

                # Client should still be closed
                assert client._client is None


class TestOpenAIClientProperty:
    """Test OpenAIClient.client property."""

    @pytest.mark.asyncio
    async def test_client_property_success(self, mock_openai_client, monkeypatch):
        """Test accessing client property when initialized."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient()
                await client.initialize()

                assert client.client == mock_openai_client

    def test_client_property_not_initialized(self, monkeypatch):
        """Test accessing client property when not initialized raises error."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            client = OpenAIClient()
            with pytest.raises(RuntimeError) as exc_info:
                _ = client.client
            assert "not initialized" in str(exc_info.value)


class TestOpenAIClientChatCompletion:
    """Test OpenAIClient.chat_completion() method."""

    @pytest.mark.asyncio
    async def test_chat_completion_success(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test successful chat completion."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                response = await client.chat_completion(
                    messages=messages, model="gpt-4"
                )

                assert response == mock_chat_completion
                mock_openai_client.chat.completions.create.assert_called_once()
                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["model"] == "gpt-4"
                assert call_args["messages"] == messages

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test chat completion with temperature parameter."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                await client.chat_completion(
                    messages=messages, model="gpt-4", temperature=0.7
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_with_max_tokens(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test chat completion with max_tokens parameter."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                await client.chat_completion(
                    messages=messages, model="gpt-4", max_tokens=100
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(
        self, mock_openai_client, mock_chat_completion_chunk, monkeypatch
    ):
        """Test streaming chat completion."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client

                # Create an async iterator for streaming
                async def stream_generator():
                    yield mock_chat_completion_chunk

                mock_openai_client.chat.completions.create.return_value = (
                    stream_generator()
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                response = await client.chat_completion(
                    messages=messages, model="gpt-4", stream=True
                )

                # Response should be an async iterator
                assert hasattr(response, "__aiter__")
                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["stream"] is True

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test chat completion with tools."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                        },
                    }
                ]
                await client.chat_completion(
                    messages=messages, model="gpt-4", tools=tools
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["tools"] == tools

    @pytest.mark.asyncio
    async def test_chat_completion_with_web_search(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test chat completion with web search enabled."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                await client.chat_completion(
                    messages=messages, model="gpt-4", enable_web_search=True
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert "tools" in call_args
                assert any(
                    tool.get("type") == "web_search" for tool in call_args["tools"]
                )

    @pytest.mark.asyncio
    async def test_chat_completion_with_reasoning_effort(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test chat completion with reasoning effort."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                await client.chat_completion(
                    messages=messages,
                    model="o1-preview",
                    reasoning_effort="high",
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_chat_completion_with_response_format(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test chat completion with response format."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                response_format = {"type": "json_object"}
                await client.chat_completion(
                    messages=messages, model="gpt-4", response_format=response_format
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["response_format"] == response_format

    @pytest.mark.asyncio
    async def test_chat_completion_with_kwargs(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test chat completion with additional kwargs."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                await client.chat_completion(
                    messages=messages, model="gpt-4", top_p=0.9, frequency_penalty=0.5
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args["top_p"] == 0.9
                assert call_args["frequency_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_chat_completion_not_initialized(self, monkeypatch):
        """Test chat completion when client is not initialized."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            client = OpenAIClient()
            messages = [{"role": "user", "content": "Hello!"}]

            with pytest.raises(RuntimeError) as exc_info:
                await client.chat_completion(messages=messages, model="gpt-4")
            assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, mock_openai_client, monkeypatch):
        """Test chat completion when API call fails."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.side_effect = Exception(
                    "API error"
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                with pytest.raises(Exception) as exc_info:
                    await client.chat_completion(messages=messages, model="gpt-4")
                assert "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_chat_completion_web_search_with_existing_tools(
        self, mock_openai_client, mock_chat_completion, monkeypatch
    ):
        """Test that web search doesn't duplicate if tools already include it."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                mock_openai_client.chat.completions.create.return_value = (
                    mock_chat_completion
                )

                client = OpenAIClient()
                await client.initialize()

                messages = [{"role": "user", "content": "Hello!"}]
                tools = [{"type": "web_search"}]
                await client.chat_completion(
                    messages=messages,
                    model="gpt-4",
                    tools=tools,
                    enable_web_search=True,
                )

                call_args = mock_openai_client.chat.completions.create.call_args[1]
                # Should only have one web_search tool
                web_search_count = sum(
                    1 for tool in call_args["tools"] if tool.get("type") == "web_search"
                )
                assert web_search_count == 1


class TestOpenAIClientIsInitialized:
    """Test OpenAIClient.is_initialized() method."""

    @pytest.mark.asyncio
    async def test_is_initialized_true_when_initialized(
        self, mock_openai_client, monkeypatch
    ):
        """Test is_initialized returns True when client is initialized."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient()
                assert not client.is_initialized()

                await client.initialize()
                assert client.is_initialized()

    @pytest.mark.asyncio
    async def test_is_initialized_false_after_close(
        self, mock_openai_client, monkeypatch
    ):
        """Test is_initialized returns False after close."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client
                client = OpenAIClient()
                await client.initialize()
                assert client.is_initialized()

                await client.close()
                assert not client.is_initialized()


class TestOpenAIClientSingleton:
    """Test OpenAIClient singleton functions."""

    @pytest.mark.asyncio
    async def test_get_client_creates_instance(self, mock_openai_client, monkeypatch):
        """Test get_client creates a new instance."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client

                # Reset singleton
                import src.llm.openai_client as openai_client_module
                from src.llm.openai_client import _default_client

                openai_client_module._default_client = None

                client = await get_client()
                assert isinstance(client, OpenAIClient)
                assert client.is_initialized()

    @pytest.mark.asyncio
    async def test_get_client_returns_existing_instance(
        self, mock_openai_client, monkeypatch
    ):
        """Test get_client returns existing instance on subsequent calls."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client

                # Reset singleton
                import src.llm.openai_client as openai_client_module

                openai_client_module._default_client = None

                client1 = await get_client()
                client2 = await get_client()

                assert client1 is client2
                # Should only initialize once
                assert mock_openai_class.call_count == 1

    @pytest.mark.asyncio
    async def test_close_default_client_success(self, mock_openai_client, monkeypatch):
        """Test close_default_client closes the singleton."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client

                # Reset singleton
                import src.llm.openai_client as openai_client_module

                openai_client_module._default_client = None

                client = await get_client()
                assert client.is_initialized()

                await close_default_client()
                assert not client.is_initialized()

                # Verify singleton is reset
                import src.llm.openai_client as openai_client_module

                assert openai_client_module._default_client is None

    @pytest.mark.asyncio
    async def test_close_default_client_when_none(self):
        """Test close_default_client when no client exists."""
        # Reset singleton
        import src.llm.openai_client as openai_client_module

        openai_client_module._default_client = None

        # Should not raise an error
        await close_default_client()

    @pytest.mark.asyncio
    async def test_get_client_after_close(self, mock_openai_client, monkeypatch):
        """Test get_client creates new instance after close."""
        with patch("src.llm.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"

            with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
                mock_openai_class.return_value = mock_openai_client

                # Reset singleton
                import src.llm.openai_client as openai_client_module

                openai_client_module._default_client = None

                client1 = await get_client()
                await close_default_client()

                client2 = await get_client()
                # Should be a new instance
                assert client1 is not client2
