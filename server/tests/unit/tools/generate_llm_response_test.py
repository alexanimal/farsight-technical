"""Unit tests for the generate_llm_response tool.

This module tests the generate_llm_response function and its various
configurations including prompts, instructions, streaming, web search,
reasoning effort, images, and files.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import (ChatCompletion, ChatCompletionChunk,
                               ChatCompletionMessage)

from src.llm.openai_client import OpenAIClient
from src.tools.generate_llm_response import generate_llm_response


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the OpenAI client singleton before each test."""
    import src.llm.openai_client as openai_client_module

    # Reset singleton before test
    openai_client_module._default_client = None
    yield
    # Clean up after test
    openai_client_module._default_client = None


@pytest.fixture(autouse=True)
def prevent_real_openai_initialization():
    """Prevent real OpenAI clients from being initialized."""
    with patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_class:
        # Make it raise an exception if someone tries to create a real client
        # This ensures we catch any code paths that bypass our mocks
        mock_openai_class.side_effect = Exception(
            "Real OpenAI client should not be created in unit tests. "
            "Ensure all tests use mocked clients via get_client patch."
        )
        yield


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAIClient instance."""
    client = MagicMock(spec=OpenAIClient)
    client.chat_completion = AsyncMock()
    client.is_initialized = MagicMock(return_value=True)
    client.client = client  # For property access if needed
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


class TestGenerateLLMResponseBasic:
    """Test basic generate_llm_response functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_simple(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test simple LLM response generation."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            response = await generate_llm_response(
                prompt="Hello!", model="gpt-4", client=None
            )

            assert response == mock_chat_completion
            mock_openai_client.chat_completion.assert_called_once()
            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["model"] == "gpt-4"
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_custom_client(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with custom client."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        response = await generate_llm_response(
            prompt="Hello!", model="gpt-4", client=mock_openai_client
        )

        assert response == mock_chat_completion
        mock_openai_client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_instructions(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with instructions."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Hello!",
                model="gpt-4",
                instructions="You are a helpful assistant.",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert len(call_args["messages"]) == 2
            assert call_args["messages"][0]["role"] == "system"
            assert call_args["messages"][0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_system_prompt(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with system prompt."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Hello!",
                model="gpt-4",
                system_prompt="You are a helpful assistant.",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert len(call_args["messages"]) == 2
            assert call_args["messages"][0]["role"] == "system"
            assert call_args["messages"][0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_llm_response_system_prompt_overrides_instructions(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test that system_prompt takes precedence over instructions."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Hello!",
                model="gpt-4",
                instructions="Old instructions",
                system_prompt="New system prompt",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["messages"][0]["content"] == "New system prompt"

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_temperature(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with temperature."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Hello!", model="gpt-4", temperature=0.7, client=None
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_max_tokens(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with max_tokens."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Hello!", model="gpt-4", max_tokens=100, client=None
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["max_tokens"] == 100


class TestGenerateLLMResponseStreaming:
    """Test generate_llm_response streaming functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_streaming(
        self, mock_openai_client, mock_chat_completion_chunk
    ):
        """Test streaming LLM response generation."""

        # Create an async iterator for streaming
        async def stream_generator():
            yield mock_chat_completion_chunk

        mock_openai_client.chat_completion.return_value = stream_generator()

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            response = await generate_llm_response(
                prompt="Hello!", model="gpt-4", stream=True, client=None
            )

            # Response should be an async iterator
            assert hasattr(response, "__aiter__")
            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["stream"] is True

    @pytest.mark.asyncio
    async def test_generate_llm_response_streaming_iteration(
        self, mock_openai_client, mock_chat_completion_chunk
    ):
        """Test iterating over streaming response."""

        # Create an async iterator for streaming
        async def stream_generator():
            yield mock_chat_completion_chunk
            yield mock_chat_completion_chunk

        mock_openai_client.chat_completion.return_value = stream_generator()

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            response = await generate_llm_response(
                prompt="Hello!", model="gpt-4", stream=True, client=None
            )

            chunks = []
            async for chunk in response:
                chunks.append(chunk)

            assert len(chunks) == 2


class TestGenerateLLMResponseWebSearch:
    """Test generate_llm_response web search functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_web_search(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with web search enabled."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="What's the weather?",
                model="gpt-4",
                enable_web_search=True,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["enable_web_search"] is True


class TestGenerateLLMResponseReasoningEffort:
    """Test generate_llm_response reasoning effort functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_reasoning_effort(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with reasoning effort."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Solve this problem",
                model="o1-preview",
                reasoning_effort="high",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_generate_llm_response_invalid_reasoning_effort(
        self, mock_openai_client
    ):
        """Test generate_llm_response with invalid reasoning effort."""
        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_response(
                    prompt="Hello!",
                    model="o1-preview",
                    reasoning_effort="invalid",
                    client=None,
                )
            assert "Invalid reasoning_effort" in str(exc_info.value)


class TestGenerateLLMResponseImages:
    """Test generate_llm_response image functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_image_url(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with image URL."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="What's in this image?",
                model="gpt-4-vision-preview",
                images=["https://example.com/image.jpg"],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert isinstance(user_message["content"], list)
            assert len(user_message["content"]) == 2  # text + image
            assert user_message["content"][0]["type"] == "text"
            assert user_message["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_multiple_images(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with multiple images."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Compare these images",
                model="gpt-4-vision-preview",
                images=[
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                ],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert len(user_message["content"]) == 3  # text + 2 images

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_image_dict(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with image dict."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            image_dict = {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            }
            await generate_llm_response(
                prompt="What's in this image?",
                model="gpt-4-vision-preview",
                images=[image_dict],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert user_message["content"][1] == image_dict

    @pytest.mark.asyncio
    async def test_generate_llm_response_invalid_image_format(self, mock_openai_client):
        """Test generate_llm_response with invalid image format."""
        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_response(
                    prompt="Hello!",
                    model="gpt-4-vision-preview",
                    images=[123],  # Invalid format
                    client=None,
                )
            assert "Invalid image format" in str(exc_info.value)


class TestGenerateLLMResponseFiles:
    """Test generate_llm_response file functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_file_id(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with file ID."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Summarize this document",
                model="gpt-4",
                files=["file-abc123"],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert "attachments" in user_message
            assert len(user_message["attachments"]) == 1
            assert user_message["attachments"][0]["file_id"] == "file-abc123"

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_multiple_files(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with multiple files."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Compare these documents",
                model="gpt-4",
                files=["file-abc123", "file-def456"],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert len(user_message["attachments"]) == 2

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_file_dict(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with file dict."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            file_dict = {"file_id": "file-abc123", "tools": [{"type": "file_search"}]}
            await generate_llm_response(
                prompt="Summarize this document",
                model="gpt-4",
                files=[file_dict],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert user_message["attachments"][0] == file_dict

    @pytest.mark.asyncio
    async def test_generate_llm_response_invalid_file_format(self, mock_openai_client):
        """Test generate_llm_response with invalid file format."""
        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_response(
                    prompt="Hello!",
                    model="gpt-4",
                    files=[123],  # Invalid format
                    client=None,
                )
            assert "Invalid file format" in str(exc_info.value)


class TestGenerateLLMResponseAdvanced:
    """Test advanced generate_llm_response functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_tools(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with tools."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                    },
                }
            ]
            await generate_llm_response(
                prompt="What's the weather?", model="gpt-4", tools=tools, client=None
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["tools"] == tools

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_tool_choice(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with tool_choice."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Hello!", model="gpt-4", tool_choice="none", client=None
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["tool_choice"] == "none"

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_response_format(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with response format."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            response_format = {"type": "json_object"}
            await generate_llm_response(
                prompt="Return JSON",
                model="gpt-4",
                response_format=response_format,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["response_format"] == response_format

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_kwargs(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with additional kwargs."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Hello!",
                model="gpt-4",
                top_p=0.9,
                frequency_penalty=0.5,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["top_p"] == 0.9
            assert call_args["frequency_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_llm_response_return_text(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with return_text option."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            response = await generate_llm_response(
                prompt="Hello!", model="gpt-4", return_text=True, client=None
            )

            assert isinstance(response, str)
            assert response == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_generate_llm_response_return_text_empty_content(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response return_text with empty content."""
        mock_chat_completion.choices[0].message.content = None
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            response = await generate_llm_response(
                prompt="Hello!", model="gpt-4", return_text=True, client=None
            )

            assert response == ""

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_images_and_files(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with both images and files."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="Analyze this",
                model="gpt-4-vision-preview",
                images=["https://example.com/image.jpg"],
                files=["file-abc123"],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            # Should use content array when files are present
            assert isinstance(user_message["content"], list)
            assert "attachments" in user_message


class TestGenerateLLMResponseErrorHandling:
    """Test generate_llm_response error handling."""

    @pytest.mark.asyncio
    async def test_generate_llm_response_api_error(self, mock_openai_client):
        """Test generate_llm_response when API call fails."""
        mock_openai_client.chat_completion.side_effect = Exception("API error")

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(Exception) as exc_info:
                await generate_llm_response(prompt="Hello!", model="gpt-4", client=None)
            # Should raise an exception (the exact message doesn't matter for this test)
            assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_generate_llm_response_empty_prompt(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with empty prompt."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            # Should handle empty prompt gracefully
            await generate_llm_response(prompt="", model="gpt-4", client=None)

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert user_message["content"] == ""

    @pytest.mark.asyncio
    async def test_generate_llm_response_no_prompt_with_images(
        self, mock_openai_client, mock_chat_completion
    ):
        """Test generate_llm_response with images but no prompt."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion

        # Patch get_client in the module where it's used
        # Get the actual module object to ensure we're patching the right namespace
        generate_llm_response_module = sys.modules["src.tools.generate_llm_response"]
        with patch.object(
            generate_llm_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_response(
                prompt="",
                model="gpt-4-vision-preview",
                images=["https://example.com/image.jpg"],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            # Should still have content array with image
            assert isinstance(user_message["content"], list)
            assert len(user_message["content"]) == 1  # Just image, no text
            assert user_message["content"][0]["type"] == "image_url"
