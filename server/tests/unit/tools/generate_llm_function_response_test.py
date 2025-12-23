"""Unit tests for the generate_llm_function_response tool.

This module tests the generate_llm_function_response function and its various
configurations including function calling, tool execution, structured outputs,
and error handling.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall

from src.llm.openai_client import OpenAIClient
from src.tools.generate_llm_function_response import generate_llm_function_response


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
def sample_tools():
    """Create sample tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]


@pytest.fixture
def mock_chat_completion_with_tool_call(sample_tools):
    """Create a mock ChatCompletion response with a tool call."""
    response = MagicMock(spec=ChatCompletion)
    response.id = "chatcmpl-123"
    response.object = "chat.completion"
    response.created = 1234567890
    response.model = "gpt-4"

    # Create tool call
    tool_call = MagicMock(spec=ChatCompletionMessageToolCall)
    tool_call.id = "call_abc123"
    tool_call.type = "function"
    tool_call.function = MagicMock()
    tool_call.function.name = "get_weather"
    tool_call.function.arguments = json.dumps({"location": "San Francisco", "unit": "fahrenheit"})

    message = MagicMock(spec=ChatCompletionMessage)
    message.role = "assistant"
    message.content = None
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.index = 0
    choice.message = message
    choice.finish_reason = "tool_calls"

    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15

    return response


@pytest.fixture
def mock_chat_completion_with_multiple_tool_calls(sample_tools):
    """Create a mock ChatCompletion response with multiple tool calls."""
    response = MagicMock(spec=ChatCompletion)
    response.id = "chatcmpl-123"
    response.object = "chat.completion"
    response.created = 1234567890
    response.model = "gpt-4"

    # Create first tool call
    tool_call_1 = MagicMock(spec=ChatCompletionMessageToolCall)
    tool_call_1.id = "call_abc123"
    tool_call_1.type = "function"
    tool_call_1.function = MagicMock()
    tool_call_1.function.name = "get_weather"
    tool_call_1.function.arguments = json.dumps({"location": "San Francisco"})

    # Create second tool call
    tool_call_2 = MagicMock(spec=ChatCompletionMessageToolCall)
    tool_call_2.id = "call_def456"
    tool_call_2.type = "function"
    tool_call_2.function = MagicMock()
    tool_call_2.function.name = "get_weather"
    tool_call_2.function.arguments = json.dumps({"location": "New York"})

    message = MagicMock(spec=ChatCompletionMessage)
    message.role = "assistant"
    message.content = None
    message.tool_calls = [tool_call_1, tool_call_2]

    choice = MagicMock()
    choice.index = 0
    choice.message = message
    choice.finish_reason = "tool_calls"

    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15

    return response


@pytest.fixture
def mock_chat_completion_no_tool_call():
    """Create a mock ChatCompletion response without tool calls."""
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


class TestGenerateLLMFunctionResponseBasic:
    """Test basic generate_llm_function_response functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_simple(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test simple function response generation."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            result = await generate_llm_function_response(
                prompt="What's the weather in San Francisco?",
                tools=sample_tools,
                model="gpt-4",
                client=None,
            )

            assert isinstance(result, dict)
            assert result["function_name"] == "get_weather"
            assert result["arguments"]["location"] == "San Francisco"
            assert result["arguments"]["unit"] == "fahrenheit"
            assert "call_id" in result

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_custom_client(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with custom client."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        result = await generate_llm_function_response(
            prompt="What's the weather?",
            tools=sample_tools,
            model="gpt-4",
            client=mock_openai_client,
        )

        assert result["function_name"] == "get_weather"
        mock_openai_client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_no_tools(self, mock_openai_client):
        """Test generate_llm_function_response without tools raises error."""
        with pytest.raises(ValueError) as exc_info:
            await generate_llm_function_response(
                prompt="Hello!", tools=[], model="gpt-4", client=mock_openai_client
            )
        assert "tools parameter is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_no_tool_calls(
        self, mock_openai_client, mock_chat_completion_no_tool_call, sample_tools
    ):
        """Test generate_llm_function_response when no tool calls are made."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_no_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            result = await generate_llm_function_response(
                prompt="Hello!", tools=sample_tools, model="gpt-4", client=None
            )

            # Should return the full ChatCompletion when no tool calls
            assert isinstance(result, ChatCompletion)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_instructions(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with instructions."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                instructions="You are a helpful assistant.",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert len(call_args["messages"]) == 2
            assert call_args["messages"][0]["role"] == "system"
            assert call_args["messages"][0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_system_prompt(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with system prompt."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                system_prompt="You are a helpful assistant.",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["messages"][0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_tool_choice_auto(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response defaults tool_choice to auto."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_tool_choice_none(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with tool_choice=none."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                tool_choice="none",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["tool_choice"] == "none"


class TestGenerateLLMFunctionResponseToolExecution:
    """Test generate_llm_function_response tool execution functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_execute_tools(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with tool execution."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        async def mock_get_weather(location: str, unit: str = "fahrenheit"):
            return {"temperature": 72, "condition": "sunny", "location": location}

        tool_executors = {"get_weather": mock_get_weather}

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            result = await generate_llm_function_response(
                prompt="What's the weather in San Francisco?",
                tools=sample_tools,
                model="gpt-4",
                execute_tools=True,
                tool_executors=tool_executors,
                client=None,
            )

            assert isinstance(result, dict)
            assert result["function_name"] == "get_weather"
            assert "result" in result
            assert result["result"]["temperature"] == 72
            assert result["result"]["condition"] == "sunny"

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_execute_tools_no_executors(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with execute_tools but no executors."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_function_response(
                    prompt="What's the weather?",
                    tools=sample_tools,
                    model="gpt-4",
                    execute_tools=True,
                    tool_executors=None,
                    client=None,
                )
            assert "tool_executors is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_execute_tools_missing_executor(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with missing tool executor."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        # Provide tool_executors with a different tool, but missing get_weather
        async def mock_other_tool():
            return {"result": "other"}

        tool_executors = {"other_tool": mock_other_tool}  # Missing get_weather

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_function_response(
                    prompt="What's the weather?",
                    tools=sample_tools,
                    model="gpt-4",
                    execute_tools=True,
                    tool_executors=tool_executors,
                    client=None,
                )
            assert "No executor found for tool" in str(exc_info.value)
            assert "get_weather" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_execute_tools_error(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response when tool execution fails."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        async def failing_get_weather(location: str, unit: str = "fahrenheit"):
            raise ValueError("Weather service unavailable")

        tool_executors = {"get_weather": failing_get_weather}

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            result = await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                execute_tools=True,
                tool_executors=tool_executors,
                client=None,
            )

            assert "error" in result
            assert result["result"] is None
            assert "Weather service unavailable" in result["error"]


class TestGenerateLLMFunctionResponseMultipleCalls:
    """Test generate_llm_function_response with multiple tool calls."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_multiple_calls(
        self,
        mock_openai_client,
        mock_chat_completion_with_multiple_tool_calls,
        sample_tools,
    ):
        """Test generate_llm_function_response with multiple tool calls."""
        mock_openai_client.chat_completion.return_value = (
            mock_chat_completion_with_multiple_tool_calls
        )

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            result = await generate_llm_function_response(
                prompt="Get weather for SF and NYC",
                tools=sample_tools,
                model="gpt-4",
                return_all_calls=True,
                client=None,
            )

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["arguments"]["location"] == "San Francisco"
            assert result[1]["arguments"]["location"] == "New York"

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_multiple_calls_execute(
        self,
        mock_openai_client,
        mock_chat_completion_with_multiple_tool_calls,
        sample_tools,
    ):
        """Test generate_llm_function_response with multiple tool calls and execution."""
        mock_openai_client.chat_completion.return_value = (
            mock_chat_completion_with_multiple_tool_calls
        )

        async def mock_get_weather(location: str, unit: str = "fahrenheit"):
            return {"temperature": 72, "condition": "sunny", "location": location}

        tool_executors = {"get_weather": mock_get_weather}

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            result = await generate_llm_function_response(
                prompt="Get weather for SF and NYC",
                tools=sample_tools,
                model="gpt-4",
                execute_tools=True,
                tool_executors=tool_executors,
                return_all_calls=True,
                client=None,
            )

            assert isinstance(result, list)
            assert len(result) == 2
            assert "result" in result[0]
            assert "result" in result[1]
            assert result[0]["result"]["location"] == "San Francisco"
            assert result[1]["result"]["location"] == "New York"


class TestGenerateLLMFunctionResponseAdvanced:
    """Test advanced generate_llm_function_response functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_temperature(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with temperature."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                temperature=0.7,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_max_tokens(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with max_tokens."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                max_tokens=100,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_web_search(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with web search enabled."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                enable_web_search=True,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["enable_web_search"] is True

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_reasoning_effort(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with reasoning effort."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="Solve this problem",
                tools=sample_tools,
                model="o1-preview",
                reasoning_effort="high",
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_invalid_reasoning_effort(
        self, mock_openai_client, sample_tools
    ):
        """Test generate_llm_function_response with invalid reasoning effort."""
        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_function_response(
                    prompt="Hello!",
                    tools=sample_tools,
                    model="o1-preview",
                    reasoning_effort="invalid",
                    client=None,
                )
            assert "Invalid reasoning_effort" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_response_format(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with response format."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            response_format = {"type": "json_object"}
            await generate_llm_function_response(
                prompt="Return JSON",
                tools=sample_tools,
                model="gpt-4",
                response_format=response_format,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["response_format"] == response_format

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_kwargs(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with additional kwargs."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's the weather?",
                tools=sample_tools,
                model="gpt-4",
                top_p=0.9,
                frequency_penalty=0.5,
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            assert call_args["top_p"] == 0.9
            assert call_args["frequency_penalty"] == 0.5


class TestGenerateLLMFunctionResponseErrorHandling:
    """Test generate_llm_function_response error handling."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_api_error(self, mock_openai_client, sample_tools):
        """Test generate_llm_function_response when API call fails."""
        mock_openai_client.chat_completion.side_effect = Exception("API error")

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(Exception) as exc_info:
                await generate_llm_function_response(
                    prompt="Hello!", tools=sample_tools, model="gpt-4", client=None
                )
            assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_invalid_json(
        self, mock_openai_client, sample_tools
    ):
        """Test generate_llm_function_response with invalid JSON in function arguments."""
        response = MagicMock(spec=ChatCompletion)
        response.id = "chatcmpl-123"
        response.object = "chat.completion"
        response.created = 1234567890
        response.model = "gpt-4"

        tool_call = MagicMock(spec=ChatCompletionMessageToolCall)
        tool_call.id = "call_abc123"
        tool_call.type = "function"
        tool_call.function = MagicMock()
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = "invalid json {location: SF}"  # Invalid JSON

        message = MagicMock(spec=ChatCompletionMessage)
        message.role = "assistant"
        message.content = None
        message.tool_calls = [tool_call]

        choice = MagicMock()
        choice.index = 0
        choice.message = message
        choice.finish_reason = "tool_calls"

        response.choices = [choice]

        mock_openai_client.chat_completion.return_value = response

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_function_response(
                    prompt="What's the weather?",
                    tools=sample_tools,
                    model="gpt-4",
                    client=None,
                )
            assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_empty_prompt(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with empty prompt."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            # Should handle empty prompt gracefully
            result = await generate_llm_function_response(
                prompt="", tools=sample_tools, model="gpt-4", client=None
            )

            assert isinstance(result, dict)
            assert result["function_name"] == "get_weather"


class TestGenerateLLMFunctionResponseImages:
    """Test generate_llm_function_response image functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_image_url(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with image URL."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="What's in this image?",
                tools=sample_tools,
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
    async def test_generate_llm_function_response_with_image_dict(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with image dict."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            image_dict = {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            }
            await generate_llm_function_response(
                prompt="What's in this image?",
                tools=sample_tools,
                model="gpt-4-vision-preview",
                images=[image_dict],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert user_message["content"][1] == image_dict

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_invalid_image_format(
        self, mock_openai_client, sample_tools
    ):
        """Test generate_llm_function_response with invalid image format."""
        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_function_response(
                    prompt="Hello!",
                    tools=sample_tools,
                    model="gpt-4-vision-preview",
                    images=[123],  # Invalid format
                    client=None,
                )
            assert "Invalid image format" in str(exc_info.value)


class TestGenerateLLMFunctionResponseFiles:
    """Test generate_llm_function_response file functionality."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_file_id(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with file ID."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="Summarize this document",
                tools=sample_tools,
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
    async def test_generate_llm_function_response_with_file_dict(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with file dict."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            file_dict = {"file_id": "file-abc123", "tools": [{"type": "file_search"}]}
            await generate_llm_function_response(
                prompt="Summarize this document",
                tools=sample_tools,
                model="gpt-4",
                files=[file_dict],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            assert user_message["attachments"][0] == file_dict

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_invalid_file_format(
        self, mock_openai_client, sample_tools
    ):
        """Test generate_llm_function_response with invalid file format."""
        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_function_response(
                    prompt="Hello!",
                    tools=sample_tools,
                    model="gpt-4",
                    files=[123],  # Invalid format
                    client=None,
                )
            assert "Invalid file format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_files_and_content_array(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test that files trigger content array format."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="Analyze this",
                tools=sample_tools,
                model="gpt-4",
                files=["file-abc123"],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            # Should use content array when files are present
            assert isinstance(user_message["content"], list)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_single_non_text_content(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with single non-text content item."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            image_dict = {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            }
            await generate_llm_function_response(
                prompt="",
                tools=sample_tools,
                model="gpt-4-vision-preview",
                images=[image_dict],
                client=None,
            )

            call_args = mock_openai_client.chat_completion.call_args[1]
            user_message = call_args["messages"][-1]
            # Single non-text content should be wrapped in array
            assert isinstance(user_message["content"], list)
            assert len(user_message["content"]) == 1


class TestGenerateLLMFunctionResponseEdgeCases:
    """Test generate_llm_function_response edge cases and error paths."""

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_unexpected_response_type(
        self, mock_openai_client, sample_tools
    ):
        """Test generate_llm_function_response with unexpected response type."""

        # Mock a non-ChatCompletion response (e.g., streaming response returned incorrectly)
        async def stream_generator():
            yield "unexpected_stream_chunk"

        mock_openai_client.chat_completion.return_value = stream_generator()

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            # Should handle gracefully and return the unexpected response
            response = await generate_llm_function_response(
                prompt="Hello!",
                tools=sample_tools,
                model="gpt-4",
                client=None,
            )

            # Should return the unexpected response type
            assert hasattr(response, "__aiter__")  # It's an async iterator

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_unexpected_tool_call_type(
        self, mock_openai_client, sample_tools
    ):
        """Test generate_llm_function_response with unexpected tool call type."""
        # Create a mock response with an unexpected tool call type
        mock_response = MagicMock(spec=ChatCompletion)
        mock_response.id = "chatcmpl-123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4"

        message = MagicMock()
        message.role = "assistant"
        message.content = None
        # Create a tool call that is not ChatCompletionMessageToolCall
        unexpected_tool_call = MagicMock()
        unexpected_tool_call.function = None
        message.tool_calls = [unexpected_tool_call]  # Wrong type

        choice = MagicMock()
        choice.index = 0
        choice.message = message
        choice.finish_reason = "tool_calls"

        mock_response.choices = [choice]

        mock_openai_client.chat_completion.return_value = mock_response

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            # Should handle gracefully and skip the unexpected tool call
            result = await generate_llm_function_response(
                prompt="Hello!",
                tools=sample_tools,
                model="gpt-4",
                client=None,
            )

            # Should return empty list or response since no valid tool calls
            # The function returns response if parsed_calls is empty
            assert isinstance(result, ChatCompletion)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_executor_not_callable(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with non-callable executor."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            # Create a non-callable executor
            tool_executors = {"get_weather": "not_a_function"}

            with pytest.raises(ValueError) as exc_info:
                await generate_llm_function_response(
                    prompt="What's the weather?",
                    tools=sample_tools,
                    model="gpt-4",
                    execute_tools=True,
                    tool_executors=tool_executors,
                    client=None,
                )
            assert "not callable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_llm_function_response_with_images_and_files(
        self, mock_openai_client, mock_chat_completion_with_tool_call, sample_tools
    ):
        """Test generate_llm_function_response with both images and files."""
        mock_openai_client.chat_completion.return_value = mock_chat_completion_with_tool_call

        generate_llm_function_response_module = sys.modules[
            "src.tools.generate_llm_function_response"
        ]
        with patch.object(
            generate_llm_function_response_module, "get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_openai_client

            await generate_llm_function_response(
                prompt="Analyze this",
                tools=sample_tools,
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
