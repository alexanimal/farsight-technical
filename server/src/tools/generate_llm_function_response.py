"""Tool for generating LLM function call responses with structured outputs.

This module provides a high-level interface for generating LLM responses with
function calling capabilities, allowing you to get structured outputs from the LLM
by defining tools/functions and parsing their calls.
"""

import json
import logging
from typing import Any, Callable, Optional

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

from src.llm.openai_client import OpenAIClient, get_client

logger = logging.getLogger(__name__)


@observe(as_type="tool")
async def generate_llm_function_response(
    prompt: str,
    tools: list[dict[str, Any]],
    model: str = "gpt-4.1-mini",
    instructions: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tool_choice: Optional[str | dict[str, Any]] = None,
    enable_web_search: bool = False,
    reasoning_effort: Optional[str] = None,
    images: Optional[list[str | dict[str, Any]]] = None,
    files: Optional[list[str | dict[str, Any]]] = None,
    response_format: Optional[dict[str, Any]] = None,
    execute_tools: bool = False,
    tool_executors: Optional[dict[str, Callable[..., Any]]] = None,
    return_all_calls: bool = False,
    client: Optional[OpenAIClient] = None,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]] | ChatCompletion:
    """Generate an LLM response with function calling and structured outputs.

    This function provides a convenient interface for generating LLM responses
    with function calling capabilities. It can parse function call arguments
    and optionally execute the tools, returning structured outputs.

    Args:
        prompt: The user prompt/message content.
        tools: List of tool/function definitions. Each tool should have a
            'type' field (typically "function") and a 'function' field with
            'name', 'description', and 'parameters' (JSON schema).
        model: Model to use for completion (default: "gpt-4").
        instructions: Optional instructions to include in the system message.
            This is combined with system_prompt if both are provided.
        system_prompt: Optional system prompt. If provided, this takes precedence
            over instructions. If neither is provided, no system message is added.
        temperature: Sampling temperature (0-2). Higher values make output
            more random. Not used with o1 models.
        max_tokens: Maximum tokens to generate.
        tool_choice: Control which tools the model can use. Can be "auto",
            "none", or a specific tool definition. Defaults to "auto" if not provided.
        enable_web_search: Enable web search capabilities (default: False).
            NOTE: Currently not supported - OpenAI API only supports 'function' and 'custom' tool types.
            This parameter is ignored and will log a warning if set to True.
        reasoning_effort: Reasoning effort for o1 models. Can be "low", "medium",
            or "high". Only applicable to o1 models.
        images: Optional list of image URLs or image dicts with 'type' and 'image_url'
            keys. Images are added to the user message content.
        files: Optional list of file IDs or file dicts. Files are attached to
            the user message.
        response_format: Optional response format (e.g., {"type": "json_object"}).
        execute_tools: If True, execute the tool calls using tool_executors and
            return results. If False, only parse and return the function call arguments.
        tool_executors: Optional dict mapping tool names to async callable functions.
            Required if execute_tools=True. Each function should accept keyword arguments
            matching the tool's parameters schema.
        return_all_calls: If True and multiple tool calls are made, return a list
            of all tool call results. If False, return only the first tool call result.
        client: Optional OpenAIClient instance. If not provided, uses the
            default singleton client.
        **kwargs: Additional parameters to pass to the chat completion API.

    Returns:
        If execute_tools=False:
            - If return_all_calls=False: dict with 'function_name' and 'arguments' (parsed)
            - If return_all_calls=True: list of dicts, one per tool call
        If execute_tools=True:
            - If return_all_calls=False: dict with 'function_name', 'arguments', and 'result'
            - If return_all_calls=True: list of dicts with execution results
        If no tool calls are made: returns the full ChatCompletion object

    Raises:
        ValueError: If invalid parameters are provided (e.g., execute_tools=True
            without tool_executors, or no tools provided).
        Exception: If API call fails or tool execution fails.

    Example:
        ```python
        # Define a tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        # Get structured output (just parse function call)
        result = await generate_llm_function_response(
            prompt="What's the weather in San Francisco?",
            tools=tools,
            model="gpt-4"
        )
        print(result["function_name"])  # "get_weather"
        print(result["arguments"])  # {"location": "San Francisco", "unit": "fahrenheit"}

        # Execute the tool and get results
        async def get_weather(location: str, unit: str = "fahrenheit"):
            return {"temperature": 72, "condition": "sunny"}

        tool_executors = {"get_weather": get_weather}
        result = await generate_llm_function_response(
            prompt="What's the weather in San Francisco?",
            tools=tools,
            model="gpt-4",
            execute_tools=True,
            tool_executors=tool_executors
        )
        print(result["result"])  # {"temperature": 72, "condition": "sunny"}

        # Multiple tool calls
        result = await generate_llm_function_response(
            prompt="Get weather for SF and NYC",
            tools=tools,
            model="gpt-4",
            execute_tools=True,
            tool_executors=tool_executors,
            return_all_calls=True
        )
        # Returns list of results for each tool call
        ```
    """
    # Validate inputs
    if not tools:
        raise ValueError("tools parameter is required for function calling")

    if execute_tools and not tool_executors:
        raise ValueError(
            "tool_executors is required when execute_tools=True. "
            "Provide a dict mapping tool names to async callable functions."
        )

    # Use provided client or get default singleton
    if client is None:
        client = await get_client()

    # Build messages list
    messages: list[dict[str, Any]] = []

    # Add system message if provided
    if system_prompt or instructions:
        system_content = system_prompt or instructions
        messages.append({"role": "system", "content": system_content})

    # Build user message content
    user_content: list[dict[str, Any]] = []

    # Add text prompt
    if prompt:
        user_content.append({"type": "text", "text": prompt})

    # Add images if provided
    if images:
        for image in images:
            if isinstance(image, str):
                # Assume it's a URL
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                    }
                )
            elif isinstance(image, dict):
                # Assume it's already formatted
                user_content.append(image)
            else:
                raise ValueError(f"Invalid image format: {image}. Expected str (URL) or dict.")

    # Build user message
    user_message: dict[str, Any] = {"role": "user"}

    # If we have multiple content items or files, use content array
    if len(user_content) > 1 or files:
        user_message["content"] = user_content
    elif len(user_content) == 1:
        # Single text content can be a string
        if user_content[0]["type"] == "text":
            user_message["content"] = user_content[0]["text"]
        else:
            user_message["content"] = user_content
    else:
        # No content (shouldn't happen, but handle gracefully)
        user_message["content"] = prompt or ""

    # Add file attachments if provided
    if files:
        attachments = []
        for file in files:
            if isinstance(file, str):
                # Assume it's a file ID
                attachments.append({"file_id": file, "tools": [{"type": "file_search"}]})
            elif isinstance(file, dict):
                # Assume it's already formatted
                attachments.append(file)
            else:
                raise ValueError(f"Invalid file format: {file}. Expected str (file_id) or dict.")
        user_message["attachments"] = attachments

    messages.append(user_message)

    # Validate reasoning_effort if provided
    if reasoning_effort is not None:
        valid_efforts = ["low", "medium", "high"]
        if reasoning_effort not in valid_efforts:
            raise ValueError(
                f"Invalid reasoning_effort: {reasoning_effort}. " f"Must be one of {valid_efforts}."
            )

    # Default tool_choice to "auto" if not provided
    if tool_choice is None:
        tool_choice = "auto"

    # Call the client's chat_completion method
    try:
        response = await client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,  # Function calls don't work well with streaming
            enable_web_search=enable_web_search,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

        # Check if response is a ChatCompletion (should always be, since stream=False)
        if not isinstance(response, ChatCompletion):
            logger.warning(
                "Unexpected response type. Expected ChatCompletion, got streaming response."
            )
            # Type narrowing: if it's not ChatCompletion, it must be one of the expected return types
            # This should not happen with stream=False, but handle it gracefully
            return response  # type: ignore[return-value]

        # Extract tool calls from the response
        message = response.choices[0].message
        tool_calls = message.tool_calls

        # If no tool calls, return the full response
        if not tool_calls:
            logger.warning("No tool calls found in LLM response. Returning full ChatCompletion.")
            return response

        # Parse tool calls
        parsed_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, ChatCompletionMessageToolCall):
                logger.warning(f"Unexpected tool call type: {type(tool_call)}")
                continue

            function_name = tool_call.function.name
            function_args_str = tool_call.function.arguments

            # Parse JSON arguments
            try:
                function_args = json.loads(function_args_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse function arguments for {function_name}: {e}")
                raise ValueError(
                    f"Invalid JSON in function arguments for {function_name}: {function_args_str}"
                ) from e

            parsed_call = {
                "function_name": function_name,
                "arguments": function_args,
                "call_id": tool_call.id,
            }

            # Execute tool if requested
            if execute_tools:
                if tool_executors is None:
                    raise ValueError("tool_executors is required when execute_tools=True")
                if function_name not in tool_executors:
                    raise ValueError(
                        f"No executor found for tool '{function_name}'. "
                        f"Available executors: {list(tool_executors.keys())}"
                    )

                executor = tool_executors[function_name]
                if not callable(executor):
                    raise ValueError(f"Tool executor for '{function_name}' is not callable")
                try:
                    # Execute the tool with parsed arguments
                    result = await executor(**function_args)
                    parsed_call["result"] = result
                except Exception as e:
                    logger.error(f"Tool execution failed for {function_name}: {e}", exc_info=True)
                    parsed_call["error"] = str(e)
                    parsed_call["result"] = None

            parsed_calls.append(parsed_call)

        # Return based on return_all_calls flag
        if return_all_calls:
            return parsed_calls
        else:
            return parsed_calls[0] if parsed_calls else response

    except Exception as e:
        logger.error(f"Failed to generate LLM function response: {e}")
        raise
