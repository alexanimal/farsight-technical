"""Tool for generating LLM responses with various configurations.

This module provides a high-level interface for generating LLM responses using
the OpenAI client, with support for prompts, instructions, streaming, web search,
reasoning effort, images, and files.
"""

import logging
from typing import Any, AsyncIterator, Optional

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.llm.openai_client import OpenAIClient, get_client

logger = logging.getLogger(__name__)


@observe(as_type="tool")
async def generate_llm_response(
    prompt: str,
    model: str = "gpt-4.1-mini",
    instructions: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    enable_web_search: bool = False,
    reasoning_effort: Optional[str] = None,
    images: Optional[list[str | dict[str, Any]]] = None,
    files: Optional[list[str | dict[str, Any]]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[str | dict[str, Any]] = None,
    response_format: Optional[dict[str, Any]] = None,
    client: Optional[OpenAIClient] = None,
    **kwargs: Any,
) -> ChatCompletion | AsyncIterator[ChatCompletionChunk] | str:
    """Generate an LLM response with flexible configuration options.

    This function provides a convenient interface for generating LLM responses
    with support for various features like streaming, web search, reasoning
    effort, images, and files.

    Args:
        prompt: The user prompt/message content.
        model: Model to use for completion (default: "gpt-4").
        instructions: Optional instructions to include in the system message.
            This is combined with system_prompt if both are provided.
        system_prompt: Optional system prompt. If provided, this takes precedence
            over instructions. If neither is provided, no system message is added.
        temperature: Sampling temperature (0-2). Higher values make output
            more random. Not used with o1 models.
        max_tokens: Maximum tokens to generate.
        stream: Whether to stream the response (default: False).
            If True, returns an AsyncIterator. If False, returns the full response.
        enable_web_search: Enable web search capabilities (default: False).
        reasoning_effort: Reasoning effort for o1 models. Can be "low", "medium",
            or "high". Only applicable to o1 models.
        images: Optional list of image URLs or image dicts with 'type' and 'image_url'
            keys. Images are added to the user message content.
        files: Optional list of file IDs or file dicts. Files are attached to
            the user message.
        tools: Optional list of tool definitions for function calling.
        tool_choice: Control which tools the model can use. Can be "auto",
            "none", or a specific tool definition.
        response_format: Optional response format (e.g., {"type": "json_object"}).
        client: Optional OpenAIClient instance. If not provided, uses the
            default singleton client.
        **kwargs: Additional parameters to pass to the chat completion API.

    Returns:
        If stream=False: ChatCompletion object or str (if return_text=True in kwargs).
        If stream=True: AsyncIterator of ChatCompletionChunk objects.

    Raises:
        ValueError: If invalid parameters are provided.
        Exception: If API call fails.

    Example:
        ```python
        # Simple text generation
        response = await generate_llm_response(
            prompt="What is the capital of France?",
            model="gpt-4"
        )
        print(response.choices[0].message.content)

        # With system instructions
        response = await generate_llm_response(
            prompt="Explain quantum computing",
            model="gpt-4",
            instructions="You are a helpful physics teacher. Explain concepts simply."
        )

        # Streaming response
        async for chunk in await generate_llm_response(
            prompt="Tell me a story",
            model="gpt-4",
            stream=True
        ):
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")

        # With web search
        response = await generate_llm_response(
            prompt="What's the latest news about AI?",
            model="gpt-4",
            enable_web_search=True
        )

        # With images
        response = await generate_llm_response(
            prompt="What's in this image?",
            model="gpt-4-vision-preview",
            images=["https://example.com/image.jpg"]
        )

        # With files
        response = await generate_llm_response(
            prompt="Summarize this document",
            model="gpt-4",
            files=["file-abc123"]
        )

        # With reasoning effort (o1 models)
        response = await generate_llm_response(
            prompt="Solve this complex math problem step by step",
            model="o1-preview",
            reasoning_effort="high"
        )
        ```
    """
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
                raise ValueError(
                    f"Invalid image format: {image}. Expected str (URL) or dict."
                )

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
                attachments.append(
                    {"file_id": file, "tools": [{"type": "file_search"}]}
                )
            elif isinstance(file, dict):
                # Assume it's already formatted
                attachments.append(file)
            else:
                raise ValueError(
                    f"Invalid file format: {file}. Expected str (file_id) or dict."
                )
        user_message["attachments"] = attachments

    messages.append(user_message)

    # Validate reasoning_effort if provided
    if reasoning_effort is not None:
        valid_efforts = ["low", "medium", "high"]
        if reasoning_effort not in valid_efforts:
            raise ValueError(
                f"Invalid reasoning_effort: {reasoning_effort}. "
                f"Must be one of {valid_efforts}."
            )

    # Extract return_text from kwargs before passing to client
    # (return_text is handled in this function, not by the OpenAI API)
    return_text = kwargs.pop("return_text", False)

    # Call the client's chat_completion method
    try:
        response = await client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            enable_web_search=enable_web_search,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

        # If streaming, return the iterator as-is
        if stream:
            return response

        # If not streaming and return_text is requested, extract text
        if return_text:
            if isinstance(response, ChatCompletion):
                return response.choices[0].message.content or ""
            return ""

        return response
    except Exception as e:
        logger.error(f"Failed to generate LLM response: {e}")
        raise
