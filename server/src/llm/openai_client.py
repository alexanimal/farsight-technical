"""OpenAI client for managing LLM connections and inference.

This module provides an OpenAI client that manages API credentials and provides
methods for generating LLM responses with various configurations including
streaming, web search, reasoning effort, images, and files.
"""

import logging
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.config import settings

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI client for LLM inference operations.

    This client manages connections to OpenAI and provides methods for
    generating chat completions with support for streaming, web search,
    reasoning effort, images, and files.

    Example:
        ```python
        client = OpenAIClient()
        await client.initialize()
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}],
            model="gpt-4"
        )
        await client.close()
        ```

        Or use as a context manager:
        ```python
        async with OpenAIClient() as client:
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello!"}],
                model="gpt-4"
            )
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the OpenAI client.

        Args:
            api_key: Optional OpenAI API key. If not provided, uses
                settings.openai_api_key.
            base_url: Optional base URL for the API. If not provided, uses
                OpenAI's default endpoint. Useful for custom endpoints or proxies.
        """
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url
        self._client: Optional[AsyncOpenAI] = None

    async def initialize(self) -> None:
        """Initialize the OpenAI client connection.

        Raises:
            ValueError: If API key is not provided.
            Exception: If client initialization fails.
        """
        if self._client is not None:
            logger.warning(
                "OpenAI client already initialized, skipping re-initialization"
            )
            return

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in environment or pass api_key parameter."
            )

        try:
            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            self._client = AsyncOpenAI(**client_kwargs)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    async def close(self) -> None:
        """Close the OpenAI client connection."""
        if self._client is not None:
            # AsyncOpenAI doesn't have an explicit close method, but we can clear the reference
            # The underlying httpx client will be cleaned up automatically
            self._client = None
            logger.info("OpenAI client closed")

    async def __aenter__(self) -> "OpenAIClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client instance.

        Returns:
            The AsyncOpenAI client instance.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if self._client is None:
            raise RuntimeError(
                "OpenAI client not initialized. Call initialize() first or use as context manager."
            )
        return self._client

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        response_format: Optional[dict[str, Any]] = None,
        enable_web_search: bool = False,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Generate a chat completion using OpenAI's API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Supports text, images, and file attachments.
            model: Model to use for completion (default: "gpt-4").
            temperature: Sampling temperature (0-2). Higher values make output
                more random. Not used with o1 models.
            max_tokens: Maximum tokens to generate.
            stream: Whether to stream the response (default: False).
            tools: Optional list of tool definitions for function calling.
            tool_choice: Control which tools the model can use. Can be "auto",
                "none", or a specific tool definition.
            response_format: Optional response format (e.g., {"type": "json_object"}).
            enable_web_search: Enable web search capabilities (default: False).
                This uses the web_search tool.
            reasoning_effort: Reasoning effort for o1 models. Can be "low", "medium",
                or "high". Only applicable to o1 models.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            ChatCompletion object if stream=False, or AsyncIterator of
            ChatCompletionChunk objects if stream=True.

        Raises:
            RuntimeError: If client is not initialized.
            Exception: If API call fails.

        Example:
            ```python
            # Simple completion
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello!"}],
                model="gpt-4"
            )
            print(response.choices[0].message.content)

            # Streaming completion
            async for chunk in await client.chat_completion(
                messages=[{"role": "user", "content": "Tell me a story"}],
                model="gpt-4",
                stream=True
            ):
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="")

            # With web search
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "What's the weather today?"}],
                model="gpt-4",
                enable_web_search=True
            )

            # With reasoning effort (o1 models)
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Solve this math problem"}],
                model="o1-preview",
                reasoning_effort="high"
            )
            ```
        """
        if self._client is None:
            raise RuntimeError(
                "OpenAI client not initialized. Call initialize() first or use as context manager."
            )

        # Build tools list if web search is enabled
        final_tools = tools or []
        if enable_web_search:
            web_search_tool = {
                "type": "web_search",
            }
            # Check if web_search tool already exists
            if not any(
                tool.get("type") == "web_search" for tool in final_tools
            ):
                final_tools.append(web_search_tool)

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if stream:
            request_params["stream"] = True
        if final_tools:
            request_params["tools"] = final_tools
        if tool_choice is not None:
            request_params["tool_choice"] = tool_choice
        if response_format is not None:
            request_params["response_format"] = response_format
        if reasoning_effort is not None:
            request_params["reasoning_effort"] = reasoning_effort

        # Add any additional kwargs
        request_params.update(kwargs)

        try:
            if stream:
                logger.debug(
                    f"Streaming chat completion with model '{model}', "
                    f"web_search={enable_web_search}, reasoning_effort={reasoning_effort}"
                )
                stream_response = await self._client.chat.completions.create(
                    **request_params
                )
                return stream_response
            else:
                logger.debug(
                    f"Chat completion with model '{model}', "
                    f"web_search={enable_web_search}, reasoning_effort={reasoning_effort}"
                )
                response = await self._client.chat.completions.create(
                    **request_params
                )
                return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise

    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-large",
    ) -> list[float]:
        """Create an embedding for the given text.

        Args:
            text: The text to embed.
            model: The embedding model to use (default: "text-embedding-3-large").

        Returns:
            List of floats representing the embedding vector.

        Raises:
            RuntimeError: If client is not initialized.
            Exception: If embedding creation fails.

        Example:
            ```python
            embedding = await client.create_embedding("Hello, world!")
            ```
        """
        if self._client is None:
            raise RuntimeError(
                "OpenAI client not initialized. Call initialize() first or use as context manager."
            )

        try:
            response = await self._client.embeddings.create(
                model=model,
                input=text,
            )
            logger.debug(f"Created embedding with model '{model}'")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise

    def is_initialized(self) -> bool:
        """Check if the OpenAI client is initialized.

        Returns:
            True if client is initialized, False otherwise.
        """
        return self._client is not None


# Singleton instance for convenience
_default_client: Optional[OpenAIClient] = None


async def get_client() -> OpenAIClient:
    """Get or create the default singleton OpenAIClient instance.

    This is a convenience function for modules that want to use a shared
    client instance without managing it themselves.

    Returns:
        The default OpenAIClient instance.

    Example:
        ```python
        client = await get_client()
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}],
            model="gpt-4"
        )
        ```
    """
    global _default_client
    if _default_client is None:
        _default_client = OpenAIClient()
        await _default_client.initialize()
    return _default_client


async def close_default_client() -> None:
    """Close the default singleton client instance."""
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None

