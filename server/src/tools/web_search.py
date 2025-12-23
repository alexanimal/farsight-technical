"""Tool for performing web search using LLM with web search capabilities.

This tool uses the generate_llm_response function with enable_web_search=True to
get additional context and information about a user's query. Instead of making
direct API calls to search engines, it leverages the LLM's built-in web search
capabilities to provide enriched context and inferred information.
"""

import logging
import time
from typing import Any, Optional

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from src.contracts.tool_io import ToolMetadata, ToolOutput, ToolParameterSchema, create_tool_output
from src.tools.generate_llm_response import generate_llm_response

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the web_search tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="web_search",
        description=(
            "Perform a web search using LLM with web search capabilities to get "
            "additional context and information about a query. This tool leverages "
            "the LLM's built-in web search to provide enriched context and inferred "
            "information without making direct API calls to search engines."
        ),
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="query",
                type="string",
                description="The search query or question to get additional context about",
                required=True,
            ),
            ToolParameterSchema(
                name="model",
                type="string",
                description="Model to use for the web search-enabled LLM call (default: 'gpt-4.1-mini')",
                required=False,
            ),
            ToolParameterSchema(
                name="system_prompt",
                type="string",
                description="Optional system prompt to guide the LLM's response",
                required=False,
            ),
            ToolParameterSchema(
                name="temperature",
                type="float",
                description="Sampling temperature (0-2). Higher values make output more random. Not used with o1 models.",
                required=False,
            ),
            ToolParameterSchema(
                name="max_tokens",
                type="integer",
                description="Maximum tokens to generate in the response",
                required=False,
            ),
        ],
        returns={
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The LLM-generated response with web search context",
                },
                "model": {
                    "type": "string",
                    "description": "The model used for the search",
                },
            },
        },
        cost_per_call=None,  # Cost depends on model and tokens used
        estimated_latency_ms=3000.0,  # Web search-enabled LLM calls typically take longer
        timeout_seconds=60.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["llm", "web-search", "context-enrichment", "read-only"],
    )


@observe(as_type="tool")
async def web_search(
    query: str,
    model: str = "gpt-4.1-mini",
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ToolOutput:
    """Perform a web search using LLM with web search capabilities.

    This tool uses the generate_llm_response function with enable_web_search=True
    to get additional context and information about a user's query. The LLM will
    use its built-in web search capabilities to provide enriched context.

    Args:
        query: The search query or question to get additional context about.
        model: Model to use for the web search-enabled LLM call (default: "gpt-4.1-mini").
        system_prompt: Optional system prompt to guide the LLM's response. If not provided,
            a default prompt will be used that instructs the LLM to provide context
            relevant to the query.
        temperature: Sampling temperature (0-2). Higher values make output more random.
            Not used with o1 models.
        max_tokens: Maximum tokens to generate in the response.

    Returns:
        ToolOutput object containing:
        - success: Whether the web search succeeded
        - result: Dictionary with:
            - response: The LLM-generated response with web search context
            - model: The model used for the search
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute the search
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Basic web search
        result = await web_search(query="What are the latest trends in AI funding?")

        # Web search with custom model and system prompt
        result = await web_search(
            query="What happened with OpenAI in 2024?",
            model="gpt-4",
            system_prompt="Provide a concise summary of recent events.",
            temperature=0.7
        )

        # Web search with token limit
        result = await web_search(
            query="Explain quantum computing",
            max_tokens=500
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate query
        if not query or not query.strip():
            error_msg = "Query cannot be empty"
            logger.error(error_msg)
            execution_time_ms = (time.time() - start_time) * 1000
            return create_tool_output(
                tool_name="web_search",
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )

        # Default system prompt if not provided
        default_system_prompt = (
            "You are a helpful assistant that uses web search to provide accurate, "
            "up-to-date information. Use web search to find relevant context about "
            "the user's query and provide a comprehensive response based on the "
            "search results. Focus on factual information and cite sources when possible."
        )
        effective_system_prompt = system_prompt or default_system_prompt

        # Build the prompt for web search
        # The prompt should encourage the LLM to use web search to find relevant information
        search_prompt = (
            f"Please search the web for information about: {query}\n\n"
            f"Provide a comprehensive response with relevant context, facts, and "
            f"up-to-date information related to this query."
        )

        logger.info(f"Performing web search for query: {query[:100]}")

        # Call generate_llm_response with web search enabled
        response = await generate_llm_response(
            prompt=search_prompt,
            model=model,
            system_prompt=effective_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_web_search=True,
            return_text=True,  # Return just the text content
        )

        # Extract response text
        response_text = response if isinstance(response, str) else ""

        execution_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Web search completed in {execution_time_ms:.2f}ms, "
            f"response length: {len(response_text)} chars"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="web_search",
            success=True,
            result={
                "response": response_text,
                "model": model,
            },
            execution_time_ms=execution_time_ms,
            metadata={
                "response_length": len(response_text),
                "model_used": model,
            },
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to perform web search: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="web_search",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
