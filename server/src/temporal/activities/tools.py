"""Temporal activities for tool execution.

This module provides execution adapters for tools from the `/tools` directory.
Activities wrap tool execution with Temporal's durable execution semantics,
including retries, timeouts, and observability.

This is an execution adapter layer - it does not contain domain logic.
All business logic lives in the tools themselves.
"""

import inspect
import logging
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

from temporalio import activity
from temporalio.common import RetryPolicy

from src.contracts.tool_io import ToolOutput
from src.tools import (
    aggregate_funding_trends,
    analyze_sector_concentration,
    calculate_funding_velocity,
    calculate_portfolio_metrics,
    compare_to_market_benchmarks,
    find_investor_portfolio,
    generate_llm_response,
    get_acquisitions,
    get_funding_rounds,
    get_organizations,
    identify_funding_patterns,
    identify_investment_patterns,
    semantic_search_organizations,
    web_search,
)

logger = logging.getLogger(__name__)

# Tool registry mapping tool names to their callable functions
# This allows dynamic tool resolution
_TOOL_REGISTRY: Dict[str, Callable] = {
    "aggregate_funding_trends": aggregate_funding_trends,
    "analyze_sector_concentration": analyze_sector_concentration,
    "calculate_funding_velocity": calculate_funding_velocity,
    "calculate_portfolio_metrics": calculate_portfolio_metrics,
    "compare_to_market_benchmarks": compare_to_market_benchmarks,
    "find_investor_portfolio": find_investor_portfolio,
    "generate_llm_response": generate_llm_response,
    "get_acquisitions": get_acquisitions,
    "get_funding_rounds": get_funding_rounds,
    "get_organizations": get_organizations,
    "identify_funding_patterns": identify_funding_patterns,
    "identify_investment_patterns": identify_investment_patterns,
    "semantic_search_organizations": semantic_search_organizations,
    "web_search": web_search,
}


def register_tool(name: str, tool_func: Callable) -> None:
    """Register a tool function in the tool registry.

    Args:
        name: The name of the tool.
        tool_func: The callable tool function.
    """
    if not callable(tool_func):
        raise ValueError(f"Tool {name} must be callable")
    _TOOL_REGISTRY[name] = tool_func
    logger.info(f"Registered tool: {name}")


def get_tool(name: str) -> Optional[Callable]:
    """Get a tool function by name.

    Args:
        name: The name of the tool.

    Returns:
        The tool function if found, None otherwise.
    """
    return _TOOL_REGISTRY.get(name)


def list_available_tools() -> list[str]:
    """List all available tool names.

    Returns:
        List of available tool names.
    """
    return list(_TOOL_REGISTRY.keys())


# Default Temporal activity options for tool execution
# These can be overridden per-activity if needed
_DEFAULT_ACTIVITY_OPTIONS = {
    "start_to_close_timeout": timedelta(seconds=300),  # 5 minutes default timeout
    "retry_policy": RetryPolicy(
        initial_interval=timedelta(seconds=1),  # Start with 1 second
        backoff_coefficient=2.0,  # Double each retry
        maximum_interval=timedelta(seconds=60),  # Max 60 seconds between retries
        maximum_attempts=3,  # Max 3 attempts
    ),
}


@activity.defn(name="execute_tool")
async def execute_tool(
    tool_name: str,
    tool_parameters: Dict[str, Any],
    timeout_seconds: Optional[float] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a tool as a Temporal activity.

    This activity:
    - Resolves the tool from the tool registry
    - Validates tool parameters
    - Executes the tool with Temporal retry/timeout semantics
    - Returns structured results or failures

    Args:
        tool_name: Name of the tool to execute (must be registered).
        tool_parameters: Dictionary of parameters to pass to the tool.
        timeout_seconds: Optional timeout in seconds (informational only;
            actual timeout is set via activity options at registration).
        retry_policy: Optional retry policy configuration (informational only;
            actual retry policy is set via activity options at registration).

    Returns:
        Dictionary containing:
        - success: bool indicating if execution succeeded
        - result: The tool's return value (if successful)
        - error: Error message (if failed)
        - tool_name: Name of the executed tool
        - metadata: Additional metadata about execution

    Raises:
        ValueError: If tool is not found or parameters are invalid.
        Exception: If tool execution fails after retries.
    """
    activity.logger.info(f"Executing tool: {tool_name} with parameters: {tool_parameters}")

    # Resolve tool
    tool_func = get_tool(tool_name)
    if tool_func is None:
        error_msg = f"Tool '{tool_name}' not found. Available tools: {list_available_tools()}"
        activity.logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "tool_name": tool_name,
            "result": None,
            "metadata": {},
        }

    # Validate tool signature
    try:
        sig = inspect.signature(tool_func)
        # Check if tool is async
        if not inspect.iscoroutinefunction(tool_func):
            error_msg = f"Tool '{tool_name}' must be an async function"
            activity.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "tool_name": tool_name,
                "result": None,
                "metadata": {},
            }
    except Exception as e:
        error_msg = f"Error inspecting tool '{tool_name}': {e}"
        activity.logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "tool_name": tool_name,
            "result": None,
            "metadata": {},
        }

    # Execute tool with error handling
    try:
        # Call the tool function with provided parameters
        # Tools are async functions, so we await them
        tool_output = await tool_func(**tool_parameters)

        # Handle ToolOutput objects (contract-compliant tools)
        if isinstance(tool_output, ToolOutput):
            activity.logger.info(
                f"Tool '{tool_name}' executed successfully "
                f"(execution_time: {tool_output.execution_time_ms:.2f}ms)"
            )
            # Convert ToolOutput to dict format for Temporal
            return {
                "success": tool_output.success,
                "result": tool_output.result,
                "error": tool_output.error,
                "tool_name": tool_output.tool_name,
                "metadata": {
                    "execution_time_ms": tool_output.execution_time_ms,
                    **tool_output.metadata,
                },
            }
        else:
            # Handle legacy tools that return raw results
            activity.logger.info(f"Tool '{tool_name}' executed successfully (legacy format)")
            return {
                "success": True,
                "result": tool_output,
                "error": None,
                "tool_name": tool_name,
                "metadata": {
                    "execution_time_ms": None,
                },
            }

    except Exception as e:
        error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
        activity.logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "tool_name": tool_name,
            "result": None,
            "metadata": {
                "exception_type": type(e).__name__,
            },
        }


@activity.defn(name="execute_tool_with_options")
async def execute_tool_with_options(
    tool_name: str,
    tool_parameters: Dict[str, Any],
    activity_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a tool with custom Temporal activity options.

    This is similar to execute_tool but allows custom activity options
    to be specified per-execution.

    Args:
        tool_name: Name of the tool to execute.
        tool_parameters: Dictionary of parameters to pass to the tool.
        activity_options: Optional dictionary of Temporal activity options.
            Can include: start_to_close_timeout, retry_policy, etc.

    Returns:
        Dictionary containing execution results (same format as execute_tool).
    """
    # For now, this is the same as execute_tool
    # Activity options are typically set at registration time, not execution time
    # But we can log them for observability
    if activity_options:
        activity.logger.info(f"Custom activity options provided: {activity_options}")

    return await execute_tool(tool_name, tool_parameters)


def get_activity_options(
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    initial_retry_interval: Optional[float] = None,
) -> Dict[str, Any]:
    """Get Temporal activity options for tool execution.

    This helper function creates activity options that can be used
    when registering activities in the worker.

    Args:
        timeout_seconds: Optional timeout in seconds (default: 300).
        max_retries: Optional maximum retry attempts (default: 3).
        initial_retry_interval: Optional initial retry interval in seconds (default: 1.0).

    Returns:
        Dictionary of activity options for Temporal.
    """
    options = _DEFAULT_ACTIVITY_OPTIONS.copy()

    if timeout_seconds is not None:
        options["start_to_close_timeout"] = timedelta(seconds=timeout_seconds)

    if max_retries is not None or initial_retry_interval is not None:
        existing_policy_raw = options.get("retry_policy")
        existing_policy: Optional[RetryPolicy] = (
            existing_policy_raw if isinstance(existing_policy_raw, RetryPolicy) else None
        )
        if existing_policy is not None:
            # Create a new RetryPolicy with updated values
            retry_policy = RetryPolicy(
                initial_interval=(
                    timedelta(seconds=initial_retry_interval)
                    if initial_retry_interval is not None
                    else existing_policy.initial_interval
                ),
                backoff_coefficient=existing_policy.backoff_coefficient,
                maximum_interval=existing_policy.maximum_interval,
                maximum_attempts=(
                    max_retries if max_retries is not None else existing_policy.maximum_attempts
                ),
            )
        else:
            retry_policy = RetryPolicy(
                initial_interval=(
                    timedelta(seconds=initial_retry_interval)
                    if initial_retry_interval is not None
                    else timedelta(seconds=1)
                ),
                maximum_attempts=max_retries if max_retries is not None else 3,
            )
        options["retry_policy"] = retry_policy

    return options
