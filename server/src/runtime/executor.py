"""Execution substrate for agent operations.

This module provides the Executor class that serves as the "hands of the system".
It executes agent invocations and tool calls, enforces execution contracts
(timeouts, cancellation), and wraps execution with tracing.

The executor provides a stable interface so orchestration logic is runtime-agnostic.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus
from src.runtime.tracing import get_tracer

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Executor:
    """Executor for agent invocations and tool calls.

    The executor is responsible for:
    - Executing agent invocations
    - Executing tool calls
    - Enforcing execution contracts (timeouts, cancellation)
    - Wrapping execution with tracing

    It does NOT handle:
    - Planning or scheduling strategy
    - Business logic
    - Result interpretation
    """

    def __init__(
        self,
        default_timeout: Optional[float] = None,
        enable_tracing: bool = True,
    ):
        """Initialize the executor.

        Args:
            default_timeout: Default timeout in seconds for executions.
                If None, no timeout is enforced by default.
            enable_tracing: Whether to enable tracing for executions.
        """
        self.default_timeout = default_timeout
        self.enable_tracing = enable_tracing

    async def execute_agent(
        self,
        agent: AgentBase,
        context: AgentContext,
        timeout: Optional[float] = None,
        cancellation_token: Optional[asyncio.CancelledError] = None,
    ) -> AgentResponse:
        """Execute an agent invocation.

        Args:
            agent: The agent instance to execute. Must have an async `execute` method
                that takes AgentContext and returns AgentResponse.
            context: The context for the agent execution.
            timeout: Timeout in seconds. If None, uses default_timeout.
            cancellation_token: Optional cancellation token for cancellation propagation.

        Returns:
            AgentResponse: The response from the agent execution.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            asyncio.CancelledError: If execution is cancelled.
            ValueError: If agent doesn't have an execute method.
        """
        timeout = timeout or self.default_timeout

        # Verify agent has execute method
        if not hasattr(agent, "execute") or not callable(getattr(agent, "execute")):
            raise ValueError(
                f"Agent {agent.name} does not have an async execute method. "
                "Agents must implement async def execute(context: AgentContext) -> AgentResponse"
            )

        # Create execution metadata
        execution_metadata = {
            "agent_name": agent.name,
            "agent_category": agent.category,
            "timeout": timeout,
        }

        # Get or create trace context
        tracer = get_tracer()
        trace_context = tracer.get_trace_context()
        trace_id = (
            trace_context.get("trace_id") if trace_context else tracer.create_trace_id()
        )
        parent_observation_id = (
            trace_context.get("observation_id") if trace_context else None
        )

        try:
            # Wrap execution with tracing if enabled
            if self.enable_tracing:
                execution_metadata["trace_id"] = trace_id
                execution_metadata["span_started"] = True

                async with tracer.async_span(
                    name=f"agent.{agent.name}",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "agent_name": agent.name,
                        "agent_category": agent.category,
                        "query": (
                            context.query[:100]
                            if len(context.query) > 100
                            else context.query
                        ),
                        "timeout": timeout,
                    },
                ) as span:
                    # Execute with timeout if specified
                    if timeout is not None:
                        response = await asyncio.wait_for(
                            self._execute_with_cancellation(
                                agent, context, cancellation_token
                            ),
                            timeout=timeout,
                        )
                    else:
                        response = await self._execute_with_cancellation(
                            agent, context, cancellation_token
                        )

                    # Add execution metadata to response
                    if response.metadata is not None:
                        response.metadata.update(execution_metadata)
                    else:
                        response.metadata = execution_metadata

                    execution_metadata["span_completed"] = True

                    return response
            else:
                # Execute without tracing
                if timeout is not None:
                    response = await asyncio.wait_for(
                        self._execute_with_cancellation(
                            agent, context, cancellation_token
                        ),
                        timeout=timeout,
                    )
                else:
                    response = await self._execute_with_cancellation(
                        agent, context, cancellation_token
                    )

                # Add execution metadata to response
                if response.metadata is not None:
                    response.metadata.update(execution_metadata)
                else:
                    response.metadata = execution_metadata

                return response

        except asyncio.TimeoutError as e:
            error_msg = f"Agent {agent.name} execution timed out after {timeout}s"
            logger.error(error_msg)
            if self.enable_tracing:
                execution_metadata["span_error"] = True
                execution_metadata["error"] = error_msg
                tracer.log_event(
                    name="agent.timeout",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "agent_name": agent.name,
                        "timeout": timeout,
                        "error": error_msg,
                    },
                )

            return AgentResponse.create_error(
                error_message=error_msg,
                agent_name=agent.name,
                agent_category=agent.category,
                metadata=execution_metadata,
            )

        except asyncio.CancelledError as e:
            error_msg = f"Agent {agent.name} execution was cancelled"
            logger.warning(error_msg)
            if self.enable_tracing:
                execution_metadata["span_cancelled"] = True
                execution_metadata["error"] = error_msg
                tracer.log_event(
                    name="agent.cancelled",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "agent_name": agent.name,
                        "error": error_msg,
                    },
                )

            return AgentResponse.create_error(
                error_message=error_msg,
                agent_name=agent.name,
                agent_category=agent.category,
                metadata=execution_metadata,
            )

        except Exception as e:
            error_msg = f"Agent {agent.name} execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.enable_tracing:
                execution_metadata["span_error"] = True
                execution_metadata["error"] = error_msg
                tracer.log_event(
                    name="agent.error",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "agent_name": agent.name,
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    },
                )

            return AgentResponse.create_error(
                error_message=error_msg,
                agent_name=agent.name,
                agent_category=agent.category,
                metadata=execution_metadata,
            )

    async def execute_tool(
        self,
        tool_name: str,
        tool_func: Callable[..., Awaitable[Any]],
        parameters: Dict[str, Any],
        timeout: Optional[float] = None,
        cancellation_token: Optional[asyncio.CancelledError] = None,
    ) -> Any:
        """Execute a tool call.

        Args:
            tool_name: Name of the tool being called.
            tool_func: Async callable that implements the tool.
            parameters: Parameters to pass to the tool function.
            timeout: Timeout in seconds. If None, uses default_timeout.
            cancellation_token: Optional cancellation token for cancellation propagation.

        Returns:
            The result from the tool execution.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            asyncio.CancelledError: If execution is cancelled.
        """
        timeout = timeout or self.default_timeout

        execution_metadata = {
            "tool_name": tool_name,
            "parameters": parameters,
            "timeout": timeout,
        }

        # Get or create trace context
        tracer = get_tracer()
        trace_context = tracer.get_trace_context()
        trace_id = (
            trace_context.get("trace_id") if trace_context else tracer.create_trace_id()
        )
        parent_observation_id = (
            trace_context.get("observation_id") if trace_context else None
        )

        try:
            if self.enable_tracing:
                execution_metadata["trace_id"] = trace_id
                execution_metadata["span_started"] = True

                async with tracer.async_span(
                    name=f"tool.{tool_name}",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "timeout": timeout,
                    },
                ) as span:
                    # Execute with timeout if specified
                    if timeout is not None:
                        result = await asyncio.wait_for(
                            self._execute_tool_with_cancellation(
                                tool_func, parameters, cancellation_token
                            ),
                            timeout=timeout,
                        )
                    else:
                        result = await self._execute_tool_with_cancellation(
                            tool_func, parameters, cancellation_token
                        )

                    execution_metadata["span_completed"] = True

                    return result
            else:
                # Execute without tracing
                if timeout is not None:
                    result = await asyncio.wait_for(
                        self._execute_tool_with_cancellation(
                            tool_func, parameters, cancellation_token
                        ),
                        timeout=timeout,
                    )
                else:
                    result = await self._execute_tool_with_cancellation(
                        tool_func, parameters, cancellation_token
                    )

                return result

        except asyncio.TimeoutError:
            error_msg = f"Tool {tool_name} execution timed out after {timeout}s"
            logger.error(error_msg)
            if self.enable_tracing:
                execution_metadata["span_error"] = True
                tracer.log_event(
                    name="tool.timeout",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "tool_name": tool_name,
                        "timeout": timeout,
                        "error": error_msg,
                    },
                )
            raise TimeoutError(error_msg)

        except asyncio.CancelledError:
            error_msg = f"Tool {tool_name} execution was cancelled"
            logger.warning(error_msg)
            if self.enable_tracing:
                execution_metadata["span_cancelled"] = True
                tracer.log_event(
                    name="tool.cancelled",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "tool_name": tool_name,
                        "error": error_msg,
                    },
                )
            raise

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.enable_tracing:
                execution_metadata["span_error"] = True
                tracer.log_event(
                    name="tool.error",
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata={
                        "tool_name": tool_name,
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    },
                )
            raise

    async def _execute_with_cancellation(
        self,
        agent: AgentBase,
        context: AgentContext,
        cancellation_token: Optional[asyncio.CancelledError],
    ) -> AgentResponse:
        """Execute agent with cancellation support.

        Args:
            agent: The agent to execute.
            context: The execution context.
            cancellation_token: Optional cancellation token.

        Returns:
            AgentResponse from the agent.
        """
        # Check for cancellation before execution
        if cancellation_token is not None:
            # In a real implementation, this would check the cancellation token
            # For now, we'll just execute normally
            pass

        # Execute the agent's execute method
        # Type check: agents must implement async execute method
        if not hasattr(agent, "execute"):
            raise ValueError(
                f"Agent {agent.name} does not implement required execute method"
            )
        execute_method = getattr(agent, "execute")
        if not callable(execute_method):
            raise ValueError(f"Agent {agent.name} execute is not callable")
        return await execute_method(context)  # type: ignore[misc]

    async def _execute_tool_with_cancellation(
        self,
        tool_func: Callable[..., Awaitable[Any]],
        parameters: Dict[str, Any],
        cancellation_token: Optional[asyncio.CancelledError],
    ) -> Any:
        """Execute tool with cancellation support.

        Args:
            tool_func: The tool function to execute.
            parameters: Parameters for the tool.
            cancellation_token: Optional cancellation token.

        Returns:
            Result from the tool execution.
        """
        # Check for cancellation before execution
        if cancellation_token is not None:
            # In a real implementation, this would check the cancellation token
            pass

        # Execute the tool function
        return await tool_func(**parameters)


# Default executor instance
_default_executor: Optional[Executor] = None


def get_executor() -> Executor:
    """Get the default executor instance.

    Returns:
        The default Executor instance.
    """
    global _default_executor
    if _default_executor is None:
        _default_executor = Executor()
    return _default_executor


def set_executor(executor: Executor) -> None:
    """Set the default executor instance.

    Args:
        executor: The Executor instance to use as default.
    """
    global _default_executor
    _default_executor = executor
