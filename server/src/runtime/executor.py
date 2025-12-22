"""Execution substrate for agent operations.

This module provides the Executor class that serves as the "hands of the system".
It executes agent invocations and tool calls, enforces execution contracts
(timeouts, cancellation), and wraps execution with tracing.

The executor provides a stable interface so orchestration logic is runtime-agnostic.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from src.contracts.agent_io import (AgentInput, AgentOutput,
                                    create_agent_output, validate_agent_input)
from src.contracts.tool_io import (ToolInput, ToolMetadata, ToolOutput,
                                   create_tool_output, validate_tool_input)
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import ResponseStatus
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
        agent_input: AgentInput,
        timeout: Optional[float] = None,
        cancellation_token: Optional[asyncio.CancelledError] = None,
    ) -> AgentOutput:
        """Execute an agent invocation.

        Args:
            agent: The agent instance to execute. Must have an async `execute` method
                that takes AgentContext and returns AgentOutput.
            agent_input: The validated agent input contract.
            timeout: Timeout in seconds. If None, uses default_timeout.
            cancellation_token: Optional cancellation token for cancellation propagation.

        Returns:
            AgentOutput: The response from the agent execution.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            asyncio.CancelledError: If execution is cancelled.
            ValueError: If agent doesn't have an execute method or input is invalid.
        """
        timeout = timeout or self.default_timeout

        # Verify agent has execute method
        if not hasattr(agent, "execute") or not callable(getattr(agent, "execute")):
            raise ValueError(
                f"Agent {agent.name} does not have an async execute method. "
                "Agents must implement async def execute(context: AgentContext) -> AgentOutput"
            )

        # Validate input using contracts
        validated_input = validate_agent_input(agent_input.to_agent_context())
        agent_context = validated_input.to_agent_context()

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
                            agent_context.query[:100]
                            if len(agent_context.query) > 100
                            else agent_context.query
                        ),
                        "timeout": timeout,
                    },
                ) as span:
                    # Set input on the observation
                    if span is not None and hasattr(span, "update"):
                        try:
                            # Convert agent_input to dict for Langfuse
                            input_dict = agent_input.model_dump() if hasattr(agent_input, "model_dump") else {
                                "query": agent_context.query,
                                "conversation_id": agent_context.conversation_id,
                                "user_id": agent_context.user_id,
                                "metadata": agent_context.metadata,
                                "shared_data": agent_context.shared_data,
                            }
                            span.update(input=input_dict)
                        except Exception as e:
                            logger.debug(f"Failed to update span input: {e}")

                    try:
                        # Execute with timeout if specified
                        if timeout is not None:
                            output = await asyncio.wait_for(
                                self._execute_with_cancellation(
                                    agent, agent_context, cancellation_token
                                ),
                                timeout=timeout,
                            )
                        else:
                            output = await self._execute_with_cancellation(
                                agent, agent_context, cancellation_token
                            )

                        # Ensure we have AgentOutput
                        if not isinstance(output, AgentOutput):
                            raise ValueError(
                                f"Agent {agent.name} returned {type(output)}, expected AgentOutput"
                            )

                        # Set output on the observation
                        if span is not None and hasattr(span, "update"):
                            try:
                                # Convert AgentOutput to dict for Langfuse
                                output_dict = output.model_dump() if hasattr(output, "model_dump") else {
                                    "content": str(output.content) if hasattr(output, "content") else "",
                                    "status": output.status.value if hasattr(output.status, "value") else str(output.status),
                                    "agent_name": output.agent_name,
                                    "agent_category": output.agent_category,
                                    "error": output.error if hasattr(output, "error") else None,
                                }
                                span.update(output=output_dict)
                            except Exception as e:
                                logger.debug(f"Failed to update span output: {e}")

                        # Add execution metadata to output
                        output.metadata.update(execution_metadata)
                        execution_metadata["span_completed"] = True

                        return output
                    except Exception as e:
                        # Capture errors that occur during execution and update the observation
                        error_msg = str(e)
                        if span is not None and hasattr(span, "update"):
                            try:
                                span.update(
                                    output={
                                        "error": error_msg,
                                        "error_type": type(e).__name__,
                                        "status": "error",
                                    }
                                )
                            except Exception as update_error:
                                logger.debug(f"Failed to update span with error: {update_error}")
                        # Re-raise to be handled by outer exception handlers
                        raise
            else:
                # Execute without tracing
                if timeout is not None:
                    output = await asyncio.wait_for(
                        self._execute_with_cancellation(
                            agent, agent_context, cancellation_token
                        ),
                        timeout=timeout,
                    )
                else:
                    output = await self._execute_with_cancellation(
                        agent, agent_context, cancellation_token
                    )

                # Ensure we have AgentOutput
                if not isinstance(output, AgentOutput):
                    raise ValueError(
                        f"Agent {agent.name} returned {type(output)}, expected AgentOutput"
                    )

                # Add execution metadata to output
                output.metadata.update(execution_metadata)

                return output

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

            return create_agent_output(
                content="",
                agent_name=agent.name,
                agent_category=agent.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
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

            return create_agent_output(
                content="",
                agent_name=agent.name,
                agent_category=agent.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
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

            return create_agent_output(
                content="",
                agent_name=agent.name,
                agent_category=agent.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
                metadata=execution_metadata,
            )

    async def execute_tool(
        self,
        tool_input: ToolInput,
        tool_func: Callable[..., Awaitable[Any]],
        timeout: Optional[float] = None,
        cancellation_token: Optional[asyncio.CancelledError] = None,
        tool_metadata: Optional[ToolMetadata] = None,
    ) -> ToolOutput:
        """Execute a tool call.

        Args:
            tool_input: The validated tool input contract.
            tool_func: Async callable that implements the tool.
            timeout: Timeout in seconds. If None, uses default_timeout.
            cancellation_token: Optional cancellation token for cancellation propagation.
            tool_metadata: Optional ToolMetadata for parameter validation. If provided,
                parameters will be validated against the tool's schema.

        Returns:
            ToolOutput: The structured output from tool execution.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            asyncio.CancelledError: If execution is cancelled.
            ValueError: If parameters don't match the tool's schema (when tool_metadata is provided).
        """
        timeout = timeout or self.default_timeout

        # Validate ToolInput
        if tool_metadata:
            validated_input = validate_tool_input(
                tool_input.tool_name,
                tool_input.parameters,
                tool_metadata,
            )
        else:
            validated_input = validate_tool_input(
                tool_input.tool_name,
                tool_input.parameters,
            )

        tool_name = validated_input.tool_name
        tool_params = validated_input.parameters
        input_metadata = validated_input.metadata

        # Merge input metadata with provided metadata
        execution_metadata = {
            "tool_name": tool_name,
            "parameters": tool_params,
            "timeout": timeout,
            **input_metadata,
        }
        if tool_metadata:
            execution_metadata["tool_metadata"] = tool_metadata.model_dump()

        # Get or create trace context
        tracer = get_tracer()
        trace_context = tracer.get_trace_context()
        trace_id = (
            trace_context.get("trace_id") if trace_context else tracer.create_trace_id()
        )
        parent_observation_id = (
            trace_context.get("observation_id") if trace_context else None
        )

        # Track execution time for ToolOutput
        start_time = time.time()

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
                        "parameters": tool_params,
                        "timeout": timeout,
                    },
                ) as span:
                    # Set input on the observation
                    if span is not None and hasattr(span, "update"):
                        try:
                            span.update(input={"tool_name": tool_name, "parameters": tool_params})
                        except Exception as e:
                            logger.debug(f"Failed to update span input: {e}")

                    try:
                        # Execute with timeout if specified
                        if timeout is not None:
                            result = await asyncio.wait_for(
                                self._execute_tool_with_cancellation(
                                    tool_func, tool_params, cancellation_token
                                ),
                                timeout=timeout,
                            )
                        else:
                            result = await self._execute_tool_with_cancellation(
                                tool_func, tool_params, cancellation_token
                            )

                        # Set output on the observation
                        if span is not None and hasattr(span, "update"):
                            try:
                                # Convert result to a serializable format
                                if isinstance(result, dict):
                                    output_data = result
                                elif hasattr(result, "model_dump"):
                                    output_data = result.model_dump()
                                else:
                                    output_data = {"result": result}
                                span.update(output=output_data)
                            except Exception as e:
                                logger.debug(f"Failed to update span output: {e}")

                        execution_metadata["span_completed"] = True

                        # Calculate execution time
                        execution_time_ms = (time.time() - start_time) * 1000

                        # Return ToolOutput
                        return create_tool_output(
                            tool_name=tool_name,
                            success=True,
                            result=result,
                            execution_time_ms=execution_time_ms,
                            metadata=execution_metadata,
                        )
                    except Exception as e:
                        # Capture errors that occur during execution and update the observation
                        error_msg = str(e)
                        if span is not None and hasattr(span, "update"):
                            try:
                                span.update(
                                    output={
                                        "error": error_msg,
                                        "error_type": type(e).__name__,
                                        "success": False,
                                    }
                                )
                            except Exception as update_error:
                                logger.debug(f"Failed to update span with error: {update_error}")
                        # Re-raise to be handled by outer exception handlers
                        raise
            else:
                # Execute without tracing
                if timeout is not None:
                    result = await asyncio.wait_for(
                        self._execute_tool_with_cancellation(
                            tool_func, tool_params, cancellation_token
                        ),
                        timeout=timeout,
                    )
                else:
                    result = await self._execute_tool_with_cancellation(
                        tool_func, tool_params, cancellation_token
                    )

                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000

                # Return ToolOutput
                return create_tool_output(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time_ms,
                    metadata=execution_metadata,
                )

        except asyncio.TimeoutError:
            error_msg = f"Tool {tool_name} execution timed out after {timeout}s"
            logger.error(error_msg)
            execution_time_ms = (time.time() - start_time) * 1000

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

            # Return ToolOutput
            return create_tool_output(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                metadata=execution_metadata,
            )

        except asyncio.CancelledError:
            error_msg = f"Tool {tool_name} execution was cancelled"
            logger.warning(error_msg)
            execution_time_ms = (time.time() - start_time) * 1000

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

            # Return ToolOutput
            return create_tool_output(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                metadata=execution_metadata,
            )

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            execution_time_ms = (time.time() - start_time) * 1000

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

            # Return ToolOutput
            return create_tool_output(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                metadata=execution_metadata,
            )

    async def _execute_with_cancellation(
        self,
        agent: AgentBase,
        context: AgentContext,
        cancellation_token: Optional[asyncio.CancelledError],
    ) -> AgentOutput:
        """Execute agent with cancellation support.

        Args:
            agent: The agent to execute.
            context: The execution context (AgentContext).
            cancellation_token: Optional cancellation token.

        Returns:
            AgentOutput from the agent.
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
