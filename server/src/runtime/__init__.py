"""Runtime layer for agent execution.

This module provides the execution substrate of the system:
- Executor: Executes agent invocations and tool calls
- AsyncManager: Manages concurrent execution with limits and cancellation
- Tracer: Provides tracing with Langfuse integration
"""

from .async_manager import AsyncManager, get_async_manager, set_async_manager
from .executor import Executor, get_executor, set_executor
from .tracing import (
    Tracer,
    create_correlation_id,
    create_trace_id,
    get_trace_context,
    get_tracer,
    log_event,
    set_tracer,
)

__all__ = [
    # Executor
    "Executor",
    "get_executor",
    "set_executor",
    # AsyncManager
    "AsyncManager",
    "get_async_manager",
    "set_async_manager",
    # Tracer
    "Tracer",
    "get_tracer",
    "set_tracer",
    "create_trace_id",
    "create_correlation_id",
    "get_trace_context",
    "log_event",
]
