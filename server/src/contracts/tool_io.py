"""Input/output contracts for tools in the system.

This module defines the contracts that all tools must adhere to, including
input parameter schemas, output result schemas, and metadata requirements.
These contracts ensure consistency, type safety, and enable tool discovery
and validation across the system.

According to the architecture plan:
- Tools are first-class, independently testable units
- Tools have declarative metadata (name, schema, cost, latency)
- Tools are callable by agents or orchestration
- Tools wrap external APIs, databases, file systems, or services
- Tools are synchronous and callable
- Tools contain no orchestration logic
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolParameterSchema(BaseModel):
    """Schema definition for a single tool parameter.

    This represents one parameter that a tool accepts, including its type,
    description, and validation rules.
    """

    name: str = Field(..., description="Name of the parameter")
    type: str = Field(
        ...,
        description="Type of the parameter (string, integer, float, boolean, object, array)",
    )
    description: str = Field(..., description="Human-readable description of the parameter")
    required: bool = Field(default=False, description="Whether this parameter is required")
    default: Optional[Any] = Field(
        default=None, description="Default value if parameter is not provided"
    )
    enum: Optional[List[Any]] = Field(
        default=None, description="Allowed values for this parameter (if applicable)"
    )


class ToolMetadata(BaseModel):
    """Declarative metadata for a tool.

    This metadata describes what a tool can do, its capabilities, constraints,
    and resource requirements. It enables dynamic discovery and routing.
    """

    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="Human-readable description of what the tool does")
    version: Optional[str] = Field(default=None, description="Version of the tool")
    parameters: List[ToolParameterSchema] = Field(
        default_factory=list, description="List of parameters this tool accepts"
    )
    returns: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Schema describing what the tool returns (JSON schema format)",
    )
    cost_per_call: Optional[float] = Field(
        default=None,
        description="Estimated cost per tool call (in tokens, API credits, etc.)",
    )
    estimated_latency_ms: Optional[float] = Field(
        default=None,
        description="Estimated latency in milliseconds for typical tool execution",
    )
    timeout_seconds: Optional[float] = Field(
        default=None, description="Recommended timeout in seconds for tool execution"
    )
    side_effects: bool = Field(
        default=True,
        description="Whether this tool has side effects (e.g., writes to database, makes API calls)",
    )
    idempotent: bool = Field(
        default=False, description="Whether this tool is idempotent (safe to retry)"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing and filtering tools"
    )


class ToolInput(BaseModel):
    """Input contract for tool execution.

    This represents the structured input that tools receive when called.
    All tools should accept parameters as a dictionary.
    """

    tool_name: str = Field(..., description="Name of the tool being called")
    parameters: Dict[str, Any] = Field(
        ..., description="Dictionary of parameters to pass to the tool function"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the tool call (e.g., caller, trace_id)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when the tool call was initiated",
    )


class ToolOutput(BaseModel):
    """Output contract for tool execution.

    This represents the structured output that tools return. Tools should
    return structured data (lists, dicts, etc.) that can be serialized.
    """

    success: bool = Field(..., description="Whether the tool execution succeeded")
    result: Optional[Any] = Field(
        default=None, description="The result from the tool execution (structured data)"
    )
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    execution_time_ms: Optional[float] = Field(
        default=None, description="Time taken to execute the tool in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution (e.g., tokens used, API calls made)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when the tool execution completed",
    )


class ToolExecutionContract(BaseModel):
    """Execution contract for tool calls.

    This defines the guarantees and constraints for tool execution, including
    timeouts, retry policies, and cancellation support.
    """

    timeout_seconds: Optional[float] = Field(
        default=None, description="Maximum time allowed for tool execution"
    )
    max_retries: int = Field(default=0, description="Maximum number of retry attempts on failure")
    retry_on_errors: List[str] = Field(
        default_factory=list,
        description="List of error types to retry on (e.g., 'TimeoutError', 'ConnectionError')",
    )
    cancellation_supported: bool = Field(
        default=False, description="Whether the tool supports cancellation"
    )
    requires_isolation: bool = Field(
        default=False,
        description="Whether the tool requires isolated execution (e.g., sandboxed)",
    )


# Type aliases for tool function signatures
ToolFunction = Callable[..., Any]  # Tool functions can have any signature
AsyncToolFunction = Callable[..., Any]  # Async tool functions (return awaitable)


def validate_tool_input(
    tool_name: str, parameters: Dict[str, Any], metadata: Optional[ToolMetadata] = None
) -> ToolInput:
    """Validate and create a ToolInput from raw parameters.

    Args:
        tool_name: Name of the tool being called
        parameters: Dictionary of parameters to pass to the tool
        metadata: Optional tool metadata for validation

    Returns:
        Validated ToolInput object

    Raises:
        ValueError: If parameters don't match the tool's schema
    """
    if metadata:
        # Validate required parameters
        required_params = {p.name for p in metadata.parameters if p.required}
        provided_params = set(parameters.keys())
        missing_params = required_params - provided_params
        if missing_params:
            raise ValueError(
                f"Tool '{tool_name}' requires parameters: {missing_params}. "
                f"Provided: {provided_params}"
            )

    return ToolInput(tool_name=tool_name, parameters=parameters, metadata={})


def create_tool_output(
    tool_name: str,
    success: bool,
    result: Optional[Any] = None,
    error: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolOutput:
    """Create a ToolOutput from tool execution results.

    Args:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        result: Result from tool execution (if successful)
        error: Error message (if failed)
        execution_time_ms: Execution time in milliseconds
        metadata: Additional metadata about execution

    Returns:
        ToolOutput object
    """
    return ToolOutput(
        tool_name=tool_name,
        success=success,
        result=result,
        error=error,
        execution_time_ms=execution_time_ms,
        metadata=metadata or {},
    )
