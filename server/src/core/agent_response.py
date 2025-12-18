"""Response data structure for agent operations.

This module provides the AgentResponse class that serves as the structured
output format for all agents in the system. It provides a consistent way
to return results, metadata, and status information from agent operations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResponseStatus(str, Enum):
    """Status enumeration for agent responses."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    PENDING = "pending"


class AgentResponse(BaseModel):
    """Structured response from an agent operation.

    This class represents the output from an agent, including the response
    content, status, metadata about what was done, and any nested responses
    from coordinated agents.

    Attributes:
        content: The main response content (text or structured data).
        status: The status of the response (success, partial, error, pending).
        agent_name: The name of the agent that generated this response.
        agent_category: The category of the agent that generated this response.
        tool_calls: Optional list of tool calls that were made during processing.
            Each tool call is a dict with 'name', 'parameters', and optionally 'result'.
        metadata: Additional metadata about the response, such as processing time,
            data sources used, confidence scores, etc.
        nested_responses: Optional list of AgentResponse objects from coordinated agents.
            This is useful for orchestration agents that coordinate multiple agents.
        error: Optional error message if status is ERROR.
        timestamp: Timestamp when the response was created.
    """

    content: Any = Field(
        ...,
        description="The main response content (text or structured data)",
    )
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="The status of the response",
    )
    agent_name: str = Field(
        ...,
        description="The name of the agent that generated this response",
    )
    agent_category: str = Field(
        ...,
        description="The category of the agent that generated this response",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of tool calls that were made during processing",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response",
    )
    nested_responses: Optional[List["AgentResponse"]] = Field(
        default=None,
        description="Optional list of AgentResponse objects from coordinated agents",
    )
    error: Optional[str] = Field(
        default=None,
        description="Optional error message if status is ERROR",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the response was created",
    )

    def add_tool_call(
        self,
        name: str,
        parameters: Dict[str, Any],
        result: Optional[Any] = None,
    ) -> None:
        """Add a tool call to the response.

        Args:
            name: The name of the tool that was called.
            parameters: The parameters that were passed to the tool.
            result: Optional result from the tool call.
        """
        if self.tool_calls is None:
            self.tool_calls = []

        tool_call: Dict[str, Any] = {
            "name": name,
            "parameters": parameters,
        }
        if result is not None:
            tool_call["result"] = result

        self.tool_calls.append(tool_call)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a value to the metadata dictionary.

        Args:
            key: The key to store the value under.
            value: The value to store.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the metadata dictionary.

        Args:
            key: The key to retrieve the value for.
            default: Default value to return if key is not found.

        Returns:
            The value associated with the key, or the default value if not found.
        """
        return self.metadata.get(key, default)

    def add_nested_response(self, response: "AgentResponse") -> None:
        """Add a nested response from a coordinated agent.

        Args:
            response: The AgentResponse object to add as a nested response.
        """
        if self.nested_responses is None:
            self.nested_responses = []
        self.nested_responses.append(response)

    def set_error(self, error_message: str) -> None:
        """Set the response status to error and store the error message.

        Args:
            error_message: The error message to store.
        """
        self.status = ResponseStatus.ERROR
        self.error = error_message

    def is_success(self) -> bool:
        """Check if the response status is success.

        Returns:
            True if status is SUCCESS, False otherwise.
        """
        return self.status == ResponseStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if the response status is error.

        Returns:
            True if status is ERROR, False otherwise.
        """
        return self.status == ResponseStatus.ERROR

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert the response to a dictionary.

        Override to ensure timestamp is serialized as ISO format string.

        Returns:
            Dictionary representation of the response.
        """
        data = super().model_dump(**kwargs)
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data

    @classmethod
    def create_success(
        cls,
        content: Any,
        agent_name: str,
        agent_category: str,
        **kwargs: Any,
    ) -> "AgentResponse":
        """Create a successful response.

        Args:
            content: The response content.
            agent_name: The name of the agent.
            agent_category: The category of the agent.
            **kwargs: Additional fields to set on the response.

        Returns:
            A new AgentResponse with SUCCESS status.
        """
        return cls(
            content=content,
            status=ResponseStatus.SUCCESS,
            agent_name=agent_name,
            agent_category=agent_category,
            **kwargs,
        )

    @classmethod
    def create_error(
        cls,
        error_message: str,
        agent_name: str,
        agent_category: str,
        **kwargs: Any,
    ) -> "AgentResponse":
        """Create an error response.

        Args:
            error_message: The error message.
            agent_name: The name of the agent.
            agent_category: The category of the agent.
            **kwargs: Additional fields to set on the response.

        Returns:
            A new AgentResponse with ERROR status.
        """
        return cls(
            content="",
            status=ResponseStatus.ERROR,
            agent_name=agent_name,
            agent_category=agent_category,
            error=error_message,
            **kwargs,
        )


# Update forward references for nested responses
AgentResponse.model_rebuild()
