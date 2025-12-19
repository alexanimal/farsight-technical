"""Input/output contracts for agents in the system.

This module defines the contracts that all agents must adhere to, including
input context schemas, output response schemas, and execution guarantees.
These contracts ensure consistency, type safety, and enable agent discovery
and coordination across the system.

According to the architecture plan:
- Agents are declarative capability providers
- Agents are domain-focused
- Agents are tool-aware but orchestration-agnostic
- Agents are stateless or lightly stateful per invocation
- Agents define input/output contracts
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import existing types to maintain compatibility
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus


class AgentInput(BaseModel):
    """Input contract for agent execution.

    This represents the structured input that agents receive. It wraps
    AgentContext to provide a contract interface while maintaining
    compatibility with existing code.

    All agents should accept AgentContext (or AgentInput) as input.
    """

    query: str = Field(..., description="The user's query or request to process")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for tracking conversation sessions",
    )
    user_id: Optional[str] = Field(
        default=None, description="Optional identifier for the user making the request"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata that can be passed between agents",
    )
    shared_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary for storing shared data between agents",
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional list of previous messages in the conversation",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when the context was created",
    )

    def to_agent_context(self) -> AgentContext:
        """Convert AgentInput to AgentContext for compatibility.

        Returns:
            AgentContext object with the same data
        """
        return AgentContext(
            query=self.query,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            metadata=self.metadata,
            shared_data=self.shared_data,
            conversation_history=self.conversation_history,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_agent_context(cls, context: AgentContext) -> "AgentInput":
        """Create AgentInput from AgentContext.

        Args:
            context: AgentContext to convert

        Returns:
            AgentInput object with the same data
        """
        return cls(
            query=context.query,
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            metadata=context.metadata,
            shared_data=context.shared_data,
            conversation_history=context.conversation_history,
            timestamp=context.timestamp,
        )


class AgentOutput(BaseModel):
    """Output contract for agent execution.

    This represents the structured output that agents return. It wraps
    AgentResponse to provide a contract interface while maintaining
    compatibility with existing code.

    All agents should return AgentResponse (or AgentOutput) as output.
    """

    content: Any = Field(
        ..., description="The main response content (text or structured data)"
    )
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS, description="The status of the response"
    )
    agent_name: str = Field(
        ..., description="The name of the agent that generated this response"
    )
    agent_category: str = Field(
        ..., description="The category of the agent that generated this response"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of tool calls that were made during processing",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )
    nested_responses: Optional[List["AgentOutput"]] = Field(
        default=None,
        description="Optional list of AgentOutput objects from coordinated agents",
    )
    error: Optional[str] = Field(
        default=None, description="Optional error message if status is ERROR"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when the response was created",
    )

    def to_agent_response(self) -> AgentResponse:
        """Convert AgentOutput to AgentResponse for compatibility.

        Returns:
            AgentResponse object with the same data
        """
        nested = None
        if self.nested_responses:
            nested = [resp.to_agent_response() for resp in self.nested_responses]

        return AgentResponse(
            content=self.content,
            status=self.status,
            agent_name=self.agent_name,
            agent_category=self.agent_category,
            tool_calls=self.tool_calls,
            metadata=self.metadata,
            nested_responses=nested,
            error=self.error,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_agent_response(cls, response: AgentResponse) -> "AgentOutput":
        """Create AgentOutput from AgentResponse.

        Args:
            response: AgentResponse to convert

        Returns:
            AgentOutput object with the same data
        """
        nested = None
        if response.nested_responses:
            nested = [
                cls.from_agent_response(resp) for resp in response.nested_responses
            ]

        return cls(
            content=response.content,
            status=response.status,
            agent_name=response.agent_name,
            agent_category=response.agent_category,
            tool_calls=response.tool_calls,
            metadata=response.metadata,
            nested_responses=nested,
            error=response.error,
            timestamp=response.timestamp,
        )


class AgentExecutionContract(BaseModel):
    """Execution contract for agent invocations.

    This defines the guarantees and constraints for agent execution, including
    timeouts, resource limits, and output guarantees.
    """

    timeout_seconds: Optional[float] = Field(
        default=None, description="Maximum time allowed for agent execution"
    )
    max_tool_calls: Optional[int] = Field(
        default=None,
        description="Maximum number of tool calls allowed during execution",
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens allowed for LLM interactions"
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tool names this agent is allowed to use (None = all tools)",
    )
    forbidden_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tool names this agent is forbidden from using",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Whether agent execution requires explicit confirmation",
    )
    cancellation_supported: bool = Field(
        default=True, description="Whether the agent supports cancellation"
    )
    output_guarantees: Dict[str, Any] = Field(
        default_factory=dict,
        description="Guarantees about the output format and content",
    )


class AgentMetadata(BaseModel):
    """Declarative metadata for an agent.

    This metadata describes what an agent can do, its capabilities, constraints,
    and resource requirements. It enables dynamic discovery and routing.
    """

    name: str = Field(..., description="Unique identifier for the agent")
    description: str = Field(
        ..., description="Human-readable description of what the agent does"
    )
    category: str = Field(
        ..., description="Category/type of agent (e.g., 'acquisition', 'orchestration')"
    )
    version: Optional[str] = Field(default=None, description="Version of the agent")
    domain: Optional[str] = Field(
        default=None, description="Domain this agent specializes in"
    )
    capabilities: List[str] = Field(
        default_factory=list, description="List of capabilities this agent provides"
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None, description="List of tool names this agent is allowed to use"
    )
    forbidden_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tool names this agent is forbidden from using",
    )
    reasoning_style: Optional[str] = Field(
        default=None,
        description="Hints about the agent's reasoning style (e.g., 'analytical', 'creative')",
    )
    resource_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource constraints (timeouts, tool calls, tokens)",
    )
    output_guarantees: Dict[str, Any] = Field(
        default_factory=dict,
        description="Guarantees about the output format and content",
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing and filtering agents"
    )


# Update forward references for nested responses
AgentOutput.model_rebuild()


def validate_agent_input(context: AgentContext) -> AgentInput:
    """Validate and convert AgentContext to AgentInput.

    Args:
        context: AgentContext to validate and convert

    Returns:
        AgentInput object

    Raises:
        ValueError: If context is invalid
    """
    if not context.query or not context.query.strip():
        raise ValueError("Agent input must have a non-empty query")

    return AgentInput.from_agent_context(context)


def create_agent_output(
    content: Any,
    agent_name: str,
    agent_category: str,
    status: ResponseStatus = ResponseStatus.SUCCESS,
    **kwargs: Any,
) -> AgentOutput:
    """Create an AgentOutput from agent execution results.

    Args:
        content: The response content
        agent_name: Name of the agent that generated the response
        agent_category: Category of the agent
        status: Status of the response
        **kwargs: Additional fields to set on the output

    Returns:
        AgentOutput object
    """
    return AgentOutput(
        content=content,
        status=status,
        agent_name=agent_name,
        agent_category=agent_category,
        **kwargs,
    )
