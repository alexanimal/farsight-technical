"""Query definitions and schemas for Temporal workflows.

This module provides query schemas that allow external systems (API, UI, etc.)
to query the state of running Temporal workflows. Queries are used to GET
information FROM workflows.

Queries enable workflows to expose:
- Current execution status
- Progress information
- Workflow state
- Agent execution status
- Error information

All queries use Pydantic models for type safety and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Enumeration of workflow execution statuses."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    PENDING = "pending"


class AgentExecutionStatus(BaseModel):
    """Status information for an agent execution within a workflow.

    Attributes:
        agent_name: Name of the agent.
        agent_category: Category of the agent.
        status: Current status of the agent execution.
        started_at: Timestamp when agent execution started.
        completed_at: Optional timestamp when agent execution completed.
        error: Optional error message if execution failed.
        metadata: Additional metadata about the agent execution.
    """

    agent_name: str = Field(
        ...,
        description="Name of the agent",
    )
    agent_category: str = Field(
        ...,
        description="Category of the agent",
    )
    status: str = Field(
        ...,
        description="Current status of the agent execution (e.g., 'running', 'completed', 'failed')",
    )
    started_at: datetime = Field(
        ...,
        description="Timestamp when agent execution started",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Optional timestamp when agent execution completed",
    )
    error: Optional[str] = Field(
        default=None,
        description="Optional error message if execution failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the agent execution",
    )


class WorkflowStatusQueryResult(BaseModel):
    """Result of a workflow status query.

    This query returns the current status and basic information
    about a workflow execution.

    Attributes:
        workflow_id: The Temporal workflow ID.
        status: Current status of the workflow.
        started_at: Timestamp when workflow started.
        completed_at: Optional timestamp when workflow completed.
        error: Optional error message if workflow failed.
        metadata: Additional metadata about the workflow.
    """

    workflow_id: str = Field(
        ...,
        description="The Temporal workflow ID",
    )
    status: WorkflowStatus = Field(
        ...,
        description="Current status of the workflow",
    )
    started_at: datetime = Field(
        ...,
        description="Timestamp when workflow started",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Optional timestamp when workflow completed",
    )
    error: Optional[str] = Field(
        default=None,
        description="Optional error message if workflow failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the workflow",
    )


class WorkflowProgressQueryResult(BaseModel):
    """Result of a workflow progress query.

    This query returns progress information about a workflow execution,
    including which agents have run, which are running, and overall progress.

    Attributes:
        workflow_id: The Temporal workflow ID.
        status: Current status of the workflow.
        total_agents: Total number of agents scheduled to run.
        completed_agents: Number of agents that have completed.
        running_agents: Number of agents currently running.
        failed_agents: Number of agents that have failed.
        agent_statuses: List of status information for each agent.
        progress_percentage: Optional progress percentage (0-100).
        current_step: Optional description of the current step.
        metadata: Additional metadata about the progress.
    """

    workflow_id: str = Field(
        ...,
        description="The Temporal workflow ID",
    )
    status: WorkflowStatus = Field(
        ...,
        description="Current status of the workflow",
    )
    total_agents: int = Field(
        default=0,
        description="Total number of agents scheduled to run",
    )
    completed_agents: int = Field(
        default=0,
        description="Number of agents that have completed",
    )
    running_agents: int = Field(
        default=0,
        description="Number of agents currently running",
    )
    failed_agents: int = Field(
        default=0,
        description="Number of agents that have failed",
    )
    agent_statuses: List[AgentExecutionStatus] = Field(
        default_factory=list,
        description="List of status information for each agent",
    )
    progress_percentage: Optional[float] = Field(
        default=None,
        description="Optional progress percentage (0-100)",
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Optional description of the current step",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the progress",
    )


class WorkflowStateQueryResult(BaseModel):
    """Result of a workflow state query.

    This query returns the full state of a workflow, including
    all context, shared data, and execution history.

    Attributes:
        workflow_id: The Temporal workflow ID.
        status: Current status of the workflow.
        context: The current AgentContext being used.
        agent_responses: List of AgentResponse objects from completed agents.
        shared_data: Dictionary of shared data between agents.
        execution_history: List of execution events.
        metadata: Additional metadata about the workflow state.
    """

    workflow_id: str = Field(
        ...,
        description="The Temporal workflow ID",
    )
    status: WorkflowStatus = Field(
        ...,
        description="Current status of the workflow",
    )
    context: Dict[str, Any] = Field(
        ...,
        description="The current AgentContext being used (serialized)",
    )
    agent_responses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of AgentResponse objects from completed agents (serialized)",
    )
    shared_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of shared data between agents",
    )
    execution_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of execution events",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the workflow state",
    )


class AgentStatusQueryResult(BaseModel):
    """Result of an agent status query.

    This query returns detailed status information for a specific
    agent within a workflow.

    Attributes:
        workflow_id: The Temporal workflow ID.
        agent_name: Name of the agent.
        agent_category: Category of the agent.
        status: Current status of the agent execution.
        started_at: Timestamp when agent execution started.
        completed_at: Optional timestamp when agent execution completed.
        response: Optional AgentResponse from the agent (serialized).
        error: Optional error message if execution failed.
        metadata: Additional metadata about the agent execution.
    """

    workflow_id: str = Field(
        ...,
        description="The Temporal workflow ID",
    )
    agent_name: str = Field(
        ...,
        description="Name of the agent",
    )
    agent_category: str = Field(
        ...,
        description="Category of the agent",
    )
    status: str = Field(
        ...,
        description="Current status of the agent execution",
    )
    started_at: datetime = Field(
        ...,
        description="Timestamp when agent execution started",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Optional timestamp when agent execution completed",
    )
    response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional AgentResponse from the agent (serialized)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Optional error message if execution failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the agent execution",
    )


# Query name constants for use in workflows
QUERY_STATUS = "workflow_status"
QUERY_PROGRESS = "workflow_progress"
QUERY_STATE = "workflow_state"
QUERY_AGENT_STATUS = "agent_status"
