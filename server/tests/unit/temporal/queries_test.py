"""Unit tests for the Temporal queries module.

This module tests all query schemas and models in queries.py, including
validation, serialization, edge cases, and error handling.
"""

from datetime import datetime
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.temporal.queries import (
    QUERY_AGENT_STATUS,
    QUERY_PROGRESS,
    QUERY_STATE,
    QUERY_STATUS,
    AgentExecutionStatus,
    AgentStatusQueryResult,
    WorkflowProgressQueryResult,
    WorkflowStateQueryResult,
    WorkflowStatus,
    WorkflowStatusQueryResult,
)


class TestWorkflowStatus:
    """Test WorkflowStatus enum."""

    def test_enum_values(self):
        """Test that all enum values are defined correctly."""
        assert WorkflowStatus.RUNNING == "running"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"
        assert WorkflowStatus.CANCELLED == "cancelled"
        assert WorkflowStatus.PAUSED == "paused"
        assert WorkflowStatus.PENDING == "pending"

    def test_enum_string_representation(self):
        """Test that enum values are strings."""
        assert isinstance(WorkflowStatus.RUNNING, str)
        assert isinstance(WorkflowStatus.COMPLETED, str)

    def test_enum_membership(self):
        """Test that enum values can be checked for membership."""
        assert "running" in WorkflowStatus.__members__.values()
        assert "completed" in WorkflowStatus.__members__.values()
        assert "invalid" not in WorkflowStatus.__members__.values()


class TestAgentExecutionStatus:
    """Test AgentExecutionStatus model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        status = AgentExecutionStatus(
            agent_name="test_agent",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        assert status.agent_name == "test_agent"
        assert status.agent_category == "test"
        assert status.status == "running"
        assert status.started_at == started_at
        assert status.completed_at is None
        assert status.error is None
        assert status.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        completed_at = datetime(2024, 1, 1, 12, 5, 0)
        status = AgentExecutionStatus(
            agent_name="test_agent",
            agent_category="test",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            error=None,
            metadata={"key": "value"},
        )
        assert status.agent_name == "test_agent"
        assert status.agent_category == "test"
        assert status.status == "completed"
        assert status.started_at == started_at
        assert status.completed_at == completed_at
        assert status.error is None
        assert status.metadata == {"key": "value"}

    def test_create_with_error(self):
        """Test creating with error field."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        completed_at = datetime(2024, 1, 1, 12, 5, 0)
        status = AgentExecutionStatus(
            agent_name="test_agent",
            agent_category="test",
            status="failed",
            started_at=started_at,
            completed_at=completed_at,
            error="Execution failed",
            metadata={},
        )
        assert status.status == "failed"
        assert status.error == "Execution failed"

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        status = AgentExecutionStatus(
            agent_name="test_agent",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        assert status.completed_at is None
        assert status.error is None
        assert status.metadata == {}

    @pytest.mark.parametrize(
        "field_name",
        ["agent_name", "agent_category", "status", "started_at"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "agent_name": "test_agent",
            "agent_category": "test",
            "status": "running",
            "started_at": datetime(2024, 1, 1, 12, 0, 0),
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            AgentExecutionStatus(**kwargs)

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        status = AgentExecutionStatus(
            agent_name="test_agent",
            agent_category="test",
            status="running",
            started_at=started_at,
            metadata={"key": "value"},
        )
        data = status.model_dump()
        assert isinstance(data, dict)
        assert data["agent_name"] == "test_agent"
        assert data["agent_category"] == "test"
        assert data["status"] == "running"
        assert data["metadata"] == {"key": "value"}

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "agent_name": "test_agent",
            "agent_category": "test",
            "status": "running",
            "started_at": "2024-01-01T12:00:00",
            "metadata": {"key": "value"},
        }
        status = AgentExecutionStatus(**data)
        assert status.agent_name == "test_agent"
        assert status.metadata == {"key": "value"}


class TestWorkflowStatusQueryResult:
    """Test WorkflowStatusQueryResult model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            started_at=started_at,
        )
        assert result.workflow_id == "workflow-123"
        assert result.status == WorkflowStatus.RUNNING
        assert result.started_at == started_at
        assert result.completed_at is None
        assert result.error is None
        assert result.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        completed_at = datetime(2024, 1, 1, 13, 0, 0)
        result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            error=None,
            metadata={"key": "value"},
        )
        assert result.workflow_id == "workflow-123"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.completed_at == completed_at
        assert result.metadata == {"key": "value"}

    def test_create_with_error(self):
        """Test creating with error field."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        completed_at = datetime(2024, 1, 1, 13, 0, 0)
        result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.FAILED,
            started_at=started_at,
            completed_at=completed_at,
            error="Workflow failed",
            metadata={},
        )
        assert result.status == WorkflowStatus.FAILED
        assert result.error == "Workflow failed"

    def test_all_status_values(self):
        """Test that all WorkflowStatus values can be used."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        for status in WorkflowStatus:
            result = WorkflowStatusQueryResult(
                workflow_id="workflow-123",
                status=status,
                started_at=started_at,
            )
            assert result.status == status

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            started_at=started_at,
        )
        assert result.completed_at is None
        assert result.error is None
        assert result.metadata == {}

    @pytest.mark.parametrize(
        "field_name",
        ["workflow_id", "status", "started_at"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "workflow_id": "workflow-123",
            "status": WorkflowStatus.RUNNING,
            "started_at": datetime(2024, 1, 1, 12, 0, 0),
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            WorkflowStatusQueryResult(**kwargs)

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            started_at=started_at,
            metadata={"key": "value"},
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["workflow_id"] == "workflow-123"
        assert data["status"] == "running"
        assert data["metadata"] == {"key": "value"}

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "workflow_id": "workflow-123",
            "status": "running",
            "started_at": "2024-01-01T12:00:00",
            "metadata": {"key": "value"},
        }
        result = WorkflowStatusQueryResult(**data)
        assert result.workflow_id == "workflow-123"
        assert result.status == WorkflowStatus.RUNNING
        assert result.metadata == {"key": "value"}


class TestWorkflowProgressQueryResult:
    """Test WorkflowProgressQueryResult model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
        )
        assert result.workflow_id == "workflow-123"
        assert result.status == WorkflowStatus.RUNNING
        assert result.total_agents == 0
        assert result.completed_agents == 0
        assert result.running_agents == 0
        assert result.failed_agents == 0
        assert result.agent_statuses == []
        assert result.progress_percentage is None
        assert result.current_step is None
        assert result.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        agent_status = AgentExecutionStatus(
            agent_name="agent1",
            agent_category="test",
            status="completed",
            started_at=started_at,
        )
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            total_agents=3,
            completed_agents=1,
            running_agents=1,
            failed_agents=0,
            agent_statuses=[agent_status],
            progress_percentage=33.33,
            current_step="Executing agent1",
            metadata={"key": "value"},
        )
        assert result.total_agents == 3
        assert result.completed_agents == 1
        assert result.running_agents == 1
        assert result.failed_agents == 0
        assert len(result.agent_statuses) == 1
        assert result.progress_percentage == 33.33
        assert result.current_step == "Executing agent1"
        assert result.metadata == {"key": "value"}

    def test_with_multiple_agent_statuses(self):
        """Test creating with multiple agent statuses."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        agent1 = AgentExecutionStatus(
            agent_name="agent1",
            agent_category="test",
            status="completed",
            started_at=started_at,
        )
        agent2 = AgentExecutionStatus(
            agent_name="agent2",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            agent_statuses=[agent1, agent2],
        )
        assert len(result.agent_statuses) == 2
        assert result.agent_statuses[0].agent_name == "agent1"
        assert result.agent_statuses[1].agent_name == "agent2"

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
        )
        assert result.total_agents == 0
        assert result.completed_agents == 0
        assert result.running_agents == 0
        assert result.failed_agents == 0
        assert result.agent_statuses == []
        assert result.progress_percentage is None
        assert result.current_step is None
        assert result.metadata == {}

    @pytest.mark.parametrize(
        "field_name",
        ["workflow_id", "status"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "workflow_id": "workflow-123",
            "status": WorkflowStatus.RUNNING,
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            WorkflowProgressQueryResult(**kwargs)

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        agent_status = AgentExecutionStatus(
            agent_name="agent1",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            total_agents=1,
            agent_statuses=[agent_status],
            progress_percentage=50.0,
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["workflow_id"] == "workflow-123"
        assert data["total_agents"] == 1
        assert data["progress_percentage"] == 50.0
        assert len(data["agent_statuses"]) == 1
        assert isinstance(data["agent_statuses"][0], dict)

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "workflow_id": "workflow-123",
            "status": "running",
            "total_agents": 2,
            "completed_agents": 1,
            "agent_statuses": [
                {
                    "agent_name": "agent1",
                    "agent_category": "test",
                    "status": "completed",
                    "started_at": "2024-01-01T12:00:00",
                }
            ],
        }
        result = WorkflowProgressQueryResult(**data)
        assert result.workflow_id == "workflow-123"
        assert result.total_agents == 2
        assert len(result.agent_statuses) == 1


class TestWorkflowStateQueryResult:
    """Test WorkflowStateQueryResult model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        result = WorkflowStateQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            context={"query": "test"},
        )
        assert result.workflow_id == "workflow-123"
        assert result.status == WorkflowStatus.RUNNING
        assert result.context == {"query": "test"}
        assert result.agent_responses == []
        assert result.shared_data == {}
        assert result.execution_history == []
        assert result.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        agent_response = {"agent_name": "agent1", "content": "response"}
        execution_event = {"event": "agent_started", "timestamp": "2024-01-01T12:00:00"}
        result = WorkflowStateQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            context={"query": "test", "conversation_id": "conv-1"},
            agent_responses=[agent_response],
            shared_data={"key": "value"},
            execution_history=[execution_event],
            metadata={"key": "value"},
        )
        assert result.context == {"query": "test", "conversation_id": "conv-1"}
        assert len(result.agent_responses) == 1
        assert result.shared_data == {"key": "value"}
        assert len(result.execution_history) == 1
        assert result.metadata == {"key": "value"}

    def test_with_multiple_agent_responses(self):
        """Test creating with multiple agent responses."""
        responses = [
            {"agent_name": "agent1", "content": "response1"},
            {"agent_name": "agent2", "content": "response2"},
        ]
        result = WorkflowStateQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            context={"query": "test"},
            agent_responses=responses,
        )
        assert len(result.agent_responses) == 2
        assert result.agent_responses[0]["agent_name"] == "agent1"
        assert result.agent_responses[1]["agent_name"] == "agent2"

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        result = WorkflowStateQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            context={"query": "test"},
        )
        assert result.agent_responses == []
        assert result.shared_data == {}
        assert result.execution_history == []
        assert result.metadata == {}

    @pytest.mark.parametrize(
        "field_name",
        ["workflow_id", "status", "context"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "workflow_id": "workflow-123",
            "status": WorkflowStatus.RUNNING,
            "context": {"query": "test"},
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            WorkflowStateQueryResult(**kwargs)

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        result = WorkflowStateQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            context={"query": "test"},
            agent_responses=[{"agent": "agent1"}],
            shared_data={"key": "value"},
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["workflow_id"] == "workflow-123"
        assert data["context"] == {"query": "test"}
        assert len(data["agent_responses"]) == 1
        assert data["shared_data"] == {"key": "value"}

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "workflow_id": "workflow-123",
            "status": "running",
            "context": {"query": "test"},
            "agent_responses": [{"agent": "agent1"}],
            "shared_data": {"key": "value"},
        }
        result = WorkflowStateQueryResult(**data)
        assert result.workflow_id == "workflow-123"
        assert result.context == {"query": "test"}
        assert len(result.agent_responses) == 1


class TestAgentStatusQueryResult:
    """Test AgentStatusQueryResult model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        result = AgentStatusQueryResult(
            workflow_id="workflow-123",
            agent_name="agent1",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        assert result.workflow_id == "workflow-123"
        assert result.agent_name == "agent1"
        assert result.agent_category == "test"
        assert result.status == "running"
        assert result.started_at == started_at
        assert result.completed_at is None
        assert result.response is None
        assert result.error is None
        assert result.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        completed_at = datetime(2024, 1, 1, 12, 5, 0)
        response = {"content": "response", "status": "success"}
        result = AgentStatusQueryResult(
            workflow_id="workflow-123",
            agent_name="agent1",
            agent_category="test",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            response=response,
            error=None,
            metadata={"key": "value"},
        )
        assert result.workflow_id == "workflow-123"
        assert result.agent_name == "agent1"
        assert result.status == "completed"
        assert result.completed_at == completed_at
        assert result.response == response
        assert result.metadata == {"key": "value"}

    def test_create_with_error(self):
        """Test creating with error field."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        completed_at = datetime(2024, 1, 1, 12, 5, 0)
        result = AgentStatusQueryResult(
            workflow_id="workflow-123",
            agent_name="agent1",
            agent_category="test",
            status="failed",
            started_at=started_at,
            completed_at=completed_at,
            error="Agent execution failed",
            metadata={},
        )
        assert result.status == "failed"
        assert result.error == "Agent execution failed"

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        result = AgentStatusQueryResult(
            workflow_id="workflow-123",
            agent_name="agent1",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        assert result.completed_at is None
        assert result.response is None
        assert result.error is None
        assert result.metadata == {}

    @pytest.mark.parametrize(
        "field_name",
        ["workflow_id", "agent_name", "agent_category", "status", "started_at"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "workflow_id": "workflow-123",
            "agent_name": "agent1",
            "agent_category": "test",
            "status": "running",
            "started_at": datetime(2024, 1, 1, 12, 0, 0),
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            AgentStatusQueryResult(**kwargs)

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        response = {"content": "response"}
        result = AgentStatusQueryResult(
            workflow_id="workflow-123",
            agent_name="agent1",
            agent_category="test",
            status="completed",
            started_at=started_at,
            response=response,
            metadata={"key": "value"},
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["workflow_id"] == "workflow-123"
        assert data["agent_name"] == "agent1"
        assert data["response"] == response
        assert data["metadata"] == {"key": "value"}

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "workflow_id": "workflow-123",
            "agent_name": "agent1",
            "agent_category": "test",
            "status": "completed",
            "started_at": "2024-01-01T12:00:00",
            "response": {"content": "response"},
            "metadata": {"key": "value"},
        }
        result = AgentStatusQueryResult(**data)
        assert result.workflow_id == "workflow-123"
        assert result.agent_name == "agent1"
        assert result.response == {"content": "response"}


class TestQueryConstants:
    """Test query name constants."""

    def test_query_status_constant(self):
        """Test QUERY_STATUS constant."""
        assert QUERY_STATUS == "workflow_status"
        assert isinstance(QUERY_STATUS, str)

    def test_query_progress_constant(self):
        """Test QUERY_PROGRESS constant."""
        assert QUERY_PROGRESS == "workflow_progress"
        assert isinstance(QUERY_PROGRESS, str)

    def test_query_state_constant(self):
        """Test QUERY_STATE constant."""
        assert QUERY_STATE == "workflow_state"
        assert isinstance(QUERY_STATE, str)

    def test_query_agent_status_constant(self):
        """Test QUERY_AGENT_STATUS constant."""
        assert QUERY_AGENT_STATUS == "agent_status"
        assert isinstance(QUERY_AGENT_STATUS, str)

    def test_all_constants_are_strings(self):
        """Test that all query constants are strings."""
        assert isinstance(QUERY_STATUS, str)
        assert isinstance(QUERY_PROGRESS, str)
        assert isinstance(QUERY_STATE, str)
        assert isinstance(QUERY_AGENT_STATUS, str)


class TestQueryModelsEdgeCases:
    """Test edge cases and special scenarios for query models."""

    def test_agent_execution_status_with_empty_metadata(self):
        """Test AgentExecutionStatus with empty metadata dict."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        status = AgentExecutionStatus(
            agent_name="test_agent",
            agent_category="test",
            status="running",
            started_at=started_at,
            metadata={},
        )
        assert status.metadata == {}

    def test_workflow_progress_with_zero_agents(self):
        """Test WorkflowProgressQueryResult with zero agents."""
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.PENDING,
            total_agents=0,
        )
        assert result.total_agents == 0
        assert result.progress_percentage is None

    def test_workflow_progress_with_100_percent(self):
        """Test WorkflowProgressQueryResult with 100% progress."""
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.COMPLETED,
            total_agents=2,
            completed_agents=2,
            progress_percentage=100.0,
        )
        assert result.progress_percentage == 100.0

    def test_workflow_state_with_empty_arrays(self):
        """Test WorkflowStateQueryResult with empty arrays."""
        result = WorkflowStateQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.PENDING,
            context={"query": "test"},
            agent_responses=[],
            execution_history=[],
        )
        assert result.agent_responses == []
        assert result.execution_history == []

    def test_agent_status_with_none_response(self):
        """Test AgentStatusQueryResult with None response."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        result = AgentStatusQueryResult(
            workflow_id="workflow-123",
            agent_name="agent1",
            agent_category="test",
            status="running",
            started_at=started_at,
            response=None,
        )
        assert result.response is None

    def test_workflow_status_with_all_enum_values(self):
        """Test that all WorkflowStatus enum values work with query results."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        for status in WorkflowStatus:
            result = WorkflowStatusQueryResult(
                workflow_id="workflow-123",
                status=status,
                started_at=started_at,
            )
            assert result.status == status

    def test_nested_agent_status_serialization(self):
        """Test that nested AgentExecutionStatus in WorkflowProgressQueryResult serializes correctly."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        agent_status = AgentExecutionStatus(
            agent_name="agent1",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            agent_statuses=[agent_status],
        )
        data = result.model_dump()
        assert len(data["agent_statuses"]) == 1
        assert data["agent_statuses"][0]["agent_name"] == "agent1"
        assert isinstance(data["agent_statuses"][0], dict)

    def test_datetime_serialization(self):
        """Test that datetime fields serialize correctly."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            started_at=started_at,
        )
        data = result.model_dump()
        assert "started_at" in data
        # Pydantic serializes datetime to ISO format string
        assert isinstance(data["started_at"], str) or isinstance(data["started_at"], datetime)

    def test_metadata_preservation(self):
        """Test that metadata dictionaries are preserved correctly."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        complex_metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "string": "test",
        }
        result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            started_at=started_at,
            metadata=complex_metadata,
        )
        assert result.metadata == complex_metadata
        data = result.model_dump()
        assert data["metadata"] == complex_metadata
