"""Unit tests for the agent response module.

This module tests the AgentResponse class, AgentInsight model, and ResponseStatus enum,
including response creation, status management, and data manipulation methods.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.core.agent_response import AgentInsight, AgentResponse, ResponseStatus


@pytest.fixture
def sample_agent_insight():
    """Create a sample AgentInsight for testing."""
    return AgentInsight(
        summary="Test insight summary",
        key_findings=["Finding 1", "Finding 2"],
        evidence={"data": "test"},
        confidence=0.95,
    )


@pytest.fixture
def sample_tool_call():
    """Create a sample tool call dictionary."""
    return {
        "name": "get_organizations",
        "parameters": {"query": "AI companies"},
        "result": [{"name": "Company A"}],
    }


class TestAgentInsight:
    """Test AgentInsight model."""

    def test_agent_insight_creation(self, sample_agent_insight):
        """Test creating AgentInsight with all fields."""
        assert sample_agent_insight.summary == "Test insight summary"
        assert len(sample_agent_insight.key_findings) == 2
        assert sample_agent_insight.evidence == {"data": "test"}
        assert sample_agent_insight.confidence == 0.95

    def test_agent_insight_minimal(self):
        """Test creating AgentInsight with only required fields."""
        insight = AgentInsight(summary="Minimal insight")
        assert insight.summary == "Minimal insight"
        assert insight.key_findings is None
        assert insight.evidence is None
        assert insight.confidence is None

    def test_agent_insight_confidence_validation(self):
        """Test AgentInsight confidence validation."""
        # Valid confidence
        insight = AgentInsight(summary="test", confidence=0.5)
        assert insight.confidence == 0.5

        # Confidence at boundaries
        insight_min = AgentInsight(summary="test", confidence=0.0)
        assert insight_min.confidence == 0.0

        insight_max = AgentInsight(summary="test", confidence=1.0)
        assert insight_max.confidence == 1.0

        # Invalid confidence (too high)
        with pytest.raises(Exception):  # Pydantic ValidationError
            AgentInsight(summary="test", confidence=1.5)

        # Invalid confidence (negative)
        with pytest.raises(Exception):  # Pydantic ValidationError
            AgentInsight(summary="test", confidence=-0.1)


class TestResponseStatus:
    """Test ResponseStatus enum."""

    def test_response_status_values(self):
        """Test ResponseStatus enum values."""
        assert ResponseStatus.SUCCESS == "success"
        assert ResponseStatus.PARTIAL == "partial"
        assert ResponseStatus.ERROR == "error"
        assert ResponseStatus.PENDING == "pending"


class TestAgentResponseCreation:
    """Test AgentResponse creation."""

    def test_create_with_agent_insight(self, sample_agent_insight):
        """Test creating AgentResponse with AgentInsight content."""
        response = AgentResponse(
            content=sample_agent_insight,
            agent_name="test_agent",
            agent_category="testing",
        )
        assert isinstance(response.content, AgentInsight)
        assert response.content.summary == "Test insight summary"
        assert response.status == ResponseStatus.SUCCESS
        assert response.agent_name == "test_agent"
        assert response.agent_category == "testing"

    def test_create_with_string_content(self):
        """Test creating AgentResponse with string content."""
        response = AgentResponse(
            content="Simple text response",
            agent_name="test_agent",
            agent_category="testing",
        )
        assert response.content == "Simple text response"
        assert response.status == ResponseStatus.SUCCESS

    def test_create_with_default_status(self):
        """Test that status defaults to SUCCESS."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        assert response.status == ResponseStatus.SUCCESS

    def test_create_with_custom_status(self):
        """Test creating AgentResponse with custom status."""
        response = AgentResponse(
            content="test",
            status=ResponseStatus.PARTIAL,
            agent_name="agent",
            agent_category="test",
        )
        assert response.status == ResponseStatus.PARTIAL

    def test_create_with_tool_calls(self, sample_tool_call):
        """Test creating AgentResponse with tool calls."""
        response = AgentResponse(
            content="test",
            agent_name="agent",
            agent_category="test",
            tool_calls=[sample_tool_call],
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_organizations"

    def test_create_with_metadata(self):
        """Test creating AgentResponse with metadata."""
        response = AgentResponse(
            content="test",
            agent_name="agent",
            agent_category="test",
            metadata={"processing_time": 1.5, "tokens": 100},
        )
        assert response.metadata["processing_time"] == 1.5
        assert response.metadata["tokens"] == 100

    def test_create_with_timestamp(self):
        """Test that timestamp is automatically set."""
        with patch("src.core.agent_response.datetime") as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            response = AgentResponse(content="test", agent_name="agent", agent_category="test")
            assert response.timestamp == mock_now

    def test_create_missing_required_fields(self):
        """Test AgentResponse validation with missing required fields."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AgentResponse(content="test")


class TestAgentResponseToolCalls:
    """Test AgentResponse tool call methods."""

    def test_add_tool_call(self):
        """Test adding a tool call."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.add_tool_call("get_organizations", {"query": "AI"}, result=[1, 2, 3])
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_organizations"
        assert response.tool_calls[0]["parameters"] == {"query": "AI"}
        assert response.tool_calls[0]["result"] == [1, 2, 3]

    def test_add_tool_call_without_result(self):
        """Test adding a tool call without result."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.add_tool_call("get_organizations", {"query": "AI"})
        assert len(response.tool_calls) == 1
        assert "result" not in response.tool_calls[0]

    def test_add_multiple_tool_calls(self):
        """Test adding multiple tool calls."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.add_tool_call("tool1", {"param1": "value1"})
        response.add_tool_call("tool2", {"param2": "value2"}, result="result2")
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0]["name"] == "tool1"
        assert response.tool_calls[1]["name"] == "tool2"

    def test_add_tool_call_initializes_list(self):
        """Test that add_tool_call initializes tool_calls list if None."""
        response = AgentResponse(
            content="test",
            agent_name="agent",
            agent_category="test",
            tool_calls=None,
        )
        response.add_tool_call("tool1", {})
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1


class TestAgentResponseMetadata:
    """Test AgentResponse metadata methods."""

    def test_add_metadata(self):
        """Test adding metadata."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.add_metadata("key1", "value1")
        assert response.metadata["key1"] == "value1"

    def test_add_metadata_overwrite(self):
        """Test overwriting existing metadata value."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.add_metadata("key1", "value1")
        response.add_metadata("key1", "value2")
        assert response.metadata["key1"] == "value2"

    def test_get_metadata_existing(self):
        """Test retrieving existing metadata value."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.add_metadata("key1", "value1")
        result = response.get_metadata("key1")
        assert result == "value1"

    def test_get_metadata_missing(self):
        """Test retrieving missing metadata value with default."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        result = response.get_metadata("missing_key")
        assert result is None

    def test_get_metadata_with_default(self):
        """Test retrieving missing metadata value with custom default."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        result = response.get_metadata("missing_key", default="default_value")
        assert result == "default_value"


class TestAgentResponseNestedResponses:
    """Test AgentResponse nested responses."""

    def test_add_nested_response(self):
        """Test adding a nested response."""
        parent = AgentResponse(content="parent", agent_name="parent_agent", agent_category="test")
        child = AgentResponse(content="child", agent_name="child_agent", agent_category="test")
        parent.add_nested_response(child)
        assert parent.nested_responses is not None
        assert len(parent.nested_responses) == 1
        assert parent.nested_responses[0] == child

    def test_add_multiple_nested_responses(self):
        """Test adding multiple nested responses."""
        parent = AgentResponse(content="parent", agent_name="parent_agent", agent_category="test")
        child1 = AgentResponse(content="child1", agent_name="child1", agent_category="test")
        child2 = AgentResponse(content="child2", agent_name="child2", agent_category="test")
        parent.add_nested_response(child1)
        parent.add_nested_response(child2)
        assert len(parent.nested_responses) == 2

    def test_add_nested_response_initializes_list(self):
        """Test that add_nested_response initializes list if None."""
        parent = AgentResponse(
            content="parent",
            agent_name="parent_agent",
            agent_category="test",
            nested_responses=None,
        )
        child = AgentResponse(content="child", agent_name="child_agent", agent_category="test")
        parent.add_nested_response(child)
        assert parent.nested_responses is not None
        assert len(parent.nested_responses) == 1


class TestAgentResponseStatus:
    """Test AgentResponse status methods."""

    def test_set_error(self):
        """Test set_error() method."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.set_error("Something went wrong")
        assert response.status == ResponseStatus.ERROR
        assert response.error == "Something went wrong"

    def test_set_error_overwrites_content(self):
        """Test that set_error doesn't change content field.

        Note: The content field is not automatically changed by set_error,
        but error responses typically have empty string content.
        """
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        original_content = response.content
        response.set_error("Error occurred")
        # Content is not automatically changed by set_error
        assert response.content == original_content
        assert response.status == ResponseStatus.ERROR

    def test_is_success(self):
        """Test is_success() method."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        assert response.is_success() is True
        response.status = ResponseStatus.PARTIAL
        assert response.is_success() is False

    def test_is_error(self):
        """Test is_error() method."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        assert response.is_error() is False
        response.set_error("Error message")
        assert response.is_error() is True

    def test_status_enum_values(self):
        """Test all status enum values."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        response.status = ResponseStatus.SUCCESS
        assert response.is_success() is True
        assert response.is_error() is False

        response.status = ResponseStatus.PARTIAL
        assert response.is_success() is False
        assert response.is_error() is False

        response.status = ResponseStatus.ERROR
        assert response.is_success() is False
        assert response.is_error() is True

        response.status = ResponseStatus.PENDING
        assert response.is_success() is False
        assert response.is_error() is False


class TestAgentResponseModelDump:
    """Test AgentResponse model_dump() method."""

    def test_model_dump_basic(self):
        """Test basic model_dump() functionality."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        data = response.model_dump()
        assert data["content"] == "test"
        assert data["agent_name"] == "agent"
        assert isinstance(data["timestamp"], str)  # Should be ISO format string

    def test_model_dump_timestamp_serialization(self):
        """Test that timestamp is serialized as ISO format string."""
        # Create response and verify timestamp is datetime object
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        assert isinstance(response.timestamp, datetime)
        original_timestamp = response.timestamp

        # Call model_dump and verify timestamp is serialized to ISO string
        data = response.model_dump()
        assert isinstance(data["timestamp"], str)
        assert data["timestamp"] == original_timestamp.isoformat()

        # Verify internal timestamp is still a datetime object
        assert isinstance(response.timestamp, datetime)
        assert response.timestamp == original_timestamp

    def test_model_dump_with_agent_insight(self, sample_agent_insight):
        """Test model_dump() with AgentInsight content."""
        response = AgentResponse(
            content=sample_agent_insight,
            agent_name="agent",
            agent_category="test",
        )
        data = response.model_dump()
        assert isinstance(data["content"], dict)  # Pydantic model becomes dict
        assert data["content"]["summary"] == "Test insight summary"

    def test_model_dump_preserves_datetime_object_internally(self):
        """Test that model_dump doesn't modify internal timestamp."""
        response = AgentResponse(content="test", agent_name="agent", agent_category="test")
        original_timestamp = response.timestamp
        data = response.model_dump()
        # Internal timestamp should still be datetime object
        assert isinstance(response.timestamp, datetime)
        assert response.timestamp == original_timestamp
        # But serialized version should be string
        assert isinstance(data["timestamp"], str)


class TestAgentResponseClassMethods:
    """Test AgentResponse class methods."""

    def test_create_success(self):
        """Test create_success() class method."""
        response = AgentResponse.create_success(
            content="Success content",
            agent_name="test_agent",
            agent_category="testing",
        )
        assert response.status == ResponseStatus.SUCCESS
        assert response.content == "Success content"
        assert response.agent_name == "test_agent"
        assert response.agent_category == "testing"

    def test_create_success_with_agent_insight(self, sample_agent_insight):
        """Test create_success() with AgentInsight content."""
        response = AgentResponse.create_success(
            content=sample_agent_insight,
            agent_name="test_agent",
            agent_category="testing",
        )
        assert response.status == ResponseStatus.SUCCESS
        assert isinstance(response.content, AgentInsight)

    def test_create_success_with_kwargs(self):
        """Test create_success() with additional kwargs."""
        response = AgentResponse.create_success(
            content="test",
            agent_name="agent",
            agent_category="test",
            metadata={"key": "value"},
            tool_calls=[{"name": "tool1"}],
        )
        assert response.metadata["key"] == "value"
        assert len(response.tool_calls) == 1

    def test_create_error(self):
        """Test create_error() class method."""
        response = AgentResponse.create_error(
            error_message="Something went wrong",
            agent_name="test_agent",
            agent_category="testing",
        )
        assert response.status == ResponseStatus.ERROR
        assert response.error == "Something went wrong"
        assert response.content == ""  # Error responses have empty content
        assert response.agent_name == "test_agent"
        assert response.agent_category == "testing"

    def test_create_error_with_kwargs(self):
        """Test create_error() with additional kwargs."""
        response = AgentResponse.create_error(
            error_message="Error occurred",
            agent_name="agent",
            agent_category="test",
            metadata={"error_code": "E001"},
        )
        assert response.metadata["error_code"] == "E001"

    def test_create_error_vs_set_error(self):
        """Test that create_error() and set_error() produce same result."""
        response1 = AgentResponse.create_error(
            error_message="Error", agent_name="agent", agent_category="test"
        )
        response2 = AgentResponse(content="", agent_name="agent", agent_category="test")
        response2.set_error("Error")
        assert response1.status == response2.status
        assert response1.error == response2.error
        assert response1.content == response2.content
