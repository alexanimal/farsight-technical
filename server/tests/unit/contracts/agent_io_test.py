"""Unit tests for the agent I/O contracts module.

This module tests all contract classes and helper functions in agent_io.py,
including validation, defaults, edge cases, conversion methods, and error handling.
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.contracts.agent_io import (
    AgentExecutionContract,
    AgentInput,
    AgentMetadata,
    AgentOutput,
    create_agent_output,
    validate_agent_input,
)
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus


class TestAgentInput:
    """Test AgentInput class."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        agent_input = AgentInput(query="What is the weather?")
        assert agent_input.query == "What is the weather?"
        assert agent_input.conversation_id is None
        assert agent_input.user_id is None
        assert agent_input.metadata == {}
        assert agent_input.shared_data == {}
        assert agent_input.conversation_history is None
        assert isinstance(agent_input.timestamp, datetime)

    def test_create_input_with_all_fields(self):
        """Test creating input with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        agent_input = AgentInput(
            query="What is the weather?",
            conversation_id="conv-123",
            user_id="user-456",
            metadata={"source": "web", "priority": "high"},
            shared_data={"previous_result": "data"},
            conversation_history=history,
            timestamp=custom_timestamp,
        )
        assert agent_input.query == "What is the weather?"
        assert agent_input.conversation_id == "conv-123"
        assert agent_input.user_id == "user-456"
        assert agent_input.metadata == {"source": "web", "priority": "high"}
        assert agent_input.shared_data == {"previous_result": "data"}
        assert agent_input.conversation_history == history
        assert agent_input.timestamp == custom_timestamp

    def test_input_defaults(self):
        """Test that optional fields have correct defaults."""
        agent_input = AgentInput(query="test")
        assert agent_input.conversation_id is None
        assert agent_input.user_id is None
        assert agent_input.metadata == {}
        assert agent_input.shared_data == {}
        assert agent_input.conversation_history is None
        assert isinstance(agent_input.timestamp, datetime)

    @pytest.mark.parametrize(
        "field_name",
        ["query"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {"query": "test"}
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            AgentInput(**kwargs)

    def test_to_agent_context(self):
        """Test conversion to AgentContext."""
        agent_input = AgentInput(
            query="test query",
            conversation_id="conv-1",
            user_id="user-1",
            metadata={"key": "value"},
            shared_data={"shared": "data"},
            conversation_history=[{"role": "user", "content": "hi"}],
        )
        context = agent_input.to_agent_context()
        assert isinstance(context, AgentContext)
        assert context.query == "test query"
        assert context.conversation_id == "conv-1"
        assert context.user_id == "user-1"
        assert context.metadata == {"key": "value"}
        assert context.shared_data == {"shared": "data"}
        assert context.conversation_history == [{"role": "user", "content": "hi"}]

    def test_from_agent_context(self):
        """Test creation from AgentContext."""
        context = AgentContext(
            query="test query",
            conversation_id="conv-1",
            user_id="user-1",
            metadata={"key": "value"},
            shared_data={"shared": "data"},
            conversation_history=[{"role": "user", "content": "hi"}],
        )
        agent_input = AgentInput.from_agent_context(context)
        assert isinstance(agent_input, AgentInput)
        assert agent_input.query == "test query"
        assert agent_input.conversation_id == "conv-1"
        assert agent_input.user_id == "user-1"
        assert agent_input.metadata == {"key": "value"}
        assert agent_input.shared_data == {"shared": "data"}
        assert agent_input.conversation_history == [{"role": "user", "content": "hi"}]

    def test_round_trip_conversion(self):
        """Test that conversion to/from AgentContext is reversible."""
        original = AgentInput(
            query="test",
            conversation_id="conv-1",
            metadata={"key": "value"},
        )
        context = original.to_agent_context()
        restored = AgentInput.from_agent_context(context)
        assert restored.query == original.query
        assert restored.conversation_id == original.conversation_id
        assert restored.metadata == original.metadata


class TestAgentOutput:
    """Test AgentOutput class."""

    def test_create_minimal_output(self):
        """Test creating output with only required fields."""
        agent_output = AgentOutput(
            content="Response content",
            agent_name="test_agent",
            agent_category="test",
        )
        assert agent_output.content == "Response content"
        assert agent_output.agent_name == "test_agent"
        assert agent_output.agent_category == "test"
        assert agent_output.status == ResponseStatus.SUCCESS
        assert agent_output.tool_calls is None
        assert agent_output.metadata == {}
        assert agent_output.nested_responses is None
        assert agent_output.error is None
        assert isinstance(agent_output.timestamp, datetime)

    def test_create_output_with_all_fields(self):
        """Test creating output with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        tool_calls = [
            {"name": "tool1", "parameters": {"param": "value"}},
            {"name": "tool2", "parameters": {}},
        ]
        nested = AgentOutput(
            content="Nested response",
            agent_name="nested_agent",
            agent_category="nested",
        )
        agent_output = AgentOutput(
            content="Main response",
            status=ResponseStatus.SUCCESS,
            agent_name="main_agent",
            agent_category="main",
            tool_calls=tool_calls,
            metadata={"key": "value"},
            nested_responses=[nested],
            error=None,
            timestamp=custom_timestamp,
        )
        assert agent_output.content == "Main response"
        assert agent_output.status == ResponseStatus.SUCCESS
        assert agent_output.agent_name == "main_agent"
        assert agent_output.agent_category == "main"
        assert agent_output.tool_calls == tool_calls
        assert agent_output.metadata == {"key": "value"}
        assert len(agent_output.nested_responses) == 1
        assert agent_output.nested_responses[0].content == "Nested response"
        assert agent_output.error is None
        assert agent_output.timestamp == custom_timestamp

    def test_create_error_output(self):
        """Test creating an error output."""
        agent_output = AgentOutput(
            content="",
            status=ResponseStatus.ERROR,
            agent_name="test_agent",
            agent_category="test",
            error="Something went wrong",
        )
        assert agent_output.status == ResponseStatus.ERROR
        assert agent_output.error == "Something went wrong"

    def test_output_defaults(self):
        """Test that optional fields have correct defaults."""
        agent_output = AgentOutput(
            content="test", agent_name="agent", agent_category="cat"
        )
        assert agent_output.status == ResponseStatus.SUCCESS
        assert agent_output.tool_calls is None
        assert agent_output.metadata == {}
        assert agent_output.nested_responses is None
        assert agent_output.error is None
        assert isinstance(agent_output.timestamp, datetime)

    @pytest.mark.parametrize(
        "field_name",
        ["content", "agent_name", "agent_category"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "content": "test",
            "agent_name": "agent",
            "agent_category": "cat",
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            AgentOutput(**kwargs)

    def test_to_agent_response(self):
        """Test conversion to AgentResponse."""
        nested = AgentOutput(
            content="Nested",
            agent_name="nested",
            agent_category="nested",
        )
        agent_output = AgentOutput(
            content="Main",
            agent_name="main",
            agent_category="main",
            tool_calls=[{"name": "tool", "parameters": {}}],
            metadata={"key": "value"},
            nested_responses=[nested],
            error=None,
        )
        response = agent_output.to_agent_response()
        assert isinstance(response, AgentResponse)
        assert response.content == "Main"
        assert response.agent_name == "main"
        assert response.agent_category == "main"
        assert len(response.tool_calls) == 1
        assert response.metadata == {"key": "value"}
        assert len(response.nested_responses) == 1
        assert response.nested_responses[0].content == "Nested"

    def test_to_agent_response_with_none_nested(self):
        """Test conversion when nested_responses is None."""
        agent_output = AgentOutput(
            content="test",
            agent_name="agent",
            agent_category="cat",
            nested_responses=None,
        )
        response = agent_output.to_agent_response()
        assert response.nested_responses is None

    def test_from_agent_response(self):
        """Test creation from AgentResponse."""
        nested_response = AgentResponse.create_success(
            content="Nested",
            agent_name="nested",
            agent_category="nested",
        )
        response = AgentResponse.create_success(
            content="Main",
            agent_name="main",
            agent_category="main",
            tool_calls=[{"name": "tool", "parameters": {}}],
            metadata={"key": "value"},
            nested_responses=[nested_response],
        )
        agent_output = AgentOutput.from_agent_response(response)
        assert isinstance(agent_output, AgentOutput)
        assert agent_output.content == "Main"
        assert agent_output.agent_name == "main"
        assert agent_output.agent_category == "main"
        assert len(agent_output.tool_calls) == 1
        assert agent_output.metadata == {"key": "value"}
        assert len(agent_output.nested_responses) == 1
        assert agent_output.nested_responses[0].content == "Nested"

    def test_from_agent_response_with_none_nested(self):
        """Test creation when nested_responses is None."""
        response = AgentResponse.create_success(
            content="test",
            agent_name="agent",
            agent_category="cat",
        )
        agent_output = AgentOutput.from_agent_response(response)
        assert agent_output.nested_responses is None

    def test_round_trip_conversion(self):
        """Test that conversion to/from AgentResponse is reversible."""
        original = AgentOutput(
            content="test",
            agent_name="agent",
            agent_category="cat",
            metadata={"key": "value"},
        )
        response = original.to_agent_response()
        restored = AgentOutput.from_agent_response(response)
        assert restored.content == original.content
        assert restored.agent_name == original.agent_name
        assert restored.agent_category == original.agent_category
        assert restored.metadata == original.metadata

    def test_deeply_nested_responses(self):
        """Test handling of deeply nested responses."""
        level3 = AgentOutput(
            content="Level 3",
            agent_name="level3",
            agent_category="test",
        )
        level2 = AgentOutput(
            content="Level 2",
            agent_name="level2",
            agent_category="test",
            nested_responses=[level3],
        )
        level1 = AgentOutput(
            content="Level 1",
            agent_name="level1",
            agent_category="test",
            nested_responses=[level2],
        )

        response = level1.to_agent_response()
        assert response.content == "Level 1"
        assert len(response.nested_responses) == 1
        assert response.nested_responses[0].content == "Level 2"
        assert len(response.nested_responses[0].nested_responses) == 1
        assert response.nested_responses[0].nested_responses[0].content == "Level 3"


class TestAgentExecutionContract:
    """Test AgentExecutionContract class."""

    def test_create_minimal_contract(self):
        """Test creating contract with defaults."""
        contract = AgentExecutionContract()
        assert contract.timeout_seconds is None
        assert contract.max_tool_calls is None
        assert contract.max_tokens is None
        assert contract.allowed_tools is None
        assert contract.forbidden_tools is None
        assert contract.requires_confirmation is False
        assert contract.cancellation_supported is True
        assert contract.output_guarantees == {}

    def test_create_contract_with_all_fields(self):
        """Test creating contract with all fields."""
        contract = AgentExecutionContract(
            timeout_seconds=60.0,
            max_tool_calls=10,
            max_tokens=10000,
            allowed_tools=["tool1", "tool2"],
            forbidden_tools=["tool3"],
            requires_confirmation=True,
            cancellation_supported=False,
            output_guarantees={"format": "json", "required_fields": ["result"]},
        )
        assert contract.timeout_seconds == 60.0
        assert contract.max_tool_calls == 10
        assert contract.max_tokens == 10000
        assert contract.allowed_tools == ["tool1", "tool2"]
        assert contract.forbidden_tools == ["tool3"]
        assert contract.requires_confirmation is True
        assert contract.cancellation_supported is False
        assert contract.output_guarantees == {
            "format": "json",
            "required_fields": ["result"],
        }

    def test_contract_defaults(self):
        """Test that fields have correct defaults."""
        contract = AgentExecutionContract()
        assert contract.timeout_seconds is None
        assert contract.max_tool_calls is None
        assert contract.max_tokens is None
        assert contract.allowed_tools is None
        assert contract.forbidden_tools is None
        assert contract.requires_confirmation is False
        assert contract.cancellation_supported is True
        assert contract.output_guarantees == {}


class TestAgentMetadata:
    """Test AgentMetadata class."""

    def test_create_minimal_metadata(self):
        """Test creating metadata with only required fields."""
        metadata = AgentMetadata(
            name="test_agent",
            description="A test agent",
            category="test",
        )
        assert metadata.name == "test_agent"
        assert metadata.description == "A test agent"
        assert metadata.category == "test"
        assert metadata.version is None
        assert metadata.domain is None
        assert metadata.capabilities == []
        assert metadata.allowed_tools is None
        assert metadata.forbidden_tools is None
        assert metadata.reasoning_style is None
        assert metadata.resource_constraints == {}
        assert metadata.output_guarantees == {}
        assert metadata.tags == []

    def test_create_metadata_with_all_fields(self):
        """Test creating metadata with all fields."""
        metadata = AgentMetadata(
            name="complex_agent",
            description="A complex agent",
            category="complex",
            version="2.0.0",
            domain="acquisitions",
            capabilities=["search", "analyze", "summarize"],
            allowed_tools=["tool1", "tool2"],
            forbidden_tools=["tool3"],
            reasoning_style="analytical",
            resource_constraints={"max_tokens": 5000, "timeout": 30},
            output_guarantees={"format": "json"},
            tags=["acquisition", "m&a"],
        )
        assert metadata.name == "complex_agent"
        assert metadata.description == "A complex agent"
        assert metadata.category == "complex"
        assert metadata.version == "2.0.0"
        assert metadata.domain == "acquisitions"
        assert metadata.capabilities == ["search", "analyze", "summarize"]
        assert metadata.allowed_tools == ["tool1", "tool2"]
        assert metadata.forbidden_tools == ["tool3"]
        assert metadata.reasoning_style == "analytical"
        assert metadata.resource_constraints == {"max_tokens": 5000, "timeout": 30}
        assert metadata.output_guarantees == {"format": "json"}
        assert metadata.tags == ["acquisition", "m&a"]

    def test_metadata_defaults(self):
        """Test that optional fields have correct defaults."""
        metadata = AgentMetadata(
            name="test", description="Test", category="test"
        )
        assert metadata.version is None
        assert metadata.domain is None
        assert metadata.capabilities == []
        assert metadata.allowed_tools is None
        assert metadata.forbidden_tools is None
        assert metadata.reasoning_style is None
        assert metadata.resource_constraints == {}
        assert metadata.output_guarantees == {}
        assert metadata.tags == []

    @pytest.mark.parametrize(
        "field_name",
        ["name", "description", "category"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {"name": "test", "description": "Test", "category": "test"}
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            AgentMetadata(**kwargs)


class TestValidateAgentInput:
    """Test validate_agent_input function."""

    def test_validate_valid_context(self):
        """Test validation with valid context."""
        context = AgentContext(query="What is the weather?")
        agent_input = validate_agent_input(context)
        assert isinstance(agent_input, AgentInput)
        assert agent_input.query == "What is the weather?"

    def test_validate_context_with_all_fields(self):
        """Test validation with context containing all fields."""
        context = AgentContext(
            query="test query",
            conversation_id="conv-1",
            user_id="user-1",
            metadata={"key": "value"},
            shared_data={"shared": "data"},
            conversation_history=[{"role": "user", "content": "hi"}],
        )
        agent_input = validate_agent_input(context)
        assert agent_input.query == "test query"
        assert agent_input.conversation_id == "conv-1"
        assert agent_input.user_id == "user-1"
        assert agent_input.metadata == {"key": "value"}
        assert agent_input.shared_data == {"shared": "data"}

    def test_validate_empty_query(self):
        """Test validation raises error for empty query."""
        context = AgentContext(query="")
        with pytest.raises(ValueError) as exc_info:
            validate_agent_input(context)
        assert "non-empty query" in str(exc_info.value).lower()

    def test_validate_whitespace_only_query(self):
        """Test validation raises error for whitespace-only query."""
        context = AgentContext(query="   ")
        with pytest.raises(ValueError) as exc_info:
            validate_agent_input(context)
        assert "non-empty query" in str(exc_info.value).lower()

    def test_validate_query_with_whitespace(self):
        """Test that query with leading/trailing whitespace is valid."""
        context = AgentContext(query="  test query  ")
        agent_input = validate_agent_input(context)
        assert agent_input.query == "  test query  "


class TestCreateAgentOutput:
    """Test create_agent_output function."""

    def test_create_successful_output(self):
        """Test creating successful output."""
        output = create_agent_output(
            content="Response content",
            agent_name="test_agent",
            agent_category="test",
        )
        assert output.content == "Response content"
        assert output.agent_name == "test_agent"
        assert output.agent_category == "test"
        assert output.status == ResponseStatus.SUCCESS
        assert output.tool_calls is None
        assert output.metadata == {}
        assert isinstance(output.timestamp, datetime)

    def test_create_output_with_custom_status(self):
        """Test creating output with custom status."""
        output = create_agent_output(
            content="Partial response",
            agent_name="test_agent",
            agent_category="test",
            status=ResponseStatus.PARTIAL,
        )
        assert output.status == ResponseStatus.PARTIAL

    def test_create_output_with_additional_fields(self):
        """Test creating output with additional kwargs."""
        tool_calls = [{"name": "tool", "parameters": {}}]
        output = create_agent_output(
            content="Response",
            agent_name="agent",
            agent_category="cat",
            tool_calls=tool_calls,
            metadata={"key": "value"},
            error=None,
        )
        assert output.tool_calls == tool_calls
        assert output.metadata == {"key": "value"}
        assert output.error is None

    def test_create_error_output(self):
        """Test creating error output."""
        output = create_agent_output(
            content="",
            agent_name="agent",
            agent_category="cat",
            status=ResponseStatus.ERROR,
            error="Something went wrong",
        )
        assert output.status == ResponseStatus.ERROR
        assert output.error == "Something went wrong"

    @patch("src.contracts.agent_io.datetime")
    def test_create_output_timestamp(self, mock_datetime):
        """Test that timestamp is set correctly."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = fixed_time

        output = create_agent_output(
            content="test",
            agent_name="agent",
            agent_category="cat",
        )
        assert output.timestamp == fixed_time


class TestAgentContractsIntegration:
    """Integration tests for agent contracts working together."""

    def test_full_agent_lifecycle(self):
        """Test a complete agent execution lifecycle."""
        # 1. Create agent metadata
        metadata = AgentMetadata(
            name="test_agent",
            description="Test agent",
            category="test",
            allowed_tools=["tool1", "tool2"],
            capabilities=["search", "analyze"],
        )

        # 2. Create execution contract
        contract = AgentExecutionContract(
            timeout_seconds=60.0,
            max_tool_calls=5,
            allowed_tools=["tool1", "tool2"],
        )

        # 3. Validate input
        context = AgentContext(query="What is the weather?")
        agent_input = validate_agent_input(context)

        # 4. Simulate execution and create output
        tool_calls = [{"name": "tool1", "parameters": {"query": "weather"}}]
        agent_output = create_agent_output(
            content="The weather is sunny",
            agent_name="test_agent",
            agent_category="test",
            tool_calls=tool_calls,
            metadata={"tokens_used": 100},
        )

        # 5. Convert to AgentResponse
        response = agent_output.to_agent_response()

        # Verify all components
        assert agent_input.query == "What is the weather?"
        assert agent_output.agent_name == "test_agent"
        assert agent_output.content == "The weather is sunny"
        assert len(agent_output.tool_calls) == 1
        assert response.agent_name == "test_agent"
        assert metadata.name == "test_agent"
        assert contract.timeout_seconds == 60.0

    def test_agent_coordination_with_nested_responses(self):
        """Test agent coordination with nested responses."""
        # Create orchestration agent output
        nested1 = create_agent_output(
            content="Result from agent 1",
            agent_name="agent1",
            agent_category="specialized",
        )
        nested2 = create_agent_output(
            content="Result from agent 2",
            agent_name="agent2",
            agent_category="specialized",
        )

        # Create orchestration output with nested responses
        orchestration_output = create_agent_output(
            content="Combined results",
            agent_name="orchestrator",
            agent_category="orchestration",
            nested_responses=[nested1, nested2],
        )

        # Convert to AgentResponse
        response = orchestration_output.to_agent_response()

        assert response.agent_name == "orchestrator"
        assert len(response.nested_responses) == 2
        assert response.nested_responses[0].content == "Result from agent 1"
        assert response.nested_responses[1].content == "Result from agent 2"

    def test_round_trip_agent_context_conversion(self):
        """Test round-trip conversion between AgentInput and AgentContext."""
        original_context = AgentContext(
            query="test",
            conversation_id="conv-1",
            metadata={"key": "value"},
        )

        # Convert to AgentInput
        agent_input = validate_agent_input(original_context)

        # Convert back to AgentContext
        restored_context = agent_input.to_agent_context()

        assert restored_context.query == original_context.query
        assert restored_context.conversation_id == original_context.conversation_id
        assert restored_context.metadata == original_context.metadata

    def test_round_trip_agent_response_conversion(self):
        """Test round-trip conversion between AgentOutput and AgentResponse."""
        original_response = AgentResponse.create_success(
            content="test",
            agent_name="agent",
            agent_category="cat",
            metadata={"key": "value"},
        )

        # Convert to AgentOutput
        agent_output = AgentOutput.from_agent_response(original_response)

        # Convert back to AgentResponse
        restored_response = agent_output.to_agent_response()

        assert restored_response.content == original_response.content
        assert restored_response.agent_name == original_response.agent_name
        assert restored_response.agent_category == original_response.agent_category
        assert restored_response.metadata == original_response.metadata

