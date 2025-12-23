"""Unit tests for the tool I/O contracts module.

This module tests all contract classes and helper functions in tool_io.py,
including validation, defaults, edge cases, and error handling.
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.contracts.tool_io import (
    ToolExecutionContract,
    ToolInput,
    ToolMetadata,
    ToolOutput,
    ToolParameterSchema,
    create_tool_output,
    validate_tool_input,
)


class TestToolParameterSchema:
    """Test ToolParameterSchema class."""

    def test_create_minimal_parameter(self):
        """Test creating a parameter with only required fields."""
        param = ToolParameterSchema(
            name="test_param",
            type="string",
            description="A test parameter",
        )
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.required is False
        assert param.default is None
        assert param.enum is None

    def test_create_parameter_with_all_fields(self):
        """Test creating a parameter with all fields."""
        param = ToolParameterSchema(
            name="status",
            type="string",
            description="Status value",
            required=True,
            default="active",
            enum=["active", "inactive", "pending"],
        )
        assert param.name == "status"
        assert param.type == "string"
        assert param.description == "Status value"
        assert param.required is True
        assert param.default == "active"
        assert param.enum == ["active", "inactive", "pending"]

    def test_parameter_defaults(self):
        """Test that optional fields have correct defaults."""
        param = ToolParameterSchema(name="test", type="integer", description="Test")
        assert param.required is False
        assert param.default is None
        assert param.enum is None

    @pytest.mark.parametrize(
        "field_name",
        ["name", "type", "description"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "name": "test",
            "type": "string",
            "description": "Test description",
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            ToolParameterSchema(**kwargs)


class TestToolMetadata:
    """Test ToolMetadata class."""

    def test_create_minimal_metadata(self):
        """Test creating metadata with only required fields."""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test tool",
        )
        assert metadata.name == "test_tool"
        assert metadata.description == "A test tool"
        assert metadata.version is None
        assert metadata.parameters == []
        assert metadata.returns is None
        assert metadata.cost_per_call is None
        assert metadata.estimated_latency_ms is None
        assert metadata.timeout_seconds is None
        assert metadata.side_effects is True
        assert metadata.idempotent is False
        assert metadata.tags == []

    def test_create_metadata_with_all_fields(self):
        """Test creating metadata with all fields."""
        param1 = ToolParameterSchema(name="input", type="string", description="Input value")
        param2 = ToolParameterSchema(
            name="count",
            type="integer",
            description="Count",
            required=True,
        )
        returns_schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }

        metadata = ToolMetadata(
            name="complex_tool",
            description="A complex tool",
            version="1.0.0",
            parameters=[param1, param2],
            returns=returns_schema,
            cost_per_call=0.001,
            estimated_latency_ms=150.5,
            timeout_seconds=30.0,
            side_effects=False,
            idempotent=True,
            tags=["database", "read-only"],
        )
        assert metadata.name == "complex_tool"
        assert metadata.description == "A complex tool"
        assert metadata.version == "1.0.0"
        assert len(metadata.parameters) == 2
        assert metadata.returns == returns_schema
        assert metadata.cost_per_call == 0.001
        assert metadata.estimated_latency_ms == 150.5
        assert metadata.timeout_seconds == 30.0
        assert metadata.side_effects is False
        assert metadata.idempotent is True
        assert metadata.tags == ["database", "read-only"]

    def test_metadata_defaults(self):
        """Test that optional fields have correct defaults."""
        metadata = ToolMetadata(name="test", description="Test")
        assert metadata.version is None
        assert metadata.parameters == []
        assert metadata.returns is None
        assert metadata.cost_per_call is None
        assert metadata.estimated_latency_ms is None
        assert metadata.timeout_seconds is None
        assert metadata.side_effects is True
        assert metadata.idempotent is False
        assert metadata.tags == []

    @pytest.mark.parametrize(
        "field_name",
        ["name", "description"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {"name": "test", "description": "Test description"}
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            ToolMetadata(**kwargs)


class TestToolInput:
    """Test ToolInput class."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        tool_input = ToolInput(
            tool_name="test_tool",
            parameters={"key": "value"},
        )
        assert tool_input.tool_name == "test_tool"
        assert tool_input.parameters == {"key": "value"}
        assert tool_input.metadata == {}
        assert isinstance(tool_input.timestamp, datetime)

    def test_create_input_with_all_fields(self):
        """Test creating input with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        tool_input = ToolInput(
            tool_name="test_tool",
            parameters={"param1": "value1", "param2": 42},
            metadata={"trace_id": "abc123", "caller": "agent1"},
            timestamp=custom_timestamp,
        )
        assert tool_input.tool_name == "test_tool"
        assert tool_input.parameters == {"param1": "value1", "param2": 42}
        assert tool_input.metadata == {"trace_id": "abc123", "caller": "agent1"}
        assert tool_input.timestamp == custom_timestamp

    def test_input_defaults(self):
        """Test that optional fields have correct defaults."""
        tool_input = ToolInput(tool_name="test", parameters={})
        assert tool_input.metadata == {}
        assert isinstance(tool_input.timestamp, datetime)

    @pytest.mark.parametrize(
        "field_name",
        ["tool_name", "parameters"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {"tool_name": "test", "parameters": {}}
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            ToolInput(**kwargs)

    def test_input_with_complex_parameters(self):
        """Test input with complex parameter types."""
        complex_params = {
            "string_param": "text",
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"},
        }
        tool_input = ToolInput(
            tool_name="complex_tool",
            parameters=complex_params,
        )
        assert tool_input.parameters == complex_params


class TestToolOutput:
    """Test ToolOutput class."""

    def test_create_successful_output(self):
        """Test creating a successful tool output."""
        result_data = {"items": [1, 2, 3], "count": 3}
        tool_output = ToolOutput(
            tool_name="test_tool",
            success=True,
            result=result_data,
            execution_time_ms=125.5,
        )
        assert tool_output.tool_name == "test_tool"
        assert tool_output.success is True
        assert tool_output.result == result_data
        assert tool_output.error is None
        assert tool_output.execution_time_ms == 125.5
        assert tool_output.metadata == {}
        assert isinstance(tool_output.timestamp, datetime)

    def test_create_failed_output(self):
        """Test creating a failed tool output."""
        error_msg = "Connection timeout"
        tool_output = ToolOutput(
            tool_name="test_tool",
            success=False,
            error=error_msg,
            execution_time_ms=5000.0,
        )
        assert tool_output.tool_name == "test_tool"
        assert tool_output.success is False
        assert tool_output.result is None
        assert tool_output.error == error_msg
        assert tool_output.execution_time_ms == 5000.0

    def test_create_output_with_all_fields(self):
        """Test creating output with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        result_data = {"status": "completed"}
        metadata = {"tokens_used": 150, "api_calls": 1}
        tool_output = ToolOutput(
            tool_name="test_tool",
            success=True,
            result=result_data,
            error=None,
            execution_time_ms=200.0,
            metadata=metadata,
            timestamp=custom_timestamp,
        )
        assert tool_output.tool_name == "test_tool"
        assert tool_output.success is True
        assert tool_output.result == result_data
        assert tool_output.error is None
        assert tool_output.execution_time_ms == 200.0
        assert tool_output.metadata == metadata
        assert tool_output.timestamp == custom_timestamp

    def test_output_defaults(self):
        """Test that optional fields have correct defaults."""
        tool_output = ToolOutput(tool_name="test", success=True)
        assert tool_output.result is None
        assert tool_output.error is None
        assert tool_output.execution_time_ms is None
        assert tool_output.metadata == {}
        assert isinstance(tool_output.timestamp, datetime)

    @pytest.mark.parametrize(
        "field_name",
        ["tool_name", "success"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {"tool_name": "test", "success": True}
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            ToolOutput(**kwargs)

    def test_output_with_complex_result(self):
        """Test output with complex result types."""
        complex_result = {
            "list": [1, 2, 3],
            "nested": {"deep": {"value": 42}},
            "mixed": [{"id": 1}, {"id": 2}],
        }
        tool_output = ToolOutput(
            tool_name="complex_tool",
            success=True,
            result=complex_result,
        )
        assert tool_output.result == complex_result


class TestToolExecutionContract:
    """Test ToolExecutionContract class."""

    def test_create_minimal_contract(self):
        """Test creating contract with defaults."""
        contract = ToolExecutionContract()
        assert contract.timeout_seconds is None
        assert contract.max_retries == 0
        assert contract.retry_on_errors == []
        assert contract.cancellation_supported is False
        assert contract.requires_isolation is False

    def test_create_contract_with_all_fields(self):
        """Test creating contract with all fields."""
        contract = ToolExecutionContract(
            timeout_seconds=30.0,
            max_retries=3,
            retry_on_errors=["TimeoutError", "ConnectionError"],
            cancellation_supported=True,
            requires_isolation=True,
        )
        assert contract.timeout_seconds == 30.0
        assert contract.max_retries == 3
        assert contract.retry_on_errors == ["TimeoutError", "ConnectionError"]
        assert contract.cancellation_supported is True
        assert contract.requires_isolation is True

    def test_contract_defaults(self):
        """Test that fields have correct defaults."""
        contract = ToolExecutionContract()
        assert contract.timeout_seconds is None
        assert contract.max_retries == 0
        assert contract.retry_on_errors == []
        assert contract.cancellation_supported is False
        assert contract.requires_isolation is False


class TestValidateToolInput:
    """Test validate_tool_input function."""

    def test_validate_without_metadata(self):
        """Test validation without metadata (no validation)."""
        tool_input = validate_tool_input(
            tool_name="test_tool",
            parameters={"param1": "value1"},
        )
        assert tool_input.tool_name == "test_tool"
        assert tool_input.parameters == {"param1": "value1"}
        assert tool_input.metadata == {}

    def test_validate_with_metadata_all_required_provided(self):
        """Test validation with metadata when all required params provided."""
        param1 = ToolParameterSchema(
            name="required_param",
            type="string",
            description="Required",
            required=True,
        )
        param2 = ToolParameterSchema(
            name="optional_param",
            type="integer",
            description="Optional",
            required=False,
        )
        metadata = ToolMetadata(
            name="test_tool",
            description="Test",
            parameters=[param1, param2],
        )

        tool_input = validate_tool_input(
            tool_name="test_tool",
            parameters={"required_param": "value"},
            metadata=metadata,
        )
        assert tool_input.tool_name == "test_tool"
        assert tool_input.parameters == {"required_param": "value"}

    def test_validate_with_metadata_missing_required(self):
        """Test validation raises error when required params missing."""
        param1 = ToolParameterSchema(
            name="required_param",
            type="string",
            description="Required",
            required=True,
        )
        param2 = ToolParameterSchema(
            name="another_required",
            type="integer",
            description="Also required",
            required=True,
        )
        metadata = ToolMetadata(
            name="test_tool",
            description="Test",
            parameters=[param1, param2],
        )

        with pytest.raises(ValueError) as exc_info:
            validate_tool_input(
                tool_name="test_tool",
                parameters={"required_param": "value"},
                metadata=metadata,
            )
        assert "requires parameters" in str(exc_info.value)
        assert "another_required" in str(exc_info.value)

    def test_validate_with_metadata_extra_params_allowed(self):
        """Test that extra parameters beyond schema are allowed."""
        param1 = ToolParameterSchema(
            name="required_param",
            type="string",
            description="Required",
            required=True,
        )
        metadata = ToolMetadata(
            name="test_tool",
            description="Test",
            parameters=[param1],
        )

        # Extra param should not cause error
        tool_input = validate_tool_input(
            tool_name="test_tool",
            parameters={"required_param": "value", "extra_param": "extra"},
            metadata=metadata,
        )
        assert "extra_param" in tool_input.parameters

    def test_validate_with_metadata_no_required_params(self):
        """Test validation when no params are required."""
        param1 = ToolParameterSchema(
            name="optional1",
            type="string",
            description="Optional",
            required=False,
        )
        metadata = ToolMetadata(
            name="test_tool",
            description="Test",
            parameters=[param1],
        )

        tool_input = validate_tool_input(
            tool_name="test_tool",
            parameters={},
            metadata=metadata,
        )
        assert tool_input.parameters == {}


class TestCreateToolOutput:
    """Test create_tool_output function."""

    def test_create_successful_output(self):
        """Test creating successful output."""
        result_data = {"status": "ok", "data": [1, 2, 3]}
        output = create_tool_output(
            tool_name="test_tool",
            success=True,
            result=result_data,
            execution_time_ms=100.0,
        )
        assert output.tool_name == "test_tool"
        assert output.success is True
        assert output.result == result_data
        assert output.error is None
        assert output.execution_time_ms == 100.0
        assert output.metadata == {}
        assert isinstance(output.timestamp, datetime)

    def test_create_failed_output(self):
        """Test creating failed output."""
        error_msg = "Tool execution failed"
        output = create_tool_output(
            tool_name="test_tool",
            success=False,
            error=error_msg,
            execution_time_ms=5000.0,
        )
        assert output.tool_name == "test_tool"
        assert output.success is False
        assert output.result is None
        assert output.error == error_msg
        assert output.execution_time_ms == 5000.0

    def test_create_output_with_metadata(self):
        """Test creating output with custom metadata."""
        metadata = {"tokens": 150, "cost": 0.001}
        output = create_tool_output(
            tool_name="test_tool",
            success=True,
            result={"result": "data"},
            metadata=metadata,
        )
        assert output.metadata == metadata

    def test_create_output_with_minimal_params(self):
        """Test creating output with only required parameters."""
        output = create_tool_output(
            tool_name="test_tool",
            success=True,
        )
        assert output.tool_name == "test_tool"
        assert output.success is True
        assert output.result is None
        assert output.error is None
        assert output.execution_time_ms is None
        assert output.metadata == {}

    def test_create_output_with_none_metadata(self):
        """Test that None metadata is converted to empty dict."""
        output = create_tool_output(
            tool_name="test_tool",
            success=True,
            metadata=None,
        )
        assert output.metadata == {}

    @patch("src.contracts.tool_io.datetime")
    def test_create_output_timestamp(self, mock_datetime):
        """Test that timestamp is set correctly."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = fixed_time

        output = create_tool_output(
            tool_name="test_tool",
            success=True,
        )
        assert output.timestamp == fixed_time


class TestToolContractsIntegration:
    """Integration tests for tool contracts working together."""

    def test_full_tool_lifecycle(self):
        """Test a complete tool execution lifecycle."""
        # 1. Define tool metadata
        param = ToolParameterSchema(
            name="query",
            type="string",
            description="Search query",
            required=True,
        )
        metadata = ToolMetadata(
            name="search_tool",
            description="Search tool",
            parameters=[param],
            timeout_seconds=10.0,
            side_effects=False,
            idempotent=True,
        )

        # 2. Validate input
        tool_input = validate_tool_input(
            tool_name="search_tool",
            parameters={"query": "test query"},
            metadata=metadata,
        )

        # 3. Create execution contract
        contract = ToolExecutionContract(
            timeout_seconds=10.0,
            max_retries=2,
            retry_on_errors=["TimeoutError"],
            cancellation_supported=True,
        )

        # 4. Simulate execution and create output
        result = {"results": ["result1", "result2"]}
        tool_output = create_tool_output(
            tool_name="search_tool",
            success=True,
            result=result,
            execution_time_ms=150.0,
            metadata={"tokens_used": 50},
        )

        # Verify all components
        assert tool_input.tool_name == "search_tool"
        assert tool_output.tool_name == "search_tool"
        assert tool_output.success is True
        assert tool_output.result == result
        assert contract.timeout_seconds == 10.0
