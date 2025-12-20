"""Unit tests for the Temporal tools activities module.

This module tests all activity functions and helper functions in tools.py,
including tool registry, tool execution, error handling, and activity options.
"""

import inspect
from datetime import timedelta
from inspect import Parameter, Signature
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio.common import RetryPolicy

from src.contracts.tool_io import ToolOutput
from src.temporal.activities.tools import (
    _DEFAULT_ACTIVITY_OPTIONS,
    execute_tool,
    execute_tool_with_options,
    get_activity_options,
    get_tool,
    list_available_tools,
    register_tool,
)


class TestToolRegistry:
    """Test tool registry functions."""

    def test_list_available_tools(self):
        """Test listing available tools."""
        tools = list_available_tools()
        assert isinstance(tools, list)
        assert "generate_llm_response" in tools
        assert "get_acquisitions" in tools
        assert "get_funding_rounds" in tools
        assert "get_organizations" in tools
        assert "semantic_search_organizations" in tools

    def test_get_tool_existing(self):
        """Test getting an existing tool."""
        tool = get_tool("generate_llm_response")
        assert tool is not None
        assert callable(tool)

    def test_get_tool_nonexistent(self):
        """Test getting a non-existent tool."""
        tool = get_tool("nonexistent_tool")
        assert tool is None

    def test_register_tool_success(self):
        """Test registering a new tool."""
        async def test_tool(param1: str) -> str:
            return f"Result: {param1}"

        try:
            register_tool("test_tool", test_tool)
            retrieved = get_tool("test_tool")
            assert retrieved == test_tool
        finally:
            # Clean up - remove the test tool
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "test_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["test_tool"]

    def test_register_tool_overwrite(self):
        """Test that registering a tool overwrites existing registration."""
        async def tool1(param: str) -> str:
            return "tool1"

        async def tool2(param: str) -> str:
            return "tool2"

        try:
            register_tool("test_tool", tool1)
            assert get_tool("test_tool") == tool1
            register_tool("test_tool", tool2)
            assert get_tool("test_tool") == tool2
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "test_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["test_tool"]

    def test_register_tool_not_callable(self):
        """Test that registering a non-callable raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            register_tool("test_tool", "not a function")
        assert "must be callable" in str(exc_info.value).lower()


class TestExecuteTool:
    """Test execute_tool activity function."""

    @pytest.mark.asyncio
    async def test_execute_tool_success_with_tool_output(self):
        """Test successful tool execution with ToolOutput."""
        async def mock_tool(param1: str) -> ToolOutput:
            return ToolOutput(
                success=True,
                result={"data": "result"},
                tool_name="test_tool",
                execution_time_ms=100.5,
                metadata={"key": "value"},
            )

        try:
            register_tool("test_tool", mock_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                result = await execute_tool(
                    "test_tool", {"param1": "value1"}
                )
                assert result["success"] is True
                assert result["result"] == {"data": "result"}
                assert result["tool_name"] == "test_tool"
                assert result["error"] is None
                assert result["metadata"]["execution_time_ms"] == 100.5
                assert result["metadata"]["key"] == "value"
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "test_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["test_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_success_with_legacy_format(self):
        """Test successful tool execution with legacy format (raw result)."""
        async def mock_tool(param1: str) -> Dict[str, Any]:
            return {"data": "legacy_result"}

        try:
            register_tool("test_tool", mock_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                result = await execute_tool(
                    "test_tool", {"param1": "value1"}
                )
                assert result["success"] is True
                assert result["result"] == {"data": "legacy_result"}
                assert result["tool_name"] == "test_tool"
                assert result["error"] is None
                assert result["metadata"]["execution_time_ms"] is None
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "test_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["test_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test tool execution when tool is not found."""
        with patch("src.temporal.activities.tools.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            result = await execute_tool("nonexistent_tool", {})
            assert result["success"] is False
            assert "not found" in result["error"].lower()
            assert result["tool_name"] == "nonexistent_tool"
            assert result["result"] is None
            assert result["metadata"] == {}
            mock_activity.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_not_async(self):
        """Test tool execution when tool is not async."""
        def sync_tool(param: str) -> str:
            return "result"

        try:
            register_tool("sync_tool", sync_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                result = await execute_tool("sync_tool", {"param": "value"})
                assert result["success"] is False
                assert "must be an async function" in result["error"].lower()
                assert result["tool_name"] == "sync_tool"
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "sync_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["sync_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_signature_inspection_error(self):
        """Test tool execution when signature inspection fails."""
        # Create a tool that will cause inspection to fail
        class BadTool:
            def __call__(self, param: str) -> str:
                return "result"

        bad_tool = BadTool()
        # Make it appear callable but break inspect.signature
        try:
            register_tool("bad_tool", bad_tool)
            with patch("src.temporal.activities.tools.inspect") as mock_inspect:
                mock_inspect.signature = MagicMock(
                    side_effect=ValueError("Inspection failed")
                )
                mock_inspect.iscoroutinefunction = MagicMock(return_value=False)
                with patch("src.temporal.activities.tools.activity") as mock_activity:
                    mock_activity.logger = MagicMock()
                    result = await execute_tool("bad_tool", {"param": "value"})
                    assert result["success"] is False
                    assert "inspecting tool" in result["error"].lower()
                    assert result["tool_name"] == "bad_tool"
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "bad_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["bad_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_execution_exception(self):
        """Test tool execution when tool raises an exception."""
        async def failing_tool(param: str) -> str:
            raise ValueError("Tool execution failed")

        try:
            register_tool("failing_tool", failing_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                result = await execute_tool("failing_tool", {"param": "value"})
                assert result["success"] is False
                assert "execution failed" in result["error"].lower()
                assert result["tool_name"] == "failing_tool"
                assert result["result"] is None
                assert result["metadata"]["exception_type"] == "ValueError"
                mock_activity.logger.error.assert_called_once()
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "failing_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["failing_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_tool_output_failure(self):
        """Test tool execution with ToolOutput indicating failure."""
        async def failing_tool(param: str) -> ToolOutput:
            return ToolOutput(
                success=False,
                result=None,
                error="Tool-specific error",
                tool_name="failing_tool",
                execution_time_ms=50.0,
            )

        try:
            register_tool("failing_tool", failing_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                result = await execute_tool("failing_tool", {"param": "value"})
                assert result["success"] is False
                assert result["error"] == "Tool-specific error"
                assert result["tool_name"] == "failing_tool"
                assert result["result"] is None
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "failing_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["failing_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_parameters(self):
        """Test tool execution with multiple parameters."""
        mock_tool = AsyncMock()
        mock_tool.return_value = ToolOutput(
            success=True,
            result={"param1": "value1", "param2": 42, "param3": True},
            tool_name="multi_param_tool",
            execution_time_ms=50.0,
        )

        try:
            register_tool("multi_param_tool", mock_tool)
            # Create a real Signature object to avoid recursion issues with MagicMock
            # when inspect functions are patched
            real_sig = Signature([
                Parameter("param1", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("param2", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("param3", Parameter.POSITIONAL_OR_KEYWORD, default=False),
            ])
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                with patch("src.temporal.activities.tools.inspect.signature") as mock_sig, \
                     patch("src.temporal.activities.tools.inspect.iscoroutinefunction") as mock_is_coro:
                    mock_sig.return_value = real_sig
                    mock_is_coro.return_value = True
                    result = await execute_tool(
                        "multi_param_tool",
                        {"param1": "value1", "param2": 42, "param3": True},
                    )
                    assert result["success"] is True, f"Tool execution failed: {result.get('error')}"
                    assert result["result"]["param1"] == "value1"
                    assert result["result"]["param2"] == 42
                    assert result["result"]["param3"] is True
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "multi_param_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["multi_param_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_logging(self):
        """Test that tool execution logs appropriately."""
        async def mock_tool(param: str) -> ToolOutput:
            return ToolOutput(
                success=True,
                result={"data": "result"},
                tool_name="test_tool",
            )

        try:
            register_tool("test_tool", mock_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                await execute_tool("test_tool", {"param": "value"})
                # Check that info was logged
                assert mock_activity.logger.info.call_count >= 1
                # Check initial log
                initial_call = mock_activity.logger.info.call_args_list[0]
                assert "Executing tool" in str(initial_call)
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "test_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["test_tool"]


class TestExecuteToolWithOptions:
    """Test execute_tool_with_options activity function."""

    @pytest.mark.asyncio
    async def test_execute_tool_with_options_success(self):
        """Test execute_tool_with_options with custom options."""
        async def mock_tool(param: str) -> ToolOutput:
            return ToolOutput(
                success=True,
                result={"data": "result"},
                tool_name="test_tool",
            )

        try:
            register_tool("test_tool", mock_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                with patch(
                    "src.temporal.activities.tools.execute_tool"
                ) as mock_execute:
                    mock_execute.return_value = {
                        "success": True,
                        "result": {"data": "result"},
                    }
                    result = await execute_tool_with_options(
                        "test_tool",
                        {"param": "value"},
                        activity_options={"timeout": 600},
                    )
                    assert result["success"] is True
                    mock_activity.logger.info.assert_called()
                    # Check that custom options were logged
                    assert any(
                        "activity options" in str(call).lower()
                        for call in mock_activity.logger.info.call_args_list
                    )
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "test_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["test_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_options_no_options(self):
        """Test execute_tool_with_options without custom options."""
        async def mock_tool(param: str) -> ToolOutput:
            return ToolOutput(
                success=True,
                result={"data": "result"},
                tool_name="test_tool",
            )

        try:
            register_tool("test_tool", mock_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                with patch(
                    "src.temporal.activities.tools.execute_tool"
                ) as mock_execute:
                    mock_execute.return_value = {"success": True}
                    result = await execute_tool_with_options(
                        "test_tool", {"param": "value"}
                    )
                    assert result["success"] is True
                    # Should not log activity options
                    assert not any(
                        "activity options" in str(call).lower()
                        for call in mock_activity.logger.info.call_args_list
                    )
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "test_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["test_tool"]


class TestGetActivityOptions:
    """Test get_activity_options helper function."""

    def test_get_activity_options_defaults(self):
        """Test get_activity_options with default values."""
        options = get_activity_options()
        assert "start_to_close_timeout" in options
        assert "retry_policy" in options
        assert options["start_to_close_timeout"] == timedelta(seconds=300)
        assert isinstance(options["retry_policy"], RetryPolicy)

    def test_get_activity_options_with_timeout(self):
        """Test get_activity_options with custom timeout."""
        options = get_activity_options(timeout_seconds=600)
        assert options["start_to_close_timeout"] == timedelta(seconds=600)
        # Retry policy should remain unchanged
        assert isinstance(options["retry_policy"], RetryPolicy)

    def test_get_activity_options_with_max_retries(self):
        """Test get_activity_options with custom max retries."""
        options = get_activity_options(max_retries=5)
        retry_policy = options["retry_policy"]
        assert isinstance(retry_policy, RetryPolicy)
        assert retry_policy.maximum_attempts == 5

    def test_get_activity_options_with_initial_retry_interval(self):
        """Test get_activity_options with custom initial retry interval."""
        options = get_activity_options(initial_retry_interval=5.0)
        retry_policy = options["retry_policy"]
        assert isinstance(retry_policy, RetryPolicy)
        assert retry_policy.initial_interval == timedelta(seconds=5.0)

    def test_get_activity_options_with_all_params(self):
        """Test get_activity_options with all custom parameters."""
        options = get_activity_options(
            timeout_seconds=600,
            max_retries=5,
            initial_retry_interval=3.0,
        )
        assert options["start_to_close_timeout"] == timedelta(seconds=600)
        retry_policy = options["retry_policy"]
        assert retry_policy.maximum_attempts == 5
        assert retry_policy.initial_interval == timedelta(seconds=3.0)

    def test_get_activity_options_retry_policy_preserves_other_fields(self):
        """Test that retry policy preserves other fields when updating."""
        options = get_activity_options(max_retries=10)
        retry_policy = options["retry_policy"]
        # Should preserve backoff_coefficient and maximum_interval
        assert retry_policy.backoff_coefficient == 2.0
        assert retry_policy.maximum_interval == timedelta(seconds=60)

    def test_get_activity_options_creates_new_dict(self):
        """Test that get_activity_options returns a new dict (not reference)."""
        options1 = get_activity_options()
        options2 = get_activity_options()
        # Should be different objects
        assert options1 is not options2
        # But should have same structure
        assert options1.keys() == options2.keys()


class TestDefaultActivityOptions:
    """Test default activity options constant."""

    def test_default_activity_options_structure(self):
        """Test that default activity options have correct structure."""
        assert isinstance(_DEFAULT_ACTIVITY_OPTIONS, dict)
        assert "start_to_close_timeout" in _DEFAULT_ACTIVITY_OPTIONS
        assert "retry_policy" in _DEFAULT_ACTIVITY_OPTIONS

    def test_default_activity_options_timeout(self):
        """Test default timeout value."""
        assert _DEFAULT_ACTIVITY_OPTIONS["start_to_close_timeout"] == timedelta(
            seconds=300
        )

    def test_default_activity_options_retry_policy(self):
        """Test default retry policy."""
        retry_policy = _DEFAULT_ACTIVITY_OPTIONS["retry_policy"]
        assert isinstance(retry_policy, RetryPolicy)
        assert retry_policy.initial_interval == timedelta(seconds=1)
        assert retry_policy.backoff_coefficient == 2.0
        assert retry_policy.maximum_interval == timedelta(seconds=60)
        assert retry_policy.maximum_attempts == 3


class TestToolExecutionEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_execute_tool_with_empty_parameters(self):
        """Test tool execution with empty parameters dict."""
        mock_tool = AsyncMock()
        mock_tool.return_value = ToolOutput(
            success=True,
            result={"data": "result"},
            tool_name="no_param_tool",
            execution_time_ms=30.0,
        )

        try:
            register_tool("no_param_tool", mock_tool)
            # Create a real Signature object to avoid recursion issues with MagicMock
            # when inspect functions are patched
            real_sig = Signature([])  # Empty signature for no-parameter function
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                with patch("src.temporal.activities.tools.inspect.signature") as mock_sig, \
                     patch("src.temporal.activities.tools.inspect.iscoroutinefunction") as mock_is_coro:
                    mock_sig.return_value = real_sig
                    mock_is_coro.return_value = True
                    result = await execute_tool("no_param_tool", {})
                    assert result["success"] is True, f"Tool execution failed: {result.get('error')}"
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "no_param_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["no_param_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_none_result(self):
        """Test tool execution that returns None."""
        async def none_tool(param: str) -> None:
            return None

        try:
            register_tool("none_tool", none_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                result = await execute_tool("none_tool", {"param": "value"})
                assert result["success"] is True
                assert result["result"] is None
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "none_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["none_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_complex_result(self):
        """Test tool execution with complex result structure."""
        async def complex_tool(param: str) -> Dict[str, Any]:
            return {
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "string": "test",
            }

        try:
            register_tool("complex_tool", complex_tool)
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                result = await execute_tool("complex_tool", {"param": "value"})
                assert result["success"] is True
                assert result["result"]["nested"]["key"] == "value"
                assert result["result"]["list"] == [1, 2, 3]
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "complex_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["complex_tool"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_tool_output_metadata(self):
        """Test tool execution preserves ToolOutput metadata."""
        mock_tool = AsyncMock()
        mock_tool.return_value = ToolOutput(
            success=True,
            result={"data": "result"},
            tool_name="metadata_tool",
            execution_time_ms=75.5,
            metadata={"custom_key": "custom_value", "count": 42},
        )

        try:
            register_tool("metadata_tool", mock_tool)
            # Create a real Signature object to avoid recursion issues with MagicMock
            # when inspect functions are patched
            real_sig = Signature([
                Parameter("param", Parameter.POSITIONAL_OR_KEYWORD),
            ])
            with patch("src.temporal.activities.tools.activity") as mock_activity:
                mock_activity.logger = MagicMock()
                with patch("src.temporal.activities.tools.inspect.signature") as mock_sig, \
                     patch("src.temporal.activities.tools.inspect.iscoroutinefunction") as mock_is_coro:
                    mock_sig.return_value = real_sig
                    mock_is_coro.return_value = True
                    result = await execute_tool("metadata_tool", {"param": "value"})
                    assert result["success"] is True, f"Tool execution failed: {result.get('error')}"
                    assert result["metadata"]["custom_key"] == "custom_value"
                    assert result["metadata"]["count"] == 42
                    assert "execution_time_ms" in result["metadata"]
        finally:
            from src.temporal.activities.tools import _TOOL_REGISTRY
            if "metadata_tool" in _TOOL_REGISTRY:
                del _TOOL_REGISTRY["metadata_tool"]

    def test_get_activity_options_with_none_values(self):
        """Test get_activity_options handles None values correctly."""
        # Should use defaults when None is passed
        options = get_activity_options(
            timeout_seconds=None,
            max_retries=None,
            initial_retry_interval=None,
        )
        # Should still have valid options
        assert "start_to_close_timeout" in options
        assert "retry_policy" in options

    def test_get_activity_options_partial_retry_params(self):
        """Test get_activity_options with partial retry parameters."""
        # Only max_retries
        options = get_activity_options(max_retries=7)
        assert options["retry_policy"].maximum_attempts == 7
        # Should preserve initial_interval from default
        assert options["retry_policy"].initial_interval == timedelta(seconds=1)

        # Only initial_retry_interval
        options = get_activity_options(initial_retry_interval=10.0)
        assert options["retry_policy"].initial_interval == timedelta(seconds=10.0)
        # Should preserve max_attempts from default
        assert options["retry_policy"].maximum_attempts == 3

