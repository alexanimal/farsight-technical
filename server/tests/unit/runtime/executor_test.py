"""Unit tests for the executor module.

This module tests the Executor class and its various methods,
including agent execution, tool execution, timeout handling, and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.contracts.agent_io import AgentInput, AgentOutput, create_agent_output
from src.contracts.tool_io import ToolInput, ToolMetadata, ToolOutput, create_tool_output
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import ResponseStatus
from src.runtime.executor import Executor, get_executor, set_executor


@pytest.fixture
def executor():
    """Create an Executor instance for testing."""
    return Executor(default_timeout=5.0, enable_tracing=False)


@pytest.fixture
def executor_with_tracing():
    """Create an Executor instance with tracing enabled."""
    return Executor(default_timeout=5.0, enable_tracing=True)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=AgentBase)
    agent.name = "test_agent"
    agent.category = "testing"
    agent.execute = AsyncMock()
    return agent


@pytest.fixture
def sample_agent_input():
    """Create sample AgentInput for testing."""
    return AgentInput(
        query="test query",
        conversation_id="conv-123",
        user_id="user-456",
        metadata={"test": "data"},
    )


@pytest.fixture
def sample_agent_output():
    """Create sample AgentOutput for testing."""
    return create_agent_output(
        content="test response",
        agent_name="test_agent",
        agent_category="testing",
        status=ResponseStatus.SUCCESS,
    )


@pytest.fixture
def sample_tool_input():
    """Create sample ToolInput for testing."""
    return ToolInput(
        tool_name="test_tool",
        parameters={"param1": "value1"},
        metadata={"test": "data"},
    )


@pytest.fixture
def sample_tool_metadata():
    """Create sample ToolMetadata for testing."""
    return ToolMetadata(
        name="test_tool",
        description="A test tool",
        parameters=[],
    )


class TestExecutorInitialization:
    """Test Executor initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        executor = Executor()
        assert executor.default_timeout is None
        assert executor.enable_tracing is True

    def test_init_with_timeout(self):
        """Test initialization with timeout."""
        executor = Executor(default_timeout=10.0)
        assert executor.default_timeout == 10.0

    def test_init_with_tracing_disabled(self):
        """Test initialization with tracing disabled."""
        executor = Executor(enable_tracing=False)
        assert executor.enable_tracing is False


class TestExecutorExecuteAgent:
    """Test Executor.execute_agent method."""

    @pytest.mark.asyncio
    async def test_execute_agent_success(
        self, executor, mock_agent, sample_agent_input, sample_agent_output
    ):
        """Test successful agent execution."""
        mock_agent.execute.return_value = sample_agent_output

        result = await executor.execute_agent(mock_agent, sample_agent_input)

        assert isinstance(result, AgentOutput)
        assert result.content == "test response"
        assert result.agent_name == "test_agent"
        assert result.status == ResponseStatus.SUCCESS
        mock_agent.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout(
        self, executor, mock_agent, sample_agent_input, sample_agent_output
    ):
        """Test agent execution with timeout."""

        async def slow_execute(context):
            await asyncio.sleep(1.0)
            return sample_agent_output

        mock_agent.execute.side_effect = slow_execute

        result = await executor.execute_agent(mock_agent, sample_agent_input, timeout=0.1)

        assert result.status == ResponseStatus.ERROR
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_agent_with_default_timeout(
        self, executor, mock_agent, sample_agent_input, sample_agent_output
    ):
        """Test agent execution uses default timeout."""

        async def slow_execute(context):
            await asyncio.sleep(10.0)
            return sample_agent_output

        mock_agent.execute.side_effect = slow_execute

        result = await executor.execute_agent(mock_agent, sample_agent_input)

        assert result.status == ResponseStatus.ERROR
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_agent_with_cancellation(
        self, executor, mock_agent, sample_agent_input, sample_agent_output
    ):
        """Test agent execution handles cancellation (via timeout).

        Note: The executor catches CancelledError internally and returns an error response.
        This test verifies that timeout (which raises CancelledError internally) is handled.
        """

        async def slow_execute(context):
            await asyncio.sleep(1.0)
            return sample_agent_output

        mock_agent.execute.side_effect = slow_execute

        # Timeout will cause CancelledError internally, which executor catches
        result = await executor.execute_agent(mock_agent, sample_agent_input, timeout=0.1)

        # Executor catches timeout (which raises internally) and returns error response
        assert result.status == ResponseStatus.ERROR
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_agent_with_exception(self, executor, mock_agent, sample_agent_input):
        """Test agent execution handles exceptions."""
        mock_agent.execute.side_effect = ValueError("test error")

        result = await executor.execute_agent(mock_agent, sample_agent_input)

        assert result.status == ResponseStatus.ERROR
        assert "test error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_agent_invalid_agent(self, executor, sample_agent_input):
        """Test execute_agent with agent missing execute method."""
        invalid_agent = MagicMock()
        invalid_agent.name = "invalid_agent"
        del invalid_agent.execute

        with pytest.raises(ValueError) as exc_info:
            await executor.execute_agent(invalid_agent, sample_agent_input)
        assert "execute method" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_agent_non_callable_execute(self, executor, sample_agent_input):
        """Test execute_agent with non-callable execute attribute."""
        invalid_agent = MagicMock()
        invalid_agent.name = "invalid_agent"
        invalid_agent.execute = "not a function"

        with pytest.raises(ValueError) as exc_info:
            await executor.execute_agent(invalid_agent, sample_agent_input)
        assert "execute method" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_agent_wrong_return_type(self, executor, mock_agent, sample_agent_input):
        """Test execute_agent with agent returning wrong type."""
        mock_agent.execute.return_value = "not an AgentOutput"

        # The executor catches ValueError and returns an error response
        result = await executor.execute_agent(mock_agent, sample_agent_input)

        assert result.status == ResponseStatus.ERROR
        # The error message is: "Agent test_agent execution failed: Agent test_agent returned <class 'str'>, expected AgentOutput"
        # When lowercased, "expected AgentOutput" becomes "expected agentoutput"
        error_lower = result.error.lower()
        assert "expected agentoutput" in error_lower or "expected AgentOutput" in result.error

    @pytest.mark.asyncio
    async def test_execute_agent_adds_metadata(
        self, executor, mock_agent, sample_agent_input, sample_agent_output
    ):
        """Test execute_agent adds execution metadata."""
        mock_agent.execute.return_value = sample_agent_output

        result = await executor.execute_agent(mock_agent, sample_agent_input)

        assert "agent_name" in result.metadata
        assert "agent_category" in result.metadata
        assert result.metadata["agent_name"] == "test_agent"
        assert result.metadata["agent_category"] == "testing"

    @pytest.mark.asyncio
    async def test_execute_agent_with_tracing(
        self, executor_with_tracing, mock_agent, sample_agent_input, sample_agent_output
    ):
        """Test execute_agent with tracing enabled."""
        mock_agent.execute.return_value = sample_agent_output

        with patch("src.runtime.executor.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_tracer.get_trace_context.return_value = None
            mock_tracer.create_trace_id.return_value = "trace-123"
            # The span object yielded by async_span should be a regular MagicMock
            # (not AsyncMock) because span.update() is a synchronous method
            mock_span = MagicMock()
            mock_span.update = MagicMock()  # Explicitly set update as sync method
            # async_span returns an async context manager that yields the span
            mock_async_cm = AsyncMock()
            mock_async_cm.__aenter__ = AsyncMock(return_value=mock_span)
            mock_async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_tracer.async_span.return_value = mock_async_cm
            mock_get_tracer.return_value = mock_tracer

            result = await executor_with_tracing.execute_agent(mock_agent, sample_agent_input)

            assert isinstance(result, AgentOutput)
            mock_tracer.async_span.assert_called_once()
            assert "trace_id" in result.metadata
            # Verify update was called (synchronously, not awaited)
            assert mock_span.update.called


class TestExecutorExecuteTool:
    """Test Executor.execute_tool method."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, executor, sample_tool_input):
        """Test successful tool execution."""

        async def tool_func(param1: str):
            return {"result": f"processed {param1}"}

        result = await executor.execute_tool(sample_tool_input, tool_func)

        assert isinstance(result, ToolOutput)
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.result["result"] == "processed value1"
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_tool_with_timeout(self, executor, sample_tool_input):
        """Test tool execution with timeout."""

        async def slow_tool_func(param1: str):
            await asyncio.sleep(1.0)
            return {"result": "slow"}

        result = await executor.execute_tool(sample_tool_input, slow_tool_func, timeout=0.1)

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_tool_with_default_timeout(self, executor, sample_tool_input):
        """Test tool execution uses default timeout."""

        async def slow_tool_func(param1: str):
            await asyncio.sleep(10.0)
            return {"result": "slow"}

        result = await executor.execute_tool(sample_tool_input, slow_tool_func)

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_with_exception(self, executor, sample_tool_input):
        """Test tool execution handles exceptions."""

        async def failing_tool_func(param1: str):
            raise ValueError("tool error")

        result = await executor.execute_tool(sample_tool_input, failing_tool_func)

        assert result.success is False
        assert "tool error" in result.error.lower()
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_tool_with_cancellation(self, executor, sample_tool_input):
        """Test tool execution handles cancellation (via timeout).

        Note: The executor catches CancelledError internally and returns an error response.
        This test verifies that timeout (which raises CancelledError internally) is handled.
        """

        async def slow_tool_func(param1: str):
            await asyncio.sleep(1.0)
            return {"result": "slow"}

        # Timeout will cause CancelledError internally, which executor catches
        result = await executor.execute_tool(sample_tool_input, slow_tool_func, timeout=0.1)

        # Executor catches timeout (which raises internally) and returns error response
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_with_metadata(
        self, executor, sample_tool_input, sample_tool_metadata
    ):
        """Test tool execution with ToolMetadata."""

        async def tool_func(param1: str):
            return {"result": "success"}

        result = await executor.execute_tool(
            sample_tool_input, tool_func, tool_metadata=sample_tool_metadata
        )

        assert result.success is True
        assert "tool_metadata" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_tool_adds_metadata(self, executor, sample_tool_input):
        """Test execute_tool adds execution metadata."""

        async def tool_func(param1: str):
            return {"result": "success"}

        result = await executor.execute_tool(sample_tool_input, tool_func)

        assert "tool_name" in result.metadata
        assert "parameters" in result.metadata
        assert result.metadata["tool_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_tool_with_tracing(self, executor_with_tracing, sample_tool_input):
        """Test execute_tool with tracing enabled."""

        async def tool_func(param1: str):
            return {"result": "success"}

        with patch("src.runtime.executor.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_tracer.get_trace_context.return_value = None
            mock_tracer.create_trace_id.return_value = "trace-123"
            # The span object yielded by async_span should be a regular MagicMock
            # (not AsyncMock) because span.update() is a synchronous method
            mock_span = MagicMock()
            mock_span.update = MagicMock()  # Explicitly set update as sync method
            # async_span returns an async context manager that yields the span
            mock_async_cm = AsyncMock()
            mock_async_cm.__aenter__ = AsyncMock(return_value=mock_span)
            mock_async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_tracer.async_span.return_value = mock_async_cm
            mock_get_tracer.return_value = mock_tracer

            result = await executor_with_tracing.execute_tool(sample_tool_input, tool_func)

            assert result.success is True
            mock_tracer.async_span.assert_called_once()
            assert "trace_id" in result.metadata
            # Verify update was called (synchronously, not awaited)
            assert mock_span.update.called

    @pytest.mark.asyncio
    async def test_execute_tool_measures_execution_time(self, executor, sample_tool_input):
        """Test execute_tool measures execution time."""

        async def tool_func(param1: str):
            await asyncio.sleep(0.1)
            return {"result": "success"}

        result = await executor.execute_tool(sample_tool_input, tool_func)

        assert result.execution_time_ms >= 100  # At least 100ms
        assert result.execution_time_ms < 200  # But less than 200ms (with some buffer)


class TestExecutorPrivateMethods:
    """Test Executor private methods."""

    @pytest.mark.asyncio
    async def test_execute_with_cancellation_success(
        self, executor, mock_agent, sample_agent_output
    ):
        """Test _execute_with_cancellation with successful execution."""
        context = AgentContext(query="test")
        mock_agent.execute.return_value = sample_agent_output

        result = await executor._execute_with_cancellation(mock_agent, context, None)

        assert isinstance(result, AgentOutput)
        mock_agent.execute.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_execute_with_cancellation_no_execute_method(self, executor):
        """Test _execute_with_cancellation with agent missing execute method."""
        invalid_agent = MagicMock()
        invalid_agent.name = "invalid"
        del invalid_agent.execute
        context = AgentContext(query="test")

        with pytest.raises(ValueError) as exc_info:
            await executor._execute_with_cancellation(invalid_agent, context, None)
        assert "execute method" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_with_cancellation_non_callable(self, executor):
        """Test _execute_with_cancellation with non-callable execute."""
        invalid_agent = MagicMock()
        invalid_agent.name = "invalid"
        invalid_agent.execute = "not callable"
        context = AgentContext(query="test")

        with pytest.raises(ValueError) as exc_info:
            await executor._execute_with_cancellation(invalid_agent, context, None)
        assert "not callable" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_tool_with_cancellation_success(self, executor):
        """Test _execute_tool_with_cancellation with successful execution."""

        async def tool_func(param1: str):
            return {"result": "success"}

        result = await executor._execute_tool_with_cancellation(
            tool_func, {"param1": "value1"}, None
        )

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_execute_tool_with_cancellation_with_exception(self, executor):
        """Test _execute_tool_with_cancellation with exception."""

        async def failing_tool_func(param1: str):
            raise ValueError("tool error")

        with pytest.raises(ValueError) as exc_info:
            await executor._execute_tool_with_cancellation(
                failing_tool_func, {"param1": "value1"}, None
            )
        assert "tool error" in str(exc_info.value)


class TestExecutorDefaultInstance:
    """Test default Executor instance functions."""

    def test_get_executor_creates_default(self):
        """Test get_executor creates default instance."""
        # Get the executor (will create if doesn't exist)
        executor = get_executor()
        assert isinstance(executor, Executor)

    def test_set_executor(self):
        """Test set_executor sets custom instance."""
        custom_executor = Executor(default_timeout=10.0, enable_tracing=False)
        set_executor(custom_executor)
        executor = get_executor()
        assert executor is custom_executor
        assert executor.default_timeout == 10.0
        assert executor.enable_tracing is False

    def test_get_executor_reuses_instance(self):
        """Test get_executor reuses existing instance."""
        executor1 = get_executor()
        executor2 = get_executor()
        assert executor1 is executor2
