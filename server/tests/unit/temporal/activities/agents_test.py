"""Unit tests for the Temporal agents activities module.

This module tests all activity functions and helper functions in agents.py,
including agent execution, error handling, and activity options.
"""

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio.common import RetryPolicy

from src.contracts.agent_io import AgentInput, AgentOutput, create_agent_output
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus
from src.temporal.activities.agents import (
    _DEFAULT_ACTIVITY_OPTIONS,
    execute_agent,
    execute_agent_with_options,
    get_activity_options,
)


class TestExecuteAgent:
    """Test execute_agent activity function."""

    @pytest.mark.asyncio
    async def test_execute_agent_success(self):
        """Test successful agent execution."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock agent output
        agent_output = create_agent_output(
            content="Test response",
            agent_name="test_agent",
            agent_category="test",
        )
        agent_response = agent_output.to_agent_response()

        # Setup mock executor
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value=agent_output)

        context_dict = {
            "query": "What is the weather?",
            "conversation_id": "conv-1",
            "user_id": "user-1",
        }

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent(
                agent_name="test_agent",
                context=context_dict,
                executor=mock_executor,
            )

            assert result["success"] is True
            assert result["error"] is None
            assert result["agent_name"] == "test_agent"
            assert result["response"] == agent_response.model_dump()
            assert result["metadata"]["agent_category"] == "test"
            assert result["metadata"]["response_status"] == ResponseStatus.SUCCESS.value

            # Verify agent was instantiated
            mock_agent_class.assert_called_once_with(config_path=Path("/fake/path/config.yaml"))
            # Verify executor was called
            mock_executor.execute_agent.assert_called_once()
            # Verify logging
            assert mock_activity.logger.info.call_count >= 2

    @pytest.mark.asyncio
    async def test_execute_agent_not_found(self):
        """Test agent execution when agent is not found."""
        context_dict = {"query": "What is the weather?"}

        with patch("src.temporal.activities.agents.get_agent", return_value=None), \
             patch("src.temporal.activities.agents.list_available_agents", return_value=["agent1", "agent2"]), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()

            result = await execute_agent(
                agent_name="nonexistent_agent",
                context=context_dict,
            )

            assert result["success"] is False
            assert "not found" in result["error"].lower()
            assert "nonexistent_agent" in result["error"]
            assert result["agent_name"] == "nonexistent_agent"
            assert result["response"] is None
            assert result["metadata"] == {}
            mock_activity.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_invalid_context(self):
        """Test agent execution with invalid context."""
        # Setup mock agent class
        mock_agent_class = MagicMock()
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_class.return_value = mock_agent_instance

        context_dict = {}  # Missing required 'query' field

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            # Mock validate_agent_input to raise an exception
            with patch("src.temporal.activities.agents.validate_agent_input") as mock_validate:
                mock_validate.side_effect = ValueError("Invalid context: query is required")

                result = await execute_agent(
                    agent_name="test_agent",
                    context=context_dict,
                )

                assert result["success"] is False
                assert "invalid context" in result["error"].lower()
                assert result["agent_name"] == "test_agent"
                assert result["response"] is None
                assert result["metadata"] == {}
                mock_activity.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_instantiation_failure(self):
        """Test agent execution when agent instantiation fails."""
        # Setup mock agent class that raises exception
        mock_agent_class = MagicMock()
        mock_agent_class.side_effect = ValueError("Failed to load config")

        context_dict = {"query": "What is the weather?"}

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent(
                agent_name="test_agent",
                context=context_dict,
            )

            assert result["success"] is False
            assert "failed to instantiate" in result["error"].lower()
            assert result["agent_name"] == "test_agent"
            assert result["response"] is None
            assert result["metadata"]["exception_type"] == "ValueError"
            mock_activity.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_execution_failure(self):
        """Test agent execution when agent execution fails."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock executor that raises exception
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(side_effect=RuntimeError("Execution failed"))

        context_dict = {"query": "What is the weather?"}

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent(
                agent_name="test_agent",
                context=context_dict,
                executor=mock_executor,
            )

            assert result["success"] is False
            assert "execution failed" in result["error"].lower()
            assert result["agent_name"] == "test_agent"
            assert result["response"] is None
            assert result["metadata"]["exception_type"] == "RuntimeError"
            mock_activity.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout(self):
        """Test agent execution with timeout parameter."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock agent output
        agent_output = create_agent_output(
            content="Test response",
            agent_name="test_agent",
            agent_category="test",
        )

        # Setup mock executor
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value=agent_output)

        context_dict = {"query": "What is the weather?"}

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent(
                agent_name="test_agent",
                context=context_dict,
                timeout_seconds=120.0,
                executor=mock_executor,
            )

            assert result["success"] is True
            # Verify executor was called with timeout
            mock_executor.execute_agent.assert_called_once()
            call_args = mock_executor.execute_agent.call_args
            assert call_args[1]["timeout"] == 120.0

    @pytest.mark.asyncio
    async def test_execute_agent_with_full_context(self):
        """Test agent execution with full context including all fields."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock agent output
        agent_output = create_agent_output(
            content="Test response",
            agent_name="test_agent",
            agent_category="test",
        )

        # Setup mock executor
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value=agent_output)

        context_dict = {
            "query": "What is the weather?",
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "metadata": {"key": "value"},
            "shared_data": {"shared": "data"},
            "conversation_history": [{"role": "user", "content": "hi"}],
        }

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent(
                agent_name="test_agent",
                context=context_dict,
                executor=mock_executor,
            )

            assert result["success"] is True
            # Verify executor was called with correct AgentInput
            mock_executor.execute_agent.assert_called_once()
            call_args = mock_executor.execute_agent.call_args
            agent_input = call_args[1]["agent_input"]  # agent_input is a keyword argument
            assert isinstance(agent_input, AgentInput)
            assert agent_input.query == "What is the weather?"
            assert agent_input.conversation_id == "conv-1"
            assert agent_input.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_execute_agent_with_partial_response(self):
        """Test agent execution with partial response status."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock agent output with partial status
        agent_output = create_agent_output(
            content="Partial response",
            agent_name="test_agent",
            agent_category="test",
            status=ResponseStatus.PARTIAL,
        )

        # Setup mock executor
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value=agent_output)

        context_dict = {"query": "What is the weather?"}

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent(
                agent_name="test_agent",
                context=context_dict,
                executor=mock_executor,
            )

            assert result["success"] is True
            assert result["metadata"]["response_status"] == ResponseStatus.PARTIAL.value

    @pytest.mark.asyncio
    async def test_execute_agent_logging(self):
        """Test that agent execution logs appropriately."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock agent output
        agent_output = create_agent_output(
            content="Test response",
            agent_name="test_agent",
            agent_category="test",
        )

        # Setup mock executor
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value=agent_output)

        context_dict = {"query": "What is the weather?"}

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            await execute_agent(
                agent_name="test_agent",
                context=context_dict,
                executor=mock_executor,
            )

            # Verify logging calls
            assert mock_activity.logger.info.call_count >= 2
            # First call should log execution start
            first_call = mock_activity.logger.info.call_args_list[0]
            assert "executing agent" in first_call[0][0].lower()
            # Second call should log success
            second_call = mock_activity.logger.info.call_args_list[1]
            assert "executed successfully" in second_call[0][0].lower()


class TestExecuteAgentWithOptions:
    """Test execute_agent_with_options activity function."""

    @pytest.mark.asyncio
    async def test_execute_agent_with_options_success(self):
        """Test execute_agent_with_options with custom options."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock agent output
        agent_output = create_agent_output(
            content="Test response",
            agent_name="test_agent",
            agent_category="test",
        )

        # Setup mock executor
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value=agent_output)

        context_dict = {"query": "What is the weather?"}
        activity_options = {
            "start_to_close_timeout": timedelta(seconds=300),
            "retry_policy": RetryPolicy(maximum_attempts=5),
        }

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent_with_options(
                agent_name="test_agent",
                context=context_dict,
                activity_options=activity_options,
                executor=mock_executor,
            )

            assert result["success"] is True
            # Verify options were logged
            mock_activity.logger.info.assert_any_call(
                f"Custom activity options provided: {activity_options}"
            )

    @pytest.mark.asyncio
    async def test_execute_agent_with_options_no_options(self):
        """Test execute_agent_with_options without custom options."""
        # Setup mock agent class
        mock_agent_instance = MagicMock(spec=AgentBase)
        mock_agent_instance.name = "test_agent"
        mock_agent_instance.category = "test"
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        # Setup mock agent output
        agent_output = create_agent_output(
            content="Test response",
            agent_name="test_agent",
            agent_category="test",
        )

        # Setup mock executor
        mock_executor = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value=agent_output)

        context_dict = {"query": "What is the weather?"}

        with patch("src.temporal.activities.agents.get_agent") as mock_get_agent, \
             patch("src.temporal.activities.agents.get_executor", return_value=mock_executor), \
             patch("src.temporal.activities.agents.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_get_agent.return_value = (mock_agent_class, Path("/fake/path/config.yaml"))

            result = await execute_agent_with_options(
                agent_name="test_agent",
                context=context_dict,
                activity_options=None,
                executor=mock_executor,
            )

            assert result["success"] is True
            # Verify no options logging occurred
            options_logged = any(
                "Custom activity options" in str(call)
                for call in mock_activity.logger.info.call_args_list
            )
            assert not options_logged


class TestGetActivityOptions:
    """Test get_activity_options helper function."""

    def test_get_activity_options_default(self):
        """Test getting default activity options."""
        options = get_activity_options()

        assert "start_to_close_timeout" in options
        assert "retry_policy" in options
        assert options["start_to_close_timeout"] == timedelta(seconds=600)
        assert isinstance(options["retry_policy"], RetryPolicy)
        assert options["retry_policy"].maximum_attempts == 3
        assert options["retry_policy"].initial_interval == timedelta(seconds=2)

    def test_get_activity_options_custom_timeout(self):
        """Test getting activity options with custom timeout."""
        options = get_activity_options(timeout_seconds=300.0)

        assert options["start_to_close_timeout"] == timedelta(seconds=300.0)
        # Retry policy should remain default
        assert isinstance(options["retry_policy"], RetryPolicy)
        assert options["retry_policy"].maximum_attempts == 3

    def test_get_activity_options_custom_max_retries(self):
        """Test getting activity options with custom max retries."""
        options = get_activity_options(max_retries=5)

        assert options["start_to_close_timeout"] == timedelta(seconds=600)
        assert isinstance(options["retry_policy"], RetryPolicy)
        assert options["retry_policy"].maximum_attempts == 5
        # Initial interval should remain default
        assert options["retry_policy"].initial_interval == timedelta(seconds=2)

    def test_get_activity_options_custom_initial_retry_interval(self):
        """Test getting activity options with custom initial retry interval."""
        options = get_activity_options(initial_retry_interval=5.0)

        assert options["start_to_close_timeout"] == timedelta(seconds=600)
        assert isinstance(options["retry_policy"], RetryPolicy)
        assert options["retry_policy"].initial_interval == timedelta(seconds=5.0)
        # Max retries should remain default
        assert options["retry_policy"].maximum_attempts == 3

    def test_get_activity_options_all_custom(self):
        """Test getting activity options with all custom values."""
        options = get_activity_options(
            timeout_seconds=900.0,
            max_retries=7,
            initial_retry_interval=10.0,
        )

        assert options["start_to_close_timeout"] == timedelta(seconds=900.0)
        assert isinstance(options["retry_policy"], RetryPolicy)
        assert options["retry_policy"].maximum_attempts == 7
        assert options["retry_policy"].initial_interval == timedelta(seconds=10.0)

    def test_get_activity_options_partial_retry_customization(self):
        """Test getting activity options with partial retry customization."""
        # Test with only max_retries
        options = get_activity_options(max_retries=4)
        assert options["retry_policy"].maximum_attempts == 4
        assert options["retry_policy"].initial_interval == timedelta(seconds=2)

        # Test with only initial_retry_interval
        options = get_activity_options(initial_retry_interval=3.0)
        assert options["retry_policy"].maximum_attempts == 3
        assert options["retry_policy"].initial_interval == timedelta(seconds=3.0)

    def test_get_activity_options_retry_policy_preserves_backoff(self):
        """Test that custom retry policy preserves backoff coefficient and max interval."""
        options = get_activity_options(max_retries=5, initial_retry_interval=1.0)

        retry_policy = options["retry_policy"]
        assert retry_policy.maximum_attempts == 5
        assert retry_policy.initial_interval == timedelta(seconds=1.0)
        # These should be preserved from default
        assert retry_policy.backoff_coefficient == 2.0
        assert retry_policy.maximum_interval == timedelta(seconds=120)

    def test_get_activity_options_returns_copy(self):
        """Test that get_activity_options returns a copy, not a reference."""
        options1 = get_activity_options()
        options2 = get_activity_options()

        # Modify one
        options1["start_to_close_timeout"] = timedelta(seconds=999)

        # Other should be unchanged
        assert options2["start_to_close_timeout"] == timedelta(seconds=600)
        assert options1["start_to_close_timeout"] == timedelta(seconds=999)

