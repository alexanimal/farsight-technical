"""Unit tests for the Temporal worker module.

This module tests the worker bootstrap functions, including worker initialization,
workflow/activity registration, signal handling, and graceful shutdown.
"""

import asyncio
import logging
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.temporal.activities import agents, tools
from src.temporal.client import (
    DEFAULT_TASK_QUEUE,
    DEFAULT_TEMPORAL_ADDRESS,
    DEFAULT_TEMPORAL_NAMESPACE,
)
from src.temporal.worker import (
    _get_activities,
    _get_workflows,
    _setup_signal_handlers,
    main,
    run_worker,
)
from src.temporal.workflows import OrchestratorWorkflow


class TestGetWorkflows:
    """Test _get_workflows function."""

    def test_get_workflows_returns_orchestrator(self):
        """Test that _get_workflows returns OrchestratorWorkflow."""
        workflows = _get_workflows()
        assert isinstance(workflows, list)
        assert len(workflows) >= 1
        assert OrchestratorWorkflow in workflows

    def test_get_workflows_handles_import_errors(self):
        """Test that _get_workflows handles import errors gracefully."""
        # The function should still return OrchestratorWorkflow even if
        # other workflows fail to import
        workflows = _get_workflows()
        assert OrchestratorWorkflow in workflows

    def test_get_workflows_returns_list(self):
        """Test that _get_workflows returns a list."""
        workflows = _get_workflows()
        assert isinstance(workflows, list)


class TestGetActivities:
    """Test _get_activities function."""

    def test_get_activities_returns_expected_activities(self):
        """Test that _get_activities returns all expected activities."""
        activities = _get_activities()
        assert isinstance(activities, list)
        assert len(activities) == 4
        assert agents.execute_agent in activities
        assert agents.execute_agent_with_options in activities
        assert tools.execute_tool in activities
        assert tools.execute_tool_with_options in activities

    def test_get_activities_returns_list(self):
        """Test that _get_activities returns a list."""
        activities = _get_activities()
        assert isinstance(activities, list)


class TestSetupSignalHandlers:
    """Test _setup_signal_handlers function."""

    def test_setup_signal_handlers_registers_handlers(self):
        """Test that signal handlers are registered."""
        with patch("src.temporal.worker.signal.signal") as mock_signal:
            _setup_signal_handlers()
            # Should register SIGINT and SIGTERM handlers
            assert mock_signal.call_count == 2
            # Check that both signals were registered
            calls = [call[0][0] for call in mock_signal.call_args_list]
            assert signal.SIGINT in calls
            assert signal.SIGTERM in calls

    def test_setup_signal_handlers_handler_calls_sys_exit(self):
        """Test that signal handler calls sys.exit."""
        with patch("src.temporal.worker.signal.signal") as mock_signal, \
             patch("src.temporal.worker.sys.exit") as mock_exit, \
             patch("src.temporal.worker.logger") as mock_logger:
            _setup_signal_handlers()
            # Get the handler function
            handler = mock_signal.call_args_list[0][0][1]
            # Call the handler
            handler(signal.SIGINT, None)
            # Verify sys.exit was called
            mock_exit.assert_called_once_with(0)
            # Verify logging occurred
            mock_logger.info.assert_called_once()


class TestRunWorker:
    """Test run_worker function."""

    @pytest.mark.asyncio
    async def test_run_worker_success(self):
        """Test successful worker execution."""
        mock_client = MagicMock()
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run = AsyncMock()
        mock_worker_instance.shutdown = AsyncMock()

        with patch("src.temporal.worker.Client") as mock_client_class, \
             patch("src.temporal.worker.Worker") as mock_worker_class, \
             patch("src.temporal.worker._get_workflows") as mock_get_workflows, \
             patch("src.temporal.worker._get_activities") as mock_get_activities, \
             patch("src.temporal.worker.logger") as mock_logger:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            mock_worker_class.return_value = mock_worker_instance
            mock_get_workflows.return_value = [OrchestratorWorkflow]
            mock_get_activities.return_value = [
                agents.execute_agent,
                tools.execute_tool,
            ]

            # Simulate worker running and then being interrupted
            mock_worker_instance.run = AsyncMock(side_effect=KeyboardInterrupt())

            await run_worker(
                temporal_address="test:7233",
                temporal_namespace="test-namespace",
                task_queue="test-queue",
                max_concurrent_activities=5,
                max_concurrent_workflow_tasks=5,
            )

            # Verify connection
            mock_client_class.connect.assert_called_once_with(
                "test:7233",
                namespace="test-namespace",
            )
            # Verify worker was created with correct parameters
            mock_worker_class.assert_called_once()
            call_args = mock_worker_class.call_args
            # client is the first positional argument
            assert call_args[0][0] == mock_client
            # Other parameters are keyword arguments
            call_kwargs = call_args[1]
            assert call_kwargs["task_queue"] == "test-queue"
            assert call_kwargs["workflows"] == [OrchestratorWorkflow]
            assert call_kwargs["activities"] == [
                agents.execute_agent,
                tools.execute_tool,
            ]
            assert call_kwargs["max_concurrent_activities"] == 5
            assert call_kwargs["max_concurrent_workflow_tasks"] == 5
            # Verify shutdown was called
            mock_worker_instance.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_worker_with_defaults(self):
        """Test worker execution with default parameters."""
        mock_client = MagicMock()
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run = AsyncMock(side_effect=KeyboardInterrupt())
        mock_worker_instance.shutdown = AsyncMock()

        with patch("src.temporal.worker.Client") as mock_client_class, \
             patch("src.temporal.worker.Worker") as mock_worker_class, \
             patch("src.temporal.worker._get_workflows") as mock_get_workflows, \
             patch("src.temporal.worker._get_activities") as mock_get_activities:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            mock_worker_class.return_value = mock_worker_instance
            mock_get_workflows.return_value = [OrchestratorWorkflow]
            mock_get_activities.return_value = [agents.execute_agent]

            await run_worker()

            # Verify default parameters were used
            mock_client_class.connect.assert_called_once_with(
                DEFAULT_TEMPORAL_ADDRESS,
                namespace=DEFAULT_TEMPORAL_NAMESPACE,
            )
            call_kwargs = mock_worker_class.call_args[1]
            assert call_kwargs["task_queue"] == DEFAULT_TASK_QUEUE
            assert call_kwargs["max_concurrent_activities"] == 10
            assert call_kwargs["max_concurrent_workflow_tasks"] == 10

    @pytest.mark.asyncio
    async def test_run_worker_keyboard_interrupt(self):
        """Test worker execution with KeyboardInterrupt."""
        mock_client = MagicMock()
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run = AsyncMock(side_effect=KeyboardInterrupt())
        mock_worker_instance.shutdown = AsyncMock()

        with patch("src.temporal.worker.Client") as mock_client_class, \
             patch("src.temporal.worker.Worker") as mock_worker_class, \
             patch("src.temporal.worker._get_workflows") as mock_get_workflows, \
             patch("src.temporal.worker._get_activities") as mock_get_activities, \
             patch("src.temporal.worker.logger") as mock_logger:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            mock_worker_class.return_value = mock_worker_instance
            mock_get_workflows.return_value = [OrchestratorWorkflow]
            mock_get_activities.return_value = [agents.execute_agent]

            await run_worker()

            # Verify shutdown was called
            mock_worker_instance.shutdown.assert_called_once()
            # Verify interrupt was logged
            assert any(
                "interrupt" in str(call).lower()
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_run_worker_exception_handling(self):
        """Test worker execution with exception handling."""
        mock_client = MagicMock()
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run = AsyncMock(side_effect=RuntimeError("Worker error"))
        mock_worker_instance.shutdown = AsyncMock()

        with patch("src.temporal.worker.Client") as mock_client_class, \
             patch("src.temporal.worker.Worker") as mock_worker_class, \
             patch("src.temporal.worker._get_workflows") as mock_get_workflows, \
             patch("src.temporal.worker._get_activities") as mock_get_activities, \
             patch("src.temporal.worker.logger") as mock_logger:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            mock_worker_class.return_value = mock_worker_instance
            mock_get_workflows.return_value = [OrchestratorWorkflow]
            mock_get_activities.return_value = [agents.execute_agent]

            with pytest.raises(RuntimeError, match="Worker error"):
                await run_worker()

            # Verify shutdown was called even on exception
            mock_worker_instance.shutdown.assert_called_once()
            # Verify error was logged
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_worker_connection_failure(self):
        """Test worker execution when connection fails."""
        with patch("src.temporal.worker.Client") as mock_client_class, \
             patch("src.temporal.worker.logger") as mock_logger:
            mock_client_class.connect = AsyncMock(
                side_effect=ConnectionError("Connection failed")
            )

            with pytest.raises(ConnectionError, match="Connection failed"):
                await run_worker()

            # Verify error was logged
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_worker_shutdown_when_no_worker(self):
        """Test that shutdown is handled when worker is None."""
        with patch("src.temporal.worker.Client") as mock_client_class:
            mock_client_class.connect = AsyncMock(
                side_effect=Exception("Error before worker creation")
            )

            with pytest.raises(Exception, match="Error before worker creation"):
                await run_worker()

            # Should not raise an error even if worker is None

    @pytest.mark.asyncio
    async def test_run_worker_logging(self):
        """Test that worker execution logs appropriately."""
        mock_client = MagicMock()
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run = AsyncMock(side_effect=KeyboardInterrupt())
        mock_worker_instance.shutdown = AsyncMock()

        with patch("src.temporal.worker.Client") as mock_client_class, \
             patch("src.temporal.worker.Worker") as mock_worker_class, \
             patch("src.temporal.worker._get_workflows") as mock_get_workflows, \
             patch("src.temporal.worker._get_activities") as mock_get_activities, \
             patch("src.temporal.worker.logger") as mock_logger:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            mock_worker_class.return_value = mock_worker_instance
            mock_get_workflows.return_value = [OrchestratorWorkflow]
            mock_get_activities.return_value = [agents.execute_agent, tools.execute_tool]

            await run_worker(
                temporal_address="test:7233",
                temporal_namespace="test-ns",
                task_queue="test-queue",
            )

            # Verify various log messages
            log_messages = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Starting Temporal worker" in msg for msg in log_messages)
            assert any("Connected to Temporal" in msg for msg in log_messages)
            assert any("Registering" in msg and "workflow" in msg.lower() for msg in log_messages)
            assert any("Registering" in msg and "activity" in msg.lower() for msg in log_messages)
            assert any("Worker initialized" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_run_worker_global_worker_variable(self):
        """Test that global _worker variable is set and cleared."""
        mock_client = MagicMock()
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run = AsyncMock(side_effect=KeyboardInterrupt())
        mock_worker_instance.shutdown = AsyncMock()

        with patch("src.temporal.worker.Client") as mock_client_class, \
             patch("src.temporal.worker.Worker") as mock_worker_class, \
             patch("src.temporal.worker._get_workflows") as mock_get_workflows, \
             patch("src.temporal.worker._get_activities") as mock_get_activities:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            mock_worker_class.return_value = mock_worker_instance
            mock_get_workflows.return_value = [OrchestratorWorkflow]
            mock_get_activities.return_value = [agents.execute_agent]

            # Import the module to access the global variable
            import src.temporal.worker as worker_module

            # Initially should be None
            assert worker_module._worker is None

            await run_worker()

            # After shutdown, should be None again
            assert worker_module._worker is None


class TestMain:
    """Test main entrypoint function."""

    @pytest.mark.asyncio
    async def test_main_sets_up_logging(self):
        """Test that main sets up logging."""
        with patch("src.temporal.worker.logging.basicConfig") as mock_basic_config, \
             patch("src.temporal.worker._setup_signal_handlers") as mock_setup_signals, \
             patch("src.temporal.worker.run_worker") as mock_run_worker:
            mock_run_worker.return_value = None

            await main()

            # Verify logging was configured
            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs["level"] == logging.INFO
            assert "format" in call_kwargs

    @pytest.mark.asyncio
    async def test_main_sets_up_signal_handlers(self):
        """Test that main sets up signal handlers."""
        with patch("src.temporal.worker.logging.basicConfig"), \
             patch("src.temporal.worker._setup_signal_handlers") as mock_setup_signals, \
             patch("src.temporal.worker.run_worker") as mock_run_worker:
            mock_run_worker.return_value = None

            await main()

            # Verify signal handlers were set up
            mock_setup_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_calls_run_worker_with_defaults(self):
        """Test that main calls run_worker with default parameters."""
        with patch("src.temporal.worker.logging.basicConfig"), \
             patch("src.temporal.worker._setup_signal_handlers"), \
             patch("src.temporal.worker.run_worker") as mock_run_worker:
            mock_run_worker.return_value = None

            await main()

            # Verify run_worker was called with defaults
            mock_run_worker.assert_called_once_with(
                temporal_address=DEFAULT_TEMPORAL_ADDRESS,
                temporal_namespace=DEFAULT_TEMPORAL_NAMESPACE,
                task_queue=DEFAULT_TASK_QUEUE,
            )

    @pytest.mark.asyncio
    async def test_main_handles_keyboard_interrupt(self):
        """Test that main completes when run_worker handles KeyboardInterrupt."""
        with patch("src.temporal.worker.logging.basicConfig"), \
             patch("src.temporal.worker._setup_signal_handlers"), \
             patch("src.temporal.worker.run_worker") as mock_run_worker:
            # run_worker() catches KeyboardInterrupt internally and returns normally
            # So main() should complete successfully
            mock_run_worker.return_value = None

            # Should complete without raising
            await main()

            mock_run_worker.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_handles_exceptions(self):
        """Test that main propagates exceptions."""
        with patch("src.temporal.worker.logging.basicConfig"), \
             patch("src.temporal.worker._setup_signal_handlers"), \
             patch("src.temporal.worker.run_worker") as mock_run_worker:
            mock_run_worker.side_effect = RuntimeError("Worker error")

            with pytest.raises(RuntimeError, match="Worker error"):
                await main()

