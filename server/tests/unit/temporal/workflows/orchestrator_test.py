"""Unit tests for the Temporal orchestrator workflow.

This module tests the OrchestratorWorkflow class, including workflow execution,
agent coordination, signal handling, query methods, and error handling.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agent_response import ResponseStatus
from src.temporal.queries import (
    AgentExecutionStatus,
    AgentStatusQueryResult,
    WorkflowProgressQueryResult,
    WorkflowStateQueryResult,
    WorkflowStatus,
    WorkflowStatusQueryResult,
)
from src.temporal.signals import CancellationSignal, UserInputSignal
from src.temporal.workflows.orchestrator import (
    AgentExecutionState,
    OrchestratorWorkflow,
    WorkflowState,
)


class TestOrchestratorWorkflowInitialization:
    """Test OrchestratorWorkflow initialization."""

    def test_init_creates_empty_state(self):
        """Test that initialization creates empty workflow state."""
        workflow = OrchestratorWorkflow()
        assert isinstance(workflow._state, WorkflowState)
        assert workflow._state.status == WorkflowStatus.PENDING
        assert workflow._state.started_at is None
        assert workflow._state.completed_at is None
        assert workflow._state.context == {}
        assert workflow._state.agent_executions == {}
        assert workflow._state.agent_responses == []
        assert workflow._state.cancellation_requested is False


class TestOrchestratorWorkflowRun:
    """Test OrchestratorWorkflow.run() method."""

    @pytest.mark.asyncio
    async def test_run_with_agent_plan_sequential(self):
        """Test workflow run with provided agent plan in sequential mode."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_plan = ["agent1", "agent2"]

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            # Mock execute_activity to return proper agent execution results
            # The workflow expects results with "success" and "response" keys
            # Note: The workflow will also call orchestration agent for consolidation,
            # so we need 3 responses: agent1, agent2, and orchestration (consolidation)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    {
                        "success": True,
                        "response": {
                            "content": {"summary": "Response from agent1"},
                            "agent_name": "agent1",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": {"summary": "Response from agent2"},
                            "agent_name": "agent2",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": {"summary": "Consolidated response"},
                            "agent_name": "orchestration",
                            "agent_category": "orchestration",
                        },
                    },
                ]
            )

            # Mock internal methods that are called during workflow execution
            with (
                patch.object(
                    workflow, "_evaluate_response_quality", new_callable=AsyncMock
                ) as mock_eval,
                patch.object(
                    workflow, "_save_conversation_history", new_callable=AsyncMock
                ) as mock_save,
                patch.object(workflow, "_append_to_history", new_callable=AsyncMock) as mock_append,
                patch.object(
                    workflow, "_update_context_with_results", new_callable=AsyncMock
                ) as mock_update_context,
            ):
                # Mock evaluation to return satisfactory=True so workflow completes
                mock_eval.return_value = {"satisfactory": True, "confidence": 0.9}
                # Mock update_context to return the context as-is
                mock_update_context.return_value = context

                result = await workflow.run(
                    context=context,
                    agent_plan=agent_plan,
                    execution_mode="sequential",
                )

                assert result["success"] is True
                assert result["workflow_id"] == "test-workflow-id"
                # agent_responses includes agent1, agent2, and orchestration (for consolidation)
                assert len(result["agent_responses"]) == 3
                assert workflow._state.status == WorkflowStatus.COMPLETED
                assert workflow._state.started_at == mock_now
                assert workflow._state.completed_at == mock_now

    @pytest.mark.asyncio
    async def test_run_with_agent_plan_parallel(self):
        """Test workflow run with provided agent plan in parallel mode."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_plan = ["agent1", "agent2"]

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            # Mock execute_activity to return proper agent execution results
            # The workflow expects results with "success" and "response" keys
            # Note: The workflow will also call orchestration agent for consolidation,
            # so we need 3 responses: agent1, agent2, and orchestration (consolidation)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    {
                        "success": True,
                        "response": {
                            "content": {"summary": "Response from agent1"},
                            "agent_name": "agent1",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": {"summary": "Response from agent2"},
                            "agent_name": "agent2",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": {"summary": "Consolidated response"},
                            "agent_name": "orchestration",
                            "agent_category": "orchestration",
                        },
                    },
                ]
            )

            # Mock internal methods that are called during workflow execution
            with (
                patch.object(
                    workflow, "_evaluate_response_quality", new_callable=AsyncMock
                ) as mock_eval,
                patch.object(
                    workflow, "_save_conversation_history", new_callable=AsyncMock
                ) as mock_save,
                patch.object(workflow, "_append_to_history", new_callable=AsyncMock) as mock_append,
                patch.object(
                    workflow, "_update_context_with_results", new_callable=AsyncMock
                ) as mock_update_context,
            ):
                # Mock evaluation to return satisfactory=True so workflow completes
                mock_eval.return_value = {"satisfactory": True, "confidence": 0.9}
                # Mock update_context to return the context as-is
                mock_update_context.return_value = context

                result = await workflow.run(
                    context=context,
                    agent_plan=agent_plan,
                    execution_mode="parallel",
                )

                assert result["success"] is True
                # agent_responses includes agent1, agent2, and orchestration (for consolidation)
                assert len(result["agent_responses"]) == 3

    @pytest.mark.asyncio
    async def test_run_without_agent_plan_determines_plan(self):
        """Test workflow run without agent plan determines plan via orchestration."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            # First call: orchestration agent to determine plan
            # Second call: execute the determined agents
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    {
                        "success": True,
                        "response": {
                            "content": "Plan determined",
                            "agent_name": "orchestration",
                            "agent_category": "orchestration",
                            "metadata": {
                                "execution_plan": {
                                    "agents": ["agent1", "agent2"],
                                },
                            },
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": "Response from agent1",
                            "agent_name": "agent1",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": "Response from agent2",
                            "agent_name": "agent2",
                            "agent_category": "test",
                        },
                    },
                ]
            )

            result = await workflow.run(context=context, agent_plan=None)

            assert result["success"] is True
            # Should have called orchestration agent first
            assert mock_workflow.execute_activity.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_with_list_context_unpacks_correctly(self):
        """Test workflow run handles list context from Temporal UI."""
        workflow = OrchestratorWorkflow()
        # Simulate Temporal UI passing arguments as a list
        context_list = [
            {"query": "What is the weather?"},
            ["agent1", "agent2"],
            "parallel",
        ]

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": True,
                    "response": {
                        "content": "Response",
                        "agent_name": "agent1",
                        "agent_category": "test",
                    },
                }
            )

            # Mock internal methods that are called during workflow execution
            with (
                patch.object(
                    workflow, "_evaluate_response_quality", new_callable=AsyncMock
                ) as mock_eval,
                patch.object(
                    workflow, "_save_conversation_history", new_callable=AsyncMock
                ) as mock_save,
                patch.object(workflow, "_append_to_history", new_callable=AsyncMock) as mock_append,
            ):
                # Mock evaluation to return satisfactory=True so workflow completes
                mock_eval.return_value = {"satisfactory": True, "confidence": 0.9}

                result = await workflow.run(context=context_list)

                assert result["success"] is True
                # Context now includes conversation_history, so check query separately
                assert workflow._state.context.get("query") == "What is the weather?"
                assert "conversation_history" in workflow._state.context
                assert workflow._state.metadata["execution_mode"] == "parallel"

    @pytest.mark.asyncio
    async def test_run_invalid_context_raises_error(self):
        """Test workflow run with invalid context raises ValueError."""
        workflow = OrchestratorWorkflow()
        context = "not a dict"  # Invalid context type

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.logger = MagicMock()

            with pytest.raises(ValueError, match="Expected context to be a dict"):
                await workflow.run(context=context)

    @pytest.mark.asyncio
    async def test_run_with_cancellation(self):
        """Test workflow run handles cancellation correctly."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_plan = ["agent1", "agent2"]

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": True,
                    "response": {
                        "content": "Response",
                        "agent_name": "agent1",
                        "agent_category": "test",
                    },
                }
            )

            # Request cancellation before running
            workflow._state.cancellation_requested = True
            workflow._state.cancellation_reason = "User cancelled"

            result = await workflow.run(
                context=context,
                agent_plan=agent_plan,
            )

            assert result["success"] is False
            assert result["metadata"]["cancelled"] is True
            assert workflow._state.status == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_run_with_agent_failure(self):
        """Test workflow run handles agent failures."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_plan = ["agent1", "agent2"]

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    {
                        "success": True,
                        "response": {
                            "content": "Response from agent1",
                            "agent_name": "agent1",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": False,
                        "error": "Agent execution failed",
                        "agent_name": "agent2",
                    },
                ]
            )

            result = await workflow.run(
                context=context,
                agent_plan=agent_plan,
            )

            assert result["success"] is False
            assert workflow._state.status == WorkflowStatus.FAILED
            assert "failed" in workflow._state.error.lower()

    @pytest.mark.asyncio
    async def test_run_with_exception_raises(self):
        """Test workflow run raises exception on workflow-level error."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            # Make _determine_agent_plan raise an exception to test workflow-level error handling
            with patch.object(
                workflow,
                "_determine_agent_plan",
                side_effect=RuntimeError("Workflow error"),
            ):

                with pytest.raises(RuntimeError, match="Workflow error"):
                    await workflow.run(context=context, agent_plan=None)

                assert workflow._state.status == WorkflowStatus.FAILED
                assert workflow._state.error is not None
                assert "Workflow execution failed" in workflow._state.error


class TestOrchestratorWorkflowDetermineAgentPlan:
    """Test _determine_agent_plan method."""

    @pytest.mark.asyncio
    async def test_determine_agent_plan_success(self):
        """Test successful agent plan determination."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": True,
                    "response": {
                        "content": "Plan",
                        "agent_name": "orchestration",
                        "agent_category": "orchestration",
                        "metadata": {
                            "execution_plan": {
                                "agents": ["agent1", "agent2", "agent3"],
                            },
                        },
                    },
                }
            )

            plan, execution_mode = await workflow._determine_agent_plan(context)

            assert plan == ["agent1", "agent2", "agent3"]
            assert execution_mode is None  # Not in test mock response
            mock_workflow.execute_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_determine_agent_plan_orchestration_fails(self):
        """Test agent plan determination when orchestration fails."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": False,
                    "error": "Orchestration failed",
                }
            )

            plan, execution_mode = await workflow._determine_agent_plan(context)

            assert plan == []
            assert execution_mode is None
            mock_workflow.logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_determine_agent_plan_no_execution_plan(self):
        """Test agent plan determination when no execution plan in response."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": True,
                    "response": {
                        "content": "Plan",
                        "agent_name": "orchestration",
                        "agent_category": "orchestration",
                        "metadata": {},  # No execution_plan
                    },
                }
            )

            plan, execution_mode = await workflow._determine_agent_plan(context)

            assert plan == []
            assert execution_mode is None
            mock_workflow.logger.warning.assert_called()


class TestOrchestratorWorkflowExecuteSingleAgent:
    """Test _execute_single_agent method."""

    @pytest.mark.asyncio
    async def test_execute_single_agent_success(self):
        """Test successful single agent execution."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_name = "test_agent"

        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": True,
                    "response": {
                        "content": "Test response",
                        "agent_name": "test_agent",
                        "agent_category": "test",
                        "metadata": {},
                    },
                }
            )

            result = await workflow._execute_single_agent(agent_name, context)

            assert result["success"] is True
            assert agent_name in workflow._state.agent_executions
            exec_state = workflow._state.agent_executions[agent_name]
            assert exec_state.status == "completed"
            assert exec_state.started_at == mock_now
            assert exec_state.completed_at == mock_now
            assert len(workflow._state.agent_responses) == 1

    @pytest.mark.asyncio
    async def test_execute_single_agent_failure(self):
        """Test single agent execution failure."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_name = "test_agent"

        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": False,
                    "error": "Agent execution failed",
                    "agent_name": agent_name,
                }
            )

            result = await workflow._execute_single_agent(agent_name, context)

            assert result["success"] is False
            exec_state = workflow._state.agent_executions[agent_name]
            assert exec_state.status == "failed"
            assert exec_state.error == "Agent execution failed"

    @pytest.mark.asyncio
    async def test_execute_single_agent_exception(self):
        """Test single agent execution with exception."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_name = "test_agent"

        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(side_effect=RuntimeError("Activity error"))

            result = await workflow._execute_single_agent(agent_name, context)

            assert result["success"] is False
            assert "exception" in result["error"].lower()
            exec_state = workflow._state.agent_executions[agent_name]
            assert exec_state.status == "failed"

    @pytest.mark.asyncio
    async def test_execute_single_agent_updates_shared_data(self):
        """Test single agent execution updates shared data."""
        workflow = OrchestratorWorkflow()
        context = {"query": "What is the weather?"}
        agent_name = "test_agent"

        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.now.return_value = mock_now
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": True,
                    "response": {
                        "content": "Test response",
                        "agent_name": "test_agent",
                        "agent_category": "test",
                        "metadata": {
                            "shared_data": {"key1": "value1", "key2": "value2"},
                        },
                    },
                }
            )

            await workflow._execute_single_agent(agent_name, context)

            assert workflow._state.shared_data == {"key1": "value1", "key2": "value2"}


class TestOrchestratorWorkflowExecuteAgentsSequential:
    """Test _execute_agents_sequential method."""

    @pytest.mark.asyncio
    async def test_execute_agents_sequential_success(self):
        """Test successful sequential agent execution."""
        workflow = OrchestratorWorkflow()
        agent_plan = ["agent1", "agent2", "agent3"]
        initial_context = {"query": "What is the weather?"}

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    {
                        "success": True,
                        "response": {
                            "content": "Response 1",
                            "agent_name": "agent1",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": "Response 2",
                            "agent_name": "agent2",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": "Response 3",
                            "agent_name": "agent3",
                            "agent_category": "test",
                        },
                    },
                ]
            )

            results = await workflow._execute_agents_sequential(agent_plan, initial_context)

            assert len(results) == 3
            assert all(r["success"] for r in results)
            assert mock_workflow.execute_activity.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_agents_sequential_with_cancellation(self):
        """Test sequential execution stops on cancellation."""
        workflow = OrchestratorWorkflow()
        agent_plan = ["agent1", "agent2", "agent3"]
        initial_context = {"query": "What is the weather?"}

        # Set cancellation after first agent
        workflow._state.cancellation_requested = True

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.logger = MagicMock()
            mock_workflow.execute_activity = AsyncMock(
                return_value={
                    "success": True,
                    "response": {
                        "content": "Response",
                        "agent_name": "agent1",
                        "agent_category": "test",
                    },
                }
            )

            results = await workflow._execute_agents_sequential(agent_plan, initial_context)

            # Should stop after checking cancellation
            assert len(results) == 0
            mock_workflow.logger.info.assert_called()


class TestOrchestratorWorkflowExecuteAgentsParallel:
    """Test _execute_agents_parallel method."""

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_success(self):
        """Test successful parallel agent execution."""
        workflow = OrchestratorWorkflow()
        agent_plan = ["agent1", "agent2", "agent3"]
        context = {"query": "What is the weather?"}

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    {
                        "success": True,
                        "response": {
                            "content": "Response 1",
                            "agent_name": "agent1",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": "Response 2",
                            "agent_name": "agent2",
                            "agent_category": "test",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "content": "Response 3",
                            "agent_name": "agent3",
                            "agent_category": "test",
                        },
                    },
                ]
            )

            results = await workflow._execute_agents_parallel(agent_plan, context)

            assert len(results) == 3
            assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_with_cancellation(self):
        """Test parallel execution returns empty on cancellation."""
        workflow = OrchestratorWorkflow()
        agent_plan = ["agent1", "agent2"]
        context = {"query": "What is the weather?"}

        workflow._state.cancellation_requested = True

        results = await workflow._execute_agents_parallel(agent_plan, context)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_handles_exceptions(self):
        """Test parallel execution handles exceptions correctly."""
        workflow = OrchestratorWorkflow()
        agent_plan = ["agent1", "agent2"]
        context = {"query": "What is the weather?"}

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    {
                        "success": True,
                        "response": {
                            "content": "Response 1",
                            "agent_name": "agent1",
                            "agent_category": "test",
                        },
                    },
                    RuntimeError("Agent error"),
                ]
            )

            results = await workflow._execute_agents_parallel(agent_plan, context)

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is False
            assert "error" in results[1]


class TestOrchestratorWorkflowQueries:
    """Test workflow query methods."""

    def test_query_status(self):
        """Test query_status query method."""
        workflow = OrchestratorWorkflow()
        workflow._state.status = WorkflowStatus.RUNNING
        workflow._state.started_at = datetime(2024, 1, 1, 12, 0, 0)
        workflow._state.metadata = {"key": "value"}

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info

            result = workflow.query_status()

            assert isinstance(result, WorkflowStatusQueryResult)
            assert result.workflow_id == "test-workflow-id"
            assert result.status == WorkflowStatus.RUNNING
            assert result.started_at == workflow._state.started_at
            assert result.metadata == {"key": "value"}

    def test_query_progress(self):
        """Test query_progress query method."""
        workflow = OrchestratorWorkflow()
        workflow._state.status = WorkflowStatus.RUNNING

        # Add some agent executions
        exec_state1 = AgentExecutionState(
            agent_name="agent1",
            agent_category="test",
            status="completed",
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 1, 0),
        )
        exec_state2 = AgentExecutionState(
            agent_name="agent2",
            agent_category="test",
            status="running",
            started_at=datetime(2024, 1, 1, 12, 1, 0),
        )
        workflow._state.agent_executions = {
            "agent1": exec_state1,
            "agent2": exec_state2,
        }

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 2, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now

            result = workflow.query_progress()

            assert isinstance(result, WorkflowProgressQueryResult)
            assert result.workflow_id == "test-workflow-id"
            assert result.total_agents == 2
            assert result.completed_agents == 1
            assert result.running_agents == 1
            assert result.failed_agents == 0
            assert result.progress_percentage == 50.0
            assert result.current_step == "Executing agent2"

    def test_query_state(self):
        """Test query_state query method."""
        workflow = OrchestratorWorkflow()
        workflow._state.context = {"query": "test"}
        workflow._state.agent_responses = [{"response": "data"}]
        workflow._state.shared_data = {"key": "value"}
        workflow._state.execution_history = [{"event": "execution"}]

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info

            result = workflow.query_state()

            assert isinstance(result, WorkflowStateQueryResult)
            assert result.workflow_id == "test-workflow-id"
            assert result.context == {"query": "test"}
            assert result.agent_responses == [{"response": "data"}]
            assert result.shared_data == {"key": "value"}
            assert result.execution_history == [{"event": "execution"}]

    def test_query_agent_status_found(self):
        """Test query_agent_status for existing agent."""
        workflow = OrchestratorWorkflow()
        exec_state = AgentExecutionState(
            agent_name="test_agent",
            agent_category="test",
            status="completed",
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 1, 0),
            response={"content": "response"},
        )
        workflow._state.agent_executions["test_agent"] = exec_state

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info

            result = workflow.query_agent_status("test_agent")

            assert isinstance(result, AgentStatusQueryResult)
            assert result.agent_name == "test_agent"
            assert result.status == "completed"
            assert result.response == {"content": "response"}

    def test_query_agent_status_not_found(self):
        """Test query_agent_status for non-existent agent."""
        workflow = OrchestratorWorkflow()

        mock_workflow_info = MagicMock()
        mock_workflow_info.workflow_id = "test-workflow-id"
        mock_now = datetime(2024, 1, 1, 12, 0, 0)

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.info.return_value = mock_workflow_info
            mock_workflow.now.return_value = mock_now

            result = workflow.query_agent_status("nonexistent_agent")

            assert isinstance(result, AgentStatusQueryResult)
            assert result.agent_name == "nonexistent_agent"
            assert result.status == "not_found"
            assert result.error is not None


class TestOrchestratorWorkflowSignals:
    """Test workflow signal handlers."""

    @pytest.mark.asyncio
    async def test_handle_cancellation(self):
        """Test handle_cancellation signal handler."""
        workflow = OrchestratorWorkflow()
        signal = CancellationSignal(
            reason="User requested cancellation",
            requested_by="user123",
        )

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.logger = MagicMock()

            await workflow.handle_cancellation(signal)

            assert workflow._state.cancellation_requested is True
            assert workflow._state.cancellation_reason == "User requested cancellation"
            assert workflow._state.metadata["cancellation_requested_by"] == "user123"
            assert "cancellation_timestamp" in workflow._state.metadata
            mock_workflow.logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_user_input(self):
        """Test handle_user_input signal handler."""
        workflow = OrchestratorWorkflow()
        signal = UserInputSignal(
            input_text="Additional information",
            user_id="user123",
        )

        with patch("src.temporal.workflows.orchestrator.workflow") as mock_workflow:
            mock_workflow.logger = MagicMock()

            await workflow.handle_user_input(signal)

            assert len(workflow._state.user_inputs) == 1
            assert "user_inputs" in workflow._state.context
            assert len(workflow._state.context["user_inputs"]) == 1
            mock_workflow.logger.info.assert_called_once()
