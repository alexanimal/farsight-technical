"""Unit tests for the Temporal client module.

This module tests the TemporalClient class and module-level functions,
including connection management, workflow operations, signal sending,
query execution, and error handling.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.temporal.client import (
    DEFAULT_TASK_QUEUE,
    DEFAULT_TEMPORAL_ADDRESS,
    DEFAULT_TEMPORAL_NAMESPACE,
    TemporalClient,
    close_client,
    get_client,
    set_client,
)
from src.temporal.queries import (
    AgentStatusQueryResult,
    WorkflowProgressQueryResult,
    WorkflowStateQueryResult,
    WorkflowStatus,
    WorkflowStatusQueryResult,
)
from src.temporal.signals import CancellationSignal, UserInputSignal


@pytest.fixture
def mock_temporal_client():
    """Create a mock Temporal Client instance."""
    client = MagicMock()
    client.get_workflow_handle = MagicMock()
    client.start_workflow = AsyncMock()
    return client


@pytest.fixture
def mock_workflow_handle():
    """Create a mock workflow handle."""
    handle = MagicMock()
    handle.id = "workflow-123"
    handle.signal = AsyncMock()
    handle.query = AsyncMock()
    handle.result = AsyncMock()
    return handle


@pytest.fixture
def temporal_client(mock_temporal_client):
    """Create a TemporalClient instance with mocked Temporal client."""
    return TemporalClient(client=mock_temporal_client)


class TestTemporalClientConstants:
    """Test module-level constants."""

    def test_default_temporal_address(self):
        """Test DEFAULT_TEMPORAL_ADDRESS constant."""
        assert DEFAULT_TEMPORAL_ADDRESS == "localhost:7233"
        assert isinstance(DEFAULT_TEMPORAL_ADDRESS, str)

    def test_default_temporal_namespace(self):
        """Test DEFAULT_TEMPORAL_NAMESPACE constant."""
        assert DEFAULT_TEMPORAL_NAMESPACE == "default"
        assert isinstance(DEFAULT_TEMPORAL_NAMESPACE, str)

    def test_default_task_queue(self):
        """Test DEFAULT_TASK_QUEUE constant."""
        assert DEFAULT_TASK_QUEUE == "orchestrator-task-queue"
        assert isinstance(DEFAULT_TASK_QUEUE, str)


class TestTemporalClientInitialization:
    """Test TemporalClient initialization."""

    def test_init_with_client(self, mock_temporal_client):
        """Test initialization with pre-configured client."""
        client = TemporalClient(client=mock_temporal_client)
        assert client._client == mock_temporal_client
        assert client._client_initialized is True
        assert client._temporal_address == DEFAULT_TEMPORAL_ADDRESS
        assert client._temporal_namespace == DEFAULT_TEMPORAL_NAMESPACE
        assert client._task_queue == DEFAULT_TASK_QUEUE

    def test_init_without_client(self):
        """Test initialization without client."""
        client = TemporalClient()
        assert client._client is None
        assert client._client_initialized is False
        assert client._temporal_address == DEFAULT_TEMPORAL_ADDRESS
        assert client._temporal_namespace == DEFAULT_TEMPORAL_NAMESPACE
        assert client._task_queue == DEFAULT_TASK_QUEUE

    def test_init_with_custom_address(self):
        """Test initialization with custom temporal address."""
        client = TemporalClient(temporal_address="custom:7233")
        assert client._temporal_address == "custom:7233"
        assert client._client is None

    def test_init_with_custom_namespace(self):
        """Test initialization with custom namespace."""
        client = TemporalClient(temporal_namespace="custom-namespace")
        assert client._temporal_namespace == "custom-namespace"
        assert client._client is None

    def test_init_with_custom_task_queue(self):
        """Test initialization with custom task queue."""
        client = TemporalClient(task_queue="custom-queue")
        assert client._task_queue == "custom-queue"
        assert client._client is None

    def test_init_with_all_custom_params(self):
        """Test initialization with all custom parameters."""
        mock_client = MagicMock()
        client = TemporalClient(
            client=mock_client,
            temporal_address="custom:7233",
            temporal_namespace="custom-ns",
            task_queue="custom-queue",
        )
        assert client._client == mock_client
        assert client._temporal_address == "custom:7233"
        assert client._temporal_namespace == "custom-ns"
        assert client._task_queue == "custom-queue"


class TestTemporalClientConnect:
    """Test TemporalClient.connect() method."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        mock_client = MagicMock()
        with patch("src.temporal.client.Client") as mock_client_class:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            client = TemporalClient()
            await client.connect()
            assert client._client == mock_client
            assert client._client_initialized is True
            mock_client_class.connect.assert_called_once_with(
                DEFAULT_TEMPORAL_ADDRESS,
                namespace=DEFAULT_TEMPORAL_NAMESPACE,
            )

    @pytest.mark.asyncio
    async def test_connect_with_custom_params(self):
        """Test connection with custom parameters."""
        mock_client = MagicMock()
        with patch("src.temporal.client.Client") as mock_client_class:
            mock_client_class.connect = AsyncMock(return_value=mock_client)
            client = TemporalClient(
                temporal_address="custom:7233",
                temporal_namespace="custom-ns",
            )
            await client.connect()
            mock_client_class.connect.assert_called_once_with(
                "custom:7233",
                namespace="custom-ns",
            )

    @pytest.mark.asyncio
    async def test_connect_when_already_connected(self, temporal_client):
        """Test connect when client is already connected."""
        # Client is already initialized via fixture
        original_client = temporal_client._client
        await temporal_client.connect()
        # Should not create a new client
        assert temporal_client._client == original_client

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        with patch("src.temporal.client.Client") as mock_client_class:
            mock_client_class.connect = AsyncMock(side_effect=Exception("Connection failed"))
            client = TemporalClient()
            with pytest.raises(RuntimeError) as exc_info:
                await client.connect()
            assert "Failed to connect to Temporal" in str(exc_info.value)
            assert "Connection failed" in str(exc_info.value)
            assert client._client is None
            assert client._client_initialized is False

    @pytest.mark.asyncio
    async def test_connect_with_preconfigured_client(self, temporal_client):
        """Test connect when client is pre-configured."""
        original_client = temporal_client._client
        await temporal_client.connect()
        # Should not attempt to create new connection
        assert temporal_client._client == original_client


class TestTemporalClientClose:
    """Test TemporalClient.close() method."""

    @pytest.mark.asyncio
    async def test_close_success(self, temporal_client):
        """Test successful close."""
        await temporal_client.close()
        assert temporal_client._client is None
        assert temporal_client._client_initialized is False

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self):
        """Test close when not connected."""
        client = TemporalClient()
        await client.close()
        assert client._client is None
        assert client._client_initialized is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self, temporal_client):
        """Test that close can be called multiple times safely."""
        await temporal_client.close()
        await temporal_client.close()
        assert temporal_client._client is None
        assert temporal_client._client_initialized is False


class TestTemporalClientEnsureConnected:
    """Test TemporalClient._ensure_connected() method."""

    def test_ensure_connected_success(self, temporal_client):
        """Test _ensure_connected when client is connected."""
        client = temporal_client._ensure_connected()
        assert client == temporal_client._client

    def test_ensure_connected_not_initialized(self):
        """Test _ensure_connected when client is not initialized."""
        client = TemporalClient()
        with pytest.raises(RuntimeError) as exc_info:
            client._ensure_connected()
        assert "not connected" in str(exc_info.value).lower()

    def test_ensure_connected_client_none(self):
        """Test _ensure_connected when client is None."""
        client = TemporalClient()
        client._client = None
        client._client_initialized = False
        with pytest.raises(RuntimeError) as exc_info:
            client._ensure_connected()
        assert "not connected" in str(exc_info.value).lower()


class TestTemporalClientStartWorkflow:
    """Test TemporalClient.start_workflow() method."""

    @pytest.mark.asyncio
    async def test_start_workflow_success(self, temporal_client, mock_workflow_handle):
        """Test successful workflow start."""
        temporal_client._client.start_workflow = AsyncMock(return_value=mock_workflow_handle)
        context = {"query": "test query"}
        workflow_id = await temporal_client.start_workflow(context)
        assert workflow_id == "workflow-123"
        temporal_client._client.start_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_workflow_with_agent_plan(self, temporal_client, mock_workflow_handle):
        """Test workflow start with agent plan."""
        temporal_client._client.start_workflow = AsyncMock(return_value=mock_workflow_handle)
        context = {"query": "test query"}
        agent_plan = ["agent1", "agent2"]
        await temporal_client.start_workflow(context, agent_plan=agent_plan)
        call_kwargs = temporal_client._client.start_workflow.call_args
        assert call_kwargs[1]["args"][1] == agent_plan

    @pytest.mark.asyncio
    async def test_start_workflow_with_execution_mode(self, temporal_client, mock_workflow_handle):
        """Test workflow start with execution mode."""
        temporal_client._client.start_workflow = AsyncMock(return_value=mock_workflow_handle)
        context = {"query": "test query"}
        await temporal_client.start_workflow(context, execution_mode="parallel")
        call_kwargs = temporal_client._client.start_workflow.call_args
        assert call_kwargs[1]["args"][2] == "parallel"

    @pytest.mark.asyncio
    async def test_start_workflow_with_workflow_id(self, temporal_client, mock_workflow_handle):
        """Test workflow start with custom workflow ID."""
        temporal_client._client.start_workflow = AsyncMock(return_value=mock_workflow_handle)
        context = {"query": "test query"}
        await temporal_client.start_workflow(context, workflow_id="custom-workflow-id")
        call_kwargs = temporal_client._client.start_workflow.call_args
        assert call_kwargs[1]["id"] == "custom-workflow-id"

    @pytest.mark.asyncio
    async def test_start_workflow_with_timeout(self, temporal_client, mock_workflow_handle):
        """Test workflow start with timeout."""
        temporal_client._client.start_workflow = AsyncMock(return_value=mock_workflow_handle)
        context = {"query": "test query"}
        timeout = timedelta(minutes=30)
        await temporal_client.start_workflow(context, workflow_timeout=timeout)
        call_kwargs = temporal_client._client.start_workflow.call_args
        assert call_kwargs[1]["execution_timeout"] == timeout

    @pytest.mark.asyncio
    async def test_start_workflow_missing_query(self, temporal_client):
        """Test workflow start with missing query in context."""
        context = {"conversation_id": "conv-123"}
        with pytest.raises(ValueError) as exc_info:
            await temporal_client.start_workflow(context)
        assert "query" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_start_workflow_not_connected(self):
        """Test workflow start when not connected."""
        client = TemporalClient()
        context = {"query": "test query"}
        with pytest.raises(RuntimeError) as exc_info:
            await client.start_workflow(context)
        assert "not connected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_start_workflow_failure(self, temporal_client, mock_workflow_handle):
        """Test workflow start failure."""
        temporal_client._client.start_workflow = AsyncMock(
            side_effect=Exception("Workflow start failed")
        )
        context = {"query": "test query"}
        with pytest.raises(Exception) as exc_info:
            await temporal_client.start_workflow(context)
        assert "Workflow start failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_start_workflow_uses_correct_task_queue(
        self, temporal_client, mock_workflow_handle
    ):
        """Test that workflow uses correct task queue."""
        temporal_client._client.start_workflow = AsyncMock(return_value=mock_workflow_handle)
        context = {"query": "test query"}
        await temporal_client.start_workflow(context)
        call_kwargs = temporal_client._client.start_workflow.call_args
        assert call_kwargs[1]["task_queue"] == DEFAULT_TASK_QUEUE


class TestTemporalClientSendCancellationSignal:
    """Test TemporalClient.send_cancellation_signal() method."""

    @pytest.mark.asyncio
    async def test_send_cancellation_signal_success(self, temporal_client, mock_workflow_handle):
        """Test successful cancellation signal."""
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        await temporal_client.send_cancellation_signal(
            "workflow-123", reason="User requested", requested_by="user-1"
        )
        temporal_client._client.get_workflow_handle.assert_called_once_with("workflow-123")
        mock_workflow_handle.signal.assert_called_once()
        call_args = mock_workflow_handle.signal.call_args
        assert call_args[0][0] == "cancellation"
        assert isinstance(call_args[0][1], CancellationSignal)
        assert call_args[0][1].reason == "User requested"
        assert call_args[0][1].requested_by == "user-1"

    @pytest.mark.asyncio
    async def test_send_cancellation_signal_minimal(self, temporal_client, mock_workflow_handle):
        """Test cancellation signal with minimal parameters."""
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        await temporal_client.send_cancellation_signal("workflow-123")
        mock_workflow_handle.signal.assert_called_once()
        call_args = mock_workflow_handle.signal.call_args
        signal = call_args[0][1]
        assert signal.reason is None
        assert signal.requested_by is None

    @pytest.mark.asyncio
    async def test_send_cancellation_signal_workflow_not_found(self, temporal_client):
        """Test cancellation signal when workflow not found."""
        temporal_client._client.get_workflow_handle = MagicMock(
            side_effect=Exception("Workflow not found")
        )
        with pytest.raises(RuntimeError) as exc_info:
            await temporal_client.send_cancellation_signal("workflow-123")
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_send_cancellation_signal_not_connected(self):
        """Test cancellation signal when not connected."""
        client = TemporalClient()
        with pytest.raises(RuntimeError) as exc_info:
            await client.send_cancellation_signal("workflow-123")
        assert "not connected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_send_cancellation_signal_other_error(
        self, temporal_client, mock_workflow_handle
    ):
        """Test cancellation signal with other error."""
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.signal = AsyncMock(side_effect=Exception("Network error"))
        with pytest.raises(Exception) as exc_info:
            await temporal_client.send_cancellation_signal("workflow-123")
        assert "Network error" in str(exc_info.value)


class TestTemporalClientSendUserInputSignal:
    """Test TemporalClient.send_user_input_signal() method."""

    @pytest.mark.asyncio
    async def test_send_user_input_signal_success(self, temporal_client, mock_workflow_handle):
        """Test successful user input signal."""
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        await temporal_client.send_user_input_signal(
            "workflow-123",
            input_text="Additional information",
            input_type="clarification",
            user_id="user-1",
            conversation_id="conv-1",
            metadata={"source": "web"},
        )
        mock_workflow_handle.signal.assert_called_once()
        call_args = mock_workflow_handle.signal.call_args
        assert call_args[0][0] == "user_input"
        assert isinstance(call_args[0][1], UserInputSignal)
        signal = call_args[0][1]
        assert signal.input_text == "Additional information"
        assert signal.input_type == "clarification"
        assert signal.user_id == "user-1"
        assert signal.conversation_id == "conv-1"
        assert signal.metadata == {"source": "web"}

    @pytest.mark.asyncio
    async def test_send_user_input_signal_minimal(self, temporal_client, mock_workflow_handle):
        """Test user input signal with minimal parameters."""
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        await temporal_client.send_user_input_signal("workflow-123", input_text="Hello")
        mock_workflow_handle.signal.assert_called_once()
        call_args = mock_workflow_handle.signal.call_args
        signal = call_args[0][1]
        assert signal.input_text == "Hello"
        assert signal.input_type is None
        assert signal.user_id is None
        assert signal.conversation_id is None
        assert signal.metadata == {}

    @pytest.mark.asyncio
    async def test_send_user_input_signal_with_none_metadata(
        self, temporal_client, mock_workflow_handle
    ):
        """Test user input signal with None metadata (should become empty dict)."""
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        await temporal_client.send_user_input_signal(
            "workflow-123", input_text="test", metadata=None
        )
        call_args = mock_workflow_handle.signal.call_args
        signal = call_args[0][1]
        assert signal.metadata == {}

    @pytest.mark.asyncio
    async def test_send_user_input_signal_workflow_not_found(self, temporal_client):
        """Test user input signal when workflow not found."""
        temporal_client._client.get_workflow_handle = MagicMock(
            side_effect=Exception("Workflow not found")
        )
        with pytest.raises(RuntimeError) as exc_info:
            await temporal_client.send_user_input_signal("workflow-123", input_text="test")
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_send_user_input_signal_not_connected(self):
        """Test user input signal when not connected."""
        client = TemporalClient()
        with pytest.raises(RuntimeError) as exc_info:
            await client.send_user_input_signal("workflow-123", input_text="test")
        assert "not connected" in str(exc_info.value).lower()


class TestTemporalClientQueryWorkflowStatus:
    """Test TemporalClient.query_workflow_status() method."""

    @pytest.mark.asyncio
    async def test_query_workflow_status_success(self, temporal_client, mock_workflow_handle):
        """Test successful workflow status query."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        query_result = WorkflowStatusQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            started_at=started_at,
        )
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=query_result)
        result = await temporal_client.query_workflow_status("workflow-123")
        assert isinstance(result, WorkflowStatusQueryResult)
        assert result.workflow_id == "workflow-123"
        assert result.status == WorkflowStatus.RUNNING
        mock_workflow_handle.query.assert_called_once_with("workflow_status")

    @pytest.mark.asyncio
    async def test_query_workflow_status_dict_result(self, temporal_client, mock_workflow_handle):
        """Test workflow status query with dict result."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        dict_result = {
            "workflow_id": "workflow-123",
            "status": "running",
            "started_at": started_at.isoformat(),
        }
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=dict_result)
        result = await temporal_client.query_workflow_status("workflow-123")
        assert isinstance(result, WorkflowStatusQueryResult)
        assert result.workflow_id == "workflow-123"

    @pytest.mark.asyncio
    async def test_query_workflow_status_workflow_not_found(self, temporal_client):
        """Test workflow status query when workflow not found."""
        temporal_client._client.get_workflow_handle = MagicMock(
            side_effect=Exception("Workflow not found")
        )
        with pytest.raises(RuntimeError) as exc_info:
            await temporal_client.query_workflow_status("workflow-123")
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_workflow_status_not_connected(self):
        """Test workflow status query when not connected."""
        client = TemporalClient()
        with pytest.raises(RuntimeError) as exc_info:
            await client.query_workflow_status("workflow-123")
        assert "not connected" in str(exc_info.value).lower()


class TestTemporalClientQueryWorkflowProgress:
    """Test TemporalClient.query_workflow_progress() method."""

    @pytest.mark.asyncio
    async def test_query_workflow_progress_success(self, temporal_client, mock_workflow_handle):
        """Test successful workflow progress query."""
        query_result = WorkflowProgressQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            total_agents=3,
            completed_agents=1,
        )
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=query_result)
        result = await temporal_client.query_workflow_progress("workflow-123")
        assert isinstance(result, WorkflowProgressQueryResult)
        assert result.workflow_id == "workflow-123"
        assert result.total_agents == 3
        mock_workflow_handle.query.assert_called_once_with("workflow_progress")

    @pytest.mark.asyncio
    async def test_query_workflow_progress_dict_result(self, temporal_client, mock_workflow_handle):
        """Test workflow progress query with dict result."""
        dict_result = {
            "workflow_id": "workflow-123",
            "status": "running",
            "total_agents": 2,
        }
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=dict_result)
        result = await temporal_client.query_workflow_progress("workflow-123")
        assert isinstance(result, WorkflowProgressQueryResult)
        assert result.total_agents == 2

    @pytest.mark.asyncio
    async def test_query_workflow_progress_workflow_not_found(self, temporal_client):
        """Test workflow progress query when workflow not found."""
        temporal_client._client.get_workflow_handle = MagicMock(
            side_effect=Exception("Workflow not found")
        )
        with pytest.raises(RuntimeError) as exc_info:
            await temporal_client.query_workflow_progress("workflow-123")
        assert "not found" in str(exc_info.value).lower()


class TestTemporalClientQueryWorkflowState:
    """Test TemporalClient.query_workflow_state() method."""

    @pytest.mark.asyncio
    async def test_query_workflow_state_success(self, temporal_client, mock_workflow_handle):
        """Test successful workflow state query."""
        query_result = WorkflowStateQueryResult(
            workflow_id="workflow-123",
            status=WorkflowStatus.RUNNING,
            context={"query": "test"},
        )
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=query_result)
        result = await temporal_client.query_workflow_state("workflow-123")
        assert isinstance(result, WorkflowStateQueryResult)
        assert result.workflow_id == "workflow-123"
        mock_workflow_handle.query.assert_called_once_with("workflow_state")

    @pytest.mark.asyncio
    async def test_query_workflow_state_dict_result(self, temporal_client, mock_workflow_handle):
        """Test workflow state query with dict result."""
        dict_result = {
            "workflow_id": "workflow-123",
            "status": "running",
            "context": {"query": "test"},
        }
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=dict_result)
        result = await temporal_client.query_workflow_state("workflow-123")
        assert isinstance(result, WorkflowStateQueryResult)
        assert result.context == {"query": "test"}

    @pytest.mark.asyncio
    async def test_query_workflow_state_workflow_not_found(self, temporal_client):
        """Test workflow state query when workflow not found."""
        temporal_client._client.get_workflow_handle = MagicMock(
            side_effect=Exception("Workflow not found")
        )
        with pytest.raises(RuntimeError) as exc_info:
            await temporal_client.query_workflow_state("workflow-123")
        assert "not found" in str(exc_info.value).lower()


class TestTemporalClientQueryAgentStatus:
    """Test TemporalClient.query_agent_status() method."""

    @pytest.mark.asyncio
    async def test_query_agent_status_success(self, temporal_client, mock_workflow_handle):
        """Test successful agent status query."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        query_result = AgentStatusQueryResult(
            workflow_id="workflow-123",
            agent_name="agent1",
            agent_category="test",
            status="running",
            started_at=started_at,
        )
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=query_result)
        result = await temporal_client.query_agent_status("workflow-123", "agent1")
        assert isinstance(result, AgentStatusQueryResult)
        assert result.agent_name == "agent1"
        mock_workflow_handle.query.assert_called_once_with("agent_status", "agent1")

    @pytest.mark.asyncio
    async def test_query_agent_status_dict_result(self, temporal_client, mock_workflow_handle):
        """Test agent status query with dict result."""
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        dict_result = {
            "workflow_id": "workflow-123",
            "agent_name": "agent1",
            "agent_category": "test",
            "status": "running",
            "started_at": started_at.isoformat(),
        }
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.query = AsyncMock(return_value=dict_result)
        result = await temporal_client.query_agent_status("workflow-123", "agent1")
        assert isinstance(result, AgentStatusQueryResult)
        assert result.agent_name == "agent1"

    @pytest.mark.asyncio
    async def test_query_agent_status_workflow_not_found(self, temporal_client):
        """Test agent status query when workflow not found."""
        temporal_client._client.get_workflow_handle = MagicMock(
            side_effect=Exception("Workflow not found")
        )
        with pytest.raises(RuntimeError) as exc_info:
            await temporal_client.query_agent_status("workflow-123", "agent1")
        assert "not found" in str(exc_info.value).lower()


class TestTemporalClientGetWorkflowResult:
    """Test TemporalClient.get_workflow_result() method."""

    @pytest.mark.asyncio
    async def test_get_workflow_result_success(self, temporal_client, mock_workflow_handle):
        """Test successful workflow result retrieval."""
        workflow_result = {"success": True, "final_response": "result"}
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.result = AsyncMock(return_value=workflow_result)
        result = await temporal_client.get_workflow_result("workflow-123")
        assert result == workflow_result
        mock_workflow_handle.result.assert_called_once_with(rpc_timeout=None)

    @pytest.mark.asyncio
    async def test_get_workflow_result_with_timeout(self, temporal_client, mock_workflow_handle):
        """Test workflow result retrieval with timeout."""
        workflow_result = {"success": True}
        temporal_client._client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)
        mock_workflow_handle.result = AsyncMock(return_value=workflow_result)
        timeout = timedelta(seconds=30)
        await temporal_client.get_workflow_result("workflow-123", timeout=timeout)
        mock_workflow_handle.result.assert_called_once_with(rpc_timeout=timeout)

    @pytest.mark.asyncio
    async def test_get_workflow_result_workflow_not_found(self, temporal_client):
        """Test workflow result retrieval when workflow not found."""
        temporal_client._client.get_workflow_handle = MagicMock(
            side_effect=Exception("Workflow not found")
        )
        with pytest.raises(RuntimeError) as exc_info:
            await temporal_client.get_workflow_result("workflow-123")
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_workflow_result_not_connected(self):
        """Test workflow result retrieval when not connected."""
        client = TemporalClient()
        with pytest.raises(RuntimeError) as exc_info:
            await client.get_workflow_result("workflow-123")
        assert "not connected" in str(exc_info.value).lower()


class TestModuleLevelFunctions:
    """Test module-level functions."""

    @pytest.mark.asyncio
    async def test_get_client_creates_new_client(self):
        """Test get_client creates a new client when none exists."""
        with patch("src.temporal.client._default_client", None):
            with patch("src.temporal.client.TemporalClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.connect = AsyncMock()
                mock_client_class.return_value = mock_client
                client = await get_client()
                assert client == mock_client
                mock_client_class.assert_called_once_with(
                    temporal_address=DEFAULT_TEMPORAL_ADDRESS,
                    temporal_namespace=DEFAULT_TEMPORAL_NAMESPACE,
                    task_queue=DEFAULT_TASK_QUEUE,
                )
                mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_returns_existing_client(self):
        """Test get_client returns existing client."""
        mock_client = MagicMock()
        with patch("src.temporal.client._default_client", mock_client):
            client = await get_client()
            assert client == mock_client

    @pytest.mark.asyncio
    async def test_get_client_with_custom_params(self):
        """Test get_client with custom parameters."""
        with patch("src.temporal.client._default_client", None):
            with patch("src.temporal.client.TemporalClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.connect = AsyncMock()
                mock_client_class.return_value = mock_client
                client = await get_client(
                    temporal_address="custom:7233",
                    temporal_namespace="custom-ns",
                    task_queue="custom-queue",
                )
                mock_client_class.assert_called_once_with(
                    temporal_address="custom:7233",
                    temporal_namespace="custom-ns",
                    task_queue="custom-queue",
                )

    def test_set_client(self):
        """Test set_client function."""
        mock_client = MagicMock()
        with patch("src.temporal.client._default_client", None):
            set_client(mock_client)
            from src.temporal.client import _default_client

            assert _default_client == mock_client

    @pytest.mark.asyncio
    async def test_close_client_success(self):
        """Test close_client function."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        with patch("src.temporal.client._default_client", mock_client):
            await close_client()
            mock_client.close.assert_called_once()
            from src.temporal.client import _default_client

            assert _default_client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self):
        """Test close_client when no client exists."""
        with patch("src.temporal.client._default_client", None):
            await close_client()
            # Should not raise an error

    @pytest.mark.asyncio
    async def test_get_client_singleton_pattern(self):
        """Test that get_client follows singleton pattern."""
        with patch("src.temporal.client._default_client", None):
            with patch("src.temporal.client.TemporalClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.connect = AsyncMock()
                mock_client_class.return_value = mock_client
                client1 = await get_client()
                client2 = await get_client()
                assert client1 == client2
                # Should only create client once
                assert mock_client_class.call_count == 1
                assert mock_client.connect.call_count == 1
