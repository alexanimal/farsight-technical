"""Unit tests for the tasks router module.

This module tests the task router endpoints, including task creation,
state queries, event streaming, signal sending, and cancellation.
All external dependencies are mocked to ensure unit test isolation.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.api.routers.tasks import (
    CancelTaskRequest,
    CancelTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    TaskSignalRequest,
    TaskSignalResponse,
    TaskStateResponse,
    cancel_task,
    create_task,
    get_task_state,
    send_task_signal,
    stream_task_events,
)
from src.temporal.queries import (
    AgentExecutionStatus,
    WorkflowProgressQueryResult,
    WorkflowStateQueryResult,
    WorkflowStatus,
    WorkflowStatusQueryResult,
)


@pytest.fixture
def mock_temporal_client():
    """Create a mock TemporalClient instance."""
    client = MagicMock()
    client.start_workflow = AsyncMock()
    client.query_workflow_status = AsyncMock()
    client.query_workflow_progress = AsyncMock()
    client.query_workflow_state = AsyncMock()
    client.send_user_input_signal = AsyncMock()
    client.send_cancellation_signal = AsyncMock()
    return client


@pytest.fixture
def mock_verify_api_key():
    """Create a mock verify_api_key dependency."""
    return MagicMock(return_value="test-api-key")


@pytest.fixture
def sample_create_task_request():
    """Create a sample CreateTaskRequest."""
    return CreateTaskRequest(
        query="Test query",
        conversation_id="conv-123",
        user_id="user-456",
        agent_plan=["agent1", "agent2"],
        execution_mode="sequential",
        metadata={"key": "value"},
    )


@pytest.fixture
def sample_workflow_status_result():
    """Create a sample WorkflowStatusQueryResult."""
    return WorkflowStatusQueryResult(
        workflow_id="workflow-123",
        status=WorkflowStatus.RUNNING,
        started_at=datetime.now(),
        completed_at=None,
        error=None,
        metadata={},
    )


@pytest.fixture
def sample_workflow_progress_result():
    """Create a sample WorkflowProgressQueryResult."""
    return WorkflowProgressQueryResult(
        workflow_id="workflow-123",
        status=WorkflowStatus.RUNNING,
        progress_percentage=50.0,
        completed_agents=1,
        total_agents=2,
        agent_statuses=[
            AgentExecutionStatus(
                agent_name="agent1",
                agent_category="test",
                status="completed",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error=None,
                metadata={},
            )
        ],
    )


@pytest.fixture
def sample_workflow_state_result():
    """Create a sample WorkflowStateQueryResult."""
    return WorkflowStateQueryResult(
        workflow_id="workflow-123",
        status=WorkflowStatus.RUNNING,
        context={"query": "Test query"},
        shared_data={},
        agent_responses=[],
        execution_history=[],
        metadata={},
    )


class TestCreateTask:
    """Test create_task endpoint."""

    @pytest.mark.asyncio
    async def test_create_task_success(
        self, mock_temporal_client, sample_create_task_request
    ):
        """Test successful task creation."""
        workflow_id = "workflow-123"
        mock_temporal_client.start_workflow.return_value = workflow_id

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = mock_temporal_client

            response = await create_task(
                request=sample_create_task_request,
                api_key="test-api-key",
            )

            assert isinstance(response, CreateTaskResponse)
            assert response.task_id == workflow_id
            assert response.status == "pending"
            assert response.message == "Task created successfully"

            # Verify workflow was started with correct context
            mock_temporal_client.start_workflow.assert_called_once()
            call_kwargs = mock_temporal_client.start_workflow.call_args[1]
            assert call_kwargs["context"]["query"] == sample_create_task_request.query
            assert (
                call_kwargs["context"]["conversation_id"]
                == sample_create_task_request.conversation_id
            )
            assert call_kwargs["context"]["user_id"] == sample_create_task_request.user_id
            assert call_kwargs["agent_plan"] == sample_create_task_request.agent_plan
            assert call_kwargs["execution_mode"] == sample_create_task_request.execution_mode

    @pytest.mark.asyncio
    async def test_create_task_with_minimal_request(self, mock_temporal_client):
        """Test task creation with minimal request (only required fields)."""
        workflow_id = "workflow-456"
        mock_temporal_client.start_workflow.return_value = workflow_id

        minimal_request = CreateTaskRequest(query="Minimal query")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = mock_temporal_client

            response = await create_task(request=minimal_request, api_key="test-api-key")

            assert response.task_id == workflow_id
            assert response.status == "pending"

            # Verify context has defaults
            call_kwargs = mock_temporal_client.start_workflow.call_args[1]
            assert call_kwargs["context"]["query"] == "Minimal query"
            assert call_kwargs["context"]["conversation_id"] is None
            assert call_kwargs["context"]["user_id"] is None
            assert call_kwargs["agent_plan"] is None
            assert call_kwargs["execution_mode"] == "sequential"

    @pytest.mark.asyncio
    async def test_create_task_logs_info(self, mock_temporal_client, sample_create_task_request):
        """Test that task creation logs info message."""
        workflow_id = "workflow-789"
        mock_temporal_client.start_workflow.return_value = workflow_id

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            await create_task(request=sample_create_task_request, api_key="test-api-key")

            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert workflow_id in log_message
            assert sample_create_task_request.query[:100] in log_message

    @pytest.mark.asyncio
    async def test_create_task_handles_workflow_error(
        self, mock_temporal_client, sample_create_task_request
    ):
        """Test task creation handles workflow start errors."""
        error_message = "Workflow start failed"
        mock_temporal_client.start_workflow.side_effect = Exception(error_message)

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await create_task(request=sample_create_task_request, api_key="test-api-key")

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert error_message in exc_info.value.detail
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_task_handles_client_error(self, sample_create_task_request):
        """Test task creation handles client initialization errors."""
        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.side_effect = Exception("Client connection failed")

            with pytest.raises(HTTPException) as exc_info:
                await create_task(request=sample_create_task_request, api_key="test-api-key")

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to create task" in exc_info.value.detail
            mock_logger.error.assert_called_once()


class TestGetTaskState:
    """Test get_task_state endpoint."""

    @pytest.mark.asyncio
    async def test_get_task_state_success(
        self, mock_temporal_client, sample_workflow_status_result
    ):
        """Test successful task state retrieval."""
        task_id = "task-123"
        mock_temporal_client.query_workflow_status.return_value = sample_workflow_status_result

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = mock_temporal_client

            response = await get_task_state(
                task_id=task_id,
                include_progress=False,
                include_state=False,
                api_key="test-api-key",
            )

            assert isinstance(response, TaskStateResponse)
            assert response.task_id == task_id
            assert response.status == sample_workflow_status_result.status.value
            assert response.progress is None
            assert response.state is None

    @pytest.mark.asyncio
    async def test_get_task_state_with_progress(
        self,
        mock_temporal_client,
        sample_workflow_status_result,
        sample_workflow_progress_result,
    ):
        """Test task state retrieval with progress information."""
        task_id = "task-123"
        mock_temporal_client.query_workflow_status.return_value = sample_workflow_status_result
        mock_temporal_client.query_workflow_progress.return_value = sample_workflow_progress_result

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = mock_temporal_client

            response = await get_task_state(
                task_id=task_id,
                include_progress=True,
                include_state=False,
                api_key="test-api-key",
            )

            assert response.progress == sample_workflow_progress_result
            mock_temporal_client.query_workflow_progress.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_get_task_state_with_full_state(
        self,
        mock_temporal_client,
        sample_workflow_status_result,
        sample_workflow_state_result,
    ):
        """Test task state retrieval with full state information."""
        task_id = "task-123"
        mock_temporal_client.query_workflow_status.return_value = sample_workflow_status_result
        mock_temporal_client.query_workflow_state.return_value = sample_workflow_state_result

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = mock_temporal_client

            response = await get_task_state(
                task_id=task_id,
                include_progress=False,
                include_state=True,
                api_key="test-api-key",
            )

            assert response.state == sample_workflow_state_result
            mock_temporal_client.query_workflow_state.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_get_task_state_progress_query_failure(
        self, mock_temporal_client, sample_workflow_status_result
    ):
        """Test task state retrieval when progress query fails."""
        task_id = "task-123"
        mock_temporal_client.query_workflow_status.return_value = sample_workflow_status_result
        mock_temporal_client.query_workflow_progress.side_effect = Exception("Progress query failed")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            response = await get_task_state(
                task_id=task_id,
                include_progress=True,
                include_state=False,
                api_key="test-api-key",
            )

            # Should still return response with None progress
            assert response.progress is None
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_task_state_state_query_failure(
        self, mock_temporal_client, sample_workflow_status_result
    ):
        """Test task state retrieval when state query fails."""
        task_id = "task-123"
        mock_temporal_client.query_workflow_status.return_value = sample_workflow_status_result
        mock_temporal_client.query_workflow_state.side_effect = Exception("State query failed")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            response = await get_task_state(
                task_id=task_id,
                include_progress=False,
                include_state=True,
                api_key="test-api-key",
            )

            # Should still return response with None state
            assert response.state is None
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_task_state_not_found(self, mock_temporal_client):
        """Test task state retrieval when task is not found."""
        task_id = "task-999"
        mock_temporal_client.query_workflow_status.side_effect = Exception("Workflow not found")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await get_task_state(
                    task_id=task_id,
                    include_progress=False,
                    include_state=False,
                    api_key="test-api-key",
                )

            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
            assert f"Task {task_id} not found" in exc_info.value.detail
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_task_state_generic_error(self, mock_temporal_client):
        """Test task state retrieval with generic error."""
        task_id = "task-123"
        mock_temporal_client.query_workflow_status.side_effect = Exception("Generic error")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await get_task_state(
                    task_id=task_id,
                    include_progress=False,
                    include_state=False,
                    api_key="test-api-key",
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to get task state" in exc_info.value.detail
            mock_logger.error.assert_called_once()


class TestStreamTaskEvents:
    """Test stream_task_events endpoint."""

    @pytest.mark.asyncio
    async def test_stream_task_events_success(
        self, mock_temporal_client, sample_workflow_status_result, sample_workflow_progress_result
    ):
        """Test successful event streaming."""
        task_id = "task-123"
        sample_workflow_status_result.status = WorkflowStatus.COMPLETED
        mock_temporal_client.query_workflow_status.return_value = sample_workflow_status_result
        mock_temporal_client.query_workflow_progress.return_value = sample_workflow_progress_result

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("asyncio.sleep", new_callable=AsyncMock):  # Mock sleep to speed up test
            mock_get_client.return_value = mock_temporal_client

            response = await stream_task_events(task_id=task_id, api_key="test-api-key")

            assert response.media_type == "text/event-stream"
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == "no-cache"

    @pytest.mark.asyncio
    async def test_stream_task_events_not_found(self, mock_temporal_client):
        """Test event streaming when task is not found."""
        task_id = "task-999"
        mock_temporal_client.query_workflow_status.side_effect = Exception("Workflow not found")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await stream_task_events(task_id=task_id, api_key="test-api-key")

            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
            assert f"Task {task_id} not found" in exc_info.value.detail
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_task_events_generic_error(self, mock_temporal_client):
        """Test event streaming with generic error."""
        task_id = "task-123"
        mock_temporal_client.query_workflow_status.side_effect = Exception("Generic error")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await stream_task_events(task_id=task_id, api_key="test-api-key")

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to stream events" in exc_info.value.detail
            mock_logger.error.assert_called_once()


class TestSendTaskSignal:
    """Test send_task_signal endpoint."""

    @pytest.mark.asyncio
    async def test_send_task_signal_success(self, mock_temporal_client):
        """Test successful signal sending."""
        task_id = "task-123"
        signal_request = TaskSignalRequest(
            input_text="User input text",
            input_type="question",
            metadata={"key": "value"},
        )

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            response = await send_task_signal(
                task_id=task_id,
                request=signal_request,
                api_key="test-api-key",
            )

            assert isinstance(response, TaskSignalResponse)
            assert response.task_id == task_id
            assert response.message == "Signal sent successfully"
            assert response.signal_sent is True

            # Verify signal was sent with correct parameters
            mock_temporal_client.send_user_input_signal.assert_called_once_with(
                workflow_id=task_id,
                input_text=signal_request.input_text,
                input_type=signal_request.input_type,
                metadata=signal_request.metadata,
            )
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_task_signal_minimal_request(self, mock_temporal_client):
        """Test signal sending with minimal request."""
        task_id = "task-123"
        signal_request = TaskSignalRequest(input_text="Minimal input")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = mock_temporal_client

            response = await send_task_signal(
                task_id=task_id,
                request=signal_request,
                api_key="test-api-key",
            )

            assert response.signal_sent is True
            call_kwargs = mock_temporal_client.send_user_input_signal.call_args[1]
            assert call_kwargs["input_text"] == "Minimal input"
            assert call_kwargs["input_type"] is None
            assert call_kwargs["metadata"] == {}

    @pytest.mark.asyncio
    async def test_send_task_signal_not_found(self, mock_temporal_client):
        """Test signal sending when task is not found."""
        task_id = "task-999"
        signal_request = TaskSignalRequest(input_text="Test input")
        mock_temporal_client.send_user_input_signal.side_effect = Exception("Workflow not found")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await send_task_signal(
                    task_id=task_id,
                    request=signal_request,
                    api_key="test-api-key",
                )

            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
            assert f"Task {task_id} not found" in exc_info.value.detail
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_task_signal_generic_error(self, mock_temporal_client):
        """Test signal sending with generic error."""
        task_id = "task-123"
        signal_request = TaskSignalRequest(input_text="Test input")
        mock_temporal_client.send_user_input_signal.side_effect = Exception("Generic error")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await send_task_signal(
                    task_id=task_id,
                    request=signal_request,
                    api_key="test-api-key",
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to send signal" in exc_info.value.detail
            mock_logger.error.assert_called_once()


class TestCancelTask:
    """Test cancel_task endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_task_success(self, mock_temporal_client):
        """Test successful task cancellation."""
        task_id = "task-123"
        cancel_request = CancelTaskRequest(reason="User requested cancellation")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            response = await cancel_task(
                task_id=task_id,
                request=cancel_request,
                api_key="test-api-key",
            )

            assert isinstance(response, CancelTaskResponse)
            assert response.task_id == task_id
            assert response.message == "Task cancellation requested"
            assert response.cancelled is True

            # Verify cancellation signal was sent
            mock_temporal_client.send_cancellation_signal.assert_called_once_with(
                workflow_id=task_id,
                reason=cancel_request.reason,
                requested_by="api",
            )
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_task_without_reason(self, mock_temporal_client):
        """Test task cancellation without reason."""
        task_id = "task-123"
        cancel_request = CancelTaskRequest(reason=None)

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = mock_temporal_client

            response = await cancel_task(
                task_id=task_id,
                request=cancel_request,
                api_key="test-api-key",
            )

            assert response.cancelled is True
            call_kwargs = mock_temporal_client.send_cancellation_signal.call_args[1]
            assert call_kwargs["reason"] is None
            assert call_kwargs["requested_by"] == "api"

    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self, mock_temporal_client):
        """Test task cancellation when task is not found."""
        task_id = "task-999"
        cancel_request = CancelTaskRequest(reason="Test reason")
        mock_temporal_client.send_cancellation_signal.side_effect = Exception("Workflow not found")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await cancel_task(
                    task_id=task_id,
                    request=cancel_request,
                    api_key="test-api-key",
                )

            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
            assert f"Task {task_id} not found" in exc_info.value.detail
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_task_generic_error(self, mock_temporal_client):
        """Test task cancellation with generic error."""
        task_id = "task-123"
        cancel_request = CancelTaskRequest(reason="Test reason")
        mock_temporal_client.send_cancellation_signal.side_effect = Exception("Generic error")

        with patch("src.api.routers.tasks.get_client", new_callable=AsyncMock) as mock_get_client, \
             patch("src.api.routers.tasks.logger") as mock_logger:
            mock_get_client.return_value = mock_temporal_client

            with pytest.raises(HTTPException) as exc_info:
                await cancel_task(
                    task_id=task_id,
                    request=cancel_request,
                    api_key="test-api-key",
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to cancel task" in exc_info.value.detail
            mock_logger.error.assert_called_once()


class TestRequestResponseModels:
    """Test request and response models."""

    def test_create_task_request_validation(self):
        """Test CreateTaskRequest model validation."""
        # Valid request
        request = CreateTaskRequest(query="Test query")
        assert request.query == "Test query"
        assert request.execution_mode == "sequential"
        assert request.metadata == {}

        # Request with all fields
        full_request = CreateTaskRequest(
            query="Full query",
            conversation_id="conv-1",
            user_id="user-1",
            agent_plan=["agent1"],
            execution_mode="parallel",
            metadata={"key": "value"},
        )
        assert full_request.conversation_id == "conv-1"
        assert full_request.agent_plan == ["agent1"]
        assert full_request.execution_mode == "parallel"

    def test_task_signal_request_validation(self):
        """Test TaskSignalRequest model validation."""
        # Minimal request
        request = TaskSignalRequest(input_text="Test input")
        assert request.input_text == "Test input"
        assert request.input_type is None
        assert request.metadata == {}

        # Full request
        full_request = TaskSignalRequest(
            input_text="Full input",
            input_type="question",
            metadata={"key": "value"},
        )
        assert full_request.input_type == "question"
        assert full_request.metadata == {"key": "value"}

    def test_cancel_task_request_validation(self):
        """Test CancelTaskRequest model validation."""
        # With reason
        request = CancelTaskRequest(reason="Test reason")
        assert request.reason == "Test reason"

        # Without reason
        request_no_reason = CancelTaskRequest(reason=None)
        assert request_no_reason.reason is None

