"""Task router for task-oriented API endpoints.

This router implements the task-oriented API surface:
- POST /tasks - Start a new task
- GET /tasks/{task_id} - Get current task state
- GET /tasks/{task_id}/events - Subscribe to task updates (SSE)
- POST /tasks/{task_id}/signal - Influence a running task
- POST /tasks/{task_id}/cancel - Cancel a task

The API is stateless and acts as a bridge to Temporal workflows.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.middleware.auth import verify_api_key
from src.temporal import (TemporalClient, WorkflowProgressQueryResult,
                          WorkflowStateQueryResult, WorkflowStatusQueryResult,
                          get_client)

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class CreateTaskRequest(BaseModel):
    """Request model for creating a new task."""

    query: str = Field(..., description="User query or request to process")
    conversation_id: Optional[str] = Field(
        default=None, description="Optional conversation identifier"
    )
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    agent_plan: Optional[List[str]] = Field(
        default=None,
        description="Optional list of agent names to execute. If not provided, orchestration agent will determine plan.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class CreateTaskResponse(BaseModel):
    """Response model for task creation."""

    task_id: str = Field(..., description="Unique task identifier (workflow ID)")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Status message")


class TaskStateResponse(BaseModel):
    """Response model for task state query."""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Current task status")
    progress: Optional[WorkflowProgressQueryResult] = Field(
        default=None, description="Detailed progress information"
    )
    state: Optional[WorkflowStateQueryResult] = Field(
        default=None, description="Full task state"
    )


class TaskSignalRequest(BaseModel):
    """Request model for sending a signal to a task."""

    input_text: str = Field(..., description="User input text")
    input_type: Optional[str] = Field(
        default=None, description="Type/category of input"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class TaskSignalResponse(BaseModel):
    """Response model for signal sending."""

    task_id: str = Field(..., description="Task identifier")
    message: str = Field(..., description="Status message")
    signal_sent: bool = Field(..., description="Whether signal was sent successfully")


class CancelTaskRequest(BaseModel):
    """Request model for cancelling a task."""

    reason: Optional[str] = Field(
        default=None, description="Optional reason for cancellation"
    )


class CancelTaskResponse(BaseModel):
    """Response model for task cancellation."""

    task_id: str = Field(..., description="Task identifier")
    message: str = Field(..., description="Status message")
    cancelled: bool = Field(..., description="Whether cancellation was successful")


@router.post("", response_model=CreateTaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    request: CreateTaskRequest,
    api_key: str = Depends(verify_api_key),
) -> CreateTaskResponse:
    """Create a new task by starting a Temporal workflow.

    Args:
        request: Task creation request.
        api_key: Verified API key (from dependency).

    Returns:
        CreateTaskResponse with task ID and status.

    Raises:
        HTTPException: If task creation fails.
    """
    try:
        client: TemporalClient = await get_client()

        # Build context from request
        context = {
            "query": request.query,
            "conversation_id": request.conversation_id,
            "user_id": request.user_id,
            "metadata": request.metadata,
            "shared_data": {},
        }

        # Start workflow
        # Note: execution_mode is not passed - let the orchestration agent decide
        workflow_id = await client.start_workflow(
            context=context,
            agent_plan=request.agent_plan,
            # execution_mode is determined by the orchestration agent, not the API
        )

        logger.info(f"Created task {workflow_id} for query: {request.query[:100]}")

        return CreateTaskResponse(
            task_id=workflow_id,
            status="pending",
            message="Task created successfully",
        )

    except Exception as e:
        logger.error(f"Failed to create task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}",
        )


@router.get("/{task_id}", response_model=TaskStateResponse)
async def get_task_state(
    task_id: str,
    include_progress: bool = Query(
        default=True, description="Include detailed progress information"
    ),
    include_state: bool = Query(default=False, description="Include full task state"),
    api_key: str = Depends(verify_api_key),
) -> TaskStateResponse:
    """Get the current state of a task.

    This is the ground truth interface - returns a snapshot of task state.

    Args:
        task_id: Task identifier (workflow ID).
        include_progress: Whether to include detailed progress.
        include_state: Whether to include full state.
        api_key: Verified API key (from dependency).

    Returns:
        TaskStateResponse with task state information.

    Raises:
        HTTPException: If task is not found or query fails.
    """
    try:
        client: TemporalClient = await get_client()

        # Query workflow status
        status_result = await client.query_workflow_status(task_id)

        # Optionally query progress
        progress = None
        if include_progress:
            try:
                progress = await client.query_workflow_progress(task_id)
            except Exception as e:
                logger.warning(f"Failed to query progress for {task_id}: {e}")

        # Optionally query full state
        full_state = None
        if include_state:
            try:
                full_state = await client.query_workflow_state(task_id)
            except Exception as e:
                logger.warning(f"Failed to query state for {task_id}: {e}")

        return TaskStateResponse(
            task_id=task_id,
            status=status_result.status.value,
            progress=progress,
            state=full_state,
        )

    except Exception as e:
        logger.error(f"Failed to get task state for {task_id}: {e}", exc_info=True)
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task state: {str(e)}",
        )


@router.get("/{task_id}/events")
async def stream_task_events(
    task_id: str,
    api_key: str = Depends(verify_api_key),
) -> StreamingResponse:
    """Stream task events via Server-Sent Events (SSE).

    This provides real-time updates for UX responsiveness.
    Anything streamed is safe to miss - the UI should query state for ground truth.

    Args:
        task_id: Task identifier (workflow ID).
        api_key: Verified API key (from dependency).

    Returns:
        StreamingResponse with SSE events.

    Raises:
        HTTPException: If task is not found.
    """
    try:
        client: TemporalClient = await get_client()

        # Verify task exists
        await client.query_workflow_status(task_id)

        async def event_generator():
            """Generate SSE events by polling workflow state."""
            last_progress = None
            last_status = None

            try:
                while True:
                    # Poll for updates
                    progress = await client.query_workflow_progress(task_id)
                    status_result = await client.query_workflow_status(task_id)

                    # Check if status changed
                    if last_status != status_result.status.value:
                        yield f"data: {json.dumps({'type': 'status', 'status': status_result.status.value, 'task_id': task_id})}\n\n"
                        last_status = status_result.status.value

                    # Check if progress changed
                    if progress != last_progress:
                        # Emit agent status updates
                        for agent_status in progress.agent_statuses:
                            yield f"data: {json.dumps({'type': 'agent_status', 'agent': agent_status.agent_name, 'status': agent_status.status, 'task_id': task_id})}\n\n"

                        # Emit progress update
                        yield f"data: {json.dumps({'type': 'progress', 'progress_percentage': progress.progress_percentage, 'completed_agents': progress.completed_agents, 'total_agents': progress.total_agents, 'task_id': task_id})}\n\n"
                        last_progress = progress

                    # If task is completed, cancelled, or failed, send final event and break
                    if status_result.status.value in [
                        "completed",
                        "cancelled",
                        "failed",
                    ]:
                        yield f"data: {json.dumps({'type': 'complete', 'status': status_result.status.value, 'task_id': task_id})}\n\n"
                        break

                    # Wait before next poll
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info(f"Event stream cancelled for task {task_id}")
            except Exception as e:
                logger.error(f"Error in event stream for {task_id}: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'task_id': task_id})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    except Exception as e:
        logger.error(f"Failed to stream events for {task_id}: {e}", exc_info=True)
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream events: {str(e)}",
        )


@router.post("/{task_id}/signal", response_model=TaskSignalResponse)
async def send_task_signal(
    task_id: str,
    request: TaskSignalRequest,
    api_key: str = Depends(verify_api_key),
) -> TaskSignalResponse:
    """Send a signal to a running task.

    Args:
        task_id: Task identifier (workflow ID).
        request: Signal request with user input.
        api_key: Verified API key (from dependency).

    Returns:
        TaskSignalResponse with confirmation.

    Raises:
        HTTPException: If task is not found or signal fails.
    """
    try:
        client: TemporalClient = await get_client()

        await client.send_user_input_signal(
            workflow_id=task_id,
            input_text=request.input_text,
            input_type=request.input_type,
            metadata=request.metadata,
        )

        logger.info(f"Sent signal to task {task_id}")

        return TaskSignalResponse(
            task_id=task_id,
            message="Signal sent successfully",
            signal_sent=True,
        )

    except Exception as e:
        logger.error(f"Failed to send signal to {task_id}: {e}", exc_info=True)
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send signal: {str(e)}",
        )


@router.post("/{task_id}/cancel", response_model=CancelTaskResponse)
async def cancel_task(
    task_id: str,
    request: CancelTaskRequest,
    api_key: str = Depends(verify_api_key),
) -> CancelTaskResponse:
    """Cancel a running task.

    Args:
        task_id: Task identifier (workflow ID).
        request: Cancellation request with optional reason.
        api_key: Verified API key (from dependency).

    Returns:
        CancelTaskResponse with confirmation.

    Raises:
        HTTPException: If task is not found or cancellation fails.
    """
    try:
        client: TemporalClient = await get_client()

        await client.send_cancellation_signal(
            workflow_id=task_id,
            reason=request.reason,
            requested_by="api",  # Could be extracted from auth context
        )

        logger.info(f"Cancelled task {task_id}")

        return CancelTaskResponse(
            task_id=task_id,
            message="Task cancellation requested",
            cancelled=True,
        )

    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}", exc_info=True)
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}",
        )
