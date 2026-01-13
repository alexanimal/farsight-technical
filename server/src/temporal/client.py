"""Temporal client for API layer integration.

This module provides a thin wrapper around the Temporal client that allows
the API layer to interact with Temporal workflows without directly importing
workflows, activities, or agents.

The client provides:
- Workflow execution (start workflows)
- Signal sending (cancellation, user input)
- Query execution (status, progress, state)
- Error handling and type safety
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleCalendarSpec,
    ScheduleHandle,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
)

# Batch operations in Temporal are created via gRPC API
# The Python SDK may not have high-level support, so we'll use the service client directly

# Note: AgentContext is not imported here to avoid workflow determinism issues
# The client methods accept Dict[str, Any] for context, not AgentContext objects
from src.temporal.queries import (
    QUERY_AGENT_STATUS,
    QUERY_PROGRESS,
    QUERY_STATE,
    QUERY_STATUS,
    AgentStatusQueryResult,
    WorkflowProgressQueryResult,
    WorkflowStateQueryResult,
    WorkflowStatusQueryResult,
)
from src.temporal.signals import (
    SIGNAL_CANCELLATION,
    SIGNAL_USER_INPUT,
    CancellationSignal,
    UserInputSignal,
)

# Import workflow lazily to avoid pulling in AgentContext during workflow validation
# The workflow is only needed when actually starting a workflow

logger = logging.getLogger(__name__)

# Default Temporal connection settings
DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"
DEFAULT_TEMPORAL_NAMESPACE = "default"
DEFAULT_TASK_QUEUE = "orchestrator-task-queue"


class TemporalClient:
    """Client for interacting with Temporal workflows.

    This client provides a clean interface for the API layer to:
    - Start workflows
    - Send signals to running workflows
    - Query workflow state
    """

    def __init__(
        self,
        client: Optional[Client] = None,
        temporal_address: str = DEFAULT_TEMPORAL_ADDRESS,
        temporal_namespace: str = DEFAULT_TEMPORAL_NAMESPACE,
        task_queue: str = DEFAULT_TASK_QUEUE,
    ):
        """Initialize the Temporal client.

        Args:
            client: Optional pre-configured Temporal client. If provided,
                other connection parameters are ignored.
            temporal_address: Temporal server address (default: localhost:7233).
            temporal_namespace: Temporal namespace (default: default).
            task_queue: Task queue name for workflows (default: orchestrator-task-queue).
        """
        self._client = client
        self._temporal_address = temporal_address
        self._temporal_namespace = temporal_namespace
        self._task_queue = task_queue
        self._client_initialized = client is not None

    async def connect(self) -> None:
        """Connect to Temporal server.

        This method must be called before using the client if a pre-configured
        client was not provided during initialization.

        Raises:
            RuntimeError: If connection fails.
        """
        if self._client is None:
            try:
                self._client = await Client.connect(
                    self._temporal_address,
                    namespace=self._temporal_namespace,
                )
                self._client_initialized = True
                logger.info(
                    f"Connected to Temporal at {self._temporal_address} "
                    f"(namespace: {self._temporal_namespace})"
                )
            except Exception as e:
                error_msg = (
                    f"Failed to connect to Temporal at {self._temporal_address}: {e}. "
                    "Make sure Temporal server is running. "
                    "For local development, run: docker-compose up -d temporal"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    async def close(self) -> None:
        """Close the Temporal client connection."""
        if self._client is not None:
            # Temporal Client doesn't have a close() method, just clear the reference
            self._client = None
            self._client_initialized = False
            logger.info("Temporal client connection closed")

    def _ensure_connected(self) -> Client:
        """Ensure client is connected and return it.

        Returns:
            The Temporal client.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if self._client is None or not self._client_initialized:
            raise RuntimeError(
                "Temporal client not connected. Call connect() first or "
                "provide a client during initialization."
            )
        return self._client

    async def start_workflow(
        self,
        context: Dict[str, Any],
        agent_plan: Optional[List[str]] = None,
        execution_mode: str = "sequential",
        workflow_id: Optional[str] = None,
        workflow_timeout: Optional[timedelta] = None,
    ) -> str:
        """Start an orchestrator workflow.

        Args:
            context: AgentContext as dictionary (query, conversation_id, etc.).
            agent_plan: Optional list of agent names to execute. If None,
                workflow will use orchestration agent to determine plan.
            execution_mode: Execution mode - "sequential" or "parallel".
            workflow_id: Optional workflow ID. If not provided, Temporal will
                generate one.
            workflow_timeout: Optional workflow execution timeout.

        Returns:
            The workflow ID.

        Raises:
            RuntimeError: If client is not connected.
            Exception: If workflow start fails.
        """
        client = self._ensure_connected()

        # Validate context has required fields
        if "query" not in context:
            raise ValueError("Context must contain 'query' field")

        # Import workflow lazily to avoid import chain issues during workflow validation
        from src.temporal.workflows.orchestrator import OrchestratorWorkflow

        try:
            # Temporal requires an 'id' parameter - generate one if not provided
            if workflow_id is None:
                workflow_id = f"workflow-{uuid.uuid4().hex[:12]}"

            # Build kwargs for start_workflow
            workflow_kwargs: Dict[str, Any] = {
                "id": workflow_id,
                "args": [context, agent_plan, execution_mode],
                "task_queue": self._task_queue,
            }
            if workflow_timeout is not None:
                workflow_kwargs["execution_timeout"] = workflow_timeout

            handle = await client.start_workflow(
                OrchestratorWorkflow.run,
                **workflow_kwargs,
            )

            logger.info(f"Started workflow: {handle.id}")
            return handle.id

        except Exception as e:
            logger.error(f"Failed to start workflow: {e}", exc_info=True)
            raise

    async def send_cancellation_signal(
        self,
        workflow_id: str,
        reason: Optional[str] = None,
        requested_by: Optional[str] = None,
    ) -> None:
        """Send a cancellation signal to a running workflow.

        Args:
            workflow_id: The workflow ID to send signal to.
            reason: Optional reason for cancellation.
            requested_by: Optional identifier of who requested cancellation.

        Raises:
            RuntimeError: If client is not connected or workflow is not found.
            Exception: If signal sending fails.
        """
        client = self._ensure_connected()

        signal = CancellationSignal(
            reason=reason,
            requested_by=requested_by,
        )

        try:
            handle = client.get_workflow_handle(workflow_id)
            await handle.signal(SIGNAL_CANCELLATION, signal)
            logger.info(f"Sent cancellation signal to workflow: {workflow_id}")
        except Exception as e:
            # Check if it's a workflow not found error
            error_str = str(e).lower()
            if "not found" in error_str or "workflow" in error_str:
                logger.warning(f"Workflow not found: {workflow_id}")
                # Re-raise as a more specific exception for API layer
                raise RuntimeError(f"Workflow {workflow_id} not found") from e
            logger.error(f"Failed to send cancellation signal: {e}", exc_info=True)
            raise

    async def send_user_input_signal(
        self,
        workflow_id: str,
        input_text: str,
        input_type: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a user input signal to a running workflow.

        Args:
            workflow_id: The workflow ID to send signal to.
            input_text: The user's input text.
            input_type: Optional type/category of input.
            user_id: Optional identifier of the user.
            conversation_id: Optional conversation identifier.
            metadata: Optional additional metadata.

        Raises:
            RuntimeError: If client is not connected or workflow is not found.
            Exception: If signal sending fails.
        """
        client = self._ensure_connected()

        signal = UserInputSignal(
            input_text=input_text,
            input_type=input_type,
            user_id=user_id,
            conversation_id=conversation_id,
            metadata=metadata or {},
        )

        try:
            handle = client.get_workflow_handle(workflow_id)
            await handle.signal(SIGNAL_USER_INPUT, signal)
            logger.info(f"Sent user input signal to workflow: {workflow_id}")
        except Exception as e:
            # Check if it's a workflow not found error
            error_str = str(e).lower()
            if "not found" in error_str or "workflow" in error_str:
                logger.warning(f"Workflow not found: {workflow_id}")
                # Re-raise as a more specific exception for API layer
                raise RuntimeError(f"Workflow {workflow_id} not found") from e
            logger.error(f"Failed to send user input signal: {e}", exc_info=True)
            raise

    async def query_workflow_status(self, workflow_id: str) -> WorkflowStatusQueryResult:
        """Query the status of a workflow.

        Args:
            workflow_id: The workflow ID to query.

        Returns:
            WorkflowStatusQueryResult with workflow status.

        Raises:
            RuntimeError: If client is not connected or workflow is not found.
            Exception: If query fails.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_workflow_handle(workflow_id)
            result = await handle.query(QUERY_STATUS)
            # Query returns the result directly (Pydantic model)
            if isinstance(result, WorkflowStatusQueryResult):
                return result
            # If it's a dict, convert it
            return WorkflowStatusQueryResult(**result)
        except Exception as e:
            # Check if it's a workflow not found error
            error_str = str(e).lower()
            if "not found" in error_str or "workflow" in error_str:
                logger.warning(f"Workflow not found: {workflow_id}")
                # Re-raise as a more specific exception for API layer
                raise RuntimeError(f"Workflow {workflow_id} not found") from e
            logger.error(f"Failed to query workflow status: {e}", exc_info=True)
            raise

    async def query_workflow_progress(self, workflow_id: str) -> WorkflowProgressQueryResult:
        """Query the progress of a workflow.

        Args:
            workflow_id: The workflow ID to query.

        Returns:
            WorkflowProgressQueryResult with progress information.

        Raises:
            RuntimeError: If client is not connected or workflow is not found.
            Exception: If query fails.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_workflow_handle(workflow_id)
            result = await handle.query(QUERY_PROGRESS)
            # Query returns the result directly (Pydantic model)
            if isinstance(result, WorkflowProgressQueryResult):
                return result
            # If it's a dict, convert it
            return WorkflowProgressQueryResult(**result)
        except Exception as e:
            # Check if it's a workflow not found error
            error_str = str(e).lower()
            if "not found" in error_str or "workflow" in error_str:
                logger.warning(f"Workflow not found: {workflow_id}")
                # Re-raise as a more specific exception for API layer
                raise RuntimeError(f"Workflow {workflow_id} not found") from e
            logger.error(f"Failed to query workflow progress: {e}", exc_info=True)
            raise

    async def query_workflow_state(self, workflow_id: str) -> WorkflowStateQueryResult:
        """Query the full state of a workflow.

        Args:
            workflow_id: The workflow ID to query.

        Returns:
            WorkflowStateQueryResult with complete workflow state.

        Raises:
            RuntimeError: If client is not connected or workflow is not found.
            Exception: If query fails.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_workflow_handle(workflow_id)
            result = await handle.query(QUERY_STATE)
            # Query returns the result directly (Pydantic model)
            if isinstance(result, WorkflowStateQueryResult):
                return result
            # If it's a dict, convert it
            return WorkflowStateQueryResult(**result)
        except Exception as e:
            # Check if it's a workflow not found error
            error_str = str(e).lower()
            if "not found" in error_str or "workflow" in error_str:
                logger.warning(f"Workflow not found: {workflow_id}")
                # Re-raise as a more specific exception for API layer
                raise RuntimeError(f"Workflow {workflow_id} not found") from e
            logger.error(f"Failed to query workflow state: {e}", exc_info=True)
            raise

    async def query_agent_status(self, workflow_id: str, agent_name: str) -> AgentStatusQueryResult:
        """Query the status of a specific agent in a workflow.

        Args:
            workflow_id: The workflow ID to query.
            agent_name: Name of the agent to query.

        Returns:
            AgentStatusQueryResult with agent status.

        Raises:
            RuntimeError: If client is not connected or workflow is not found.
            Exception: If query fails.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_workflow_handle(workflow_id)
            result = await handle.query(QUERY_AGENT_STATUS, agent_name)
            # Query returns the result directly (Pydantic model)
            if isinstance(result, AgentStatusQueryResult):
                return result
            # If it's a dict, convert it
            return AgentStatusQueryResult(**result)
        except Exception as e:
            # Check if it's a workflow not found error
            error_str = str(e).lower()
            if "not found" in error_str or "workflow" in error_str:
                logger.warning(f"Workflow not found: {workflow_id}")
                # Re-raise as a more specific exception for API layer
                raise RuntimeError(f"Workflow {workflow_id} not found") from e
            logger.error(f"Failed to query agent status: {e}", exc_info=True)
            raise

    async def get_workflow_result(
        self, workflow_id: str, timeout: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get the result of a completed workflow.

        Args:
            workflow_id: The workflow ID.
            timeout: Optional timeout for waiting for result.

        Returns:
            Dictionary containing workflow result.

        Raises:
            RuntimeError: If client is not connected or workflow is not found.
            Exception: If getting result fails.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_workflow_handle(workflow_id)
            result = await handle.result(rpc_timeout=timeout)
            return result
        except Exception as e:
            # Check if it's a workflow not found error
            error_str = str(e).lower()
            if "not found" in error_str or "workflow" in error_str:
                logger.warning(f"Workflow not found: {workflow_id}")
                # Re-raise as a more specific exception for API layer
                raise RuntimeError(f"Workflow {workflow_id} not found") from e
            logger.error(f"Failed to get workflow result: {e}", exc_info=True)
            raise

    async def create_schedule(
        self,
        schedule_id: str,
        workflow_type: str,
        workflow_args: List[Any],
        schedule_spec: ScheduleSpec,
        task_queue: Optional[str] = None,
        workflow_id_template: Optional[str] = None,
        enabled: bool = True,
    ) -> ScheduleHandle:
        """Create a schedule that automatically triggers workflows.

        Args:
            schedule_id: Unique identifier for the schedule.
            workflow_type: Type of workflow to start ("orchestrator" or "pipeline").
            workflow_args: Arguments to pass to the workflow.
            schedule_spec: ScheduleSpec defining when to trigger (cron, interval, etc.).
            task_queue: Optional task queue name. Uses default if not provided.
            workflow_id_template: Optional template for workflow IDs. Use {timestamp}
                placeholder for unique IDs per run. Default: "{schedule_id}-{timestamp}".
            enabled: Whether the schedule is enabled (default: True).

        Returns:
            ScheduleHandle for the created schedule.

        Raises:
            RuntimeError: If client is not connected.
            ValueError: If workflow_type is invalid.
            Exception: If schedule creation fails.
        """
        client = self._ensure_connected()

        # Import workflows lazily
        if workflow_type == "pipeline":
            from src.temporal.workflows.pipeline import PipelineWorkflow

            workflow_func = PipelineWorkflow.run
        elif workflow_type == "orchestrator":
            from src.temporal.workflows.orchestrator import OrchestratorWorkflow

            workflow_func = OrchestratorWorkflow.run
        else:
            raise ValueError(
                f"Invalid workflow_type: {workflow_type}. Must be 'orchestrator' or 'pipeline'"
            )

        try:
            # Build workflow ID template
            if workflow_id_template is None:
                workflow_id_template = f"{schedule_id}-{{timestamp}}"

            # Create schedule action
            action = ScheduleActionStartWorkflow(
                workflow_func,
                id=workflow_id_template,
                task_queue=task_queue or self._task_queue,
                args=workflow_args,
            )

            # Create schedule
            # ScheduleState uses 'note' for description and 'paused' for enabled/disabled state
            # paused=False means enabled, paused=True means disabled
            schedule = Schedule(
                action=action,
                spec=schedule_spec,
                state=ScheduleState(note="", paused=not enabled),
            )

            handle = await client.create_schedule(
                schedule_id,
                schedule,
                trigger_immediately=False,
            )

            logger.info(f"Created schedule: {schedule_id} (enabled: {enabled})")
            return handle

        except Exception as e:
            logger.error(f"Failed to create schedule {schedule_id}: {e}", exc_info=True)
            raise

    async def get_schedule(self, schedule_id: str) -> ScheduleHandle:
        """Get a schedule handle.

        Args:
            schedule_id: The schedule ID.

        Returns:
            ScheduleHandle for the schedule.

        Raises:
            RuntimeError: If client is not connected or schedule not found.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_schedule_handle(schedule_id)
            return handle
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "schedule" in error_str:
                logger.warning(f"Schedule not found: {schedule_id}")
                raise RuntimeError(f"Schedule {schedule_id} not found") from e
            logger.error(f"Failed to get schedule: {e}", exc_info=True)
            raise

    async def update_schedule(
        self,
        schedule_id: str,
        schedule_spec: Optional[ScheduleSpec] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update a schedule.

        Args:
            schedule_id: The schedule ID to update.
            schedule_spec: Optional new schedule specification.
            enabled: Optional new enabled state.

        Raises:
            RuntimeError: If client is not connected or schedule not found.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_schedule_handle(schedule_id)

            # Update requires an updater function that takes the schedule description
            # and returns an updated Schedule
            def updater(description: Any) -> Schedule:
                """Updater function for schedule modification."""
                # Get the current schedule from the description
                # The description object has a 'schedule' attribute
                current_schedule = description.schedule
                
                # Create a new Schedule with updates
                updated_schedule = Schedule(
                    action=current_schedule.action,
                    spec=schedule_spec if schedule_spec is not None else current_schedule.spec,
                    state=ScheduleState(
                        note=current_schedule.state.note if hasattr(current_schedule.state, 'note') else "",
                        paused=not enabled if enabled is not None else current_schedule.state.paused,
                    ),
                )
                
                return updated_schedule

            await handle.update(updater)

            logger.info(f"Updated schedule: {schedule_id}")
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "schedule" in error_str:
                logger.warning(f"Schedule not found: {schedule_id}")
                raise RuntimeError(f"Schedule {schedule_id} not found") from e
            logger.error(f"Failed to update schedule: {e}", exc_info=True)
            raise

    async def delete_schedule(self, schedule_id: str) -> None:
        """Delete a schedule.

        Args:
            schedule_id: The schedule ID to delete.

        Raises:
            RuntimeError: If client is not connected or schedule not found.
        """
        client = self._ensure_connected()

        try:
            handle = client.get_schedule_handle(schedule_id)
            await handle.delete()
            logger.info(f"Deleted schedule: {schedule_id}")
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "schedule" in error_str:
                logger.warning(f"Schedule not found: {schedule_id}")
                raise RuntimeError(f"Schedule {schedule_id} not found") from e
            logger.error(f"Failed to delete schedule: {e}", exc_info=True)
            raise

    async def list_schedules(self) -> List[ScheduleHandle]:
        """List all schedules in the namespace.

        Returns:
            List of ScheduleHandle objects.

        Raises:
            RuntimeError: If client is not connected.
        """
        client = self._ensure_connected()

        try:
            # Note: Temporal Python SDK may not have a direct list_schedules method
            # This is a placeholder - may need to use gRPC directly or check SDK version
            # For now, we'll raise NotImplementedError and document the limitation
            raise NotImplementedError(
                "list_schedules is not directly supported in Temporal Python SDK. "
                "Use Temporal UI or tctl to list schedules, or implement using gRPC client."
            )
        except Exception as e:
            logger.error(f"Failed to list schedules: {e}", exc_info=True)
            raise

    async def start_batch_operation(
        self,
        workflow_type: str,
        items: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        task_queue: Optional[str] = None,
    ) -> str:
        """Start a batch operation to process multiple items.

        Args:
            workflow_type: Type of workflow to run ("orchestrator" or "pipeline").
            items: List of context dictionaries, one per workflow execution.
            batch_id: Optional batch operation ID. Generated if not provided.
            task_queue: Optional task queue name. Uses default if not provided.

        Returns:
            The batch operation ID.

        Raises:
            RuntimeError: If client is not connected.
            ValueError: If workflow_type is invalid or items is empty.
            Exception: If batch operation start fails.
        """
        client = self._ensure_connected()

        if not items:
            raise ValueError("items list cannot be empty")

        # Import workflows lazily
        if workflow_type == "pipeline":
            from src.temporal.workflows.pipeline import PipelineWorkflow

            workflow_func = PipelineWorkflow.run
        elif workflow_type == "orchestrator":
            from src.temporal.workflows.orchestrator import OrchestratorWorkflow

            workflow_func = OrchestratorWorkflow.run
        else:
            raise ValueError(
                f"Invalid workflow_type: {workflow_type}. Must be 'orchestrator' or 'pipeline'"
            )

        try:
            if batch_id is None:
                batch_id = f"batch-{uuid.uuid4().hex[:12]}"

            # Create batch operation using Temporal's gRPC API
            # This creates a proper batch operation that shows up in the Temporal UI
            # Reference: https://python.temporal.io/temporalio.bridge.services_generated.WorkflowService.html#start_batch_operation
            try:
                from temporalio.api.workflowservice.v1 import StartBatchOperationRequest
                from temporalio.api.batch.v1 import BatchOperationJob
                from temporalio.converter import DataConverter
                
                # Build batch operation jobs
                operations = []
                converter = DataConverter.default
                
                for i, item_context in enumerate(items):
                    workflow_id = f"{batch_id}-item-{i}"

                    # Build workflow args based on type
                    if workflow_type == "pipeline":
                        workflow_args = [
                            item_context.get("pipeline_type", ""),
                            item_context.get("context", {}),
                            item_context.get("pipeline_config"),
                        ]
                    else:
                        workflow_args = [
                            item_context,
                            item_context.get("agent_plan"),
                            item_context.get("execution_mode", "sequential"),
                        ]

                    # Serialize workflow arguments to payloads
                    payloads = [converter.to_payload(arg) for arg in workflow_args]

                    # Create batch operation job
                    job = BatchOperationJob(
                        workflow_id=workflow_id,
                        workflow_type=workflow_type,
                        task_queue=task_queue or self._task_queue,
                        arguments=payloads,
                    )
                    operations.append(job)

                # Create batch operation request
                # The request structure based on Temporal gRPC API
                # Note: For starting workflows, we use the start_workflow_operation field
                # The operation field is for other batch operation types (terminate, cancel, etc.)
                request = StartBatchOperationRequest(
                    job_id=batch_id,
                    namespace=client.namespace,
                    visibility_query=f'WorkflowId LIKE "{batch_id}-%"',
                    # operation field is not needed for start_workflow_operation
                    # It's only used for terminate, cancel, signal, delete, reset, etc.
                )
                
                # Set the start workflow operation with the jobs
                # The start_workflow_operation field contains the operations list
                request.start_workflow_operation.operations.extend(operations)

                # Execute batch operation via service client
                # The service_client is accessed through client.service_client
                response = await client.service_client.start_batch_operation(request)
                
                logger.info(
                    f"Started batch operation {batch_id} with {len(items)} workflow(s). "
                    f"This will appear in the Temporal UI Batch Operations page. "
                    f"Response: {response}"
                )
                return batch_id
                
            except (ImportError, AttributeError, NotImplementedError) as e:
                # Fallback: create individual workflows (won't show as batch in UI)
                logger.warning(
                    f"Batch operations API not available ({e}). "
                    f"Creating {len(items)} individual workflows with batch ID {batch_id}."
                )
                for i, item_context in enumerate(items):
                    workflow_id = f"{batch_id}-item-{i}"

                    # Build workflow args based on type
                    if workflow_type == "pipeline":
                        workflow_args = [
                            item_context.get("pipeline_type", ""),
                            item_context.get("context", {}),
                            item_context.get("pipeline_config"),
                        ]
                    else:
                        workflow_args = [
                            item_context,
                            item_context.get("agent_plan"),
                            item_context.get("execution_mode", "sequential"),
                        ]

                    await client.start_workflow(
                        workflow_func,
                        id=workflow_id,
                        task_queue=task_queue or self._task_queue,
                        args=workflow_args,
                    )

                logger.info(
                    f"Started {len(items)} individual workflow(s) with batch ID {batch_id}. "
                    "Note: These will not appear as a batch operation in the Temporal UI."
                )
                return batch_id

        except Exception as e:
            logger.error(f"Failed to start batch operation: {e}", exc_info=True)
            raise

    async def get_batch_operation_status(
        self, batch_id: str, workflow_type: str = "pipeline"
    ) -> Dict[str, Any]:
        """Get the status of a batch operation.

        Note: This method queries individual workflows in the batch. For a true
        batch operation status, you may need to use Temporal's batch operation
        API directly (which may require gRPC access).

        Args:
            batch_id: The batch operation ID.
            workflow_type: Type of workflow used in the batch.

        Returns:
            Dictionary with batch operation status including:
            - batch_id: The batch ID
            - total_workflows: Total number of workflows
            - completed: Number of completed workflows
            - running: Number of running workflows
            - failed: Number of failed workflows
            - workflow_ids: List of workflow IDs in the batch

        Raises:
            RuntimeError: If client is not connected.
        """
        client = self._ensure_connected()

        try:
            # Query workflows by ID pattern
            # Note: This is a simplified implementation
            # In production, you might want to store batch metadata in Redis/DB
            # or use Temporal's batch operation API if available

            # For now, we'll need to track workflow IDs differently
            # This is a placeholder that shows the pattern
            # In practice, you'd store the workflow_ids when creating the batch

            logger.warning(
                "get_batch_operation_status is a simplified implementation. "
                "Consider storing batch metadata in Redis/DB for production use."
            )

            return {
                "batch_id": batch_id,
                "total_workflows": 0,
                "completed": 0,
                "running": 0,
                "failed": 0,
                "workflow_ids": [],
                "note": "Batch operation tracking requires storing workflow IDs. "
                "Consider implementing batch metadata storage.",
            }

        except Exception as e:
            logger.error(f"Failed to get batch operation status: {e}", exc_info=True)
            raise


# Singleton instance - can be initialized and reused
_default_client: Optional[TemporalClient] = None


async def get_client(
    temporal_address: str = DEFAULT_TEMPORAL_ADDRESS,
    temporal_namespace: str = DEFAULT_TEMPORAL_NAMESPACE,
    task_queue: str = DEFAULT_TASK_QUEUE,
) -> TemporalClient:
    """Get or create the default Temporal client.

    Args:
        temporal_address: Temporal server address.
        temporal_namespace: Temporal namespace.
        task_queue: Task queue name.

    Returns:
        The default TemporalClient instance.
    """
    global _default_client

    if _default_client is None:
        _default_client = TemporalClient(
            temporal_address=temporal_address,
            temporal_namespace=temporal_namespace,
            task_queue=task_queue,
        )
        await _default_client.connect()

    return _default_client


def set_client(client: TemporalClient) -> None:
    """Set the default Temporal client.

    Args:
        client: The TemporalClient instance to use as default.
    """
    global _default_client
    _default_client = client


async def close_client() -> None:
    """Close the default Temporal client connection."""
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None
