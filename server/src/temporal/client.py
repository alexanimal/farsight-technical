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
from datetime import timedelta
from typing import Any, Dict, List, Optional

from temporalio.client import Client

# Note: AgentContext is not imported here to avoid workflow determinism issues
# The client methods accept Dict[str, Any] for context, not AgentContext objects
from src.temporal.queries import (QUERY_AGENT_STATUS, QUERY_PROGRESS,
                                  QUERY_STATE, QUERY_STATUS,
                                  AgentStatusQueryResult,
                                  WorkflowProgressQueryResult,
                                  WorkflowStateQueryResult,
                                  WorkflowStatusQueryResult)
from src.temporal.signals import (SIGNAL_CANCELLATION, SIGNAL_USER_INPUT,
                                  CancellationSignal, UserInputSignal)

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
            # Build kwargs, only include id if provided
            workflow_kwargs: Dict[str, Any] = {
                "args": [context, agent_plan, execution_mode],
                "task_queue": self._task_queue,
            }
            if workflow_id is not None:
                workflow_kwargs["id"] = workflow_id
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

    async def query_workflow_status(
        self, workflow_id: str
    ) -> WorkflowStatusQueryResult:
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

    async def query_workflow_progress(
        self, workflow_id: str
    ) -> WorkflowProgressQueryResult:
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

    async def query_agent_status(
        self, workflow_id: str, agent_name: str
    ) -> AgentStatusQueryResult:
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
