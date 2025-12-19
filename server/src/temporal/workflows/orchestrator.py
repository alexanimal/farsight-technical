"""Main orchestration workflow for agent execution.

This workflow coordinates agent execution, handles signals and queries,
and manages workflow state. It contains decision logic only - all
execution happens through activities.

Workflow responsibilities:
- Decide which agents to run
- Decide execution order and parallelism
- Track task state
- Respond to signals
- Expose queries
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus
from src.temporal.queries import (QUERY_AGENT_STATUS, QUERY_PROGRESS,
                                  QUERY_STATE, QUERY_STATUS,
                                  AgentExecutionStatus, AgentStatusQueryResult,
                                  WorkflowProgressQueryResult,
                                  WorkflowStateQueryResult, WorkflowStatus,
                                  WorkflowStatusQueryResult)
from src.temporal.signals import (SIGNAL_CANCELLATION, SIGNAL_USER_INPUT,
                                  CancellationSignal, UserInputSignal)

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutionState:
    """State tracking for a single agent execution."""

    agent_name: str
    agent_category: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """Internal workflow state."""

    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    agent_executions: Dict[str, AgentExecutionState] = field(default_factory=dict)
    agent_responses: List[Dict[str, Any]] = field(default_factory=list)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    cancellation_requested: bool = False
    cancellation_reason: Optional[str] = None
    user_inputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@workflow.defn(name="orchestrator")
class OrchestratorWorkflow:
    """Main orchestration workflow for coordinating agent execution.

    This workflow:
    1. Receives initial context and agent execution plan
    2. Executes agents in sequence or parallel based on plan
    3. Handles signals (cancellation, user input)
    4. Exposes queries (status, progress, state)
    5. Returns final results
    """

    def __init__(self) -> None:
        """Initialize the workflow."""
        self._state = WorkflowState()
        # Signals are handled via @workflow.signal decorators, not ExternalSignal

    @workflow.run
    async def run(
        self,
        context: Dict[str, Any],
        agent_plan: Optional[List[str]] = None,
        execution_mode: str = "sequential",  # sequential, parallel
    ) -> Dict[str, Any]:
        """Run the orchestration workflow.

        Args:
            context: Initial AgentContext as dictionary (query, conversation_id, etc.).
            agent_plan: Optional list of agent names to execute in order.
                If None, starts with orchestration agent to determine plan.
            execution_mode: Execution mode - "sequential" or "parallel".

        Returns:
            Dictionary containing:
            - success: bool indicating if workflow completed successfully
            - final_response: Final AgentResponse (if successful)
            - agent_responses: List of all agent responses
            - workflow_id: Workflow ID
            - metadata: Additional metadata
        """
        workflow_id = workflow.info().workflow_id
        self._state.context = context
        self._state.status = WorkflowStatus.RUNNING
        self._state.metadata["workflow_id"] = workflow_id
        self._state.metadata["execution_mode"] = execution_mode

        workflow.logger.info(
            f"Starting orchestrator workflow {workflow_id} with query: {context.get('query', 'N/A')[:100]}"
        )

        try:
            # If no agent plan provided, start with orchestration agent
            if agent_plan is None:
                agent_plan = await self._determine_agent_plan(context)

            # Execute agents according to plan
            if execution_mode == "parallel":
                results = await self._execute_agents_parallel(agent_plan, context)
            else:
                results = await self._execute_agents_sequential(agent_plan, context)

            # Determine final status
            if self._state.cancellation_requested:
                self._state.status = WorkflowStatus.CANCELLED
                self._state.completed_at = datetime.utcnow()
                return {
                    "success": False,
                    "final_response": None,
                    "agent_responses": self._state.agent_responses,
                    "workflow_id": workflow_id,
                    "metadata": {
                        "cancelled": True,
                        "cancellation_reason": self._state.cancellation_reason,
                    },
                }

            # Check if any agents failed
            failed_agents = [
                name
                for name, exec_state in self._state.agent_executions.items()
                if exec_state.status == "failed"
            ]

            if failed_agents:
                self._state.status = WorkflowStatus.FAILED
                self._state.error = f"Agents failed: {', '.join(failed_agents)}"
            else:
                self._state.status = WorkflowStatus.COMPLETED

            self._state.completed_at = datetime.utcnow()

            # Get final response (last successful agent response, or orchestration response)
            final_response = None
            if self._state.agent_responses:
                # Try to find orchestration agent response first
                for response in reversed(self._state.agent_responses):
                    if response.get("agent_category") == "orchestration":
                        final_response = response
                        break
                if final_response is None:
                    final_response = self._state.agent_responses[-1]

            return {
                "success": self._state.status == WorkflowStatus.COMPLETED,
                "final_response": final_response,
                "agent_responses": self._state.agent_responses,
                "workflow_id": workflow_id,
                "metadata": self._state.metadata,
            }

        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            workflow.logger.error(error_msg, exc_info=True)
            self._state.status = WorkflowStatus.FAILED
            self._state.error = error_msg
            self._state.completed_at = datetime.utcnow()
            raise

    async def _determine_agent_plan(self, context: Dict[str, Any]) -> List[str]:
        """Determine which agents to execute using orchestration agent.

        Args:
            context: Initial context.

        Returns:
            List of agent names to execute.
        """
        workflow.logger.info("Determining agent plan using orchestration agent")

        # Execute orchestration agent to determine plan
        result = await self._execute_single_agent("orchestration", context)

        if not result.get("success"):
            # Fallback: if orchestration fails, return empty plan
            workflow.logger.warning("Orchestration agent failed, using empty plan")
            return []

        # Extract agent plan from orchestration response
        # This is a simplified version - in practice, the orchestration agent
        # would analyze the query and return a list of agents to execute
        response = result.get("response", {})
        content = response.get("content", {})

        # Try to extract agent plan from response
        # Format could be: {"agents": ["acquisition"], "reasoning": "..."}
        if isinstance(content, dict) and "agents" in content:
            agent_plan = content["agents"]
        elif isinstance(content, str):
            # Simple heuristic: if query mentions acquisition, use acquisition agent
            query = context.get("query", "").lower()
            agent_plan = []
            if "acquisition" in query or "acquired" in query:
                agent_plan.append("acquisition")
        else:
            agent_plan = []

        workflow.logger.info(f"Determined agent plan: {agent_plan}")
        return agent_plan

    async def _execute_single_agent(
        self, agent_name: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent and update state.

        Args:
            agent_name: Name of agent to execute.
            context: Context to pass to agent.

        Returns:
            Execution result dictionary.
        """
        # Create execution state
        exec_state = AgentExecutionState(
            agent_name=agent_name,
            agent_category="unknown",  # Will be updated after execution
            status="running",
            started_at=datetime.utcnow(),
        )
        self._state.agent_executions[agent_name] = exec_state

        workflow.logger.info(f"Executing agent: {agent_name}")

        try:
            # Execute agent via activity
            # Use activity name as string to avoid importing activities module
            # (which would pull in non-deterministic dependencies like langfuse/httpx)
            result = await workflow.execute_activity(
                "execute_agent",
                args=[agent_name, context],
                start_to_close_timeout=timedelta(seconds=600),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=2),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(seconds=120),
                    maximum_attempts=3,
                ),
            )

            exec_state.completed_at = datetime.utcnow()

            if result.get("success"):
                exec_state.status = "completed"
                response = result.get("response", {})
                exec_state.response = response
                exec_state.agent_category = response.get("agent_category", "unknown")

                # Add response to state
                self._state.agent_responses.append(response)

                # Update shared data if agent provides it
                if "shared_data" in response.get("metadata", {}):
                    self._state.shared_data.update(response["metadata"]["shared_data"])

                # Add to execution history
                self._state.execution_history.append(
                    {
                        "agent_name": agent_name,
                        "status": "completed",
                        "timestamp": exec_state.completed_at.isoformat(),
                    }
                )
            else:
                exec_state.status = "failed"
                exec_state.error = result.get("error", "Unknown error")

                # Add to execution history
                self._state.execution_history.append(
                    {
                        "agent_name": agent_name,
                        "status": "failed",
                        "error": exec_state.error,
                        "timestamp": exec_state.completed_at.isoformat(),
                    }
                )

            return result

        except Exception as e:
            error_msg = f"Agent {agent_name} execution raised exception: {str(e)}"
            workflow.logger.error(error_msg, exc_info=True)
            exec_state.status = "failed"
            exec_state.error = error_msg
            exec_state.completed_at = datetime.utcnow()

            self._state.execution_history.append(
                {
                    "agent_name": agent_name,
                    "status": "failed",
                    "error": error_msg,
                    "timestamp": exec_state.completed_at.isoformat(),
                }
            )

            return {
                "success": False,
                "error": error_msg,
                "agent_name": agent_name,
                "response": None,
            }

    async def _execute_agents_sequential(
        self, agent_plan: List[str], initial_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute agents sequentially, passing context between them.

        Args:
            agent_plan: List of agent names to execute.
            initial_context: Initial context.

        Returns:
            List of execution results.
        """
        results = []
        context = initial_context.copy()

        for agent_name in agent_plan:
            # Check for cancellation
            if self._state.cancellation_requested:
                workflow.logger.info("Cancellation requested, stopping execution")
                break

            # Check for user input
            await self._check_user_input(context)

            # Execute agent
            result = await self._execute_single_agent(agent_name, context)

            results.append(result)

            # Update context with agent response for next agent
            if result.get("success") and result.get("response"):
                response = result["response"]
                # Add agent response to context metadata
                if "metadata" not in context:
                    context["metadata"] = {}
                context["metadata"][f"{agent_name}_response"] = response

                # Update shared data in context
                if "shared_data" in response.get("metadata", {}):
                    context.setdefault("shared_data", {}).update(
                        response["metadata"]["shared_data"]
                    )

        return results

    async def _execute_agents_parallel(
        self, agent_plan: List[str], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute agents in parallel.

        Args:
            agent_plan: List of agent names to execute.
            context: Context to pass to all agents.

        Returns:
            List of execution results.
        """
        # Check for cancellation before starting
        if self._state.cancellation_requested:
            return []

        # Execute all agents in parallel using asyncio.gather
        # Note: In Temporal workflows, we use workflow.execute_activity
        # which handles parallelism correctly
        tasks = [
            self._execute_single_agent(agent_name, context) for agent_name in agent_plan
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "success": False,
                        "error": str(result),
                        "agent_name": agent_plan[i],
                        "response": None,
                    }
                )
            elif isinstance(result, dict):
                processed_results.append(result)
            else:
                # Fallback for unexpected types
                processed_results.append(
                    {
                        "success": False,
                        "error": f"Unexpected result type: {type(result)}",
                        "agent_name": agent_plan[i],
                        "response": None,
                    }
                )

        return processed_results

    async def _check_user_input(self, context: Dict[str, Any]) -> None:
        """Check for and process user input signals.

        Args:
            context: Context to potentially update with user input.
        """
        # This is a simplified version - in a real implementation,
        # you might wait for user input or process it differently
        # For now, we just log that we're checking
        pass

    @workflow.query(name=QUERY_STATUS)
    def query_status(self) -> WorkflowStatusQueryResult:
        """Query workflow status.

        Returns:
            WorkflowStatusQueryResult with current status.
        """
        workflow_id = workflow.info().workflow_id
        return WorkflowStatusQueryResult(
            workflow_id=workflow_id,
            status=self._state.status,
            started_at=self._state.started_at,
            completed_at=self._state.completed_at,
            error=self._state.error,
            metadata=self._state.metadata,
        )

    @workflow.query(name=QUERY_PROGRESS)
    def query_progress(self) -> WorkflowProgressQueryResult:
        """Query workflow progress.

        Returns:
            WorkflowProgressQueryResult with progress information.
        """
        workflow_id = workflow.info().workflow_id

        # Calculate progress metrics
        total_agents = len(self._state.agent_executions)
        completed_agents = sum(
            1
            for exec_state in self._state.agent_executions.values()
            if exec_state.status == "completed"
        )
        running_agents = sum(
            1
            for exec_state in self._state.agent_executions.values()
            if exec_state.status == "running"
        )
        failed_agents = sum(
            1
            for exec_state in self._state.agent_executions.values()
            if exec_state.status == "failed"
        )

        # Calculate progress percentage
        progress_percentage = None
        if total_agents > 0:
            progress_percentage = (completed_agents / total_agents) * 100.0

        # Build agent statuses
        agent_statuses = []
        for exec_state in self._state.agent_executions.values():
            agent_statuses.append(
                AgentExecutionStatus(
                    agent_name=exec_state.agent_name,
                    agent_category=exec_state.agent_category,
                    status=exec_state.status,
                    started_at=exec_state.started_at or datetime.utcnow(),
                    completed_at=exec_state.completed_at,
                    error=exec_state.error,
                    metadata=exec_state.metadata,
                )
            )

        # Determine current step
        current_step = None
        if running_agents > 0:
            running_agent = next(
                (
                    exec_state
                    for exec_state in self._state.agent_executions.values()
                    if exec_state.status == "running"
                ),
                None,
            )
            if running_agent:
                current_step = f"Executing {running_agent.agent_name}"

        return WorkflowProgressQueryResult(
            workflow_id=workflow_id,
            status=self._state.status,
            total_agents=total_agents,
            completed_agents=completed_agents,
            running_agents=running_agents,
            failed_agents=failed_agents,
            agent_statuses=agent_statuses,
            progress_percentage=progress_percentage,
            current_step=current_step,
            metadata=self._state.metadata,
        )

    @workflow.query(name=QUERY_STATE)
    def query_state(self) -> WorkflowStateQueryResult:
        """Query full workflow state.

        Returns:
            WorkflowStateQueryResult with complete state.
        """
        workflow_id = workflow.info().workflow_id
        return WorkflowStateQueryResult(
            workflow_id=workflow_id,
            status=self._state.status,
            context=self._state.context,
            agent_responses=self._state.agent_responses,
            shared_data=self._state.shared_data,
            execution_history=self._state.execution_history,
            metadata=self._state.metadata,
        )

    @workflow.query(name=QUERY_AGENT_STATUS)
    def query_agent_status(self, agent_name: str) -> AgentStatusQueryResult:
        """Query status of a specific agent.

        Args:
            agent_name: Name of the agent to query.

        Returns:
            AgentStatusQueryResult with agent status.
        """
        workflow_id = workflow.info().workflow_id

        exec_state = self._state.agent_executions.get(agent_name)
        if exec_state is None:
            return AgentStatusQueryResult(
                workflow_id=workflow_id,
                agent_name=agent_name,
                agent_category="unknown",
                status="not_found",
                started_at=datetime.utcnow(),
                error=f"Agent {agent_name} not found in workflow",
            )

        return AgentStatusQueryResult(
            workflow_id=workflow_id,
            agent_name=exec_state.agent_name,
            agent_category=exec_state.agent_category,
            status=exec_state.status,
            started_at=exec_state.started_at or datetime.utcnow(),
            completed_at=exec_state.completed_at,
            response=exec_state.response,
            error=exec_state.error,
            metadata=exec_state.metadata,
        )

    @workflow.signal(name=SIGNAL_CANCELLATION)
    async def handle_cancellation(self, signal: CancellationSignal) -> None:
        """Handle cancellation signal.

        Args:
            signal: Cancellation signal.
        """
        workflow.logger.info(
            f"Cancellation requested: {signal.reason} by {signal.requested_by}"
        )
        self._state.cancellation_requested = True
        self._state.cancellation_reason = signal.reason or "User requested cancellation"
        self._state.metadata["cancellation_requested_by"] = signal.requested_by
        self._state.metadata["cancellation_timestamp"] = signal.timestamp.isoformat()

    @workflow.signal(name=SIGNAL_USER_INPUT)
    async def handle_user_input(self, signal: UserInputSignal) -> None:
        """Handle user input signal.

        Args:
            signal: User input signal.
        """
        workflow.logger.info(f"User input received: {signal.input_text[:100]}")
        self._state.user_inputs.append(signal.model_dump())

        # Update context with user input
        if "user_inputs" not in self._state.context:
            self._state.context["user_inputs"] = []
        self._state.context["user_inputs"].append(signal.model_dump())
