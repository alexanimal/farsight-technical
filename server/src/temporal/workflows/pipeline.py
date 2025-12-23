"""Pipeline workflows for deterministic, scripted analysis pipelines.

Pipeline workflows are pre-defined, deterministic sequences of steps (tools/agents)
that produce consistent, repeatable analysis. They differ from orchestrator workflows
in that they:

1. Have fixed execution sequences (no dynamic planning)
2. Call tools/agents in a predetermined order
3. Are faster and more predictable
4. Produce consistent outputs for the same inputs
5. Are ideal for standardized analysis patterns

Pipeline workflows are useful for:
- Standardized sector analysis
- Regular reporting pipelines
- Batch processing jobs
- Deterministic data transformations

Architecture:
    Pipeline Workflow → Activities (tools/agents) → Results
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from temporalio import workflow
from temporalio.common import RetryPolicy

from src.temporal.pipelines.base import PipelineBase, PipelineStep
from src.temporal.pipelines.registry import get_pipeline, list_available_pipelines
from src.temporal.queries import (
    QUERY_AGENT_STATUS,
    QUERY_PROGRESS,
    QUERY_STATE,
    QUERY_STATUS,
    AgentExecutionStatus,
    AgentStatusQueryResult,
    WorkflowProgressQueryResult,
    WorkflowStateQueryResult,
    WorkflowStatus,
    WorkflowStatusQueryResult,
)
from src.temporal.signals import (
    SIGNAL_CANCELLATION,
    SIGNAL_USER_INPUT,
    CancellationSignal,
    UserInputSignal,
)

logger = logging.getLogger(__name__)


class PipelineState(BaseModel):
    """Internal pipeline workflow state."""

    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING, description="Current workflow status"
    )
    started_at: Optional[datetime] = Field(
        default=None, description="Timestamp when workflow started"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Timestamp when workflow completed"
    )
    error: Optional[str] = Field(default=None, description="Error message if workflow failed")
    pipeline_type: str = Field(default="", description="Type of pipeline (e.g., 'sector_analysis')")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Initial context with pipeline inputs"
    )
    steps: List[PipelineStep] = Field(
        default_factory=list, description="List of pipeline steps to execute"
    )
    step_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Results from each step execution"
    )
    final_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Final consolidated pipeline result"
    )
    cancellation_requested: bool = Field(
        default=False, description="Whether cancellation has been requested"
    )
    cancellation_reason: Optional[str] = Field(
        default=None, description="Reason for cancellation if requested"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the pipeline"
    )


@workflow.defn(name="pipeline")
class PipelineWorkflow:
    """Pipeline workflow for deterministic, scripted analysis pipelines.

    This workflow executes a pre-defined sequence of steps (tools/agents)
    in a fixed order. Unlike the orchestrator, it does not use LLM-based
    planning or dynamic agent selection.

    Pipeline workflows are ideal for:
    - Standardized analysis patterns
    - Repeatable data processing
    - Batch jobs with fixed steps
    - Deterministic transformations

    Example pipeline types:
    - sector_analysis: Analyze funding trends for a sector
    - portfolio_analysis: Analyze an investor's portfolio
    - acquisition_analysis: Identify acquisition targets
    """

    def __init__(self) -> None:
        """Initialize the pipeline workflow."""
        self._state = PipelineState()
        self._pipeline_instance: Optional[PipelineBase] = (
            None  # Will be set when pipeline is loaded
        )

    @workflow.run
    async def run(
        self,
        pipeline_type: Any = None,
        context: Any = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a pipeline workflow.

        Args:
            pipeline_type: Type of pipeline to execute (e.g., "sector_analysis").
            context: Initial context dictionary with pipeline inputs.
            pipeline_config: Optional configuration to override defaults.

        Returns:
            Dictionary containing:
            - success: bool indicating if pipeline completed successfully
            - final_result: Final pipeline result
            - step_results: Results from each step
            - workflow_id: Workflow ID
            - metadata: Additional metadata
        """
        # Handle case where Temporal UI passes arguments in different formats
        # When Temporal UI passes a single JSON object, it may pass it as the first argument only
        # We need to detect this and unpack it

        # Case 1: pipeline_type is a dict containing all parameters (from Temporal UI JSON input)
        # This is the most common case when using Temporal UI - single JSON object passed as first arg
        if isinstance(pipeline_type, dict) and "pipeline_type" in pipeline_type:
            # Unpack from dict: {"pipeline_type": "...", "context": {...}, "pipeline_config": {...}}
            input_dict = pipeline_type
            pipeline_type = input_dict.get("pipeline_type", "")
            context = input_dict.get("context", {})
            pipeline_config = input_dict.get("pipeline_config") or pipeline_config
            workflow.logger.info(
                f"Detected single dict input from Temporal UI, unpacked: "
                f"pipeline_type={pipeline_type}, context_keys={list(context.keys())}"
            )
        # Case 2: pipeline_type is a dict but context is None/missing (Temporal passed single dict)
        elif isinstance(pipeline_type, dict) and context is None:
            # Temporal UI passed the JSON object as first arg, context wasn't provided
            if "pipeline_type" in pipeline_type:
                input_dict = pipeline_type
                pipeline_type = input_dict.get("pipeline_type", "")
                context = input_dict.get("context", {})
                pipeline_config = input_dict.get("pipeline_config") or pipeline_config
                workflow.logger.info(
                    f"Detected single dict input (context was None), unpacked: "
                    f"pipeline_type={pipeline_type}, context_keys={list(context.keys())}"
                )
            else:
                # Dict doesn't have pipeline_type key - might be malformed
                workflow.logger.warning(
                    f"pipeline_type is a dict but missing 'pipeline_type' key. "
                    f"Keys: {list(pipeline_type.keys())}"
                )
                # Try to use it as context and require pipeline_type to be set
                context = pipeline_type
                pipeline_type = ""  # Will fail validation with helpful error
        # Case 3: Arguments passed as a list
        elif isinstance(pipeline_type, list) and len(pipeline_type) >= 1:
            # Unpack: [pipeline_type, context, pipeline_config]
            pipeline_type, context, pipeline_config = (
                pipeline_type[0] if len(pipeline_type) > 0 else "",
                pipeline_type[1] if len(pipeline_type) > 1 else {},
                pipeline_type[2] if len(pipeline_type) > 2 else None,
            )
            workflow.logger.info(
                f"Detected list input, unpacked: pipeline_type={pipeline_type}, "
                f"context_keys={list(context.keys()) if isinstance(context, dict) else 'N/A'}"
            )

        # Set defaults if still None
        if context is None:
            context = {}
        if pipeline_type is None:
            pipeline_type = ""

        # Validate pipeline_type is a string
        if not isinstance(pipeline_type, str):
            raise ValueError(
                f"Expected pipeline_type to be a string, got {type(pipeline_type).__name__}. "
                f"Pipeline type must be a string like 'sector_analysis'. "
                f'If passing from Temporal UI, use format: {{"pipeline_type": "...", "context": {{...}}}}'
            )

        # Validate context is a dict
        if not isinstance(context, dict):
            raise ValueError(
                f"Expected context to be a dict, got {type(context).__name__}. "
                "Context must be a dictionary with pipeline inputs."
            )

        workflow_id = workflow.info().workflow_id
        self._state.pipeline_type = pipeline_type
        self._state.context = context
        self._state.status = WorkflowStatus.RUNNING
        self._state.started_at = workflow.now()
        self._state.metadata["workflow_id"] = workflow_id

        workflow.logger.info(f"Starting pipeline workflow {workflow_id} of type: {pipeline_type}")

        try:
            # Get pipeline class from registry
            pipeline_class = get_pipeline(pipeline_type)
            if pipeline_class is None:
                raise ValueError(
                    f"Unknown pipeline type: {pipeline_type}. "
                    f"Available pipelines: {list_available_pipelines()}"
                )

            # Instantiate pipeline
            self._pipeline_instance = pipeline_class()
            workflow.logger.info(
                f"Loaded pipeline: {self._pipeline_instance.name} "
                f"(version: {self._pipeline_instance.config.version})"
            )

            # Build pipeline steps using the pipeline instance
            steps = self._pipeline_instance.build_steps(context, pipeline_config)
            self._state.steps = steps

            workflow.logger.info(f"Pipeline {pipeline_type} has {len(steps)} steps to execute")

            # Execute steps sequentially (pipelines are deterministic)
            step_results: Dict[str, Dict[str, Any]] = {}
            pipeline_data = context.copy()  # Accumulate data through pipeline

            for step in steps:
                # Check for cancellation
                if self._state.cancellation_requested:
                    workflow.logger.info("Cancellation requested, stopping pipeline")
                    break

                # Execute step
                step.status = "running"
                step.started_at = workflow.now()

                try:
                    if step.step_type == "tool":
                        result = await self._execute_tool_step(step, pipeline_data, step_results)
                    elif step.step_type == "agent":
                        result = await self._execute_agent_step(step, pipeline_data, step_results)
                    else:
                        raise ValueError(f"Unknown step type: {step.step_type}")

                    step.status = "completed" if result.get("success") else "failed"
                    step.result = result
                    step.error = result.get("error")
                    step.completed_at = workflow.now()

                    # Store step result (even if failed) so subsequent steps can access it
                    # Tools return result["result"], agents return result["response"]
                    step_result = None
                    if step.step_type == "agent" and result.get("response") is not None:
                        step_result = result["response"]
                    elif result.get("result") is not None:
                        step_result = result["result"]

                    if step_result is not None:
                        step_results[step.step_id] = step_result
                        # Merge successful step result into pipeline_data for subsequent steps
                        # Only merge if result is a dictionary (lists can't be merged with update())
                        if result.get("success") and isinstance(step_result, dict):
                            pipeline_data.update(step_result)
                        elif result.get("success") and isinstance(step_result, list):
                            # Store lists under the step_id key for reference
                            pipeline_data[step.step_id] = step_result

                    workflow.logger.info(
                        f"Step {step.step_id} ({step.name}) completed: " f"{step.status}"
                    )

                except Exception as e:
                    error_msg = f"Step {step.step_id} failed: {str(e)}"
                    workflow.logger.error(error_msg, exc_info=True)
                    step.status = "failed"
                    step.error = error_msg
                    step.completed_at = workflow.now()

                    # Decide whether to continue or fail pipeline
                    # For now, fail fast on any step failure
                    self._state.error = error_msg
                    self._state.status = WorkflowStatus.FAILED
                    self._state.completed_at = workflow.now()
                    return {
                        "success": False,
                        "final_result": None,
                        "step_results": step_results,
                        "workflow_id": workflow_id,
                        "error": error_msg,
                        "metadata": self._state.metadata,
                    }

            # All steps completed successfully
            self._state.step_results = step_results
            if self._pipeline_instance is None:
                raise RuntimeError("Pipeline instance not initialized. This should not happen.")
            self._state.final_result = self._pipeline_instance.build_final_result(
                step_results, pipeline_data, context
            )
            self._state.status = WorkflowStatus.COMPLETED
            self._state.completed_at = workflow.now()

            workflow.logger.info(
                f"Pipeline {pipeline_type} completed successfully with "
                f"{len(step_results)} step results"
            )

            return {
                "success": True,
                "final_result": self._state.final_result,
                "step_results": step_results,
                "workflow_id": workflow_id,
                "metadata": self._state.metadata,
            }

        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            workflow.logger.error(error_msg, exc_info=True)
            self._state.status = WorkflowStatus.FAILED
            self._state.error = error_msg
            self._state.completed_at = workflow.now()
            raise

    async def _execute_tool_step(
        self,
        step: PipelineStep,
        pipeline_data: Dict[str, Any],
        step_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a tool step.

        Args:
            step: PipelineStep to execute.
            pipeline_data: Accumulated data from previous steps.
            step_results: Dictionary mapping step_id to step results from previous steps.

        Returns:
            Execution result dictionary.
        """
        # Resolve parameters using the pipeline instance's method
        if self._pipeline_instance is None:
            raise RuntimeError("Pipeline instance not initialized. This should not happen.")
        resolved_params = self._pipeline_instance.resolve_step_parameters(
            step, pipeline_data, step_results
        )

        workflow.logger.info(
            f"Executing tool step {step.step_id}: {step.name} with parameters: "
            f"{resolved_params}"
        )

        # Execute tool via activity
        result = await workflow.execute_activity(
            "execute_tool",
            args=[step.name, resolved_params],
            start_to_close_timeout=timedelta(seconds=300),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=60),
                maximum_attempts=3,
            ),
        )

        return result

    async def _execute_agent_step(
        self,
        step: PipelineStep,
        pipeline_data: Dict[str, Any],
        step_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute an agent step.

        Args:
            step: PipelineStep to execute.
            pipeline_data: Accumulated data from previous steps.
            step_results: Dictionary mapping step_id to step results from previous steps.

        Returns:
            Execution result dictionary.
        """
        # Build agent context from pipeline_data
        agent_context = {
            "query": step.parameters.get("query", ""),
            "conversation_history": [],
            "metadata": {
                "pipeline_type": self._state.pipeline_type,
                "pipeline_data": pipeline_data,
            },
        }

        workflow.logger.info(
            f"Executing agent step {step.step_id}: {step.name} with context: "
            f"{agent_context.get('query', '')[:100]}"
        )

        # Execute agent via activity
        result = await workflow.execute_activity(
            "execute_agent",
            args=[step.name, agent_context],
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=120),
                maximum_attempts=3,
            ),
        )

        return result

    @workflow.query(name=QUERY_STATUS)
    def query_status(self) -> WorkflowStatusQueryResult:
        """Query pipeline workflow status."""
        workflow_id = workflow.info().workflow_id
        return WorkflowStatusQueryResult(
            workflow_id=workflow_id,
            status=self._state.status,
            started_at=self._state.started_at or workflow.now(),
            completed_at=self._state.completed_at,
            error=self._state.error,
            metadata={
                **self._state.metadata,
                "pipeline_type": self._state.pipeline_type,
            },
        )

    @workflow.query(name=QUERY_PROGRESS)
    def query_progress(self) -> WorkflowProgressQueryResult:
        """Query pipeline workflow progress."""
        workflow_id = workflow.info().workflow_id

        total_steps = len(self._state.steps)
        completed_steps = sum(1 for step in self._state.steps if step.status == "completed")
        running_steps = sum(1 for step in self._state.steps if step.status == "running")
        failed_steps = sum(1 for step in self._state.steps if step.status == "failed")

        progress_percentage = (completed_steps / total_steps * 100.0) if total_steps > 0 else None

        # Build agent statuses from steps
        agent_statuses = []
        for step in self._state.steps:
            if step.step_type == "agent":
                agent_statuses.append(
                    AgentExecutionStatus(
                        agent_name=step.name,
                        agent_category="pipeline",
                        status=step.status,
                        started_at=step.started_at or workflow.now(),
                        completed_at=step.completed_at,
                        error=step.error,
                        metadata=step.metadata,
                    )
                )

        current_step = None
        running_step = next((step for step in self._state.steps if step.status == "running"), None)
        if running_step:
            current_step = f"Executing {running_step.step_id}: {running_step.name}"

        return WorkflowProgressQueryResult(
            workflow_id=workflow_id,
            status=self._state.status,
            total_agents=len([s for s in self._state.steps if s.step_type == "agent"]),
            completed_agents=len(
                [s for s in self._state.steps if s.step_type == "agent" and s.status == "completed"]
            ),
            running_agents=running_steps,
            failed_agents=failed_steps,
            agent_statuses=agent_statuses,
            progress_percentage=progress_percentage,
            current_step=current_step,
            iteration_number=None,
            metadata={
                **self._state.metadata,
                "pipeline_type": self._state.pipeline_type,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
            },
        )

    @workflow.query(name=QUERY_STATE)
    def query_state(self) -> WorkflowStateQueryResult:
        """Query full pipeline workflow state."""
        workflow_id = workflow.info().workflow_id
        return WorkflowStateQueryResult(
            workflow_id=workflow_id,
            status=self._state.status,
            context=self._state.context,
            agent_responses=[
                response
                for step in self._state.steps
                if step.step_type == "agent"
                and step.result is not None
                and isinstance(step.result, dict)
                for response in [step.result.get("response")]
                if response is not None
            ],
            shared_data=self._state.step_results,
            execution_history=[
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type,
                    "name": step.name,
                    "status": step.status,
                    "timestamp": (step.completed_at.isoformat() if step.completed_at else None),
                }
                for step in self._state.steps
            ],
            metadata={
                **self._state.metadata,
                "pipeline_type": self._state.pipeline_type,
                "final_result": self._state.final_result,
            },
        )

    @workflow.signal(name=SIGNAL_CANCELLATION)
    async def handle_cancellation(self, signal: CancellationSignal) -> None:
        """Handle cancellation signal."""
        workflow.logger.info(f"Cancellation requested: {signal.reason} by {signal.requested_by}")
        self._state.cancellation_requested = True
        self._state.cancellation_reason = signal.reason or "User requested cancellation"
        self._state.metadata["cancellation_requested_by"] = signal.requested_by
        self._state.metadata["cancellation_timestamp"] = signal.timestamp.isoformat()
