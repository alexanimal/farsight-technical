"""Base class for pipeline definitions.

Pipelines define deterministic sequences of steps (tools/agents) that
produce consistent, repeatable analysis. They are separate from workflows
to maintain clean separation between execution logic and step definitions.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineStep(BaseModel):
    """Represents a single step in a pipeline execution."""

    step_id: str = Field(..., description="Unique identifier for the step")
    step_type: str = Field(..., description="Type of step: 'tool' or 'agent'")
    name: str = Field(..., description="Tool name or agent name to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters to pass to the tool or agent"
    )
    status: str = Field(
        default="pending",
        description="Current status: 'pending', 'running', 'completed', or 'failed'",
    )
    started_at: Optional[datetime] = Field(
        default=None, description="Timestamp when step execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Timestamp when step execution completed"
    )
    result: Optional[Dict[str, Any]] = Field(default=None, description="Result from step execution")
    error: Optional[str] = Field(default=None, description="Error message if step failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the step"
    )


class PipelineConfig(BaseModel):
    """Configuration for a pipeline definition."""

    name: str = Field(..., description="Unique identifier for the pipeline")
    description: str = Field(
        ..., description="Human-readable description of what the pipeline does"
    )
    version: str = Field(default="1.0.0", description="Pipeline version for compatibility tracking")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the pipeline"
    )


class PipelineBase(ABC):
    """Base class for all pipeline definitions.

    Pipelines define:
    - A fixed sequence of steps (tools/agents)
    - How to resolve step parameters from context and previous results
    - How to consolidate final results from step outputs

    Pipelines are deterministic and do not contain Temporal-specific logic.
    They can be tested in isolation without Temporal.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline.

        Args:
            config: Optional pipeline configuration. If not provided,
                uses default configuration.
        """
        self.config = config or PipelineConfig(
            name=self.__class__.__name__.lower().replace("pipeline", ""),
            description=f"Pipeline: {self.__class__.__name__}",
        )

    @property
    def name(self) -> str:
        """Get the pipeline's name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get the pipeline's description."""
        return self.config.description

    @abstractmethod
    def build_steps(
        self,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[PipelineStep]:
        """Build the list of steps for this pipeline.

        This method defines the fixed sequence of steps that the pipeline
        will execute. Steps are executed in order, with each step's results
        available to subsequent steps.

        Args:
            context: Initial context dictionary with pipeline inputs.
                Contains user-provided parameters like sector_name, time_period, etc.
            config: Optional configuration overrides for the pipeline.

        Returns:
            List of PipelineStep objects in execution order.
        """
        pass

    def resolve_step_parameters(
        self,
        step: PipelineStep,
        pipeline_data: Dict[str, Any],
        step_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve step parameters that reference previous step results.

        This method allows steps to reference data from previous steps.
        Override this method to customize parameter resolution logic
        for specific pipeline types.

        Args:
            step: The PipelineStep to resolve parameters for.
            pipeline_data: Accumulated data from all previous steps.
            step_results: Dictionary mapping step_id to step results.

        Returns:
            Resolved parameters dictionary.
        """
        resolved = {}
        for key, value in step.parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Reference to previous step result: ${step_id.field}
                ref_path = value[2:-1]  # Remove ${ and }
                parts = ref_path.split(".", 1)
                if len(parts) == 2:
                    step_id, field = parts
                    if step_id in step_results:
                        step_result = step_results[step_id]
                        if field in step_result:
                            resolved[key] = step_result[field]
                        else:
                            resolved[key] = value  # Keep unresolved
                    else:
                        resolved[key] = value  # Keep unresolved
                else:
                    # Direct reference to pipeline_data
                    if ref_path in pipeline_data:
                        resolved[key] = pipeline_data[ref_path]
                    else:
                        resolved[key] = value  # Keep unresolved
            else:
                resolved[key] = value

        return resolved

    def build_final_result(
        self,
        step_results: Dict[str, Dict[str, Any]],
        pipeline_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the final consolidated result from pipeline step results.

        This method consolidates all step results into a final output.
        Override this method to customize result consolidation for specific
        pipeline types.

        Args:
            step_results: Dictionary mapping step_id to step results.
            pipeline_data: Accumulated data from all steps.
            context: Original pipeline context.

        Returns:
            Final consolidated result dictionary.
        """
        return {
            "pipeline_name": self.name,
            "pipeline_version": self.config.version,
            "step_results": step_results,
            "pipeline_data": pipeline_data,
            "context": context,
        }
