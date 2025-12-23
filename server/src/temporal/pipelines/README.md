# Pipeline Architecture

This directory contains pipeline definitions that are separate from Temporal workflows. This separation provides:

- **Clean separation of concerns**: Pipelines define *what* to execute, workflows handle *how* to execute
- **Testability**: Pipelines can be tested in isolation without Temporal
- **Reusability**: Pipeline logic can be reused in different contexts
- **Maintainability**: Each pipeline type is in its own file

## Structure

```
pipelines/
├── __init__.py          # Module exports
├── base.py             # PipelineBase abstract class
├── registry.py         # Pipeline discovery and registration
├── sector_analysis.py  # Sector analysis pipeline implementation
└── README.md          # This file
```

## Pipeline Base Class

All pipelines inherit from `PipelineBase` which defines the interface:

- `build_steps()`: Define the sequence of steps
- `resolve_step_parameters()`: Resolve parameters from previous step results
- `build_final_result()`: Consolidate step results into final output

## Creating a New Pipeline

1. Create a new file: `pipelines/your_pipeline.py`
2. Inherit from `PipelineBase`:

```python
from src.temporal.pipelines.base import PipelineBase, PipelineConfig, PipelineStep

class YourPipeline(PipelineBase):
    def __init__(self, config: Optional[PipelineConfig] = None):
        if config is None:
            config = PipelineConfig(
                name="your_pipeline",
                description="Description of your pipeline",
                version="1.0.0",
            )
        super().__init__(config)

    def build_steps(
        self,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[PipelineStep]:
        # Define your steps here
        return [
            PipelineStep(
                step_id="step_1",
                step_type="tool",
                name="tool_name",
                parameters={...},
            ),
            # ... more steps
        ]

    def resolve_step_parameters(
        self,
        step: PipelineStep,
        pipeline_data: Dict[str, Any],
        step_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Override if you need custom parameter resolution
        return super().resolve_step_parameters(step, pipeline_data, step_results)

    def build_final_result(
        self,
        step_results: Dict[str, Dict[str, Any]],
        pipeline_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Override to customize final result structure
        return super().build_final_result(step_results, pipeline_data, context)
```

3. Add your pipeline name to `registry.py` in the `known_pipelines` list
4. The registry will auto-discover and register your pipeline

## Pipeline Registry

The registry automatically discovers pipelines from the `pipelines/` directory. Pipelines are registered by name and can be retrieved by workflows.

### Usage in Workflows

```python
from src.temporal.pipelines.registry import get_pipeline

# Get pipeline class
pipeline_class = get_pipeline("sector_analysis")

# Instantiate
pipeline = pipeline_class()

# Build steps
steps = pipeline.build_steps(context, config)
```

## Example: Sector Analysis Pipeline

The `sector_analysis.py` pipeline demonstrates:

1. **Step Definition**: Fixed sequence of 5 steps
2. **Parameter Resolution**: Custom logic to resolve `org_uuids` and `trend_data` from previous steps
3. **Result Consolidation**: Structured final result with all analysis components

## Benefits of This Architecture

1. **Separation**: Pipeline logic is separate from Temporal workflow code
2. **Testability**: Pipelines can be unit tested without Temporal
3. **Extensibility**: Easy to add new pipeline types
4. **Maintainability**: Each pipeline is self-contained
5. **Reusability**: Pipeline logic can be reused in different contexts

## Workflow Integration

The `PipelineWorkflow` in `workflows/pipeline.py` uses the registry to:

1. Load the appropriate pipeline class
2. Instantiate the pipeline
3. Use the pipeline's methods to build steps, resolve parameters, and build results

This keeps the workflow focused on execution semantics while pipelines handle domain logic.

