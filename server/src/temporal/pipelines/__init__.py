"""Pipeline definitions for deterministic analysis pipelines.

This module provides the base pipeline interface and registry for discovering
and executing pipeline definitions. Pipelines are separated from workflows
to maintain clean separation of concerns.

Pipelines define:
- Fixed sequences of steps (tools/agents)
- Step parameter resolution
- Result consolidation logic

Workflows handle:
- Temporal execution semantics
- State management
- Query/signal handling
"""

from src.temporal.pipelines.base import PipelineBase, PipelineStep
from src.temporal.pipelines.registry import (
    discover_pipelines,
    get_pipeline,
    list_available_pipelines,
    register_pipeline,
)

__all__ = [
    "PipelineBase",
    "PipelineStep",
    "discover_pipelines",
    "get_pipeline",
    "list_available_pipelines",
    "register_pipeline",
]
