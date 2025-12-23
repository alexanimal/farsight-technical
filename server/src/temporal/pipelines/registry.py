"""Pipeline registry for discovering available pipelines.

This module provides pipeline discovery and registration functionality.
Pipelines are discovered from the pipelines directory and can be used
by workflows without direct imports.

According to the architecture:
- Pipelines must not import Temporal
- Pipeline discovery should be available to workflows
- Pipelines should be testable in isolation
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from src.temporal.pipelines.base import PipelineBase

logger = logging.getLogger(__name__)

# Pipeline registry mapping pipeline names to their classes
# Format: {pipeline_name: pipeline_class}
_PIPELINE_REGISTRY: Dict[str, Type[PipelineBase]] = {}


def register_pipeline(name: str, pipeline_class: Type[PipelineBase]) -> None:
    """Register a pipeline class in the pipeline registry.

    Args:
        name: The name of the pipeline (e.g., "sector_analysis").
        pipeline_class: The pipeline class (must inherit from PipelineBase).
    """
    if not issubclass(pipeline_class, PipelineBase):
        raise ValueError(f"Pipeline {name} must inherit from PipelineBase")

    _PIPELINE_REGISTRY[name] = pipeline_class
    logger.info(f"Registered pipeline: {name}")


def get_pipeline(name: str) -> Optional[Type[PipelineBase]]:
    """Get a pipeline class by name.

    Args:
        name: The name of the pipeline.

    Returns:
        The pipeline class if found, None otherwise.
    """
    return _PIPELINE_REGISTRY.get(name)


def list_available_pipelines() -> list[str]:
    """List all available pipeline names.

    Returns:
        List of available pipeline names.
    """
    return list(_PIPELINE_REGISTRY.keys())


def get_pipeline_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a pipeline by name.

    This is useful for workflows that need to know about available
    pipelines without instantiating them.

    Args:
        name: The name of the pipeline.

    Returns:
        Dictionary with pipeline metadata (name, description, version, etc.)
        or None if pipeline not found.
    """
    pipeline_class = get_pipeline(name)
    if pipeline_class is None:
        return None

    # Instantiate pipeline to get config
    try:
        pipeline = pipeline_class()
        return {
            "name": pipeline.name,
            "description": pipeline.description,
            "version": pipeline.config.version,
            "metadata": pipeline.config.metadata,
        }
    except Exception as e:
        logger.warning(f"Failed to get metadata for pipeline {name}: {e}")
        return {
            "name": name,
            "description": f"Pipeline: {name}",
            "version": "1.0.0",
            "metadata": {},
        }


def discover_pipelines() -> None:
    """Discover and register pipelines from the pipelines directory.

    This function attempts to auto-discover pipelines by looking in the
    src/temporal/pipelines directory. Pipelines should be in:
    src/temporal/pipelines/{pipeline_name}.py

    This function is defensive and will skip pipelines that can't be imported
    or don't have the expected structure.

    This function can be called multiple times safely (idempotent).
    """
    # Base path: src/temporal/pipelines
    pipelines_base = Path(__file__).parent

    # Known pipeline names (can be extended)
    # Pipelines can be registered manually via register_pipeline() if auto-discovery fails
    known_pipelines = [
        "sector_analysis",
        # Add more pipeline names here as they are created
    ]

    for pipeline_name in known_pipelines:
        # Skip if already registered
        if pipeline_name in _PIPELINE_REGISTRY:
            logger.debug(f"Pipeline {pipeline_name} already registered, skipping")
            continue

        try:
            # Try to import the pipeline module
            module_path = f"src.temporal.pipelines.{pipeline_name}"
            try:
                pipeline_module = importlib.import_module(module_path)
            except (ImportError, ModuleNotFoundError) as e:
                logger.debug(
                    f"Could not import pipeline module {module_path}: {e}. "
                    "Pipeline may not be implemented yet."
                )
                continue

            # Try to find the pipeline class
            # Look for classes that end with "Pipeline" or match the pipeline name
            pipeline_class = None
            for attr_name in dir(pipeline_module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(pipeline_module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, PipelineBase)
                    and attr != PipelineBase
                ):
                    pipeline_class = attr
                    break

            if pipeline_class is None:
                logger.debug(
                    f"No PipelineBase subclass found in {module_path}. "
                    "Pipeline may not be implemented yet."
                )
                continue

            # Register the pipeline
            register_pipeline(pipeline_name, pipeline_class)

        except Exception as e:
            logger.warning(f"Failed to register pipeline {pipeline_name} during discovery: {e}")


# Initialize pipeline registry on module load
try:
    discover_pipelines()
except Exception as e:
    logger.warning(f"Failed to auto-discover pipelines: {e}")
