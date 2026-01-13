"""Batch operation utilities for Temporal workflows.

This module provides helper functions for creating and managing batch operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_pipeline_batch_items(
    pipeline_type: str,
    contexts: List[Dict[str, Any]],
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build batch items for pipeline workflow execution.

    Args:
        pipeline_type: Type of pipeline to run (e.g., "sector_analysis").
        contexts: List of context dictionaries, one per batch item.
        pipeline_config: Optional pipeline configuration to apply to all items.

    Returns:
        List of item dictionaries ready for batch operation.

    Example:
        ```python
        contexts = [
            {"sector_name": "Technology", "time_period_start": "...", "time_period_end": "..."},
            {"sector_name": "Healthcare", "time_period_start": "...", "time_period_end": "..."},
        ]
        items = build_pipeline_batch_items("sector_analysis", contexts)
        ```
    """
    items = []
    for context in contexts:
        item = {
            "pipeline_type": pipeline_type,
            "context": context,
            "pipeline_config": pipeline_config,
        }
        items.append(item)

    return items


def build_orchestrator_batch_items(
    contexts: List[Dict[str, Any]],
    agent_plans: Optional[List[Optional[List[str]]]] = None,
    execution_modes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build batch items for orchestrator workflow execution.

    Args:
        contexts: List of context dictionaries, one per batch item.
        agent_plans: Optional list of agent plans (one per context, or None for all).
        execution_modes: Optional list of execution modes (one per context, or None for all).

    Returns:
        List of item dictionaries ready for batch operation.

    Example:
        ```python
        contexts = [
            {"query": "Analyze Technology sector", "conversation_id": "conv-1"},
            {"query": "Analyze Healthcare sector", "conversation_id": "conv-2"},
        ]
        items = build_orchestrator_batch_items(contexts)
        ```
    """
    items = []
    for i, context in enumerate(contexts):
        item = context.copy()

        # Add agent_plan if provided
        if agent_plans is not None and i < len(agent_plans):
            item["agent_plan"] = agent_plans[i]

        # Add execution_mode if provided
        if execution_modes is not None and i < len(execution_modes):
            item["execution_mode"] = execution_modes[i]
        else:
            item["execution_mode"] = "sequential"

        items.append(item)

    return items


def build_sector_analysis_contexts(
    sectors: List[str],
    time_period_start: Optional[str] = None,
    time_period_end: Optional[str] = None,
    granularity: str = "quarterly",
    min_funding_amount: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build context dictionaries for sector analysis batch operations.

    Args:
        sectors: List of sector names to analyze.
        time_period_start: Start date in ISO format. If None, uses last 12 months.
        time_period_end: End date in ISO format. If None, uses current date.
        granularity: Time granularity ("monthly", "quarterly", "yearly"). Default: "quarterly".
        min_funding_amount: Optional minimum funding amount filter.

    Returns:
        List of context dictionaries for sector analysis.

    Example:
        ```python
        contexts = build_sector_analysis_contexts(
            sectors=["Technology", "Healthcare"],
            granularity="quarterly"
        )
        ```
    """
    # Set default time period if not provided
    if time_period_end is None:
        time_period_end = datetime.utcnow().isoformat() + "Z"

    if time_period_start is None:
        # Default to last 12 months
        end_date = datetime.fromisoformat(time_period_end.replace("Z", "+00:00"))
        start_date = end_date - timedelta(days=365)
        time_period_start = start_date.isoformat().replace("+00:00", "Z")

    contexts = []
    for sector in sectors:
        context = {
            "sector_name": sector,
            "time_period_start": time_period_start,
            "time_period_end": time_period_end,
            "granularity": granularity,
        }

        if min_funding_amount is not None:
            context["min_funding_amount"] = min_funding_amount

        contexts.append(context)

    return contexts
