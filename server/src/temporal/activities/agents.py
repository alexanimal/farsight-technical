"""Temporal activities for agent execution.

This module provides execution adapters for agents from the `/agents` directory.
Activities wrap agent execution with Temporal's durable execution semantics,
including retries, timeouts, and observability.

This is an execution adapter layer - it does not contain domain logic.
All business logic lives in the agents themselves.

Execution flow:
    Workflow → Activity → Runtime Executor → Agent
"""

import importlib
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from temporalio import activity
from temporalio.common import RetryPolicy

from src.contracts.agent_io import AgentOutput, validate_agent_input
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse
from src.db import get_redis_client
from src.runtime.executor import Executor, get_executor

# Import agent registry from agents module (no Temporal dependency)
# This ensures agents can use the same registry without importing Temporal
from src.agents.registry import (
    discover_agents,
    get_agent,
    list_available_agents,
    register_agent,
)

logger = logging.getLogger(__name__)

# Initialize agent registry on module load
# This ensures agents are discovered when Temporal activities are loaded
try:
    discover_agents()
except Exception as e:
    logger.warning(f"Failed to auto-discover agents: {e}")


# Default Temporal activity options for agent execution
# Agents typically take longer than tools, so longer timeout
_DEFAULT_ACTIVITY_OPTIONS = {
    "start_to_close_timeout": timedelta(seconds=600),  # 10 minutes default timeout
    "retry_policy": RetryPolicy(
        initial_interval=timedelta(seconds=2),  # Start with 2 seconds
        backoff_coefficient=2.0,  # Double each retry
        maximum_interval=timedelta(seconds=120),  # Max 2 minutes between retries
        maximum_attempts=3,  # Max 3 attempts
    ),
}


@activity.defn(name="execute_agent")
async def execute_agent(
    agent_name: str,
    context: Dict[str, Any],
    timeout_seconds: Optional[float] = None,
    executor: Optional[Executor] = None,
) -> Dict[str, Any]:
    """Execute an agent as a Temporal activity.

    This activity:
    - Resolves the agent from the agent registry
    - Instantiates the agent
    - Creates AgentContext from the provided context dict
    - Executes the agent via the runtime executor
    - Returns structured results or failures

    Args:
        agent_name: Name of the agent to execute (must be registered).
        context: Dictionary containing AgentContext fields (query, conversation_id, etc.).
            This will be converted to an AgentContext object.
        timeout_seconds: Optional timeout in seconds (informational only;
            actual timeout is set via activity options at registration).
        executor: Optional Executor instance (uses default if not provided).

    Returns:
        Dictionary containing:
        - success: bool indicating if execution succeeded
        - response: The AgentResponse object (serialized) (if successful)
        - error: Error message (if failed)
        - agent_name: Name of the executed agent
        - metadata: Additional metadata about execution

    Raises:
        ValueError: If agent is not found or context is invalid.
        Exception: If agent execution fails after retries.
    """
    activity.logger.info(
        f"Executing agent: {agent_name} with context: {context.get('query', 'N/A')[:100]}"
    )

    # Resolve agent
    agent_info = get_agent(agent_name)
    if agent_info is None:
        error_msg = (
            f"Agent '{agent_name}' not found. "
            f"Available agents: {list_available_agents()}"
        )
        activity.logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "agent_name": agent_name,
            "response": None,
            "metadata": {},
        }

    agent_class, config_path = agent_info

    # Validate and create AgentContext using contract validation
    try:
        # Use contract validation helper for consistency
        agent_input = validate_agent_input(AgentContext(**context))
        agent_context = agent_input.to_agent_context()
    except Exception as e:
        error_msg = f"Invalid context for agent '{agent_name}': {e}"
        activity.logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "agent_name": agent_name,
            "response": None,
            "metadata": {},
        }

    # Get executor (use provided or default)
    exec = executor or get_executor()

    # Instantiate agent
    try:
        agent = agent_class(config_path=config_path)
    except Exception as e:
        error_msg = f"Failed to instantiate agent '{agent_name}': {e}"
        activity.logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "agent_name": agent_name,
            "response": None,
            "metadata": {
                "exception_type": type(e).__name__,
            },
        }

    # Execute agent via executor
    try:
        # Convert AgentContext to AgentInput for contract compliance
        from src.contracts.agent_io import AgentInput

        agent_input = AgentInput.from_agent_context(agent_context)

        # Use executor timeout if provided, otherwise let executor use its default
        output = await exec.execute_agent(
            agent=agent,
            agent_input=agent_input,
            timeout=timeout_seconds,
        )

        # Executor now returns AgentOutput, convert to AgentResponse for serialization
        response = output.to_agent_response()

        activity.logger.info(
            f"Agent '{agent_name}' executed successfully with status: {response.status}"
        )

        # Serialize response for return
        response_dict = response.model_dump()

        return {
            "success": True,
            "response": response_dict,
            "error": None,
            "agent_name": agent_name,
            "metadata": {
                "agent_category": agent.category,
                "response_status": response.status.value,
            },
        }

    except Exception as e:
        error_msg = f"Agent '{agent_name}' execution failed: {str(e)}"
        activity.logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "agent_name": agent_name,
            "response": None,
            "metadata": {
                "exception_type": type(e).__name__,
            },
        }


@activity.defn(name="execute_agent_with_options")
async def execute_agent_with_options(
    agent_name: str,
    context: Dict[str, Any],
    activity_options: Optional[Dict[str, Any]] = None,
    executor: Optional[Executor] = None,
) -> Dict[str, Any]:
    """Execute an agent with custom Temporal activity options.

    This is similar to execute_agent but allows custom activity options
    to be specified per-execution.

    Args:
        agent_name: Name of the agent to execute.
        context: Dictionary containing AgentContext fields.
        activity_options: Optional dictionary of Temporal activity options.
            Can include: start_to_close_timeout, retry_policy, etc.
        executor: Optional Executor instance.

    Returns:
        Dictionary containing execution results (same format as execute_agent).
    """
    # For now, this is the same as execute_agent
    # Activity options are typically set at registration time, not execution time
    # But we can log them for observability
    if activity_options:
        activity.logger.info(f"Custom activity options provided: {activity_options}")

    return await execute_agent(agent_name, context, executor=executor)


@activity.defn(name="save_conversation_history")
async def save_conversation_history(
    conversation_id: str,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Save conversation history to Redis.

    This activity saves the conversation history to Redis for persistence.
    It's called from workflows to persist conversation state.

    Args:
        conversation_id: The conversation identifier.
        history: List of message dictionaries with 'role', 'content', and 'timestamp' keys.

    Returns:
        Dictionary containing:
        - success: bool indicating if save succeeded
        - error: Error message (if failed)
        - conversation_id: The conversation ID
    """
    activity.logger.info(
        f"Saving conversation history for {conversation_id} ({len(history)} messages)"
    )

    try:
        redis_client = await get_redis_client()
        await redis_client.save_conversation_history(conversation_id, history)

        activity.logger.info(f"Successfully saved conversation history for {conversation_id}")
        return {
            "success": True,
            "error": None,
            "conversation_id": conversation_id,
        }
    except Exception as e:
        error_msg = f"Failed to save conversation history for {conversation_id}: {str(e)}"
        activity.logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "conversation_id": conversation_id,
        }


@activity.defn(name="enrich_query")
async def enrich_query(
    query: str,
    conversation_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Enrich query with context and extract metadata.

    This activity performs query enrichment using the QueryEnrichmentService.
    It's called from workflows to improve user queries and extract structured metadata.

    Args:
        query: The user's query string.
        conversation_history: List of message dictionaries with 'role' and 'content' keys.
            Can be empty list for first-time users.

    Returns:
        Dictionary containing:
        - success: bool indicating if enrichment succeeded
        - result: Dictionary with improved_query, original_query, and metadata (if successful)
        - error: Error message (if failed)
    """
    activity.logger.info(
        f"Enriching query: '{query[:100]}...' (history: {len(conversation_history)} messages)"
    )

    try:
        # Import here to avoid non-deterministic imports at module level
        from src.tools.query_enrichment import QueryEnrichmentService

        service = QueryEnrichmentService()

        # Call enrichment service
        result = await service.enrich_query(
            query=query,
            conversation_history=conversation_history if conversation_history else None,
        )

        activity.logger.info(
            f"Query enriched successfully: '{query[:50]}...' -> '{result.get('improved_query', '')[:50]}...'"
        )

        return {
            "success": True,
            "result": result,
            "error": None,
        }
    except Exception as e:
        error_msg = f"Query enrichment failed: {str(e)}"
        activity.logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "result": None,
            "error": error_msg,
        }


def get_activity_options(
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    initial_retry_interval: Optional[float] = None,
) -> Dict[str, Any]:
    """Get Temporal activity options for agent execution.

    This helper function creates activity options that can be used
    when registering activities in the worker.

    Args:
        timeout_seconds: Optional timeout in seconds (default: 600).
        max_retries: Optional maximum retry attempts (default: 3).
        initial_retry_interval: Optional initial retry interval in seconds (default: 2.0).

    Returns:
        Dictionary of activity options for Temporal.
    """
    options = _DEFAULT_ACTIVITY_OPTIONS.copy()

    if timeout_seconds is not None:
        options["start_to_close_timeout"] = timedelta(seconds=timeout_seconds)

    if max_retries is not None or initial_retry_interval is not None:
        existing_policy_raw = options.get("retry_policy")
        existing_policy: Optional[RetryPolicy] = (
            existing_policy_raw
            if isinstance(existing_policy_raw, RetryPolicy)
            else None
        )
        if existing_policy is not None:
            # Create a new RetryPolicy with updated values
            retry_policy = RetryPolicy(
                initial_interval=(
                    timedelta(seconds=initial_retry_interval)
                    if initial_retry_interval is not None
                    else existing_policy.initial_interval
                ),
                backoff_coefficient=existing_policy.backoff_coefficient,
                maximum_interval=existing_policy.maximum_interval,
                maximum_attempts=(
                    max_retries
                    if max_retries is not None
                    else existing_policy.maximum_attempts
                ),
            )
        else:
            retry_policy = RetryPolicy(
                initial_interval=(
                    timedelta(seconds=initial_retry_interval)
                    if initial_retry_interval is not None
                    else timedelta(seconds=2)
                ),
                maximum_attempts=max_retries if max_retries is not None else 3,
            )
        options["retry_policy"] = retry_policy

    return options
