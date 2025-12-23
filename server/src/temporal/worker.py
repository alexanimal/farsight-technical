"""Temporal worker process entrypoint.

This module provides the worker bootstrap that registers workflows and activities
and starts polling Temporal task queues. This is a process entrypoint, not a
library module.

The worker runs as a separate service from the API, allowing independent scaling.
"""

import asyncio
import logging
import signal
import sys
from typing import List, Optional

from temporalio.client import Client
from temporalio.worker import Worker

from src.temporal.activities import agents, tools
from src.temporal.client import (DEFAULT_TASK_QUEUE, DEFAULT_TEMPORAL_ADDRESS,
                                 DEFAULT_TEMPORAL_NAMESPACE)
from src.temporal.workflows import OrchestratorWorkflow

logger = logging.getLogger(__name__)

# Global worker instance for graceful shutdown
_worker: Optional[Worker] = None


async def run_worker(
    temporal_address: str = DEFAULT_TEMPORAL_ADDRESS,
    temporal_namespace: str = DEFAULT_TEMPORAL_NAMESPACE,
    task_queue: str = DEFAULT_TASK_QUEUE,
    max_concurrent_activities: int = 10,
    max_concurrent_workflow_tasks: int = 10,
) -> None:
    """Run the Temporal worker.

    This function:
    1. Connects to Temporal
    2. Registers workflows and activities
    3. Starts polling task queues
    4. Runs until interrupted

    Args:
        temporal_address: Temporal server address.
        temporal_namespace: Temporal namespace.
        task_queue: Task queue name to poll.
        max_concurrent_activities: Maximum concurrent activities.
        max_concurrent_workflow_tasks: Maximum concurrent workflow tasks.
    """
    global _worker

    logger.info(
        f"Starting Temporal worker (address: {temporal_address}, "
        f"namespace: {temporal_namespace}, task_queue: {task_queue})"
    )

    try:
        # Connect to Temporal
        client = await Client.connect(
            temporal_address,
            namespace=temporal_namespace,
        )
        logger.info(f"Connected to Temporal at {temporal_address}")

        # Register workflows
        workflows = _get_workflows()
        logger.info(f"Registering {len(workflows)} workflow(s)")

        # Register activities
        activities = _get_activities()
        logger.info(f"Registering {len(activities)} activity/activities")

        # Create worker
        # Note: Config module is not imported in src/__init__.py to avoid
        # non-deterministic operations (Path.expanduser) in workflows
        _worker = Worker(
            client,
            task_queue=task_queue,
            workflows=workflows,
            activities=activities,
            max_concurrent_activities=max_concurrent_activities,
            max_concurrent_workflow_tasks=max_concurrent_workflow_tasks,
        )

        logger.info("Worker initialized, starting to poll task queue...")
        logger.info(f"  - Task queue: {task_queue}")
        logger.info(f"  - Max concurrent activities: {max_concurrent_activities}")
        logger.info(
            f"  - Max concurrent workflow tasks: {max_concurrent_workflow_tasks}"
        )

        # Run worker (this blocks until interrupted)
        await _worker.run()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down worker...")
    except Exception as e:
        logger.error(f"Worker error: {e}", exc_info=True)
        raise
    finally:
        if _worker is not None:
            logger.info("Shutting down worker...")
            await _worker.shutdown()
            _worker = None
            logger.info("Worker shut down complete")


def _get_workflows() -> List[type]:
    """Get list of workflows to register.

    Returns:
        List of workflow classes.
    """
    workflows = [OrchestratorWorkflow]

    # Optionally include other workflows if they exist
    # These are optional and may not be implemented yet
    try:
        from src.temporal.workflows import pipeline, swarm

        # Only add if they have workflow definitions
        # For now, we'll just register orchestrator
        # Uncomment when other workflows are implemented:
        # if hasattr(swarm, 'SwarmWorkflow'):
        #     workflows.append(swarm.SwarmWorkflow)
        # if hasattr(pipeline, 'PipelineWorkflow'):
        #     workflows.append(pipeline.PipelineWorkflow)
    except ImportError:
        # Other workflows not implemented yet, that's fine
        pass

    return workflows  # type: ignore[return-value]


def _get_activities() -> List:
    """Get list of activities to register.

    Returns:
        List of activity functions.
    """
    activities = [
        agents.execute_agent,
        agents.execute_agent_with_options,
        agents.save_conversation_history,
        agents.enrich_query,
        tools.execute_tool,
        tools.execute_tool_with_options,
    ]

    return activities


def _setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        # The worker will be interrupted in the run_worker loop
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main() -> None:
    """Main entrypoint for the worker process.

    This function can be called directly or used as a CLI entrypoint.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set up signal handlers
    _setup_signal_handlers()

    # Configuration can come from environment variables or defaults
    # For production, these should be set via environment variables
    import os
    temporal_address = os.getenv("TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)
    temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", DEFAULT_TEMPORAL_NAMESPACE)
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", DEFAULT_TASK_QUEUE)

    # Run the worker
    await run_worker(
        temporal_address=temporal_address,
        temporal_namespace=temporal_namespace,
        task_queue=task_queue,
    )


if __name__ == "__main__":
    # Run as script
    asyncio.run(main())
