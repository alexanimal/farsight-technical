"""Temporal integration layer for durable workflow orchestration.

This module provides the Temporal integration for the system, including:
- Client for API layer to interact with workflows
- Worker process for executing workflows and activities
- Signal and query definitions
- Workflow and activity implementations

Architecture:
- API layer should only import from this module's client interface
- Workflows and activities are registered by the worker process
- Signals and queries provide type-safe communication
"""

# Client interface (for API layer)
# Import lazily to avoid pulling in client when workflows import activities
# Workflows should not import client - it's only for the API layer
_client_module = None


def _get_client_module():
    """Lazy import of client module to avoid workflow determinism issues."""
    global _client_module
    if _client_module is None:
        from . import client as _client_module
    return _client_module


# Export symbols lazily via __getattr__ (PEP 562)
# This allows workflows to import queries/signals without triggering
# client/worker/activities imports (which pull in non-deterministic dependencies)
def __getattr__(name: str):
    """Lazy import of symbols to avoid import chain issues during workflow validation."""
    # Client symbols
    if name in {
        "DEFAULT_TASK_QUEUE",
        "DEFAULT_TEMPORAL_ADDRESS",
        "DEFAULT_TEMPORAL_NAMESPACE",
        "TemporalClient",
        "close_client",
        "get_client",
        "set_client",
    }:
        module = _get_client_module()
        return getattr(module, name)

    # Worker symbols
    if name in {"main", "run_worker"}:
        from .worker import main, run_worker

        if name == "main":
            return main
        elif name == "run_worker":
            return run_worker

    # Activities/workflows submodules (for worker registration only)
    if name == "activities":
        from . import activities

        return activities
    if name == "workflows":
        from . import workflows

        return workflows

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For backwards compatibility, try to import directly but catch errors
# This allows the module to work when not in workflow validation context
try:
    from .client import (
        DEFAULT_TASK_QUEUE,
        DEFAULT_TEMPORAL_ADDRESS,
        DEFAULT_TEMPORAL_NAMESPACE,
        TemporalClient,
        close_client,
        get_client,
        set_client,
    )
except Exception:
    # If import fails (e.g., during workflow validation), the symbols
    # will be available via __getattr__ when actually needed (by API layer)
    pass

# Worker (for process entrypoint) - imported lazily via __getattr__ only
# Do NOT import here to avoid sys.modules warning when running as module

# Queries (for API layer to query workflows)
from .queries import (
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

# Signals (for API layer to send signals)
from .signals import (
    SIGNAL_CANCELLATION,
    SIGNAL_CONFIGURATION_CHANGE,
    SIGNAL_STATUS_UPDATE,
    SIGNAL_USER_INPUT,
    CancellationSignal,
    ConfigurationChangeSignal,
    StatusUpdateSignal,
    UserInputSignal,
)

# Submodules (for worker registration, not for API layer)
# DO NOT import at module level - this triggers the import chain during workflow validation
# Activities import runtime.executor which imports langfuse/httpx (restricted)
# These are only imported lazily via __getattr__ when actually needed (by worker)

__all__ = [
    # Client
    "TemporalClient",
    "get_client",
    "set_client",
    "close_client",
    "DEFAULT_TEMPORAL_ADDRESS",
    "DEFAULT_TEMPORAL_NAMESPACE",
    "DEFAULT_TASK_QUEUE",
    # Worker
    "run_worker",
    "main",
    # Signals
    "SIGNAL_CANCELLATION",
    "SIGNAL_USER_INPUT",
    "SIGNAL_STATUS_UPDATE",
    "SIGNAL_CONFIGURATION_CHANGE",
    "CancellationSignal",
    "UserInputSignal",
    "StatusUpdateSignal",
    "ConfigurationChangeSignal",
    # Queries
    "QUERY_STATUS",
    "QUERY_PROGRESS",
    "QUERY_STATE",
    "QUERY_AGENT_STATUS",
    "WorkflowStatus",
    "WorkflowStatusQueryResult",
    "WorkflowProgressQueryResult",
    "WorkflowStateQueryResult",
    "AgentStatusQueryResult",
    "AgentExecutionStatus",
    # Submodules (for worker, not API)
    "activities",
    "workflows",
]
