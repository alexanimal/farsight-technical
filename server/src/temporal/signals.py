"""Signal definitions and schemas for Temporal workflows.

This module provides signal schemas that allow external systems (API, UI, etc.)
to communicate with running Temporal workflows. Signals are used to send
asynchronous messages TO workflows.

Signals enable workflows to:
- Receive cancellation requests
- Accept additional user input during execution
- Respond to configuration changes
- Handle status updates from external systems

All signals use Pydantic models for type safety and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Enumeration of signal types."""

    CANCELLATION = "cancellation"
    USER_INPUT = "user_input"
    STATUS_UPDATE = "status_update"
    CONFIGURATION_CHANGE = "configuration_change"


class CancellationSignal(BaseModel):
    """Signal to request cancellation of a workflow execution.

    This signal is sent when a user or system wants to cancel
    an in-progress workflow. The workflow should gracefully
    handle cancellation and clean up resources.

    Attributes:
        reason: Optional reason for cancellation.
        requested_by: Optional identifier of who requested cancellation.
        timestamp: Timestamp when cancellation was requested.
    """

    reason: Optional[str] = Field(
        default=None,
        description="Optional reason for cancellation",
    )
    requested_by: Optional[str] = Field(
        default=None,
        description="Optional identifier of who requested cancellation (user_id, system, etc.)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when cancellation was requested",
    )


class UserInputSignal(BaseModel):
    """Signal to provide additional user input to a running workflow.

    This signal allows workflows to receive additional input from users
    during execution. Useful for interactive workflows that may need
    clarification or additional information.

    Attributes:
        input_text: The user's input text.
        input_type: Optional type/category of input (e.g., 'clarification', 'confirmation').
        user_id: Optional identifier of the user providing input.
        conversation_id: Optional conversation identifier for context.
        metadata: Additional metadata about the input.
        timestamp: Timestamp when input was received.
    """

    input_text: str = Field(
        ...,
        description="The user's input text",
    )
    input_type: Optional[str] = Field(
        default=None,
        description="Optional type/category of input (e.g., 'clarification', 'confirmation')",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional identifier of the user providing input",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation identifier for context",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the input",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when input was received",
    )


class StatusUpdateSignal(BaseModel):
    """Signal to update workflow status from external systems.

    This signal allows external systems to notify workflows about
    status changes or events that may affect workflow execution.

    Attributes:
        status: The new status or status update.
        status_code: Optional status code for programmatic handling.
        message: Optional human-readable message.
        source: Optional identifier of the system sending the update.
        metadata: Additional metadata about the status update.
        timestamp: Timestamp when status was updated.
    """

    status: str = Field(
        ...,
        description="The new status or status update",
    )
    status_code: Optional[str] = Field(
        default=None,
        description="Optional status code for programmatic handling",
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional human-readable message",
    )
    source: Optional[str] = Field(
        default=None,
        description="Optional identifier of the system sending the update",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the status update",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when status was updated",
    )


class ConfigurationChangeSignal(BaseModel):
    """Signal to notify workflow of configuration changes.

    This signal allows workflows to respond to runtime configuration
    changes, such as updated agent settings, tool configurations, or
    execution parameters.

    Attributes:
        config_key: The configuration key that changed.
        config_value: The new configuration value.
        config_type: Optional type of configuration (e.g., 'agent', 'tool', 'execution').
        metadata: Additional metadata about the configuration change.
        timestamp: Timestamp when configuration was changed.
    """

    config_key: str = Field(
        ...,
        description="The configuration key that changed",
    )
    config_value: Any = Field(
        ...,
        description="The new configuration value",
    )
    config_type: Optional[str] = Field(
        default=None,
        description="Optional type of configuration (e.g., 'agent', 'tool', 'execution')",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the configuration change",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when configuration was changed",
    )


# Signal name constants for use in workflows
SIGNAL_CANCELLATION = "cancellation"
SIGNAL_USER_INPUT = "user_input"
SIGNAL_STATUS_UPDATE = "status_update"
SIGNAL_CONFIGURATION_CHANGE = "configuration_change"
