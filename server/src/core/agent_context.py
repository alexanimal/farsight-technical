"""Context data structure for agent operations.

This module provides the AgentContext class that serves as the input data
and shared state container for all agents in the system. It allows agents
to pass context and shared data between each other.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Context data structure for agent operations.

    This class represents the input data and shared state that is passed
    between agents. It contains the user query, conversation history,
    metadata, and any shared data that agents may need to coordinate.

    Attributes:
        query: The user's query or request that the agent should process.
        conversation_id: Optional identifier for tracking conversation sessions.
        user_id: Optional identifier for the user making the request.
        metadata: Additional metadata that can be passed between agents.
            This can include routing hints, previous agent outputs, etc.
        shared_data: Dictionary for storing shared data between agents.
            This allows agents to pass data to each other during coordination.
        conversation_history: Optional list of previous messages in the conversation.
            Each message is a dict with 'role' and 'content' keys.
        timestamp: Timestamp when the context was created.
    """

    query: str = Field(..., description="The user's query or request to process")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for tracking conversation sessions",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for the user making the request",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata that can be passed between agents",
    )
    shared_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary for storing shared data between agents",
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional list of previous messages in the conversation",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the context was created",
    )

    def add_to_shared_data(self, key: str, value: Any) -> None:
        """Add a value to the shared data dictionary.

        Args:
            key: The key to store the value under.
            value: The value to store.
        """
        self.shared_data[key] = value

    def get_from_shared_data(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the shared data dictionary.

        Args:
            key: The key to retrieve the value for.
            default: Default value to return if key is not found.

        Returns:
            The value associated with the key, or the default value if not found.
        """
        return self.shared_data.get(key, default)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a value to the metadata dictionary.

        Args:
            key: The key to store the value under.
            value: The value to store.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the metadata dictionary.

        Args:
            key: The key to retrieve the value for.
            default: Default value to return if key is not found.

        Returns:
            The value associated with the key, or the default value if not found.
        """
        return self.metadata.get(key, default)

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant', 'system').
            content: The content of the message.
        """
        if self.conversation_history is None:
            self.conversation_history = []
        self.conversation_history.append({"role": role, "content": content})

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert the context to a dictionary.

        Override to ensure timestamp is serialized as ISO format string.

        Returns:
            Dictionary representation of the context.
        """
        data = super().model_dump(**kwargs)
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data

