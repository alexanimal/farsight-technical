"""Redis client for managing conversation history and workflow mappings.

This module provides a connection pool-based Redis client that can be used
for storing and retrieving conversation history and workflow state mappings.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool, Redis

from src.config import settings

logger = logging.getLogger(__name__)

# Constants for Redis key prefixes and TTLs
CONVERSATION_HISTORY_TTL = 604800  # 7 days in seconds
WORKFLOW_MAPPING_TTL = 86400  # 24 hours in seconds


class RedisClient:
    """Redis client with connection pool management.

    This client creates and manages a Redis connection pool for efficient
    conversation history and workflow mapping operations.

    Example:
        ```python
        client = RedisClient()
        await client.initialize()
        history = await client.get_conversation_history("conv-123")
        await client.close()
        ```

        Or use as a context manager:
        ```python
        async with RedisClient() as client:
            history = await client.get_conversation_history("conv-123")
        ```
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Redis client.

        Args:
            config: Optional Redis configuration dict. If not provided,
                uses settings from src.config. Should contain: host, port,
                db, password (optional), max_connections.
        """
        self._config = config or settings.redis_config
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None

    async def initialize(self) -> None:
        """Initialize the connection pool.

        Raises:
            redis.RedisError: If connection to Redis fails.
        """
        if self._pool is not None:
            logger.warning("Redis pool already initialized, skipping re-initialization")
            return

        try:
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                self._build_connection_url(),
                max_connections=self._config.get("max_connections", 10),
                decode_responses=True,  # Automatically decode responses to strings
            )

            # Create Redis client from pool
            self._redis = Redis(connection_pool=self._pool)

            # Test connection
            await self._redis.ping()

            logger.info(
                f"Redis connection pool initialized: "
                f"{self._config['host']}:{self._config['port']}/{self._config['db']}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool and release all connections."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None
            logger.info("Redis connection pool closed")

    async def __aenter__(self) -> "RedisClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def _build_connection_url(self) -> str:
        """Build Redis connection URL from config.

        Returns:
            Redis connection URL string.
        """
        password = self._config.get("password")
        host = self._config["host"]
        port = self._config["port"]
        db = self._config["db"]

        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"

    def _ensure_connected(self) -> Redis:
        """Ensure client is connected and return it.

        Returns:
            The Redis client.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if self._redis is None or self._pool is None:
            raise RuntimeError(
                "Redis client not connected. Call initialize() first or use as context manager."
            )
        return self._redis

    # Conversation History Methods

    async def get_conversation_history(
        self, conversation_id: str
    ) -> List[Dict[str, str]]:
        """Get conversation history for a given conversation ID.

        Args:
            conversation_id: The conversation identifier.

        Returns:
            List of message dictionaries with 'role' and 'content' keys.
            Returns empty list if conversation doesn't exist or on error.

        Example:
            ```python
            history = await client.get_conversation_history("conv-123")
            # Returns: [{"role": "user", "content": "...", "timestamp": "..."}, ...]
            ```
        """
        try:
            redis = self._ensure_connected()
            key = f"conversation:{conversation_id}:history"
            data = await redis.get(key)

            if data is None:
                return []

            # Parse JSON string
            history = json.loads(data)
            if not isinstance(history, list):
                logger.warning(
                    f"Invalid conversation history format for {conversation_id}, returning empty list"
                )
                return []

            return history
        except Exception as e:
            logger.error(
                f"Failed to get conversation history for {conversation_id}: {e}",
                exc_info=True,
            )
            # Return empty list on error (graceful degradation)
            return []

    async def save_conversation_history(
        self, conversation_id: str, history: List[Dict[str, str]]
    ) -> None:
        """Save conversation history for a given conversation ID.

        Args:
            conversation_id: The conversation identifier.
            history: List of message dictionaries with 'role' and 'content' keys.

        Raises:
            RuntimeError: If client is not initialized.
            redis.RedisError: If save operation fails.

        Example:
            ```python
            history = [
                {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "assistant", "content": "Hi there!", "timestamp": "2024-01-01T00:00:01Z"}
            ]
            await client.save_conversation_history("conv-123", history)
            ```
        """
        try:
            redis = self._ensure_connected()
            key = f"conversation:{conversation_id}:history"
            data = json.dumps(history)

            # Save with TTL
            await redis.setex(key, CONVERSATION_HISTORY_TTL, data)
            logger.debug(f"Saved conversation history for {conversation_id}")
        except Exception as e:
            logger.error(
                f"Failed to save conversation history for {conversation_id}: {e}",
                exc_info=True,
            )
            raise

    async def append_to_conversation(
        self, conversation_id: str, role: str, content: str, timestamp: Optional[str] = None
    ) -> None:
        """Append a message to conversation history.

        This method loads existing history, appends the new message, and saves it back.

        Args:
            conversation_id: The conversation identifier.
            role: Message role ('user', 'assistant', 'system').
            content: Message content.
            timestamp: Optional timestamp (ISO format). If not provided, current time is used.

        Example:
            ```python
            await client.append_to_conversation(
                "conv-123",
                "user",
                "What is the weather?",
                "2024-01-01T00:00:00Z"
            )
            ```
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"

        # Load existing history
        history = await self.get_conversation_history(conversation_id)

        # Append new message
        history.append(
            {
                "role": role,
                "content": content,
                "timestamp": timestamp,
            }
        )

        # Save updated history
        await self.save_conversation_history(conversation_id, history)

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete conversation history for a given conversation ID.

        Args:
            conversation_id: The conversation identifier.

        Example:
            ```python
            await client.delete_conversation("conv-123")
            ```
        """
        try:
            redis = self._ensure_connected()
            key = f"conversation:{conversation_id}:history"
            await redis.delete(key)
            logger.debug(f"Deleted conversation history for {conversation_id}")
        except Exception as e:
            logger.error(
                f"Failed to delete conversation history for {conversation_id}: {e}",
                exc_info=True,
            )
            # Don't raise - deletion is best effort

    # Workflow Mapping Methods

    async def get_workflow_id_for_conversation(
        self, conversation_id: str
    ) -> Optional[str]:
        """Get the current workflow ID for a conversation.

        Args:
            conversation_id: The conversation identifier.

        Returns:
            Workflow ID if found, None otherwise.

        Example:
            ```python
            workflow_id = await client.get_workflow_id_for_conversation("conv-123")
            ```
        """
        try:
            redis = self._ensure_connected()
            key = f"conversation:{conversation_id}:workflow_id"
            workflow_id = await redis.get(key)
            return workflow_id
        except Exception as e:
            logger.error(
                f"Failed to get workflow ID for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            return None

    async def set_workflow_id_for_conversation(
        self, conversation_id: str, workflow_id: str, ttl: Optional[int] = None
    ) -> None:
        """Set the workflow ID for a conversation.

        Args:
            conversation_id: The conversation identifier.
            workflow_id: The workflow ID to store.
            ttl: Optional TTL in seconds. Defaults to WORKFLOW_MAPPING_TTL (24 hours).

        Example:
            ```python
            await client.set_workflow_id_for_conversation("conv-123", "workflow-456")
            ```
        """
        try:
            redis = self._ensure_connected()
            key = f"conversation:{conversation_id}:workflow_id"
            ttl = ttl or WORKFLOW_MAPPING_TTL
            await redis.setex(key, ttl, workflow_id)
            logger.debug(
                f"Set workflow ID {workflow_id} for conversation {conversation_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to set workflow ID for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise

    async def delete_workflow_mapping(self, conversation_id: str) -> None:
        """Delete the workflow ID mapping for a conversation.

        Args:
            conversation_id: The conversation identifier.

        Example:
            ```python
            await client.delete_workflow_mapping("conv-123")
            ```
        """
        try:
            redis = self._ensure_connected()
            key = f"conversation:{conversation_id}:workflow_id"
            await redis.delete(key)
            logger.debug(f"Deleted workflow mapping for {conversation_id}")
        except Exception as e:
            logger.error(
                f"Failed to delete workflow mapping for {conversation_id}: {e}",
                exc_info=True,
            )
            # Don't raise - deletion is best effort

    async def is_connected(self) -> bool:
        """Check if Redis client is connected.

        Returns:
            True if connected, False otherwise.
        """
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False


# Singleton instance - can be initialized and reused
_default_client: Optional[RedisClient] = None


async def get_redis_client(config: Optional[Dict[str, Any]] = None) -> RedisClient:
    """Get or create the default Redis client.

    Args:
        config: Optional Redis configuration dict. If not provided,
            uses settings from src.config.

    Returns:
        The default RedisClient instance.
    """
    global _default_client

    if _default_client is None:
        _default_client = RedisClient(config=config)
        await _default_client.initialize()

    return _default_client


def set_redis_client(client: RedisClient) -> None:
    """Set the default Redis client.

    Args:
        client: The RedisClient instance to use as default.
    """
    global _default_client
    _default_client = client


async def close_redis_client() -> None:
    """Close the default Redis client connection."""
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None

