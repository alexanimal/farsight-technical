"""PostgreSQL client for managing database connections and queries.

This module provides a connection pool-based PostgreSQL client that can be used
as a base class or singleton instance for querying the database.
"""

import asyncio
import logging
from typing import Any, Optional, Sequence

import asyncpg  # type: ignore[import-untyped]
from asyncpg import Pool, Record

from src.config import settings

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL client with connection pool management.

    This client creates and manages an asyncpg connection pool for efficient
    database querying. It provides a generic query method that can be used
    by other modules in the library.

    Example:
        ```python
        client = PostgresClient()
        await client.initialize()
        results = await client.query("SELECT * FROM companies WHERE id = $1", 123)
        await client.close()
        ```

        Or use as a context manager:
        ```python
        async with PostgresClient() as client:
            results = await client.query("SELECT * FROM companies LIMIT 10")
        ```
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        min_size: int = 2,
        max_size: int = 10,
    ):
        """Initialize the PostgreSQL client.

        Args:
            config: Optional database configuration dict. If not provided,
                uses settings from src.config. Should contain: host, port,
                user, password, database.
            min_size: Minimum number of connections in the pool.
            max_size: Maximum number of connections in the pool.
        """
        self._config = config or settings.postgres_config
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[Pool] = None

    async def initialize(self) -> None:
        """Initialize the connection pool.

        Raises:
            asyncpg.exceptions.PostgresError: If connection to database fails.
        """
        if self._pool is not None:
            logger.warning("Pool already initialized, skipping re-initialization")
            return

        try:
            self._pool = await asyncpg.create_pool(
                host=self._config["host"],
                port=self._config["port"],
                user=self._config["user"],
                password=self._config["password"],
                database=self._config["database"],
                min_size=self._min_size,
                max_size=self._max_size,
            )
            logger.info(
                f"PostgreSQL connection pool initialized: "
                f"{self._config['host']}:{self._config['port']}/{self._config['database']}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool and release all connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    async def __aenter__(self) -> "PostgresClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def pool(self) -> Pool:
        """Get the connection pool.

        Returns:
            The asyncpg connection pool.

        Raises:
            RuntimeError: If pool is not initialized.
        """
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not initialized. Call initialize() first or use as context manager."
            )
        return self._pool

    async def query(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> list[Record]:
        """Execute a SELECT query and return results.

        This method is designed for read-only queries. It uses the connection
        pool to execute queries efficiently.

        Args:
            query: SQL query string. Use $1, $2, etc. for parameterized queries.
            *args: Query parameters to bind to the query.
            timeout: Optional timeout in seconds for the query.

        Returns:
            List of Record objects containing query results.

        Raises:
            RuntimeError: If pool is not initialized.
            asyncpg.exceptions.PostgresError: If query execution fails.

        Example:
            ```python
            # Simple query
            results = await client.query("SELECT * FROM companies LIMIT 10")

            # Parameterized query
            results = await client.query(
                "SELECT * FROM companies WHERE id = $1 AND status = $2",
                123, "active"
            )
            ```
        """
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not initialized. Call initialize() first or use as context manager."
            )

        try:
            async with self._pool.acquire() as connection:
                records = await connection.fetch(query, *args, timeout=timeout)
                logger.debug(f"Query executed successfully: {query[:50]}...")
                return list(records)
        except Exception as e:
            logger.error(f"Query execution failed: {query[:50]}... Error: {e}")
            raise

    async def query_one(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> Optional[Record]:
        """Execute a SELECT query and return a single result.

        Args:
            query: SQL query string. Use $1, $2, etc. for parameterized queries.
            *args: Query parameters to bind to the query.
            timeout: Optional timeout in seconds for the query.

        Returns:
            A single Record object if found, None otherwise.

        Raises:
            RuntimeError: If pool is not initialized.
            asyncpg.exceptions.PostgresError: If query execution fails.

        Example:
            ```python
            result = await client.query_one(
                "SELECT * FROM companies WHERE id = $1",
                123
            )
            ```
        """
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not initialized. Call initialize() first or use as context manager."
            )

        try:
            async with self._pool.acquire() as connection:
                record = await connection.fetchrow(query, *args, timeout=timeout)
                logger.debug(f"Query executed successfully: {query[:50]}...")
                return record
        except Exception as e:
            logger.error(f"Query execution failed: {query[:50]}... Error: {e}")
            raise

    async def query_value(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a SELECT query and return a single value.

        Useful for queries like COUNT(*) or SELECT column FROM table LIMIT 1.

        Args:
            query: SQL query string. Use $1, $2, etc. for parameterized queries.
            *args: Query parameters to bind to the query.
            timeout: Optional timeout in seconds for the query.

        Returns:
            The single value from the query result.

        Raises:
            RuntimeError: If pool is not initialized.
            asyncpg.exceptions.PostgresError: If query execution fails.

        Example:
            ```python
            count = await client.query_value("SELECT COUNT(*) FROM companies")
            ```
        """
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not initialized. Call initialize() first or use as context manager."
            )

        try:
            async with self._pool.acquire() as connection:
                value = await connection.fetchval(query, *args, timeout=timeout)
                logger.debug(f"Query executed successfully: {query[:50]}...")
                return value
        except Exception as e:
            logger.error(f"Query execution failed: {query[:50]}... Error: {e}")
            raise

    def is_connected(self) -> bool:
        """Check if the connection pool is initialized.

        Returns:
            True if pool is initialized, False otherwise.
        """
        return self._pool is not None


# Singleton instance for convenience
_default_client: Optional[PostgresClient] = None
_client_lock = asyncio.Lock()


async def get_client() -> PostgresClient:
    """Get or create the default singleton PostgresClient instance.

    This is a convenience function for modules that want to use a shared
    client instance without managing it themselves. Uses a lock to prevent
    race conditions when multiple coroutines try to initialize the client
    simultaneously.

    Returns:
        The default PostgresClient instance.

    Example:
        ```python
        client = await get_client()
        results = await client.query("SELECT * FROM companies")
        ```
    """
    global _default_client
    async with _client_lock:
        if _default_client is None:
            _default_client = PostgresClient()
            await _default_client.initialize()
        elif not _default_client.is_connected():
            # Client exists but pool is not initialized, re-initialize
            await _default_client.initialize()
    return _default_client


async def close_default_client() -> None:
    """Close the default singleton client instance."""
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None
