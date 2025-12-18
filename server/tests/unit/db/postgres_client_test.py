"""Unit tests for the PostgreSQL client module.

This module tests the PostgresClient class and its various methods,
including connection pool management, query execution, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg  # type: ignore[import-untyped]
import pytest

from src.db.postgres_client import (PostgresClient, close_default_client,
                                    get_client)


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg connection pool."""
    pool = MagicMock(spec=asyncpg.Pool)
    pool.close = AsyncMock()
    return pool


def _setup_pool_acquire(pool, connection):
    """Helper to set up pool.acquire() as async context manager."""
    acquire_context = AsyncMock()
    acquire_context.__aenter__ = AsyncMock(return_value=connection)
    acquire_context.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_context)


@pytest.fixture
def mock_connection():
    """Create a mock asyncpg connection."""
    connection = MagicMock()
    connection.fetch = AsyncMock()
    connection.fetchrow = AsyncMock()
    connection.fetchval = AsyncMock()
    return connection


@pytest.fixture
def mock_record():
    """Create a mock asyncpg Record."""
    record = MagicMock(spec=asyncpg.Record)
    record.items = MagicMock(return_value=[("id", 1), ("name", "test")])
    return record


@pytest.fixture
def test_config():
    """Create a test database configuration."""
    return {
        "host": "localhost",
        "port": 5432,
        "user": "testuser",
        "password": "testpass",
        "database": "testdb",
    }


class TestPostgresClientInitialization:
    """Test PostgresClient initialization."""

    def test_init_with_default_config(self, monkeypatch):
        """Test initialization with default config from settings."""
        with patch("src.db.postgres_client.settings") as mock_settings:
            mock_settings.postgres_config = {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres",
                "database": "farsight",
            }
            client = PostgresClient()
            assert client._config == mock_settings.postgres_config
            assert client._min_size == 2
            assert client._max_size == 10
            assert client._pool is None

    def test_init_with_custom_config(self, test_config):
        """Test initialization with custom config."""
        client = PostgresClient(config=test_config)
        assert client._config == test_config
        assert client._pool is None

    def test_init_with_custom_pool_sizes(self, test_config):
        """Test initialization with custom pool sizes."""
        client = PostgresClient(
            config=test_config,
            min_size=5,
            max_size=20,
        )
        assert client._min_size == 5
        assert client._max_size == 20

    def test_init_pool_not_initialized(self, test_config):
        """Test that pool is None after initialization."""
        client = PostgresClient(config=test_config)
        assert client._pool is None
        assert not client.is_connected()


class TestPostgresClientInitialize:
    """Test PostgresClient.initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, test_config, mock_pool):
        """Test successful pool initialization."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config)
            await client.initialize()

            mock_create_pool.assert_called_once_with(
                host=test_config["host"],
                port=test_config["port"],
                user=test_config["user"],
                password=test_config["password"],
                database=test_config["database"],
                min_size=2,
                max_size=10,
            )
            assert client._pool == mock_pool

    @pytest.mark.asyncio
    async def test_initialize_with_custom_pool_sizes(self, test_config, mock_pool):
        """Test initialization with custom pool sizes."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config, min_size=5, max_size=15)
            await client.initialize()

            mock_create_pool.assert_called_once_with(
                host=test_config["host"],
                port=test_config["port"],
                user=test_config["user"],
                password=test_config["password"],
                database=test_config["database"],
                min_size=5,
                max_size=15,
            )

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, test_config, mock_pool):
        """Test that re-initialization is skipped with warning."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config)
            await client.initialize()
            await client.initialize()  # Second call

            # Should only be called once
            assert mock_create_pool.call_count == 1
            assert client._pool == mock_pool

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, test_config):
        """Test initialization failure raises exception."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.side_effect = asyncpg.PostgresError("Connection failed")
            client = PostgresClient(config=test_config)

            with pytest.raises(asyncpg.PostgresError) as exc_info:
                await client.initialize()
            assert "Connection failed" in str(exc_info.value)
            assert client._pool is None

    @pytest.mark.asyncio
    async def test_initialize_generic_error(self, test_config):
        """Test that generic errors during initialization are raised."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.side_effect = ValueError("Unexpected error")
            client = PostgresClient(config=test_config)

            with pytest.raises(ValueError):
                await client.initialize()


class TestPostgresClientClose:
    """Test PostgresClient.close() method."""

    @pytest.mark.asyncio
    async def test_close_success(self, test_config, mock_pool):
        """Test successful pool closure."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config)
            await client.initialize()
            await client.close()

            mock_pool.close.assert_called_once()
            assert client._pool is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self, test_config):
        """Test closing when pool is not initialized."""
        client = PostgresClient(config=test_config)
        # Should not raise an error
        await client.close()
        assert client._pool is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, test_config, mock_pool):
        """Test that close can be called multiple times safely."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config)
            await client.initialize()
            await client.close()
            await client.close()  # Second call

            # Should only close once
            assert mock_pool.close.call_count == 1


class TestPostgresClientContextManager:
    """Test PostgresClient as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self, test_config, mock_pool):
        """Test using client as async context manager."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            async with PostgresClient(config=test_config) as client:
                assert client._pool == mock_pool
                assert client.is_connected()

            # Pool should be closed after exiting context
            mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self, test_config, mock_pool):
        """Test that context manager returns the client instance."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            async with PostgresClient(config=test_config) as client:
                assert isinstance(client, PostgresClient)

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, test_config, mock_pool):
        """Test context manager properly closes pool even on exception."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            try:
                async with PostgresClient(config=test_config) as client:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Pool should still be closed despite exception
            mock_pool.close.assert_called_once()


class TestPostgresClientPoolProperty:
    """Test PostgresClient.pool property."""

    @pytest.mark.asyncio
    async def test_pool_property_success(self, test_config, mock_pool):
        """Test accessing pool property when initialized."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config)
            await client.initialize()

            pool = client.pool
            assert pool == mock_pool

    def test_pool_property_not_initialized(self, test_config):
        """Test accessing pool property when not initialized raises error."""
        client = PostgresClient(config=test_config)
        with pytest.raises(RuntimeError) as exc_info:
            _ = client.pool
        assert "not initialized" in str(exc_info.value).lower()


class TestPostgresClientQuery:
    """Test PostgresClient.query() method."""

    @pytest.mark.asyncio
    async def test_query_success(
        self, test_config, mock_pool, mock_connection, mock_record
    ):
        """Test successful query execution."""
        mock_records = [mock_record, MagicMock(spec=asyncpg.Record)]
        mock_connection.fetch.return_value = mock_records
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            query = "SELECT * FROM companies LIMIT 10"
            results = await client.query(query)

            assert results == list(mock_records)
            mock_connection.fetch.assert_called_once_with(query, timeout=None)

    @pytest.mark.asyncio
    async def test_query_with_parameters(
        self, test_config, mock_pool, mock_connection, mock_record
    ):
        """Test query with parameters."""
        mock_connection.fetch.return_value = [mock_record]
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            query = "SELECT * FROM companies WHERE id = $1 AND status = $2"
            results = await client.query(query, 123, "active")

            assert len(results) == 1
            mock_connection.fetch.assert_called_once_with(
                query, 123, "active", timeout=None
            )

    @pytest.mark.asyncio
    async def test_query_with_timeout(
        self, test_config, mock_pool, mock_connection, mock_record
    ):
        """Test query with timeout parameter."""
        mock_connection.fetch.return_value = [mock_record]
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            query = "SELECT * FROM companies"
            results = await client.query(query, timeout=30.0)

            mock_connection.fetch.assert_called_once_with(query, timeout=30.0)

    @pytest.mark.asyncio
    async def test_query_not_initialized(self, test_config):
        """Test query when pool is not initialized."""
        client = PostgresClient(config=test_config)
        with pytest.raises(RuntimeError) as exc_info:
            await client.query("SELECT * FROM companies")
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_database_error(self, test_config, mock_pool, mock_connection):
        """Test query with database error."""
        mock_connection.fetch.side_effect = asyncpg.PostgresError("Query failed")
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            with pytest.raises(asyncpg.PostgresError) as exc_info:
                await client.query("SELECT * FROM companies")
            assert "Query failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_empty_result(self, test_config, mock_pool, mock_connection):
        """Test query returning empty result."""
        mock_connection.fetch.return_value = []
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            results = await client.query("SELECT * FROM companies WHERE id = $1", 999)
            assert results == []


class TestPostgresClientQueryOne:
    """Test PostgresClient.query_one() method."""

    @pytest.mark.asyncio
    async def test_query_one_success(
        self, test_config, mock_pool, mock_connection, mock_record
    ):
        """Test successful query_one execution."""
        mock_connection.fetchrow.return_value = mock_record
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            query = "SELECT * FROM companies WHERE id = $1"
            result = await client.query_one(query, 123)

            assert result == mock_record
            mock_connection.fetchrow.assert_called_once_with(query, 123, timeout=None)

    @pytest.mark.asyncio
    async def test_query_one_no_result(self, test_config, mock_pool, mock_connection):
        """Test query_one when no result is found."""
        mock_connection.fetchrow.return_value = None
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            result = await client.query_one(
                "SELECT * FROM companies WHERE id = $1", 999
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_query_one_with_timeout(
        self, test_config, mock_pool, mock_connection, mock_record
    ):
        """Test query_one with timeout parameter."""
        mock_connection.fetchrow.return_value = mock_record
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            await client.query_one("SELECT * FROM companies", timeout=15.0)
            mock_connection.fetchrow.assert_called_once_with(
                "SELECT * FROM companies", timeout=15.0
            )

    @pytest.mark.asyncio
    async def test_query_one_not_initialized(self, test_config):
        """Test query_one when pool is not initialized."""
        client = PostgresClient(config=test_config)
        with pytest.raises(RuntimeError) as exc_info:
            await client.query_one("SELECT * FROM companies")
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_one_database_error(
        self, test_config, mock_pool, mock_connection
    ):
        """Test query_one with database error."""
        mock_connection.fetchrow.side_effect = asyncpg.PostgresError("Query failed")
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            with pytest.raises(asyncpg.PostgresError):
                await client.query_one("SELECT * FROM companies")


class TestPostgresClientQueryValue:
    """Test PostgresClient.query_value() method."""

    @pytest.mark.asyncio
    async def test_query_value_success(self, test_config, mock_pool, mock_connection):
        """Test successful query_value execution."""
        mock_connection.fetchval.return_value = 42
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            query = "SELECT COUNT(*) FROM companies"
            result = await client.query_value(query)

            assert result == 42
            mock_connection.fetchval.assert_called_once_with(query, timeout=None)

    @pytest.mark.asyncio
    async def test_query_value_string_result(
        self, test_config, mock_pool, mock_connection
    ):
        """Test query_value returning string."""
        mock_connection.fetchval.return_value = "test_value"
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            result = await client.query_value("SELECT name FROM companies LIMIT 1")
            assert result == "test_value"

    @pytest.mark.asyncio
    async def test_query_value_with_parameters(
        self, test_config, mock_pool, mock_connection
    ):
        """Test query_value with parameters."""
        mock_connection.fetchval.return_value = 5
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            result = await client.query_value(
                "SELECT COUNT(*) FROM companies WHERE status = $1",
                "active",
            )
            assert result == 5
            mock_connection.fetchval.assert_called_once_with(
                "SELECT COUNT(*) FROM companies WHERE status = $1",
                "active",
                timeout=None,
            )

    @pytest.mark.asyncio
    async def test_query_value_with_timeout(
        self, test_config, mock_pool, mock_connection
    ):
        """Test query_value with timeout parameter."""
        mock_connection.fetchval.return_value = 100
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            await client.query_value("SELECT COUNT(*) FROM companies", timeout=20.0)
            mock_connection.fetchval.assert_called_once_with(
                "SELECT COUNT(*) FROM companies",
                timeout=20.0,
            )

    @pytest.mark.asyncio
    async def test_query_value_not_initialized(self, test_config):
        """Test query_value when pool is not initialized."""
        client = PostgresClient(config=test_config)
        with pytest.raises(RuntimeError) as exc_info:
            await client.query_value("SELECT COUNT(*) FROM companies")
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_value_database_error(
        self, test_config, mock_pool, mock_connection
    ):
        """Test query_value with database error."""
        mock_connection.fetchval.side_effect = asyncpg.PostgresError("Query failed")
        _setup_pool_acquire(mock_pool, mock_connection)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            client = PostgresClient(config=test_config)
            await client.initialize()

            with pytest.raises(asyncpg.PostgresError):
                await client.query_value("SELECT COUNT(*) FROM companies")


class TestPostgresClientIsConnected:
    """Test PostgresClient.is_connected() method."""

    def test_is_connected_false_when_not_initialized(self, test_config):
        """Test is_connected returns False when pool is not initialized."""
        client = PostgresClient(config=test_config)
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_is_connected_true_when_initialized(self, test_config, mock_pool):
        """Test is_connected returns True when pool is initialized."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config)
            await client.initialize()
            assert client.is_connected()

    @pytest.mark.asyncio
    async def test_is_connected_false_after_close(self, test_config, mock_pool):
        """Test is_connected returns False after closing pool."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            client = PostgresClient(config=test_config)
            await client.initialize()
            assert client.is_connected()
            await client.close()
            assert not client.is_connected()


class TestPostgresClientSingleton:
    """Test singleton client functions."""

    @pytest.mark.asyncio
    async def test_get_client_creates_instance(self, mock_pool):
        """Test get_client creates a new instance if none exists."""
        with patch("src.db.postgres_client.PostgresClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock()
            mock_client_class.return_value = mock_client

            # Reset the global variable
            import src.db.postgres_client as pg_module

            pg_module._default_client = None

            client = await get_client()

            assert client == mock_client
            mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_returns_existing_instance(self, mock_pool):
        """Test get_client returns existing instance if already created."""
        with patch("src.db.postgres_client.PostgresClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock()
            mock_client_class.return_value = mock_client

            # Reset the global variable
            import src.db.postgres_client as pg_module

            pg_module._default_client = None

            client1 = await get_client()
            client2 = await get_client()

            assert client1 is client2
            # Should only initialize once
            assert mock_client.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_close_default_client_success(self, mock_pool):
        """Test close_default_client closes the singleton instance."""
        with patch("src.db.postgres_client.PostgresClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            # Reset the global variable
            import src.db.postgres_client as pg_module

            pg_module._default_client = None

            await get_client()
            await close_default_client()

            mock_client.close.assert_called_once()
            assert pg_module._default_client is None

    @pytest.mark.asyncio
    async def test_close_default_client_when_none(self):
        """Test close_default_client when no client exists."""
        # Reset the global variable
        import src.db.postgres_client as pg_module

        pg_module._default_client = None

        # Should not raise an error
        await close_default_client()
        assert pg_module._default_client is None

    @pytest.mark.asyncio
    async def test_get_client_after_close(self, mock_pool):
        """Test get_client creates new instance after close."""
        with patch("src.db.postgres_client.PostgresClient") as mock_client_class:
            mock_client1 = MagicMock()
            mock_client1.initialize = AsyncMock()
            mock_client1.close = AsyncMock()

            mock_client2 = MagicMock()
            mock_client2.initialize = AsyncMock()

            mock_client_class.side_effect = [mock_client1, mock_client2]

            # Reset the global variable
            import src.db.postgres_client as pg_module

            pg_module._default_client = None

            client1 = await get_client()
            await close_default_client()
            client2 = await get_client()

            assert client1 is not client2
            assert mock_client_class.call_count == 2
