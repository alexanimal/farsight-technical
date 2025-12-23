"""Unit tests for the Pinecone client module.

This module tests the PineconeClient class and its various methods,
including connection management, query execution, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.db.pinecone_client import PineconeClient, close_default_client, get_client


@pytest.fixture
def mock_pinecone_client():
    """Create a mock Pinecone client instance."""
    client = MagicMock()
    client.list_indexes = MagicMock()
    client.Index = MagicMock()
    return client


@pytest.fixture
def mock_index():
    """Create a mock Pinecone Index instance."""
    index = MagicMock()
    index.query = MagicMock()
    index.fetch = MagicMock()
    index.describe_index_stats = MagicMock()
    return index


@pytest.fixture
def mock_query_response():
    """Create a mock query response."""
    response = MagicMock()
    match1 = MagicMock()
    match1.id = "id1"
    match1.score = 0.95
    match1.metadata = {"name": "Company 1"}
    match2 = MagicMock()
    match2.id = "id2"
    match2.score = 0.87
    match2.metadata = {"name": "Company 2"}
    response.matches = [match1, match2]
    return response


@pytest.fixture
def mock_fetch_response():
    """Create a mock fetch response."""
    response = MagicMock()
    response.vectors = {
        "id1": {"values": [0.1, 0.2, 0.3], "metadata": {"name": "Company 1"}},
        "id2": {"values": [0.4, 0.5, 0.6], "metadata": {"name": "Company 2"}},
    }
    return response


@pytest.fixture
def mock_index_stats():
    """Create a mock index stats object."""
    stats = MagicMock()
    stats.dimension = 1536
    stats.index_fullness = 0.5
    stats.total_vector_count = 1000
    return stats


class TestPineconeClientInitialization:
    """Test PineconeClient initialization."""

    def test_init_with_default_settings(self, monkeypatch):
        """Test initialization with default settings."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-api-key"
            mock_settings.pinecone_index = "default-index"
            client = PineconeClient()
            assert client._api_key == "test-api-key"
            assert client._default_index == "default-index"
            assert client._client is None

    def test_init_with_custom_api_key(self, monkeypatch):
        """Test initialization with custom API key."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "default-key"
            mock_settings.pinecone_index = "default-index"
            client = PineconeClient(api_key="custom-key")
            assert client._api_key == "custom-key"
            assert client._default_index == "default-index"

    def test_init_with_custom_default_index(self, monkeypatch):
        """Test initialization with custom default index."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "default-index"
            client = PineconeClient(default_index="custom-index")
            assert client._api_key == "test-key"
            assert client._default_index == "custom-index"

    def test_init_with_both_custom(self, monkeypatch):
        """Test initialization with both custom API key and index."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "default-key"
            mock_settings.pinecone_index = "default-index"
            client = PineconeClient(api_key="custom-key", default_index="custom-index")
            assert client._api_key == "custom-key"
            assert client._default_index == "custom-index"

    def test_init_client_not_initialized(self, monkeypatch):
        """Test that client is None after initialization."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"
            client = PineconeClient()
            assert client._client is None
            assert not client.is_connected()


class TestPineconeClientInitialize:
    """Test PineconeClient.initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_pinecone_client, monkeypatch):
        """Test successful client initialization."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-api-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient()
                await client.initialize()

                mock_pinecone_class.assert_called_once_with(api_key="test-api-key")
                assert client._client == mock_pinecone_client

    @pytest.mark.asyncio
    async def test_initialize_with_custom_api_key(self, mock_pinecone_client, monkeypatch):
        """Test initialization with custom API key."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "default-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient(api_key="custom-key")
                await client.initialize()

                mock_pinecone_class.assert_called_once_with(api_key="custom-key")

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_pinecone_client, monkeypatch):
        """Test that re-initialization is skipped with warning."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient()
                await client.initialize()
                await client.initialize()  # Second call

                # Should only be called once
                assert mock_pinecone_class.call_count == 1
                assert client._client == mock_pinecone_client

    @pytest.mark.asyncio
    async def test_initialize_no_api_key(self, monkeypatch):
        """Test initialization failure when API key is missing."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = None
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            with pytest.raises(ValueError) as exc_info:
                await client.initialize()
            assert "API key is required" in str(exc_info.value)
            assert client._client is None

    @pytest.mark.asyncio
    async def test_initialize_empty_api_key(self, monkeypatch):
        """Test initialization failure when API key is empty string."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = ""
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            with pytest.raises(ValueError) as exc_info:
                await client.initialize()
            assert "API key is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, monkeypatch):
        """Test initialization failure raises exception."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.side_effect = Exception("Connection failed")
                client = PineconeClient()

                with pytest.raises(Exception) as exc_info:
                    await client.initialize()
                assert "Connection failed" in str(exc_info.value)
                assert client._client is None


class TestPineconeClientClose:
    """Test PineconeClient.close() method."""

    @pytest.mark.asyncio
    async def test_close_success(self, mock_pinecone_client, monkeypatch):
        """Test successful client closure."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient()
                await client.initialize()
                await client.close()

                assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self, monkeypatch):
        """Test closing when client is not initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            # Should not raise an error
            await client.close()
            assert client._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_pinecone_client, monkeypatch):
        """Test that close can be called multiple times safely."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient()
                await client.initialize()
                await client.close()
                await client.close()  # Second call

                assert client._client is None


class TestPineconeClientContextManager:
    """Test PineconeClient as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self, mock_pinecone_client, monkeypatch):
        """Test using client as async context manager."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                async with PineconeClient() as client:
                    assert client._client == mock_pinecone_client
                    assert client.is_connected()

                # Client should be closed after exiting context
                assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self, mock_pinecone_client, monkeypatch):
        """Test that context manager returns the client instance."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                async with PineconeClient() as client:
                    assert isinstance(client, PineconeClient)

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, mock_pinecone_client, monkeypatch):
        """Test context manager properly closes client even on exception."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                try:
                    async with PineconeClient() as client:
                        raise ValueError("Test exception")
                except ValueError:
                    pass

                # Client should still be closed despite exception
                assert client._client is None


class TestPineconeClientClientProperty:
    """Test PineconeClient.client property."""

    @pytest.mark.asyncio
    async def test_client_property_success(self, mock_pinecone_client, monkeypatch):
        """Test accessing client property when initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient()
                await client.initialize()

                pinecone_client = client.client
                assert pinecone_client == mock_pinecone_client

    def test_client_property_not_initialized(self, monkeypatch):
        """Test accessing client property when not initialized raises error."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            with pytest.raises(RuntimeError) as exc_info:
                _ = client.client
            assert "not initialized" in str(exc_info.value).lower()


class TestPineconeClientListIndexes:
    """Test PineconeClient.list_indexes() method."""

    def test_list_indexes_success_with_names_method(
        self, mock_pinecone_client, mock_index, monkeypatch
    ):
        """Test successful list_indexes when response has names() method."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_response = MagicMock()
            mock_response.names = MagicMock(return_value=["index1", "index2", "index3"])
            mock_pinecone_client.list_indexes.return_value = mock_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            indexes = client.list_indexes()
            assert indexes == ["index1", "index2", "index3"]
            mock_pinecone_client.list_indexes.assert_called_once()

    def test_list_indexes_success_with_list(self, mock_pinecone_client, mock_index, monkeypatch):
        """Test successful list_indexes when response is a list."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_index1 = MagicMock()
            mock_index1.name = "index1"
            mock_index2 = MagicMock()
            mock_index2.name = "index2"
            mock_pinecone_client.list_indexes.return_value = [mock_index1, mock_index2]

            client = PineconeClient()
            client._client = mock_pinecone_client

            indexes = client.list_indexes()
            assert indexes == ["index1", "index2"]

    def test_list_indexes_success_with_iterable(
        self, mock_pinecone_client, mock_index, monkeypatch
    ):
        """Test successful list_indexes when response is iterable."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_index1 = MagicMock()
            mock_index1.name = "index1"
            mock_index2 = MagicMock()
            mock_index2.name = "index2"
            mock_response = iter([mock_index1, mock_index2])
            mock_pinecone_client.list_indexes.return_value = mock_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            indexes = client.list_indexes()
            assert indexes == ["index1", "index2"]

    def test_list_indexes_empty_result(self, mock_pinecone_client, monkeypatch):
        """Test list_indexes with empty result."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.list_indexes.return_value = []

            client = PineconeClient()
            client._client = mock_pinecone_client

            indexes = client.list_indexes()
            assert indexes == []

    def test_list_indexes_not_initialized(self, monkeypatch):
        """Test list_indexes when client is not initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            with pytest.raises(RuntimeError) as exc_info:
                client.list_indexes()
            assert "not initialized" in str(exc_info.value).lower()

    def test_list_indexes_error(self, mock_pinecone_client, monkeypatch):
        """Test list_indexes with error."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.list_indexes.side_effect = Exception("List failed")

            client = PineconeClient()
            client._client = mock_pinecone_client

            with pytest.raises(Exception) as exc_info:
                client.list_indexes()
            assert "List failed" in str(exc_info.value)


class TestPineconeClientQuery:
    """Test PineconeClient.query() method."""

    @pytest.mark.asyncio
    async def test_query_success(
        self, mock_pinecone_client, mock_index, mock_query_response, monkeypatch
    ):
        """Test successful query execution."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.query.return_value = mock_query_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            query_vector = [0.1, 0.2, 0.3]
            results = await client.query(query_vector, top_k=5)

            assert results == mock_query_response
            mock_pinecone_client.Index.assert_called_once_with("test-index")
            mock_index.query.assert_called_once_with(
                vector=query_vector,
                top_k=5,
                include_metadata=True,
                include_values=False,
            )

    @pytest.mark.asyncio
    async def test_query_with_custom_index(
        self, mock_pinecone_client, mock_index, mock_query_response, monkeypatch
    ):
        """Test query with custom index name."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "default-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.query.return_value = mock_query_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            query_vector = [0.1, 0.2, 0.3]
            await client.query(query_vector, index_name="custom-index", top_k=10)

            mock_pinecone_client.Index.assert_called_once_with("custom-index")

    @pytest.mark.asyncio
    async def test_query_with_metadata_filter(
        self, mock_pinecone_client, mock_index, mock_query_response, monkeypatch
    ):
        """Test query with metadata filter."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.query.return_value = mock_query_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            query_vector = [0.1, 0.2, 0.3]
            metadata_filter = {"sector": "AI", "status": "active"}
            await client.query(query_vector, metadata_filter=metadata_filter)

            mock_index.query.assert_called_once_with(
                vector=query_vector,
                top_k=10,
                include_metadata=True,
                include_values=False,
                filter=metadata_filter,
            )

    @pytest.mark.asyncio
    async def test_query_with_namespace(
        self, mock_pinecone_client, mock_index, mock_query_response, monkeypatch
    ):
        """Test query with namespace."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.query.return_value = mock_query_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            query_vector = [0.1, 0.2, 0.3]
            await client.query(query_vector, namespace="ns1")

            mock_index.query.assert_called_once_with(
                vector=query_vector,
                top_k=10,
                include_metadata=True,
                include_values=False,
                namespace="ns1",
            )

    @pytest.mark.asyncio
    async def test_query_with_include_values(
        self, mock_pinecone_client, mock_index, mock_query_response, monkeypatch
    ):
        """Test query with include_values=True."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.query.return_value = mock_query_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            query_vector = [0.1, 0.2, 0.3]
            await client.query(query_vector, include_values=True)

            mock_index.query.assert_called_once_with(
                vector=query_vector,
                top_k=10,
                include_metadata=True,
                include_values=True,
            )

    @pytest.mark.asyncio
    async def test_query_not_initialized(self, monkeypatch):
        """Test query when client is not initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            with pytest.raises(RuntimeError) as exc_info:
                await client.query([0.1, 0.2, 0.3])
            assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_no_index_name(self, mock_pinecone_client, monkeypatch):
        """Test query when no index name is provided and no default."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = None

            client = PineconeClient()
            client._client = mock_pinecone_client

            with pytest.raises(ValueError) as exc_info:
                await client.query([0.1, 0.2, 0.3])
            assert "index_name must be provided" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_error(self, mock_pinecone_client, mock_index, monkeypatch):
        """Test query with error."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.query.side_effect = Exception("Query failed")

            client = PineconeClient()
            client._client = mock_pinecone_client

            with pytest.raises(Exception) as exc_info:
                await client.query([0.1, 0.2, 0.3])
            assert "Query failed" in str(exc_info.value)


class TestPineconeClientFetch:
    """Test PineconeClient.fetch() method."""

    @pytest.mark.asyncio
    async def test_fetch_success(
        self, mock_pinecone_client, mock_index, mock_fetch_response, monkeypatch
    ):
        """Test successful fetch execution."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.fetch.return_value = mock_fetch_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            ids = ["id1", "id2"]
            results = await client.fetch(ids)

            assert results == mock_fetch_response
            mock_pinecone_client.Index.assert_called_once_with("test-index")
            mock_index.fetch.assert_called_once_with(ids=ids)

    @pytest.mark.asyncio
    async def test_fetch_with_custom_index(
        self, mock_pinecone_client, mock_index, mock_fetch_response, monkeypatch
    ):
        """Test fetch with custom index name."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "default-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.fetch.return_value = mock_fetch_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            ids = ["id1", "id2"]
            await client.fetch(ids, index_name="custom-index")

            mock_pinecone_client.Index.assert_called_once_with("custom-index")

    @pytest.mark.asyncio
    async def test_fetch_with_namespace(
        self, mock_pinecone_client, mock_index, mock_fetch_response, monkeypatch
    ):
        """Test fetch with namespace."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.fetch.return_value = mock_fetch_response

            client = PineconeClient()
            client._client = mock_pinecone_client

            ids = ["id1", "id2"]
            await client.fetch(ids, namespace="ns1")

            mock_index.fetch.assert_called_once_with(ids=ids, namespace="ns1")

    @pytest.mark.asyncio
    async def test_fetch_not_initialized(self, monkeypatch):
        """Test fetch when client is not initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            with pytest.raises(RuntimeError) as exc_info:
                await client.fetch(["id1"])
            assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_fetch_no_index_name(self, mock_pinecone_client, monkeypatch):
        """Test fetch when no index name is provided and no default."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = None

            client = PineconeClient()
            client._client = mock_pinecone_client

            with pytest.raises(ValueError) as exc_info:
                await client.fetch(["id1"])
            assert "index_name must be provided" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_fetch_error(self, mock_pinecone_client, mock_index, monkeypatch):
        """Test fetch with error."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.fetch.side_effect = Exception("Fetch failed")

            client = PineconeClient()
            client._client = mock_pinecone_client

            with pytest.raises(Exception) as exc_info:
                await client.fetch(["id1"])
            assert "Fetch failed" in str(exc_info.value)


class TestPineconeClientIsConnected:
    """Test PineconeClient.is_connected() method."""

    def test_is_connected_false_when_not_initialized(self, monkeypatch):
        """Test is_connected returns False when client is not initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_is_connected_true_when_initialized(self, mock_pinecone_client, monkeypatch):
        """Test is_connected returns True when client is initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient()
                await client.initialize()
                assert client.is_connected()

    @pytest.mark.asyncio
    async def test_is_connected_false_after_close(self, mock_pinecone_client, monkeypatch):
        """Test is_connected returns False after closing client."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client
                client = PineconeClient()
                await client.initialize()
                assert client.is_connected()
                await client.close()
                assert not client.is_connected()


class TestPineconeClientGetIndexInfo:
    """Test PineconeClient.get_index_info() method."""

    def test_get_index_info_success(
        self, mock_pinecone_client, mock_index, mock_index_stats, monkeypatch
    ):
        """Test successful get_index_info execution."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.describe_index_stats.return_value = mock_index_stats

            client = PineconeClient()
            client._client = mock_pinecone_client

            info = client.get_index_info()

            assert info["name"] == "test-index"
            assert info["dimension"] == 1536
            assert info["index_fullness"] == 0.5
            assert info["total_vector_count"] == 1000
            mock_pinecone_client.Index.assert_called_once_with("test-index")
            mock_index.describe_index_stats.assert_called_once()

    def test_get_index_info_with_custom_index(
        self, mock_pinecone_client, mock_index, mock_index_stats, monkeypatch
    ):
        """Test get_index_info with custom index name."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "default-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.describe_index_stats.return_value = mock_index_stats

            client = PineconeClient()
            client._client = mock_pinecone_client

            info = client.get_index_info("custom-index")

            assert info["name"] == "custom-index"
            mock_pinecone_client.Index.assert_called_once_with("custom-index")

    def test_get_index_info_with_missing_attributes(
        self, mock_pinecone_client, mock_index, monkeypatch
    ):
        """Test get_index_info when stats object has missing attributes."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            # Use spec=[] to prevent MagicMock from auto-creating attributes
            # This ensures hasattr() returns False for missing attributes
            mock_stats = Mock(spec=[])
            mock_pinecone_client.Index.return_value = mock_index
            mock_index.describe_index_stats.return_value = mock_stats

            client = PineconeClient()
            client._client = mock_pinecone_client

            info = client.get_index_info()

            assert info["name"] == "test-index"
            assert info["dimension"] is None
            assert info["index_fullness"] is None
            assert info["total_vector_count"] is None

    def test_get_index_info_not_initialized(self, monkeypatch):
        """Test get_index_info when client is not initialized."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            client = PineconeClient()
            with pytest.raises(RuntimeError) as exc_info:
                client.get_index_info()
            assert "not initialized" in str(exc_info.value).lower()

    def test_get_index_info_no_index_name(self, mock_pinecone_client, monkeypatch):
        """Test get_index_info when no index name is provided and no default."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = None

            client = PineconeClient()
            client._client = mock_pinecone_client

            with pytest.raises(ValueError) as exc_info:
                client.get_index_info()
            assert "index_name must be provided" in str(exc_info.value).lower()

    def test_get_index_info_error(self, mock_pinecone_client, mock_index, monkeypatch):
        """Test get_index_info with error."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            mock_pinecone_client.Index.return_value = mock_index
            mock_index.describe_index_stats.side_effect = Exception("Stats failed")

            client = PineconeClient()
            client._client = mock_pinecone_client

            with pytest.raises(Exception) as exc_info:
                client.get_index_info()
            assert "Stats failed" in str(exc_info.value)


class TestPineconeClientSingleton:
    """Test singleton client functions."""

    @pytest.mark.asyncio
    async def test_get_client_creates_instance(self, mock_pinecone_client, monkeypatch):
        """Test get_client creates a new instance if none exists."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client

                with patch("src.db.pinecone_client.PineconeClient") as mock_client_class:
                    mock_client = MagicMock()
                    mock_client.initialize = AsyncMock()
                    mock_client_class.return_value = mock_client

                    # Reset the global variable
                    import src.db.pinecone_client as pc_module

                    pc_module._default_client = None

                    client = await get_client()

                    assert client == mock_client
                    mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_returns_existing_instance(self, mock_pinecone_client, monkeypatch):
        """Test get_client returns existing instance if already created."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client

                with patch("src.db.pinecone_client.PineconeClient") as mock_client_class:
                    mock_client = MagicMock()
                    mock_client.initialize = AsyncMock()
                    mock_client_class.return_value = mock_client

                    # Reset the global variable
                    import src.db.pinecone_client as pc_module

                    pc_module._default_client = None

                    client1 = await get_client()
                    client2 = await get_client()

                    assert client1 is client2
                    # Should only initialize once
                    assert mock_client.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_close_default_client_success(self, mock_pinecone_client, monkeypatch):
        """Test close_default_client closes the singleton instance."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client

                with patch("src.db.pinecone_client.PineconeClient") as mock_client_class:
                    mock_client = MagicMock()
                    mock_client.initialize = AsyncMock()
                    mock_client.close = AsyncMock()
                    mock_client_class.return_value = mock_client

                    # Reset the global variable
                    import src.db.pinecone_client as pc_module

                    pc_module._default_client = None

                    await get_client()
                    await close_default_client()

                    mock_client.close.assert_called_once()
                    assert pc_module._default_client is None

    @pytest.mark.asyncio
    async def test_close_default_client_when_none(self):
        """Test close_default_client when no client exists."""
        # Reset the global variable
        import src.db.pinecone_client as pc_module

        pc_module._default_client = None

        # Should not raise an error
        await close_default_client()
        assert pc_module._default_client is None

    @pytest.mark.asyncio
    async def test_get_client_after_close(self, mock_pinecone_client, monkeypatch):
        """Test get_client creates new instance after close."""
        with patch("src.db.pinecone_client.settings") as mock_settings:
            mock_settings.pinecone_api_key = "test-key"
            mock_settings.pinecone_index = "test-index"

            with patch("src.db.pinecone_client.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_client

                with patch("src.db.pinecone_client.PineconeClient") as mock_client_class:
                    mock_client1 = MagicMock()
                    mock_client1.initialize = AsyncMock()
                    mock_client1.close = AsyncMock()

                    mock_client2 = MagicMock()
                    mock_client2.initialize = AsyncMock()

                    mock_client_class.side_effect = [mock_client1, mock_client2]

                    # Reset the global variable
                    import src.db.pinecone_client as pc_module

                    pc_module._default_client = None

                    client1 = await get_client()
                    await close_default_client()
                    client2 = await get_client()

                    assert client1 is not client2
                    assert mock_client_class.call_count == 2
