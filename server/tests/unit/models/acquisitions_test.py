"""Unit tests for the acquisitions model module.

This module tests the AcquisitionModel class and its various methods,
including query building, filtering, and error handling.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.models.acquisitions import Acquisition, AcquisitionModel


@pytest.fixture
def mock_postgres_client():
    """Create a mock PostgresClient instance."""
    client = MagicMock()
    client.query = AsyncMock()
    client.query_value = AsyncMock()
    return client


@pytest.fixture
def sample_acquisition_uuid():
    """Create a sample acquisition UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_acquiree_uuid():
    """Create a sample acquiree UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_acquirer_uuid():
    """Create a sample acquirer UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_acquisition_record(
    sample_acquisition_uuid, sample_acquiree_uuid, sample_acquirer_uuid
):
    """Create a sample acquisition record (as would be returned from database)."""

    # Create a dict-like object that can be converted to a dict
    # asyncpg.Record objects are dict-like, so dict(record) works
    # dict() constructor works with objects that are iterable over (key, value) pairs
    class Record:
        def __init__(self, **kwargs):
            self._data = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __iter__(self):
            # Make it iterable over (key, value) pairs for dict() constructor
            return iter(self._data.items())

    return Record(
        acquisition_uuid=sample_acquisition_uuid,
        acquiree_uuid=sample_acquiree_uuid,
        acquirer_uuid=sample_acquirer_uuid,
        acquisition_type="merger",
        acquisition_announce_date=datetime(2023, 6, 15),
        acquisition_price_usd=50000000,
        terms="Cash and stock",
        acquirer_type="public_company",
    )


@pytest.fixture
def sample_acquisition(
    sample_acquisition_uuid, sample_acquiree_uuid, sample_acquirer_uuid
):
    """Create a sample Acquisition Pydantic model instance."""
    return Acquisition(
        acquisition_uuid=sample_acquisition_uuid,
        acquiree_uuid=sample_acquiree_uuid,
        acquirer_uuid=sample_acquirer_uuid,
        acquisition_type="merger",
        acquisition_announce_date=datetime(2023, 6, 15),
        acquisition_price_usd=50000000,
        terms="Cash and stock",
        acquirer_type="public_company",
    )


class TestAcquisitionModelInitialization:
    """Test AcquisitionModel initialization."""

    def test_init_with_custom_client(self, mock_postgres_client):
        """Test initialization with custom PostgresClient."""
        model = AcquisitionModel(client=mock_postgres_client)
        assert model._client == mock_postgres_client
        assert not model._use_default_client

    def test_init_without_client(self):
        """Test initialization without client (uses default)."""
        model = AcquisitionModel()
        assert model._client is None
        assert model._use_default_client

    @pytest.mark.asyncio
    async def test_initialize_with_default_client(self):
        """Test initialize() when using default client."""
        mock_client = MagicMock()
        with patch(
            "src.models.acquisitions.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            model = AcquisitionModel()
            await model.initialize()
            assert model._client == mock_client
            mock_get_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_client(self, mock_postgres_client):
        """Test initialize() when using custom client (should not call get_postgres_client)."""
        with patch(
            "src.models.acquisitions.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            model = AcquisitionModel(client=mock_postgres_client)
            await model.initialize()
            assert model._client == mock_postgres_client
            mock_get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize() can be called multiple times safely."""
        mock_client = MagicMock()
        with patch(
            "src.models.acquisitions.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            model = AcquisitionModel()
            await model.initialize()
            await model.initialize()  # Second call
            # Should only call get_postgres_client once
            assert mock_get_client.call_count == 1


class TestAcquisitionModelGet:
    """Test AcquisitionModel.get() method."""

    @pytest.mark.asyncio
    async def test_get_all_acquisitions(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting all acquisitions without filters."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert len(results) == 1
        assert isinstance(results[0], Acquisition)
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        assert "SELECT * FROM acquisitions" in call_args[0][0]
        assert "WHERE" not in call_args[0][0]
        assert "ORDER BY acquisition_announce_date DESC NULLS LAST" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_by_acquisition_uuid(
        self, mock_postgres_client, sample_acquisition_uuid, sample_acquisition_record
    ):
        """Test getting acquisition by UUID."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(acquisition_uuid=sample_acquisition_uuid)

        assert len(results) == 1
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_uuid = $1" in query
        assert str(sample_acquisition_uuid) in params

    @pytest.mark.asyncio
    async def test_get_by_acquiree_uuid(
        self, mock_postgres_client, sample_acquiree_uuid, sample_acquisition_record
    ):
        """Test getting acquisitions by acquiree UUID."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(acquiree_uuid=sample_acquiree_uuid)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquiree_uuid = $1" in query
        assert str(sample_acquiree_uuid) in params

    @pytest.mark.asyncio
    async def test_get_by_acquirer_uuid(
        self, mock_postgres_client, sample_acquirer_uuid, sample_acquisition_record
    ):
        """Test getting acquisitions by acquirer UUID."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(acquirer_uuid=sample_acquirer_uuid)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquirer_uuid = $1" in query
        assert str(sample_acquirer_uuid) in params

    @pytest.mark.asyncio
    async def test_get_by_acquisition_type(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by type."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(acquisition_type="merger")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_type = $1" in query
        assert "merger" in params

    @pytest.mark.asyncio
    async def test_get_by_announce_date(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by exact announce date."""
        announce_date = datetime(2023, 6, 15)
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(acquisition_announce_date=announce_date)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_announce_date = $1" in query
        assert announce_date in params

    @pytest.mark.asyncio
    async def test_get_by_announce_date_range(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by announce date range."""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            acquisition_announce_date_from=date_from,
            acquisition_announce_date_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_announce_date >= $1" in query
        assert "acquisition_announce_date <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_price_exact(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by exact price."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(acquisition_price_usd=50000000)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_price_usd = $1" in query
        assert 50000000 in params

    @pytest.mark.asyncio
    async def test_get_by_price_range(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by price range."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            acquisition_price_usd_min=1000000,
            acquisition_price_usd_max=100000000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_price_usd >= $1" in query
        assert "acquisition_price_usd <= $2" in query
        assert 1000000 in params
        assert 100000000 in params

    @pytest.mark.asyncio
    async def test_get_by_terms_exact(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by exact terms match."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(terms="Cash and stock")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "terms = $1" in query
        assert "Cash and stock" in params

    @pytest.mark.asyncio
    async def test_get_by_terms_ilike(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by case-insensitive terms search."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(terms_ilike="cash")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "terms ILIKE $1" in query
        assert "%cash%" in params

    @pytest.mark.asyncio
    async def test_get_by_acquirer_type(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions by acquirer type."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(acquirer_type="public_company")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquirer_type = $1" in query
        assert "public_company" in params

    @pytest.mark.asyncio
    async def test_get_with_multiple_filters(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions with multiple filters."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            acquisition_type="merger",
            acquisition_price_usd_min=1000000,
            acquirer_type="public_company",
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_type = $1" in query
        assert "acquisition_price_usd >= $2" in query
        assert "acquirer_type = $3" in query
        assert "merger" in params
        assert 1000000 in params
        assert "public_company" in params

    @pytest.mark.asyncio
    async def test_get_with_limit(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions with limit."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(limit=10)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "LIMIT $1" in query
        assert 10 in params

    @pytest.mark.asyncio
    async def test_get_with_offset(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions with offset."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(offset=20)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "OFFSET $1" in query
        assert 20 in params

    @pytest.mark.asyncio
    async def test_get_with_limit_and_offset(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test getting acquisitions with both limit and offset."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(limit=10, offset=20)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "LIMIT $1" in query
        assert "OFFSET $2" in query
        assert 10 in params
        assert 20 in params

    @pytest.mark.asyncio
    async def test_get_empty_result(self, mock_postgres_client):
        """Test getting acquisitions when no results are found."""
        mock_postgres_client.query.return_value = []
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert results == []
        mock_postgres_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_multiple_results(
        self,
        mock_postgres_client,
        sample_acquisition_record,
        sample_acquiree_uuid,
        sample_acquirer_uuid,
    ):
        """Test getting multiple acquisitions."""

        # Create a second record with a different UUID
        class Record:
            def __init__(self, **kwargs):
                self._data = kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def __iter__(self):
                # Make it iterable over (key, value) pairs for dict() constructor
                return iter(self._data.items())

        record2 = Record(
            acquisition_uuid=uuid4(),
            acquiree_uuid=sample_acquiree_uuid,
            acquirer_uuid=sample_acquirer_uuid,
            acquisition_type="acquisition",
            acquisition_announce_date=datetime(2023, 7, 20),
            acquisition_price_usd=75000000,
            terms="All cash",
            acquirer_type="private_equity",
        )

        mock_postgres_client.query.return_value = [sample_acquisition_record, record2]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert len(results) == 2
        assert all(isinstance(r, Acquisition) for r in results)

    @pytest.mark.asyncio
    async def test_get_not_initialized(self):
        """Test get() raises RuntimeError when client is not initialized."""
        model = AcquisitionModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.get()
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_database_error(self, mock_postgres_client):
        """Test get() raises exception when database query fails."""
        mock_postgres_client.query.side_effect = Exception("Database connection failed")
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.get()
        assert "Database connection failed" in str(exc_info.value)


class TestAcquisitionModelGetByUuid:
    """Test AcquisitionModel.get_by_uuid() method."""

    @pytest.mark.asyncio
    async def test_get_by_uuid_success(
        self, mock_postgres_client, sample_acquisition_uuid, sample_acquisition_record
    ):
        """Test successfully getting acquisition by UUID."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        result = await model.get_by_uuid(sample_acquisition_uuid)

        assert result is not None
        assert isinstance(result, Acquisition)
        assert result.acquisition_uuid == sample_acquisition_uuid
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_uuid = $1" in query
        assert "LIMIT $2" in query
        assert str(sample_acquisition_uuid) in params
        assert 1 in params

    @pytest.mark.asyncio
    async def test_get_by_uuid_not_found(
        self, mock_postgres_client, sample_acquisition_uuid
    ):
        """Test get_by_uuid() when acquisition is not found."""
        mock_postgres_client.query.return_value = []
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        result = await model.get_by_uuid(sample_acquisition_uuid)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_uuid_not_initialized(self, sample_acquisition_uuid):
        """Test get_by_uuid() raises RuntimeError when client is not initialized."""
        model = AcquisitionModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.get_by_uuid(sample_acquisition_uuid)
        assert "not initialized" in str(exc_info.value).lower()


class TestAcquisitionModelCount:
    """Test AcquisitionModel.count() method."""

    @pytest.mark.asyncio
    async def test_count_all(self, mock_postgres_client):
        """Test counting all acquisitions."""
        mock_postgres_client.query_value.return_value = 42
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count()

        assert count == 42
        mock_postgres_client.query_value.assert_called_once()
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        assert "SELECT COUNT(*) FROM acquisitions" in query
        assert "WHERE" not in query

    @pytest.mark.asyncio
    async def test_count_by_acquisition_uuid(
        self, mock_postgres_client, sample_acquisition_uuid
    ):
        """Test counting acquisitions by UUID."""
        mock_postgres_client.query_value.return_value = 1
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(acquisition_uuid=sample_acquisition_uuid)

        assert count == 1
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_uuid = $1" in query
        assert str(sample_acquisition_uuid) in params

    @pytest.mark.asyncio
    async def test_count_by_acquiree_uuid(
        self, mock_postgres_client, sample_acquiree_uuid
    ):
        """Test counting acquisitions by acquiree UUID."""
        mock_postgres_client.query_value.return_value = 3
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(acquiree_uuid=sample_acquiree_uuid)

        assert count == 3
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquiree_uuid = $1" in query
        assert str(sample_acquiree_uuid) in params

    @pytest.mark.asyncio
    async def test_count_by_acquirer_uuid(
        self, mock_postgres_client, sample_acquirer_uuid
    ):
        """Test counting acquisitions by acquirer UUID."""
        mock_postgres_client.query_value.return_value = 5
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(acquirer_uuid=sample_acquirer_uuid)

        assert count == 5
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquirer_uuid = $1" in query
        assert str(sample_acquirer_uuid) in params

    @pytest.mark.asyncio
    async def test_count_by_acquisition_type(self, mock_postgres_client):
        """Test counting acquisitions by type."""
        mock_postgres_client.query_value.return_value = 10
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(acquisition_type="merger")

        assert count == 10
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_type = $1" in query
        assert "merger" in params

    @pytest.mark.asyncio
    async def test_count_by_date_range(self, mock_postgres_client):
        """Test counting acquisitions by date range."""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        mock_postgres_client.query_value.return_value = 15
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            acquisition_announce_date_from=date_from,
            acquisition_announce_date_to=date_to,
        )

        assert count == 15
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_announce_date >= $1" in query
        assert "acquisition_announce_date <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_count_by_price_range(self, mock_postgres_client):
        """Test counting acquisitions by price range."""
        mock_postgres_client.query_value.return_value = 8
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            acquisition_price_usd_min=1000000,
            acquisition_price_usd_max=100000000,
        )

        assert count == 8
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_price_usd >= $1" in query
        assert "acquisition_price_usd <= $2" in query
        assert 1000000 in params
        assert 100000000 in params

    @pytest.mark.asyncio
    async def test_count_by_acquirer_type(self, mock_postgres_client):
        """Test counting acquisitions by acquirer type."""
        mock_postgres_client.query_value.return_value = 12
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(acquirer_type="public_company")

        assert count == 12
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquirer_type = $1" in query
        assert "public_company" in params

    @pytest.mark.asyncio
    async def test_count_with_multiple_filters(self, mock_postgres_client):
        """Test counting acquisitions with multiple filters."""
        mock_postgres_client.query_value.return_value = 7
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            acquisition_type="merger",
            acquisition_price_usd_min=1000000,
            acquirer_type="public_company",
        )

        assert count == 7
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_type = $1" in query
        assert "acquisition_price_usd >= $2" in query
        assert "acquirer_type = $3" in query
        assert "merger" in params
        assert 1000000 in params
        assert "public_company" in params

    @pytest.mark.asyncio
    async def test_count_zero(self, mock_postgres_client):
        """Test counting when no acquisitions match."""
        mock_postgres_client.query_value.return_value = 0
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(acquisition_type="nonexistent")

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_none_result(self, mock_postgres_client):
        """Test counting when query returns None (should return 0)."""
        mock_postgres_client.query_value.return_value = None
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_not_initialized(self):
        """Test count() raises RuntimeError when client is not initialized."""
        model = AcquisitionModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.count()
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_count_database_error(self, mock_postgres_client):
        """Test count() raises exception when database query fails."""
        mock_postgres_client.query_value.side_effect = Exception(
            "Database connection failed"
        )
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.count()
        assert "Database connection failed" in str(exc_info.value)


class TestAcquisitionModelEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_get_with_none_values_ignored(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test that None filter values are ignored in query."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            acquisition_uuid=None,
            acquisition_type=None,
            acquisition_price_usd=None,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        # None values should not appear in WHERE clause
        assert "acquisition_uuid = $" not in query
        assert "acquisition_type = $" not in query
        assert "acquisition_price_usd = $" not in query

    @pytest.mark.asyncio
    async def test_get_with_partial_none_values(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test that only non-None values are used in query."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            acquisition_type="merger",
            acquisition_price_usd=None,  # Should be ignored
            acquirer_type="public_company",
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "acquisition_type = $1" in query
        assert "acquirer_type = $2" in query
        assert "acquisition_price_usd = $" not in query
        assert "merger" in params
        assert "public_company" in params
        assert None not in params

    @pytest.mark.asyncio
    async def test_get_query_ordering(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test that query includes proper ORDER BY clause."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        await model.get()

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        assert "ORDER BY acquisition_announce_date DESC NULLS LAST" in query

    @pytest.mark.asyncio
    async def test_get_parameter_ordering(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test that query parameters are correctly ordered."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(
            acquisition_type="merger",
            acquisition_price_usd_min=1000000,
            acquirer_type="public_company",
        )

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        # Check that parameters are numbered sequentially
        assert "$1" in query
        assert "$2" in query
        assert "$3" in query
        # Check that limit/offset use correct parameter indices
        # Since we have 3 filters, limit should be $4 and offset $5 if present
        # But we don't have limit/offset here, so max should be $3

    @pytest.mark.asyncio
    async def test_get_with_limit_parameter_index(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test that limit parameter uses correct index after filters."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(acquisition_type="merger", limit=10)

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # acquisition_type = $1, limit = $2
        assert "LIMIT $2" in query
        assert 10 in params
        assert len(params) == 2  # acquisition_type and limit

    @pytest.mark.asyncio
    async def test_get_with_offset_parameter_index(
        self, mock_postgres_client, sample_acquisition_record
    ):
        """Test that offset parameter uses correct index after filters and limit."""
        mock_postgres_client.query.return_value = [sample_acquisition_record]
        model = AcquisitionModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(acquisition_type="merger", limit=10, offset=20)

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # acquisition_type = $1, limit = $2, offset = $3
        assert "LIMIT $2" in query
        assert "OFFSET $3" in query
        assert len(params) == 3
