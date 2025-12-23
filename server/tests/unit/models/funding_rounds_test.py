"""Unit tests for the funding rounds model module.

This module tests the FundingRoundModel class and its various methods,
including query building, filtering, and error handling.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.models.funding_rounds import FundingRound, FundingRoundModel


@pytest.fixture
def mock_postgres_client():
    """Create a mock PostgresClient instance."""
    client = MagicMock()
    client.query = AsyncMock()
    client.query_value = AsyncMock()
    return client


@pytest.fixture
def sample_funding_round_uuid():
    """Create a sample funding round UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_funding_round_record(sample_funding_round_uuid, sample_org_uuid):
    """Create a sample funding round record (as would be returned from database)."""

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
        funding_round_uuid=sample_funding_round_uuid,
        investment_date=datetime(2023, 6, 15),
        org_uuid=sample_org_uuid,
        general_funding_stage="Series A",
        stage="Series A-1",
        investors=["Sequoia Capital", "Andreessen Horowitz"],
        lead_investors=["Sequoia Capital"],
        fundraise_amount_usd=10000000,
        valuation_usd=50000000,
    )


@pytest.fixture
def sample_funding_round(sample_funding_round_uuid, sample_org_uuid):
    """Create a sample FundingRound Pydantic model instance."""
    return FundingRound(
        funding_round_uuid=sample_funding_round_uuid,
        investment_date=datetime(2023, 6, 15),
        org_uuid=sample_org_uuid,
        general_funding_stage="Series A",
        stage="Series A-1",
        investors=["Sequoia Capital", "Andreessen Horowitz"],
        lead_investors=["Sequoia Capital"],
        fundraise_amount_usd=10000000,
        valuation_usd=50000000,
    )


class TestFundingRoundModelInitialization:
    """Test FundingRoundModel initialization."""

    def test_init_with_custom_client(self, mock_postgres_client):
        """Test initialization with custom PostgresClient."""
        model = FundingRoundModel(client=mock_postgres_client)
        assert model._client == mock_postgres_client
        assert not model._use_default_client

    def test_init_without_client(self):
        """Test initialization without client (uses default)."""
        model = FundingRoundModel()
        assert model._client is None
        assert model._use_default_client

    @pytest.mark.asyncio
    async def test_initialize_with_default_client(self):
        """Test initialize() when using default client."""
        mock_client = MagicMock()
        with patch(
            "src.models.funding_rounds.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            model = FundingRoundModel()
            await model.initialize()
            assert model._client == mock_client
            mock_get_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_client(self, mock_postgres_client):
        """Test initialize() when using custom client (should not call get_postgres_client)."""
        with patch(
            "src.models.funding_rounds.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            model = FundingRoundModel(client=mock_postgres_client)
            await model.initialize()
            assert model._client == mock_postgres_client
            mock_get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize() can be called multiple times safely."""
        mock_client = MagicMock()
        with patch(
            "src.models.funding_rounds.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            model = FundingRoundModel()
            await model.initialize()
            await model.initialize()  # Second call
            # Should only call get_postgres_client once
            assert mock_get_client.call_count == 1


class TestFundingRoundModelGet:
    """Test FundingRoundModel.get() method."""

    @pytest.mark.asyncio
    async def test_get_all_funding_rounds(self, mock_postgres_client, sample_funding_round_record):
        """Test getting all funding rounds without filters."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert len(results) == 1
        assert isinstance(results[0], FundingRound)
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        assert "SELECT * FROM fundingrounds" in call_args[0][0]
        assert "WHERE" not in call_args[0][0]
        assert "ORDER BY investment_date DESC NULLS LAST" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_by_funding_round_uuid(
        self,
        mock_postgres_client,
        sample_funding_round_uuid,
        sample_funding_round_record,
    ):
        """Test getting funding round by UUID."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(funding_round_uuid=sample_funding_round_uuid)

        assert len(results) == 1
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "funding_round_uuid = $1" in query
        assert str(sample_funding_round_uuid) in params

    @pytest.mark.asyncio
    async def test_get_by_org_uuid(
        self, mock_postgres_client, sample_org_uuid, sample_funding_round_record
    ):
        """Test getting funding rounds by organization UUID."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_uuid=sample_org_uuid)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_uuid = $1" in query
        assert str(sample_org_uuid) in params

    @pytest.mark.asyncio
    async def test_get_by_org_uuids_empty_list(self, mock_postgres_client):
        """Test getting funding rounds with empty org_uuids list returns empty list."""
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_uuids=[])

        assert results == []
        # Should return early without querying database
        mock_postgres_client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_org_uuids_single(
        self, mock_postgres_client, sample_org_uuid, sample_funding_round_record
    ):
        """Test getting funding rounds by single org_uuid in org_uuids list uses equality."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_uuids=[sample_org_uuid])

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # Single UUID should use equality for better index usage
        assert "org_uuid = $1" in query
        assert str(sample_org_uuid) in params
        # Should NOT use ANY clause for single UUID
        assert "ANY" not in query

    @pytest.mark.asyncio
    async def test_get_by_org_uuids_multiple(
        self,
        mock_postgres_client,
        sample_org_uuid,
        sample_funding_round_record,
    ):
        """Test getting funding rounds by multiple org_uuids uses ANY clause."""
        org_uuid_2 = uuid4()
        org_uuid_3 = uuid4()
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_uuids=[sample_org_uuid, org_uuid_2, org_uuid_3])

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # Multiple UUIDs should use ANY clause with array
        assert "org_uuid = ANY($1::uuid[])" in query
        # Params should contain list of string UUIDs
        assert len(params) == 1
        assert isinstance(params[0], list)
        assert str(sample_org_uuid) in params[0]
        assert str(org_uuid_2) in params[0]
        assert str(org_uuid_3) in params[0]

    @pytest.mark.asyncio
    async def test_get_by_investment_date(self, mock_postgres_client, sample_funding_round_record):
        """Test getting funding rounds by exact investment date."""
        investment_date = datetime(2023, 6, 15)
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(investment_date=investment_date)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "investment_date = $1" in query
        assert investment_date in params

    @pytest.mark.asyncio
    async def test_get_by_investment_date_range(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds by investment date range."""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            investment_date_from=date_from,
            investment_date_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "investment_date >= $1" in query
        assert "investment_date <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_general_funding_stage(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds by general funding stage."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(general_funding_stage="Series A")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "general_funding_stage = $1" in query
        assert "Series A" in params

    @pytest.mark.asyncio
    async def test_get_by_stage(self, mock_postgres_client, sample_funding_round_record):
        """Test getting funding rounds by specific stage."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(stage="Series A-1")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "stage = $1" in query
        assert "Series A-1" in params

    @pytest.mark.asyncio
    async def test_get_by_investors_contains(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds by investors array contains."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(investors_contains="Sequoia Capital")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(investors)" in query
        assert "Sequoia Capital" in params

    @pytest.mark.asyncio
    async def test_get_by_lead_investors_contains(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds by lead_investors array contains."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(lead_investors_contains="Sequoia Capital")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(lead_investors)" in query
        assert "Sequoia Capital" in params

    @pytest.mark.asyncio
    async def test_get_by_fundraise_amount_exact(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds by exact fundraise amount."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(fundraise_amount_usd=10000000)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "fundraise_amount_usd = $1" in query
        assert 10000000 in params

    @pytest.mark.asyncio
    async def test_get_by_fundraise_amount_range(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds by fundraise amount range."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            fundraise_amount_usd_min=1000000,
            fundraise_amount_usd_max=50000000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "fundraise_amount_usd >= $1" in query
        assert "fundraise_amount_usd <= $2" in query
        assert 1000000 in params
        assert 50000000 in params

    @pytest.mark.asyncio
    async def test_get_by_valuation_exact(self, mock_postgres_client, sample_funding_round_record):
        """Test getting funding rounds by exact valuation."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(valuation_usd=50000000)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_usd = $1" in query
        assert 50000000 in params

    @pytest.mark.asyncio
    async def test_get_by_valuation_range(self, mock_postgres_client, sample_funding_round_record):
        """Test getting funding rounds by valuation range."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            valuation_usd_min=10000000,
            valuation_usd_max=100000000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_usd >= $1" in query
        assert "valuation_usd <= $2" in query
        assert 10000000 in params
        assert 100000000 in params

    @pytest.mark.asyncio
    async def test_get_with_multiple_filters(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds with multiple filters."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            general_funding_stage="Series A",
            fundraise_amount_usd_min=1000000,
            investors_contains="Sequoia Capital",
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "general_funding_stage = $1" in query
        assert "$2 = ANY(investors)" in query
        assert "fundraise_amount_usd >= $3" in query
        assert "Series A" in params
        assert "Sequoia Capital" in params
        assert 1000000 in params

    @pytest.mark.asyncio
    async def test_get_with_limit(self, mock_postgres_client, sample_funding_round_record):
        """Test getting funding rounds with limit."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(limit=10)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "LIMIT $1" in query
        assert 10 in params

    @pytest.mark.asyncio
    async def test_get_with_offset(self, mock_postgres_client, sample_funding_round_record):
        """Test getting funding rounds with offset."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
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
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test getting funding rounds with both limit and offset."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
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
        """Test getting funding rounds when no results are found."""
        mock_postgres_client.query.return_value = []
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert results == []
        mock_postgres_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_multiple_results(
        self, mock_postgres_client, sample_funding_round_record, sample_org_uuid
    ):
        """Test getting multiple funding rounds."""

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
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 7, 20),
            org_uuid=sample_org_uuid,
            general_funding_stage="Series B",
            stage="Series B-1",
            investors=["Accel Partners"],
            lead_investors=["Accel Partners"],
            fundraise_amount_usd=20000000,
            valuation_usd=100000000,
        )

        mock_postgres_client.query.return_value = [sample_funding_round_record, record2]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert len(results) == 2
        assert all(isinstance(r, FundingRound) for r in results)

    @pytest.mark.asyncio
    async def test_get_not_initialized(self):
        """Test get() raises RuntimeError when client is not initialized."""
        model = FundingRoundModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.get()
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_database_error(self, mock_postgres_client):
        """Test get() raises exception when database query fails."""
        mock_postgres_client.query.side_effect = Exception("Database connection failed")
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.get()
        assert "Database connection failed" in str(exc_info.value)


class TestFundingRoundModelGetByUuid:
    """Test FundingRoundModel.get_by_uuid() method."""

    @pytest.mark.asyncio
    async def test_get_by_uuid_success(
        self,
        mock_postgres_client,
        sample_funding_round_uuid,
        sample_funding_round_record,
    ):
        """Test successfully getting funding round by UUID."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        result = await model.get_by_uuid(sample_funding_round_uuid)

        assert result is not None
        assert isinstance(result, FundingRound)
        assert result.funding_round_uuid == sample_funding_round_uuid
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "funding_round_uuid = $1" in query
        assert "LIMIT $2" in query
        assert str(sample_funding_round_uuid) in params
        assert 1 in params

    @pytest.mark.asyncio
    async def test_get_by_uuid_not_found(self, mock_postgres_client, sample_funding_round_uuid):
        """Test get_by_uuid() when funding round is not found."""
        mock_postgres_client.query.return_value = []
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        result = await model.get_by_uuid(sample_funding_round_uuid)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_uuid_not_initialized(self, sample_funding_round_uuid):
        """Test get_by_uuid() raises RuntimeError when client is not initialized."""
        model = FundingRoundModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.get_by_uuid(sample_funding_round_uuid)
        assert "not initialized" in str(exc_info.value).lower()


class TestFundingRoundModelCount:
    """Test FundingRoundModel.count() method."""

    @pytest.mark.asyncio
    async def test_count_all(self, mock_postgres_client):
        """Test counting all funding rounds."""
        mock_postgres_client.query_value.return_value = 42
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count()

        assert count == 42
        mock_postgres_client.query_value.assert_called_once()
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        assert "SELECT COUNT(*) FROM fundingrounds" in query
        assert "WHERE" not in query

    @pytest.mark.asyncio
    async def test_count_by_funding_round_uuid(
        self, mock_postgres_client, sample_funding_round_uuid
    ):
        """Test counting funding rounds by UUID."""
        mock_postgres_client.query_value.return_value = 1
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(funding_round_uuid=sample_funding_round_uuid)

        assert count == 1
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "funding_round_uuid = $1" in query
        assert str(sample_funding_round_uuid) in params

    @pytest.mark.asyncio
    async def test_count_by_org_uuid(self, mock_postgres_client, sample_org_uuid):
        """Test counting funding rounds by organization UUID."""
        mock_postgres_client.query_value.return_value = 3
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(org_uuid=sample_org_uuid)

        assert count == 3
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_uuid = $1" in query
        assert str(sample_org_uuid) in params

    @pytest.mark.asyncio
    async def test_count_by_general_funding_stage(self, mock_postgres_client):
        """Test counting funding rounds by general funding stage."""
        mock_postgres_client.query_value.return_value = 10
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(general_funding_stage="Series A")

        assert count == 10
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "general_funding_stage = $1" in query
        assert "Series A" in params

    @pytest.mark.asyncio
    async def test_count_by_stage(self, mock_postgres_client):
        """Test counting funding rounds by specific stage."""
        mock_postgres_client.query_value.return_value = 5
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(stage="Series A-1")

        assert count == 5
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "stage = $1" in query
        assert "Series A-1" in params

    @pytest.mark.asyncio
    async def test_count_by_investors_contains(self, mock_postgres_client):
        """Test counting funding rounds by investors array contains."""
        mock_postgres_client.query_value.return_value = 8
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(investors_contains="Sequoia Capital")

        assert count == 8
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(investors)" in query
        assert "Sequoia Capital" in params

    @pytest.mark.asyncio
    async def test_count_by_lead_investors_contains(self, mock_postgres_client):
        """Test counting funding rounds by lead_investors array contains."""
        mock_postgres_client.query_value.return_value = 6
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(lead_investors_contains="Sequoia Capital")

        assert count == 6
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(lead_investors)" in query
        assert "Sequoia Capital" in params

    @pytest.mark.asyncio
    async def test_count_by_date_range(self, mock_postgres_client):
        """Test counting funding rounds by date range."""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        mock_postgres_client.query_value.return_value = 15
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            investment_date_from=date_from,
            investment_date_to=date_to,
        )

        assert count == 15
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "investment_date >= $1" in query
        assert "investment_date <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_count_by_fundraise_amount_range(self, mock_postgres_client):
        """Test counting funding rounds by fundraise amount range."""
        mock_postgres_client.query_value.return_value = 12
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            fundraise_amount_usd_min=1000000,
            fundraise_amount_usd_max=50000000,
        )

        assert count == 12
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "fundraise_amount_usd >= $1" in query
        assert "fundraise_amount_usd <= $2" in query
        assert 1000000 in params
        assert 50000000 in params

    @pytest.mark.asyncio
    async def test_count_by_valuation_range(self, mock_postgres_client):
        """Test counting funding rounds by valuation range."""
        mock_postgres_client.query_value.return_value = 9
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            valuation_usd_min=10000000,
            valuation_usd_max=100000000,
        )

        assert count == 9
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_usd >= $1" in query
        assert "valuation_usd <= $2" in query
        assert 10000000 in params
        assert 100000000 in params

    @pytest.mark.asyncio
    async def test_count_with_multiple_filters(self, mock_postgres_client):
        """Test counting funding rounds with multiple filters."""
        mock_postgres_client.query_value.return_value = 7
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            general_funding_stage="Series A",
            fundraise_amount_usd_min=1000000,
            investors_contains="Sequoia Capital",
        )

        assert count == 7
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "general_funding_stage = $1" in query
        assert "$2 = ANY(investors)" in query
        assert "fundraise_amount_usd >= $3" in query
        assert "Series A" in params
        assert "Sequoia Capital" in params
        assert 1000000 in params

    @pytest.mark.asyncio
    async def test_count_zero(self, mock_postgres_client):
        """Test counting when no funding rounds match."""
        mock_postgres_client.query_value.return_value = 0
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(general_funding_stage="nonexistent")

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_none_result(self, mock_postgres_client):
        """Test counting when query returns None (should return 0)."""
        mock_postgres_client.query_value.return_value = None
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_not_initialized(self):
        """Test count() raises RuntimeError when client is not initialized."""
        model = FundingRoundModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.count()
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_count_database_error(self, mock_postgres_client):
        """Test count() raises exception when database query fails."""
        mock_postgres_client.query_value.side_effect = Exception("Database connection failed")
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.count()
        assert "Database connection failed" in str(exc_info.value)


class TestFundingRoundModelEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_get_with_none_values_ignored(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test that None filter values are ignored in query."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            funding_round_uuid=None,
            general_funding_stage=None,
            fundraise_amount_usd=None,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        # None values should not appear in WHERE clause
        assert "funding_round_uuid = $" not in query
        assert "general_funding_stage = $" not in query
        assert "fundraise_amount_usd = $" not in query

    @pytest.mark.asyncio
    async def test_get_with_partial_none_values(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test that only non-None values are used in query."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            general_funding_stage="Series A",
            fundraise_amount_usd=None,  # Should be ignored
            investors_contains="Sequoia Capital",
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "general_funding_stage = $1" in query
        assert "$2 = ANY(investors)" in query
        assert "fundraise_amount_usd = $" not in query
        assert "Series A" in params
        assert "Sequoia Capital" in params
        assert None not in params

    @pytest.mark.asyncio
    async def test_get_query_ordering(self, mock_postgres_client, sample_funding_round_record):
        """Test that query includes proper ORDER BY clause."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        await model.get()

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        assert "ORDER BY investment_date DESC NULLS LAST" in query

    @pytest.mark.asyncio
    async def test_get_parameter_ordering(self, mock_postgres_client, sample_funding_round_record):
        """Test that query parameters are correctly ordered."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(
            general_funding_stage="Series A",
            fundraise_amount_usd_min=1000000,
            investors_contains="Sequoia Capital",
        )

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        # Check that parameters are numbered sequentially
        assert "$1" in query
        assert "$2" in query
        assert "$3" in query

    @pytest.mark.asyncio
    async def test_get_with_limit_parameter_index(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test that limit parameter uses correct index after filters."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(general_funding_stage="Series A", limit=10)

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # general_funding_stage = $1, limit = $2
        assert "LIMIT $2" in query
        assert 10 in params
        assert len(params) == 2  # general_funding_stage and limit

    @pytest.mark.asyncio
    async def test_get_with_offset_parameter_index(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test that offset parameter uses correct index after filters and limit."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(general_funding_stage="Series A", limit=10, offset=20)

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # general_funding_stage = $1, limit = $2, offset = $3
        assert "LIMIT $2" in query
        assert "OFFSET $3" in query
        assert len(params) == 3

    @pytest.mark.asyncio
    async def test_get_array_fields_handled_correctly(
        self, mock_postgres_client, sample_funding_round_record
    ):
        """Test that array fields (investors, lead_investors) are handled correctly."""
        mock_postgres_client.query.return_value = [sample_funding_round_record]
        model = FundingRoundModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(investors_contains="Sequoia Capital")

        assert len(results) == 1
        # Verify the result has the array fields properly set
        assert results[0].investors == ["Sequoia Capital", "Andreessen Horowitz"]
        assert results[0].lead_investors == ["Sequoia Capital"]
