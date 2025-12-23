"""Unit tests for the Pinecone organization model module.

This module tests the PineconeOrganizationModel class and its various methods,
including query building, filtering, embedding generation, and error handling.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.models.pinecone_organizations import PineconeOrganization, PineconeOrganizationModel


@pytest.fixture
def mock_pinecone_client():
    """Create a mock PineconeClient instance."""
    client = MagicMock()
    client.query = AsyncMock()
    return client


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAIClient instance."""
    client = MagicMock()
    client.create_embedding = AsyncMock()
    return client


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim embedding


@pytest.fixture
def sample_pinecone_match(sample_org_uuid):
    """Create a sample Pinecone match object."""
    match = MagicMock()
    match.id = str(sample_org_uuid)
    match.score = 0.95
    match.metadata = {
        "org_uuid": str(sample_org_uuid),
        "name": "Test Company",
        "categories": ["AI", "Healthcare"],
        "org_status": "operating",
        "total_funding_usd": 10000000.0,
        "founding_date": "2020-01-15T00:00:00",
        "last_fundraise_date": "2023-06-15T00:00:00",
        "employee_count": "51-100",
        "org_type": "company",
        "stage": "series_a",
        "valuation_usd": 50000000.0,
        "investors": [str(uuid4()), str(uuid4())],
        "general_funding_stage": "late_stage_venture",
        "num_acquisitions": 3,
        "revenue_range": "10M-50M",
    }
    return match


@pytest.fixture
def sample_pinecone_response(sample_pinecone_match):
    """Create a sample Pinecone query response."""
    response = MagicMock()
    response.matches = [sample_pinecone_match]
    return response


class TestPineconeOrganizationModelInitialization:
    """Test PineconeOrganizationModel initialization."""

    def test_init_with_custom_clients(self, mock_pinecone_client, mock_openai_client):
        """Test initialization with custom clients."""
        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        assert model._pinecone_client == mock_pinecone_client
        assert model._openai_client == mock_openai_client
        assert not model._use_default_pinecone
        assert not model._use_default_openai

    def test_init_without_clients(self):
        """Test initialization without clients (uses defaults)."""
        model = PineconeOrganizationModel()
        assert model._pinecone_client is None
        assert model._openai_client is None
        assert model._use_default_pinecone
        assert model._use_default_openai

    def test_init_with_only_pinecone_client(self, mock_pinecone_client):
        """Test initialization with only Pinecone client."""
        model = PineconeOrganizationModel(pinecone_client=mock_pinecone_client)
        assert model._pinecone_client == mock_pinecone_client
        assert model._openai_client is None
        assert not model._use_default_pinecone
        assert model._use_default_openai

    def test_init_with_only_openai_client(self, mock_openai_client):
        """Test initialization with only OpenAI client."""
        model = PineconeOrganizationModel(openai_client=mock_openai_client)
        assert model._pinecone_client is None
        assert model._openai_client == mock_openai_client
        assert model._use_default_pinecone
        assert not model._use_default_openai

    @pytest.mark.asyncio
    async def test_initialize_with_default_clients(self):
        """Test initialize() when using default clients."""
        mock_pinecone = MagicMock()
        mock_openai = MagicMock()
        with (
            patch(
                "src.models.pinecone_organizations.get_pinecone_client",
                new_callable=AsyncMock,
            ) as mock_get_pinecone,
            patch(
                "src.models.pinecone_organizations.get_openai_client",
                new_callable=AsyncMock,
            ) as mock_get_openai,
        ):
            mock_get_pinecone.return_value = mock_pinecone
            mock_get_openai.return_value = mock_openai
            model = PineconeOrganizationModel()
            await model.initialize()
            assert model._pinecone_client == mock_pinecone
            assert model._openai_client == mock_openai
            mock_get_pinecone.assert_called_once()
            mock_get_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_clients(self, mock_pinecone_client, mock_openai_client):
        """Test initialize() when using custom clients (should not call getters)."""
        with (
            patch(
                "src.models.pinecone_organizations.get_pinecone_client",
                new_callable=AsyncMock,
            ) as mock_get_pinecone,
            patch(
                "src.models.pinecone_organizations.get_openai_client",
                new_callable=AsyncMock,
            ) as mock_get_openai,
        ):
            model = PineconeOrganizationModel(
                pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
            )
            await model.initialize()
            assert model._pinecone_client == mock_pinecone_client
            assert model._openai_client == mock_openai_client
            mock_get_pinecone.assert_not_called()
            mock_get_openai.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize() can be called multiple times safely."""
        mock_pinecone = MagicMock()
        mock_openai = MagicMock()
        with (
            patch(
                "src.models.pinecone_organizations.get_pinecone_client",
                new_callable=AsyncMock,
            ) as mock_get_pinecone,
            patch(
                "src.models.pinecone_organizations.get_openai_client",
                new_callable=AsyncMock,
            ) as mock_get_openai,
        ):
            mock_get_pinecone.return_value = mock_pinecone
            mock_get_openai.return_value = mock_openai
            model = PineconeOrganizationModel()
            await model.initialize()
            await model.initialize()  # Second call
            # Should only call getters once
            assert mock_get_pinecone.call_count == 1
            assert mock_get_openai.call_count == 1


class TestPineconeOrganizationModelQuery:
    """Test PineconeOrganizationModel.query() method."""

    @pytest.mark.asyncio
    async def test_query_basic(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test basic query with just text."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        results = await model.query(text="AI companies", top_k=10)

        assert len(results) == 1
        assert isinstance(results[0], PineconeOrganization)
        mock_openai_client.create_embedding.assert_called_once_with(
            text="AI companies", model="text-embedding-3-large"
        )
        mock_pinecone_client.query.assert_called_once()
        call_kwargs = mock_pinecone_client.query.call_args[1]
        assert call_kwargs["query_vector"] == sample_embedding
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["include_metadata"] is True
        assert call_kwargs["metadata_filter"] is None

    @pytest.mark.asyncio
    async def test_query_with_org_uuid(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_org_uuid,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with org_uuid filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", org_uuid=sample_org_uuid)

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["org_uuid"] == {"$eq": str(sample_org_uuid)}

    @pytest.mark.asyncio
    async def test_query_with_name(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with name filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", name="Test Company")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["name"] == {"$eq": "Test Company"}

    @pytest.mark.asyncio
    async def test_query_with_categories_contains(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with categories_contains filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", categories_contains="AI")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["categories"] == {"$in": ["AI"]}

    @pytest.mark.asyncio
    async def test_query_with_org_status(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with org_status filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", org_status="operating")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["org_status"] == {"$eq": "operating"}

    @pytest.mark.asyncio
    async def test_query_with_funding_range(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with total_funding_usd range filters."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(
            text="test", total_funding_usd_min=1000000, total_funding_usd_max=50000000
        )

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["total_funding_usd"]["$gte"] == 1000000
        assert metadata_filter["total_funding_usd"]["$lte"] == 50000000

    @pytest.mark.asyncio
    async def test_query_with_funding_min_only(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with only total_funding_usd_min."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", total_funding_usd_min=1000000)

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["total_funding_usd"]["$gte"] == 1000000
        assert "$lte" not in metadata_filter["total_funding_usd"]

    @pytest.mark.asyncio
    async def test_query_with_funding_max_only(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with only total_funding_usd_max."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", total_funding_usd_max=50000000)

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["total_funding_usd"]["$lte"] == 50000000
        assert "$gte" not in metadata_filter["total_funding_usd"]

    @pytest.mark.asyncio
    async def test_query_with_founding_date_range(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with founding_date range filters."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        date_from = datetime(2020, 1, 1)
        date_to = datetime(2023, 12, 31)
        await model.query(text="test", founding_date_from=date_from, founding_date_to=date_to)

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["founding_date"]["$gte"] == date_from.isoformat()
        assert metadata_filter["founding_date"]["$lte"] == date_to.isoformat()

    @pytest.mark.asyncio
    async def test_query_with_last_fundraise_date_range(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with last_fundraise_date range filters."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        date_from = datetime(2022, 1, 1)
        date_to = datetime(2023, 12, 31)
        await model.query(
            text="test",
            last_fundraise_date_from=date_from,
            last_fundraise_date_to=date_to,
        )

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["last_fundraise_date"]["$gte"] == date_from.isoformat()
        assert metadata_filter["last_fundraise_date"]["$lte"] == date_to.isoformat()

    @pytest.mark.asyncio
    async def test_query_with_employee_count(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with employee_count filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", employee_count="51-100")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["employee_count"] == {"$eq": "51-100"}

    @pytest.mark.asyncio
    async def test_query_with_org_type(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with org_type filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", org_type="company")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["org_type"] == {"$eq": "company"}

    @pytest.mark.asyncio
    async def test_query_with_stage(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with stage filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", stage="series_a")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["stage"] == {"$eq": "series_a"}

    @pytest.mark.asyncio
    async def test_query_with_valuation_range(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with valuation_usd range filters."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", valuation_usd_min=10000000, valuation_usd_max=100000000)

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["valuation_usd"]["$gte"] == 10000000
        assert metadata_filter["valuation_usd"]["$lte"] == 100000000

    @pytest.mark.asyncio
    async def test_query_with_investors_contains(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_org_uuid,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with investors_contains filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", investors_contains=sample_org_uuid)

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["investors"] == {"$in": [str(sample_org_uuid)]}

    @pytest.mark.asyncio
    async def test_query_with_general_funding_stage(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with general_funding_stage filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", general_funding_stage="late_stage_venture")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["general_funding_stage"] == {"$eq": "late_stage_venture"}

    @pytest.mark.asyncio
    async def test_query_with_acquisitions_range(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with num_acquisitions range filters."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", num_acquisitions_min=1, num_acquisitions_max=10)

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["num_acquisitions"]["$gte"] == 1
        assert metadata_filter["num_acquisitions"]["$lte"] == 10

    @pytest.mark.asyncio
    async def test_query_with_revenue_range(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with revenue_range filter."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", revenue_range="10M-50M")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["revenue_range"] == {"$eq": "10M-50M"}

    @pytest.mark.asyncio
    async def test_query_with_multiple_filters(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_org_uuid,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with multiple filters."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(
            text="healthcare startups",
            org_status="operating",
            total_funding_usd_min=1000000,
            general_funding_stage="late_stage_venture",
            top_k=5,
        )

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        assert metadata_filter["org_status"] == {"$eq": "operating"}
        assert metadata_filter["total_funding_usd"]["$gte"] == 1000000
        assert metadata_filter["general_funding_stage"] == {"$eq": "late_stage_venture"}
        assert call_kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_query_with_custom_index_name(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test query with custom index_name."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test", index_name="custom_index")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        assert call_kwargs["index_name"] == "custom_index"

    @pytest.mark.asyncio
    async def test_query_result_conversion(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_org_uuid,
        sample_embedding,
    ):
        """Test that query results are properly converted to PineconeOrganization objects."""
        # Create a match with string UUIDs and dates
        investor_uuid_1 = uuid4()
        investor_uuid_2 = uuid4()
        match = MagicMock()
        match.id = str(sample_org_uuid)
        match.score = 0.95
        match.metadata = {
            "org_uuid": str(sample_org_uuid),
            "name": "Test Company",
            "investors": [str(investor_uuid_1), str(investor_uuid_2)],
            "founding_date": "2020-01-15T00:00:00",
            "last_fundraise_date": "2023-06-15T00:00:00",
        }

        response = MagicMock()
        response.matches = [match]

        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        results = await model.query(text="test")

        assert len(results) == 1
        org = results[0]
        assert isinstance(org, PineconeOrganization)
        assert org.org_uuid == sample_org_uuid
        assert org.name == "Test Company"
        assert org.score == 0.95
        assert len(org.investors) == 2
        assert org.investors[0] == investor_uuid_1
        assert org.investors[1] == investor_uuid_2
        assert isinstance(org.founding_date, datetime)
        assert isinstance(org.last_fundraise_date, datetime)

    @pytest.mark.asyncio
    async def test_query_result_with_none_metadata(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_org_uuid,
        sample_embedding,
    ):
        """Test query result handling when match has None metadata."""
        match = MagicMock()
        match.id = str(sample_org_uuid)
        match.score = 0.85
        match.metadata = None

        response = MagicMock()
        response.matches = [match]

        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        # When metadata is None, it becomes empty dict, but org_uuid is required
        # The code will raise a ValidationError when trying to create PineconeOrganization
        with pytest.raises(Exception) as exc_info:
            await model.query(text="test")
        # Should raise ValidationError from Pydantic
        assert "validation error" in str(exc_info.value).lower() or "org_uuid" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_query_result_with_invalid_uuid(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
    ):
        """Test query result handling with invalid UUID in metadata."""
        match = MagicMock()
        match.id = "test-id"
        match.score = 0.85
        match.metadata = {"org_uuid": "invalid-uuid"}

        response = MagicMock()
        response.matches = [match]

        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        # Invalid UUID string stays in metadata, Pydantic will reject it
        with pytest.raises(Exception) as exc_info:
            await model.query(text="test")
        # Should raise ValidationError from Pydantic
        assert (
            "validation error" in str(exc_info.value).lower()
            or "uuid" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_query_result_with_invalid_date(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_org_uuid,
        sample_embedding,
    ):
        """Test query result handling with invalid date format."""
        match = MagicMock()
        match.id = str(sample_org_uuid)
        match.score = 0.85
        match.metadata = {
            "org_uuid": str(sample_org_uuid),
            "founding_date": "invalid-date-format",
        }

        response = MagicMock()
        response.matches = [match]

        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        # Invalid date string stays in metadata, Pydantic will reject it
        with pytest.raises(Exception) as exc_info:
            await model.query(text="test")
        # Should raise ValidationError from Pydantic
        assert (
            "validation error" in str(exc_info.value).lower()
            or "datetime" in str(exc_info.value).lower()
            or "date" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_query_empty_results(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
    ):
        """Test query when no results are found."""
        response = MagicMock()
        response.matches = []

        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        results = await model.query(text="test")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_multiple_results(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
    ):
        """Test query with multiple results."""
        match1 = MagicMock()
        match1.id = "id1"
        match1.score = 0.95
        match1.metadata = {"org_uuid": str(uuid4()), "name": "Company 1"}

        match2 = MagicMock()
        match2.id = "id2"
        match2.score = 0.90
        match2.metadata = {"org_uuid": str(uuid4()), "name": "Company 2"}

        response = MagicMock()
        response.matches = [match1, match2]

        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        results = await model.query(text="test")

        assert len(results) == 2
        assert all(isinstance(org, PineconeOrganization) for org in results)
        assert results[0].score == 0.95
        assert results[1].score == 0.90

    @pytest.mark.asyncio
    async def test_query_not_initialized_pinecone(self, mock_openai_client):
        """Test query raises RuntimeError when PineconeClient is not initialized."""
        model = PineconeOrganizationModel(openai_client=mock_openai_client)
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.query(text="test")
        assert "PineconeClient not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_not_initialized_openai(self, mock_pinecone_client):
        """Test query raises RuntimeError when OpenAIClient is not initialized."""
        model = PineconeOrganizationModel(pinecone_client=mock_pinecone_client)
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.query(text="test")
        assert "OpenAIClient not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_embedding_failure(
        self,
        mock_pinecone_client,
        mock_openai_client,
    ):
        """Test query raises exception when embedding generation fails."""
        mock_openai_client.create_embedding.side_effect = Exception("Embedding generation failed")

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.query(text="test")
        assert "Embedding generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_pinecone_failure(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
    ):
        """Test query raises exception when Pinecone query fails."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.side_effect = Exception("Pinecone query failed")

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.query(text="test")
        assert "Pinecone query failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_with_none_filters(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test that None filter values are ignored."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(
            text="test",
            org_uuid=None,
            name=None,
            org_status=None,
            total_funding_usd_min=None,
            total_funding_usd_max=None,
        )

        call_kwargs = mock_pinecone_client.query.call_args[1]
        metadata_filter = call_kwargs["metadata_filter"]
        # When all filters are None, metadata_filter should be None
        assert metadata_filter is None

    @pytest.mark.asyncio
    async def test_query_empty_metadata_filter(
        self,
        mock_pinecone_client,
        mock_openai_client,
        sample_embedding,
        sample_pinecone_response,
    ):
        """Test that empty metadata filter is passed as None to Pinecone."""
        mock_openai_client.create_embedding.return_value = sample_embedding
        mock_pinecone_client.query.return_value = sample_pinecone_response

        model = PineconeOrganizationModel(
            pinecone_client=mock_pinecone_client, openai_client=mock_openai_client
        )
        await model.initialize()

        await model.query(text="test")

        call_kwargs = mock_pinecone_client.query.call_args[1]
        # When no filters are provided, metadata_filter should be None
        assert call_kwargs["metadata_filter"] is None


class TestPineconeOrganizationModel:
    """Test PineconeOrganization Pydantic model."""

    def test_pinecone_organization_creation(self, sample_org_uuid):
        """Test creating a PineconeOrganization instance."""
        org = PineconeOrganization(
            org_uuid=sample_org_uuid,
            name="Test Company",
            categories=["AI", "Healthcare"],
            org_status="operating",
            total_funding_usd=10000000.0,
            score=0.95,
        )

        assert org.org_uuid == sample_org_uuid
        assert org.name == "Test Company"
        assert org.categories == ["AI", "Healthcare"]
        assert org.org_status == "operating"
        assert org.total_funding_usd == 10000000.0
        assert org.score == 0.95

    def test_pinecone_organization_optional_fields(self, sample_org_uuid):
        """Test PineconeOrganization with only required fields."""
        org = PineconeOrganization(org_uuid=sample_org_uuid)

        assert org.org_uuid == sample_org_uuid
        assert org.name is None
        assert org.categories is None
        assert org.score is None
