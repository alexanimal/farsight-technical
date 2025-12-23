"""Unit tests for the organizations model module.

This module tests the OrganizationModel class and its various methods,
including query building, filtering, and error handling.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.models.organizations import Organization, OrganizationModel


@pytest.fixture
def mock_postgres_client():
    """Create a mock PostgresClient instance."""
    client = MagicMock()
    client.query = AsyncMock()
    client.query_value = AsyncMock()
    return client


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_organization_record(sample_org_uuid):
    """Create a sample organization record (as would be returned from database)."""

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
        org_uuid=sample_org_uuid,
        cb_url="https://www.crunchbase.com/organization/example",
        categories=["Artificial Intelligence", "Machine Learning"],
        category_groups=["Technology"],
        closed_on=None,
        closed_on_precision=None,
        company_profit_type="for_profit",
        created_at=datetime(2020, 1, 15),
        raw_description="A leading AI company",
        web_scrape=None,
        rewritten_description="A leading artificial intelligence company",
        total_funding_native=50000000,
        total_funding_currency="USD",
        total_funding_usd=50000000,
        exited_on=None,
        exited_on_precision=None,
        founding_date=datetime(2018, 5, 10),
        founding_date_precision="day",
        general_funding_stage="Series B",
        logo_url="https://example.com/logo.png",
        ipo_status="private",
        last_fundraise_date=datetime(2023, 6, 15),
        last_funding_total_native=20000000,
        last_funding_total_currency="USD",
        last_funding_total_usd=20000000,
        stage="Series B-1",
        org_type="company",
        city="San Francisco",
        state="California",
        country="United States",
        continent="North America",
        name="Example AI Corp",
        num_acquisitions=2,
        employee_count="51-100",
        num_funding_rounds=3,
        num_investments=0,
        num_portfolio_organizations=0,
        operating_status="operating",
        cb_rank=1500,
        revenue_range="$10M-$50M",
        org_status="active",
        updated_at=datetime(2023, 12, 1),
        valuation_native=200000000,
        valuation_currency="USD",
        valuation_usd=200000000,
        valuation_date=datetime(2023, 6, 15),
        org_domain="example.com",
    )


@pytest.fixture
def sample_organization(sample_org_uuid):
    """Create a sample Organization Pydantic model instance."""
    return Organization(
        org_uuid=sample_org_uuid,
        cb_url="https://www.crunchbase.com/organization/example",
        categories=["Artificial Intelligence", "Machine Learning"],
        category_groups=["Technology"],
        closed_on=None,
        closed_on_precision=None,
        company_profit_type="for_profit",
        created_at=datetime(2020, 1, 15),
        raw_description="A leading AI company",
        web_scrape=None,
        rewritten_description="A leading artificial intelligence company",
        total_funding_native=50000000,
        total_funding_currency="USD",
        total_funding_usd=50000000,
        exited_on=None,
        exited_on_precision=None,
        founding_date=datetime(2018, 5, 10),
        founding_date_precision="day",
        general_funding_stage="Series B",
        logo_url="https://example.com/logo.png",
        ipo_status="private",
        last_fundraise_date=datetime(2023, 6, 15),
        last_funding_total_native=20000000,
        last_funding_total_currency="USD",
        last_funding_total_usd=20000000,
        stage="Series B-1",
        org_type="company",
        city="San Francisco",
        state="California",
        country="United States",
        continent="North America",
        name="Example AI Corp",
        num_acquisitions=2,
        employee_count="51-100",
        num_funding_rounds=3,
        num_investments=0,
        num_portfolio_organizations=0,
        operating_status="operating",
        cb_rank=1500,
        revenue_range="$10M-$50M",
        org_status="active",
        updated_at=datetime(2023, 12, 1),
        valuation_native=200000000,
        valuation_currency="USD",
        valuation_usd=200000000,
        valuation_date=datetime(2023, 6, 15),
        org_domain="example.com",
    )


class TestOrganizationModelInitialization:
    """Test OrganizationModel initialization."""

    def test_init_with_custom_client(self, mock_postgres_client):
        """Test initialization with custom PostgresClient."""
        model = OrganizationModel(client=mock_postgres_client)
        assert model._client == mock_postgres_client
        assert not model._use_default_client

    def test_init_without_client(self):
        """Test initialization without client (uses default)."""
        model = OrganizationModel()
        assert model._client is None
        assert model._use_default_client

    @pytest.mark.asyncio
    async def test_initialize_with_default_client(self):
        """Test initialize() when using default client."""
        mock_client = MagicMock()
        with patch(
            "src.models.organizations.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            model = OrganizationModel()
            await model.initialize()
            assert model._client == mock_client
            mock_get_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_client(self, mock_postgres_client):
        """Test initialize() when using custom client (should not call get_postgres_client)."""
        with patch(
            "src.models.organizations.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            model = OrganizationModel(client=mock_postgres_client)
            await model.initialize()
            assert model._client == mock_postgres_client
            mock_get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize() can be called multiple times safely."""
        mock_client = MagicMock()
        with patch(
            "src.models.organizations.get_postgres_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            model = OrganizationModel()
            await model.initialize()
            await model.initialize()  # Second call
            # Should only call get_postgres_client once
            assert mock_get_client.call_count == 1


class TestOrganizationModelGet:
    """Test OrganizationModel.get() method."""

    @pytest.mark.asyncio
    async def test_get_all_organizations(self, mock_postgres_client, sample_organization_record):
        """Test getting all organizations without filters."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert len(results) == 1
        assert isinstance(results[0], Organization)
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        assert "SELECT * FROM organizations" in call_args[0][0]
        assert "WHERE" not in call_args[0][0]
        assert "ORDER BY updated_at DESC NULLS LAST" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_by_org_uuid(
        self, mock_postgres_client, sample_org_uuid, sample_organization_record
    ):
        """Test getting organization by UUID."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_uuid=sample_org_uuid)

        assert len(results) == 1
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_uuid = $1" in query
        assert str(sample_org_uuid) in params

    @pytest.mark.asyncio
    async def test_get_by_cb_url(self, mock_postgres_client, sample_organization_record):
        """Test getting organization by Crunchbase URL."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(cb_url="https://www.crunchbase.com/organization/example")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "cb_url = $1" in query
        assert "https://www.crunchbase.com/organization/example" in params

    @pytest.mark.asyncio
    async def test_get_by_categories_contains(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by categories array contains."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(categories_contains="Artificial Intelligence")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(categories)" in query
        assert "Artificial Intelligence" in params

    @pytest.mark.asyncio
    async def test_get_by_category_groups_contains(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by category_groups array contains."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(category_groups_contains="Technology")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(category_groups)" in query
        assert "Technology" in params

    @pytest.mark.asyncio
    async def test_get_by_country(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by country."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(country="United States")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "country = $1" in query
        assert "United States" in params

    @pytest.mark.asyncio
    async def test_get_by_continent(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by continent."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(continent="North America")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "continent = $1" in query
        assert "North America" in params

    @pytest.mark.asyncio
    async def test_get_by_city(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by city."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(city="San Francisco")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "city = $1" in query
        assert "San Francisco" in params

    @pytest.mark.asyncio
    async def test_get_by_state(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by state."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(state="California")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "state = $1" in query
        assert "California" in params

    @pytest.mark.asyncio
    async def test_get_by_name_exact(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exact name match."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(name="Example AI Corp")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "name = $1" in query
        assert "Example AI Corp" in params

    @pytest.mark.asyncio
    async def test_get_by_name_ilike(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by case-insensitive name search."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(name_ilike="example")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "name ILIKE $1" in query
        assert "%example%" in params

    @pytest.mark.asyncio
    async def test_get_by_org_domain(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exact domain match."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_domain="example.com")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_domain = $1" in query
        assert "example.com" in params

    @pytest.mark.asyncio
    async def test_get_by_org_domain_ilike(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by case-insensitive domain search."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_domain_ilike="example")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_domain ILIKE $1" in query
        assert "%example%" in params

    @pytest.mark.asyncio
    async def test_get_by_raw_description_ilike(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by case-insensitive raw description search."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(raw_description_ilike="AI")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "raw_description ILIKE $1" in query
        assert "%AI%" in params

    @pytest.mark.asyncio
    async def test_get_by_rewritten_description_ilike(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by case-insensitive rewritten description search."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(rewritten_description_ilike="artificial")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "rewritten_description ILIKE $1" in query
        assert "%artificial%" in params

    @pytest.mark.asyncio
    async def test_get_by_general_funding_stage(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by general funding stage."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(general_funding_stage="Series B")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "general_funding_stage = $1" in query
        assert "Series B" in params

    @pytest.mark.asyncio
    async def test_get_by_stage(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by specific stage."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(stage="Series B-1")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "stage = $1" in query
        assert "Series B-1" in params

    @pytest.mark.asyncio
    async def test_get_by_org_type(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by organization type."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_type="company")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_type = $1" in query
        assert "company" in params

    @pytest.mark.asyncio
    async def test_get_by_operating_status(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by operating status."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(operating_status="operating")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "operating_status = $1" in query
        assert "operating" in params

    @pytest.mark.asyncio
    async def test_get_by_org_status(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by organization status."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(org_status="active")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_status = $1" in query
        assert "active" in params

    @pytest.mark.asyncio
    async def test_get_by_ipo_status(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by IPO status."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(ipo_status="private")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "ipo_status = $1" in query
        assert "private" in params

    @pytest.mark.asyncio
    async def test_get_by_company_profit_type(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by company profit type."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(company_profit_type="for_profit")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "company_profit_type = $1" in query
        assert "for_profit" in params

    @pytest.mark.asyncio
    async def test_get_by_employee_count(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by employee count."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(employee_count="51-100")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "employee_count = $1" in query
        assert "51-100" in params

    @pytest.mark.asyncio
    async def test_get_by_revenue_range(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by revenue range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(revenue_range="$10M-$50M")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "revenue_range = $1" in query
        assert "$10M-$50M" in params

    @pytest.mark.asyncio
    async def test_get_by_founding_date(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exact founding date."""
        founding_date = datetime(2018, 5, 10)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(founding_date=founding_date)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "founding_date = $1" in query
        assert founding_date in params

    @pytest.mark.asyncio
    async def test_get_by_founding_date_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by founding date range."""
        date_from = datetime(2018, 1, 1)
        date_to = datetime(2020, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            founding_date_from=date_from,
            founding_date_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "founding_date >= $1" in query
        assert "founding_date <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_total_funding_usd_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by total funding USD range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            total_funding_usd_min=1000000,
            total_funding_usd_max=100000000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "total_funding_usd >= $1" in query
        assert "total_funding_usd <= $2" in query
        assert 1000000 in params
        assert 100000000 in params

    @pytest.mark.asyncio
    async def test_get_by_valuation_usd_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by valuation USD range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            valuation_usd_min=10000000,
            valuation_usd_max=500000000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_usd >= $1" in query
        assert "valuation_usd <= $2" in query
        assert 10000000 in params
        assert 500000000 in params

    @pytest.mark.asyncio
    async def test_get_by_num_funding_rounds_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by number of funding rounds range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            num_funding_rounds_min=1,
            num_funding_rounds_max=10,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_funding_rounds >= $1" in query
        assert "num_funding_rounds <= $2" in query
        assert 1 in params
        assert 10 in params

    @pytest.mark.asyncio
    async def test_get_by_num_acquisitions_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by number of acquisitions range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            num_acquisitions_min=1,
            num_acquisitions_max=5,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_acquisitions >= $1" in query
        assert "num_acquisitions <= $2" in query
        assert 1 in params
        assert 5 in params

    @pytest.mark.asyncio
    async def test_get_by_cb_rank_range(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by Crunchbase rank range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            cb_rank_min=1000,
            cb_rank_max=2000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "cb_rank >= $1" in query
        assert "cb_rank <= $2" in query
        assert 1000 in params
        assert 2000 in params

    @pytest.mark.asyncio
    async def test_get_by_last_fundraise_date_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by last fundraise date range."""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            last_fundraise_date_from=date_from,
            last_fundraise_date_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "last_fundraise_date >= $1" in query
        assert "last_fundraise_date <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_created_at_range(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by created_at date range."""
        date_from = datetime(2020, 1, 1)
        date_to = datetime(2020, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            created_at_from=date_from,
            created_at_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "created_at >= $1" in query
        assert "created_at <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_updated_at_range(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by updated_at date range."""
        date_from = datetime(2023, 11, 1)
        date_to = datetime(2023, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            updated_at_from=date_from,
            updated_at_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "updated_at >= $1" in query
        assert "updated_at <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_closed_on(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exact closed_on date."""
        closed_date = datetime(2020, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(closed_on=closed_date)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "closed_on = $1" in query
        assert closed_date in params

    @pytest.mark.asyncio
    async def test_get_by_closed_on_range(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by closed_on date range."""
        date_from = datetime(2020, 1, 1)
        date_to = datetime(2020, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            closed_on_from=date_from,
            closed_on_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "closed_on >= $1" in query
        assert "closed_on <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_closed_on_precision(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by closed_on precision."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(closed_on_precision="day")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "closed_on_precision = $1" in query
        assert "day" in params

    @pytest.mark.asyncio
    async def test_get_by_total_funding_native(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact total_funding_native."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(total_funding_native=50000000)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "total_funding_native = $1" in query
        assert 50000000 in params

    @pytest.mark.asyncio
    async def test_get_by_total_funding_native_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by total_funding_native range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            total_funding_native_min=1000000,
            total_funding_native_max=100000000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "total_funding_native >= $1" in query
        assert "total_funding_native <= $2" in query
        assert 1000000 in params
        assert 100000000 in params

    @pytest.mark.asyncio
    async def test_get_by_total_funding_currency(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by total_funding_currency."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(total_funding_currency="USD")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "total_funding_currency = $1" in query
        assert "USD" in params

    @pytest.mark.asyncio
    async def test_get_by_total_funding_usd_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact total_funding_usd."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(total_funding_usd=50000000)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "total_funding_usd = $1" in query
        assert 50000000 in params

    @pytest.mark.asyncio
    async def test_get_by_exited_on(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exact exited_on date."""
        exited_date = datetime(2021, 6, 30)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(exited_on=exited_date)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "exited_on = $1" in query
        assert exited_date in params

    @pytest.mark.asyncio
    async def test_get_by_exited_on_range(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exited_on date range."""
        date_from = datetime(2021, 1, 1)
        date_to = datetime(2021, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            exited_on_from=date_from,
            exited_on_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "exited_on >= $1" in query
        assert "exited_on <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_by_exited_on_precision(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exited_on precision."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(exited_on_precision="month")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "exited_on_precision = $1" in query
        assert "month" in params

    @pytest.mark.asyncio
    async def test_get_by_founding_date_precision(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by founding_date precision."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(founding_date_precision="day")

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "founding_date_precision = $1" in query
        assert "day" in params

    @pytest.mark.asyncio
    async def test_get_by_last_fundraise_date_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact last_fundraise_date."""
        fundraise_date = datetime(2023, 6, 15)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(last_fundraise_date=fundraise_date)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "last_fundraise_date = $1" in query
        assert fundraise_date in params

    @pytest.mark.asyncio
    async def test_get_by_last_funding_total_usd_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact last_funding_total_usd."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(last_funding_total_usd=20000000)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "last_funding_total_usd = $1" in query
        assert 20000000 in params

    @pytest.mark.asyncio
    async def test_get_by_last_funding_total_usd_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by last_funding_total_usd range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            last_funding_total_usd_min=1000000,
            last_funding_total_usd_max=50000000,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "last_funding_total_usd >= $1" in query
        assert "last_funding_total_usd <= $2" in query
        assert 1000000 in params
        assert 50000000 in params

    @pytest.mark.asyncio
    async def test_get_by_num_funding_rounds_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact num_funding_rounds."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(num_funding_rounds=3)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_funding_rounds = $1" in query
        assert 3 in params

    @pytest.mark.asyncio
    async def test_get_by_num_investments_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact num_investments."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(num_investments=0)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_investments = $1" in query
        assert 0 in params

    @pytest.mark.asyncio
    async def test_get_by_num_investments_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by num_investments range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            num_investments_min=0,
            num_investments_max=10,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_investments >= $1" in query
        assert "num_investments <= $2" in query
        assert 0 in params
        assert 10 in params

    @pytest.mark.asyncio
    async def test_get_by_num_portfolio_organizations_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact num_portfolio_organizations."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(num_portfolio_organizations=0)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_portfolio_organizations = $1" in query
        assert 0 in params

    @pytest.mark.asyncio
    async def test_get_by_num_portfolio_organizations_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by num_portfolio_organizations range."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            num_portfolio_organizations_min=0,
            num_portfolio_organizations_max=50,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_portfolio_organizations >= $1" in query
        assert "num_portfolio_organizations <= $2" in query
        assert 0 in params
        assert 50 in params

    @pytest.mark.asyncio
    async def test_get_by_cb_rank_exact(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exact cb_rank."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(cb_rank=1500)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "cb_rank = $1" in query
        assert 1500 in params

    @pytest.mark.asyncio
    async def test_get_by_num_acquisitions_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact num_acquisitions."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(num_acquisitions=2)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "num_acquisitions = $1" in query
        assert 2 in params

    @pytest.mark.asyncio
    async def test_get_by_valuation_usd_exact(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by exact valuation_usd."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(valuation_usd=200000000)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_usd = $1" in query
        assert 200000000 in params

    @pytest.mark.asyncio
    async def test_get_by_valuation_date(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations by exact valuation_date."""
        valuation_date = datetime(2023, 6, 15)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(valuation_date=valuation_date)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_date = $1" in query
        assert valuation_date in params

    @pytest.mark.asyncio
    async def test_get_by_valuation_date_range(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations by valuation_date range."""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            valuation_date_from=date_from,
            valuation_date_to=date_to,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_date >= $1" in query
        assert "valuation_date <= $2" in query
        assert date_from in params
        assert date_to in params

    @pytest.mark.asyncio
    async def test_get_with_multiple_filters(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations with multiple filters."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            country="United States",
            total_funding_usd_min=1000000,
            categories_contains="Artificial Intelligence",
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # Check that all filters are present
        assert "country = $" in query
        assert "total_funding_usd >= $" in query
        assert "$" in query and "= ANY(categories)" in query
        assert "United States" in params
        assert 1000000 in params
        assert "Artificial Intelligence" in params

    @pytest.mark.asyncio
    async def test_get_with_limit(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations with limit."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(limit=10)

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "LIMIT $1" in query
        assert 10 in params

    @pytest.mark.asyncio
    async def test_get_with_offset(self, mock_postgres_client, sample_organization_record):
        """Test getting organizations with offset."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
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
        self, mock_postgres_client, sample_organization_record
    ):
        """Test getting organizations with both limit and offset."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
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
        """Test getting organizations when no results are found."""
        mock_postgres_client.query.return_value = []
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert results == []
        mock_postgres_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_multiple_results(self, mock_postgres_client, sample_organization_record):
        """Test getting multiple organizations."""

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
            org_uuid=uuid4(),
            cb_url="https://www.crunchbase.com/organization/example2",
            categories=["Fintech"],
            category_groups=["Financial Services"],
            closed_on=None,
            closed_on_precision=None,
            company_profit_type="for_profit",
            created_at=datetime(2019, 3, 20),
            raw_description="A fintech company",
            web_scrape=None,
            rewritten_description="A financial technology company",
            total_funding_native=30000000,
            total_funding_currency="USD",
            total_funding_usd=30000000,
            exited_on=None,
            exited_on_precision=None,
            founding_date=datetime(2017, 8, 15),
            founding_date_precision="day",
            general_funding_stage="Series A",
            logo_url=None,
            ipo_status="private",
            last_fundraise_date=datetime(2022, 4, 10),
            last_funding_total_native=10000000,
            last_funding_total_currency="USD",
            last_funding_total_usd=10000000,
            stage="Series A-1",
            org_type="company",
            city="New York",
            state="New York",
            country="United States",
            continent="North America",
            name="Example Fintech Inc",
            num_acquisitions=0,
            employee_count="11-50",
            num_funding_rounds=2,
            num_investments=0,
            num_portfolio_organizations=0,
            operating_status="operating",
            cb_rank=2500,
            revenue_range="$1M-$10M",
            org_status="active",
            updated_at=datetime(2023, 11, 15),
            valuation_native=50000000,
            valuation_currency="USD",
            valuation_usd=50000000,
            valuation_date=datetime(2022, 4, 10),
            org_domain="example2.com",
        )

        mock_postgres_client.query.return_value = [sample_organization_record, record2]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get()

        assert len(results) == 2
        assert all(isinstance(r, Organization) for r in results)

    @pytest.mark.asyncio
    async def test_get_not_initialized(self):
        """Test get() raises RuntimeError when client is not initialized."""
        model = OrganizationModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.get()
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_database_error(self, mock_postgres_client):
        """Test get() raises exception when database query fails."""
        mock_postgres_client.query.side_effect = Exception("Database connection failed")
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.get()
        assert "Database connection failed" in str(exc_info.value)


class TestOrganizationModelGetByUuid:
    """Test OrganizationModel.get_by_uuid() method."""

    @pytest.mark.asyncio
    async def test_get_by_uuid_success(
        self, mock_postgres_client, sample_org_uuid, sample_organization_record
    ):
        """Test successfully getting organization by UUID."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        result = await model.get_by_uuid(sample_org_uuid)

        assert result is not None
        assert isinstance(result, Organization)
        assert result.org_uuid == sample_org_uuid
        mock_postgres_client.query.assert_called_once()
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_uuid = $1" in query
        assert "LIMIT $2" in query
        assert str(sample_org_uuid) in params
        assert 1 in params

    @pytest.mark.asyncio
    async def test_get_by_uuid_not_found(self, mock_postgres_client, sample_org_uuid):
        """Test get_by_uuid() when organization is not found."""
        mock_postgres_client.query.return_value = []
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        result = await model.get_by_uuid(sample_org_uuid)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_uuid_not_initialized(self, sample_org_uuid):
        """Test get_by_uuid() raises RuntimeError when client is not initialized."""
        model = OrganizationModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.get_by_uuid(sample_org_uuid)
        assert "not initialized" in str(exc_info.value).lower()


class TestOrganizationModelCount:
    """Test OrganizationModel.count() method."""

    @pytest.mark.asyncio
    async def test_count_all(self, mock_postgres_client):
        """Test counting all organizations."""
        mock_postgres_client.query_value.return_value = 100
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count()

        assert count == 100
        mock_postgres_client.query_value.assert_called_once()
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        assert "SELECT COUNT(*) FROM organizations" in query
        assert "WHERE" not in query

    @pytest.mark.asyncio
    async def test_count_by_org_uuid(self, mock_postgres_client, sample_org_uuid):
        """Test counting organizations by UUID."""
        mock_postgres_client.query_value.return_value = 1
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(org_uuid=sample_org_uuid)

        assert count == 1
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_uuid = $1" in query
        assert str(sample_org_uuid) in params

    @pytest.mark.asyncio
    async def test_count_by_categories_contains(self, mock_postgres_client):
        """Test counting organizations by categories array contains."""
        mock_postgres_client.query_value.return_value = 25
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(categories_contains="Artificial Intelligence")

        assert count == 25
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(categories)" in query
        assert "Artificial Intelligence" in params

    @pytest.mark.asyncio
    async def test_count_by_category_groups_contains(self, mock_postgres_client):
        """Test counting organizations by category_groups array contains."""
        mock_postgres_client.query_value.return_value = 30
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(category_groups_contains="Technology")

        assert count == 30
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "$1 = ANY(category_groups)" in query
        assert "Technology" in params

    @pytest.mark.asyncio
    async def test_count_by_country(self, mock_postgres_client):
        """Test counting organizations by country."""
        mock_postgres_client.query_value.return_value = 50
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(country="United States")

        assert count == 50
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "country = $1" in query
        assert "United States" in params

    @pytest.mark.asyncio
    async def test_count_by_continent(self, mock_postgres_client):
        """Test counting organizations by continent."""
        mock_postgres_client.query_value.return_value = 60
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(continent="North America")

        assert count == 60
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "continent = $1" in query
        assert "North America" in params

    @pytest.mark.asyncio
    async def test_count_by_general_funding_stage(self, mock_postgres_client):
        """Test counting organizations by general funding stage."""
        mock_postgres_client.query_value.return_value = 15
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(general_funding_stage="Series B")

        assert count == 15
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "general_funding_stage = $1" in query
        assert "Series B" in params

    @pytest.mark.asyncio
    async def test_count_by_stage(self, mock_postgres_client):
        """Test counting organizations by specific stage."""
        mock_postgres_client.query_value.return_value = 8
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(stage="Series B-1")

        assert count == 8
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "stage = $1" in query
        assert "Series B-1" in params

    @pytest.mark.asyncio
    async def test_count_by_org_type(self, mock_postgres_client):
        """Test counting organizations by organization type."""
        mock_postgres_client.query_value.return_value = 40
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(org_type="company")

        assert count == 40
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_type = $1" in query
        assert "company" in params

    @pytest.mark.asyncio
    async def test_count_by_operating_status(self, mock_postgres_client):
        """Test counting organizations by operating status."""
        mock_postgres_client.query_value.return_value = 35
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(operating_status="operating")

        assert count == 35
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "operating_status = $1" in query
        assert "operating" in params

    @pytest.mark.asyncio
    async def test_count_by_org_status(self, mock_postgres_client):
        """Test counting organizations by organization status."""
        mock_postgres_client.query_value.return_value = 45
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(org_status="active")

        assert count == 45
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_status = $1" in query
        assert "active" in params

    @pytest.mark.asyncio
    async def test_count_by_total_funding_usd_range(self, mock_postgres_client):
        """Test counting organizations by total funding USD range."""
        mock_postgres_client.query_value.return_value = 20
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            total_funding_usd_min=1000000,
            total_funding_usd_max=100000000,
        )

        assert count == 20
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "total_funding_usd >= $1" in query
        assert "total_funding_usd <= $2" in query
        assert 1000000 in params
        assert 100000000 in params

    @pytest.mark.asyncio
    async def test_count_by_valuation_usd_range(self, mock_postgres_client):
        """Test counting organizations by valuation USD range."""
        mock_postgres_client.query_value.return_value = 12
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            valuation_usd_min=10000000,
            valuation_usd_max=500000000,
        )

        assert count == 12
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "valuation_usd >= $1" in query
        assert "valuation_usd <= $2" in query
        assert 10000000 in params
        assert 500000000 in params

    @pytest.mark.asyncio
    async def test_count_by_name_ilike(self, mock_postgres_client):
        """Test counting organizations by case-insensitive name search."""
        mock_postgres_client.query_value.return_value = 5
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(name_ilike="example")

        assert count == 5
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "name ILIKE $1" in query
        assert "%example%" in params

    @pytest.mark.asyncio
    async def test_count_by_org_domain_ilike(self, mock_postgres_client):
        """Test counting organizations by case-insensitive domain search."""
        mock_postgres_client.query_value.return_value = 3
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(org_domain_ilike="example")

        assert count == 3
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "org_domain ILIKE $1" in query
        assert "%example%" in params

    @pytest.mark.asyncio
    async def test_count_with_multiple_filters(self, mock_postgres_client):
        """Test counting organizations with multiple filters."""
        mock_postgres_client.query_value.return_value = 7
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(
            country="United States",
            total_funding_usd_min=1000000,
            categories_contains="Artificial Intelligence",
        )

        assert count == 7
        call_args = mock_postgres_client.query_value.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # Check that all filters are present
        assert "country = $" in query
        assert "total_funding_usd >= $" in query
        assert "$" in query and "= ANY(categories)" in query
        assert "United States" in params
        assert 1000000 in params
        assert "Artificial Intelligence" in params

    @pytest.mark.asyncio
    async def test_count_zero(self, mock_postgres_client):
        """Test counting when no organizations match."""
        mock_postgres_client.query_value.return_value = 0
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count(country="Nonexistent Country")

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_none_result(self, mock_postgres_client):
        """Test counting when query returns None (should return 0)."""
        mock_postgres_client.query_value.return_value = None
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        count = await model.count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_not_initialized(self):
        """Test count() raises RuntimeError when client is not initialized."""
        model = OrganizationModel()
        # Don't call initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await model.count()
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_count_database_error(self, mock_postgres_client):
        """Test count() raises exception when database query fails."""
        mock_postgres_client.query_value.side_effect = Exception("Database connection failed")
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        with pytest.raises(Exception) as exc_info:
            await model.count()
        assert "Database connection failed" in str(exc_info.value)


class TestOrganizationModelEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_get_with_none_values_ignored(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test that None filter values are ignored in query."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            org_uuid=None,
            country=None,
            total_funding_usd=None,
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        # None values should not appear in WHERE clause
        assert "org_uuid = $" not in query
        assert "country = $" not in query
        assert "total_funding_usd = $" not in query

    @pytest.mark.asyncio
    async def test_get_with_partial_none_values(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test that only non-None values are used in query."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(
            country="United States",
            total_funding_usd=None,  # Should be ignored
            categories_contains="Artificial Intelligence",
        )

        assert len(results) == 1
        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        assert "country = $" in query
        assert "$" in query and "= ANY(categories)" in query
        assert "total_funding_usd = $" not in query
        assert "United States" in params
        assert "Artificial Intelligence" in params
        assert None not in params

    @pytest.mark.asyncio
    async def test_get_query_ordering(self, mock_postgres_client, sample_organization_record):
        """Test that query includes proper ORDER BY clause."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        await model.get()

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        assert "ORDER BY updated_at DESC NULLS LAST" in query

    @pytest.mark.asyncio
    async def test_get_parameter_ordering(self, mock_postgres_client, sample_organization_record):
        """Test that query parameters are correctly ordered."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(
            country="United States",
            total_funding_usd_min=1000000,
            categories_contains="Artificial Intelligence",
        )

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        # Check that parameters are numbered sequentially
        assert "$1" in query
        assert "$2" in query
        assert "$3" in query

    @pytest.mark.asyncio
    async def test_get_with_limit_parameter_index(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test that limit parameter uses correct index after filters."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(country="United States", limit=10)

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # country = $1, limit = $2
        assert "LIMIT $2" in query
        assert 10 in params
        assert len(params) == 2  # country and limit

    @pytest.mark.asyncio
    async def test_get_with_offset_parameter_index(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test that offset parameter uses correct index after filters and limit."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        await model.get(country="United States", limit=10, offset=20)

        call_args = mock_postgres_client.query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]
        # country = $1, limit = $2, offset = $3
        assert "LIMIT $2" in query
        assert "OFFSET $3" in query
        assert len(params) == 3

    @pytest.mark.asyncio
    async def test_get_array_fields_handled_correctly(
        self, mock_postgres_client, sample_organization_record
    ):
        """Test that array fields (categories, category_groups) are handled correctly."""
        mock_postgres_client.query.return_value = [sample_organization_record]
        model = OrganizationModel(client=mock_postgres_client)
        await model.initialize()

        results = await model.get(categories_contains="Artificial Intelligence")

        assert len(results) == 1
        # Verify the result has the array fields properly set
        assert results[0].categories == ["Artificial Intelligence", "Machine Learning"]
        assert results[0].category_groups == ["Technology"]
