"""Unit tests for the get_organizations tool.

This module tests the get_organizations function and its various
configurations including filtering, parameter conversion, error handling,
and pagination.
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.contracts.tool_io import ToolOutput
from src.models.organizations import Organization, OrganizationModel
from src.tools.get_organizations import get_organizations, get_tool_metadata


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_organization(sample_org_uuid):
    """Create a sample Organization Pydantic model instance."""
    return Organization(
        org_uuid=sample_org_uuid,
        cb_url="https://www.crunchbase.com/organization/example",
        categories=["Technology", "Software"],
        category_groups=["B2B"],
        closed_on=None,
        closed_on_precision=None,
        company_profit_type="for_profit",
        created_at=datetime(2020, 1, 15),
        raw_description="A technology company",
        web_scrape=None,
        rewritten_description="A leading technology company",
        total_funding_native=10000000,
        total_funding_currency="USD",
        total_funding_usd=10000000,
        exited_on=None,
        exited_on_precision=None,
        founding_date=datetime(2015, 5, 10),
        founding_date_precision="day",
        general_funding_stage="Series A",
        logo_url=None,
        ipo_status="private",
        last_fundraise_date=datetime(2023, 3, 20),
        last_funding_total_native=5000000,
        last_funding_total_currency="USD",
        last_funding_total_usd=5000000,
        stage="Series A",
        org_type="company",
        city="San Francisco",
        state="California",
        country="United States",
        continent="North America",
        name="Example Tech Inc",
        num_acquisitions=2,
        employee_count="51-100",
        num_funding_rounds=3,
        num_investments=0,
        num_portfolio_organizations=0,
        operating_status="operating",
        cb_rank=5000,
        revenue_range="1M-10M",
        org_status="active",
        updated_at=datetime(2023, 12, 1),
        valuation_usd=50000000,
        valuation_date=datetime(2023, 3, 20),
        valuation_date_precision="day",
        org_domain="example.com",
    )


@pytest.fixture
def mock_organization_model():
    """Create a mock OrganizationModel instance."""
    model = MagicMock(spec=OrganizationModel)
    model.initialize = AsyncMock()
    model.get = AsyncMock()
    return model


class TestGetToolMetadata:
    """Test get_tool_metadata function."""

    def test_get_tool_metadata_structure(self):
        """Test that get_tool_metadata returns correct structure."""
        metadata = get_tool_metadata()

        assert metadata.name == "get_organizations"
        assert metadata.description is not None
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.parameters, list)
        assert len(metadata.parameters) > 0  # Has many parameters
        assert metadata.returns is not None
        assert metadata.cost_per_call is None
        assert metadata.estimated_latency_ms == 150.0
        assert metadata.timeout_seconds == 30.0
        assert metadata.side_effects is False
        assert metadata.idempotent is True
        assert "database" in metadata.tags
        assert "organizations" in metadata.tags
        assert "read-only" in metadata.tags

    def test_get_tool_metadata_has_expected_parameters(self):
        """Test that get_tool_metadata includes key expected parameters."""
        metadata = get_tool_metadata()
        param_names = {param.name for param in metadata.parameters}

        # Check for some key parameters (not all, as there are 80+)
        expected_params = {
            "org_uuid",
            "cb_url",
            "name",
            "name_ilike",
            "country",
            "city",
            "total_funding_usd",
            "total_funding_usd_min",
            "total_funding_usd_max",
            "limit",
            "offset",
        }

        assert expected_params.issubset(param_names)

    def test_get_tool_metadata_no_required_params(self):
        """Test that all parameters are optional."""
        metadata = get_tool_metadata()
        required_params = [p for p in metadata.parameters if p.required]
        assert len(required_params) == 0

    def test_get_tool_metadata_returns_schema(self):
        """Test that returns schema is correctly defined."""
        metadata = get_tool_metadata()
        assert metadata.returns["type"] == "array"
        assert metadata.returns["items"]["type"] == "object"


class TestGetOrganizationsBasic:
    """Test basic get_organizations functionality."""

    @pytest.mark.asyncio
    async def test_get_organizations_no_filters(self, mock_organization_model, sample_organization):
        """Test getting organizations with no filters."""
        mock_organization_model.get.return_value = [sample_organization]

        # Patch OrganizationModel in the module where it's used
        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert isinstance(result, ToolOutput)
            assert result.success is True
            assert result.tool_name == "get_organizations"
            assert result.result is not None
            assert isinstance(result.result, list)
            assert len(result.result) == 1
            assert result.error is None
            assert result.execution_time_ms is not None
            assert result.metadata["num_results"] == 1

            # Verify model was initialized and called
            mock_organization_model.initialize.assert_called_once()
            # Note: get() is called with all parameters (even None values)
            # so we just verify it was called, not the exact arguments
            mock_organization_model.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_organizations_empty_result(self, mock_organization_model):
        """Test getting organizations when no results are found."""
        mock_organization_model.get.return_value = []

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert result.success is True
            assert result.result == []
            assert result.metadata["num_results"] == 0

    @pytest.mark.asyncio
    async def test_get_organizations_multiple_results(
        self, mock_organization_model, sample_organization, sample_org_uuid
    ):
        """Test getting multiple organizations."""
        org2 = Organization(
            org_uuid=uuid4(),
            cb_url="https://www.crunchbase.com/organization/example2",
            categories=["Healthcare"],
            category_groups=["B2C"],
            name="Example Healthcare Inc",
            country="United States",
            total_funding_usd=20000000,
        )

        mock_organization_model.get.return_value = [sample_organization, org2]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert result.success is True
            assert len(result.result) == 2
            assert result.metadata["num_results"] == 2

    @pytest.mark.asyncio
    async def test_get_organizations_result_format(
        self, mock_organization_model, sample_organization
    ):
        """Test that result contains properly formatted dictionaries."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert isinstance(result.result[0], dict)
            # Verify it's a model dump (contains all fields)
            assert "org_uuid" in result.result[0]
            assert "name" in result.result[0]
            assert "country" in result.result[0]
            assert "total_funding_usd" in result.result[0]


class TestGetOrganizationsFiltering:
    """Test get_organizations with various filters."""

    @pytest.mark.asyncio
    async def test_get_organizations_by_uuid(
        self, mock_organization_model, sample_organization, sample_org_uuid
    ):
        """Test getting organizations by UUID."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(org_uuid=str(sample_org_uuid))

            assert result.success is True
            mock_organization_model.get.assert_called_once()
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["org_uuid"] == sample_org_uuid
            assert isinstance(call_kwargs["org_uuid"], UUID)

    @pytest.mark.asyncio
    async def test_get_organizations_by_name(self, mock_organization_model, sample_organization):
        """Test getting organizations by exact name."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(name="Example Tech Inc")

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["name"] == "Example Tech Inc"

    @pytest.mark.asyncio
    async def test_get_organizations_by_name_ilike(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by case-insensitive name search."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(name_ilike="example")

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["name_ilike"] == "example"

    @pytest.mark.asyncio
    async def test_get_organizations_by_country(self, mock_organization_model, sample_organization):
        """Test getting organizations by country."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(country="United States")

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["country"] == "United States"

    @pytest.mark.asyncio
    async def test_get_organizations_by_funding_range(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by funding range."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(
                total_funding_usd_min=1000000,
                total_funding_usd_max=100000000,
            )

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["total_funding_usd_min"] == 1000000
            assert call_kwargs["total_funding_usd_max"] == 100000000

    @pytest.mark.asyncio
    async def test_get_organizations_by_categories_contains(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by category."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(categories_contains="Technology")

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["categories_contains"] == "Technology"

    @pytest.mark.asyncio
    async def test_get_organizations_by_stage(self, mock_organization_model, sample_organization):
        """Test getting organizations by stage."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(stage="Series A")

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["stage"] == "Series A"

    @pytest.mark.asyncio
    async def test_get_organizations_by_org_type(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by organization type."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(org_type="company")

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["org_type"] == "company"

    @pytest.mark.asyncio
    async def test_get_organizations_with_multiple_filters(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations with multiple filters."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(
                country="United States",
                total_funding_usd_min=1000000,
                stage="Series A",
            )

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["country"] == "United States"
            assert call_kwargs["total_funding_usd_min"] == 1000000
            assert call_kwargs["stage"] == "Series A"


class TestGetOrganizationsDateFiltering:
    """Test get_organizations with date filters."""

    @pytest.mark.asyncio
    async def test_get_organizations_by_founding_date(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by exact founding date."""
        date_str = "2015-05-10T00:00:00"
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(founding_date=date_str)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["founding_date"] == datetime(2015, 5, 10)
            assert isinstance(call_kwargs["founding_date"], datetime)

    @pytest.mark.asyncio
    async def test_get_organizations_by_founding_date_range(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by founding date range."""
        date_from_str = "2010-01-01T00:00:00"
        date_to_str = "2020-12-31T23:59:59"
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(
                founding_date_from=date_from_str,
                founding_date_to=date_to_str,
            )

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["founding_date_from"] == datetime(2010, 1, 1)
            assert call_kwargs["founding_date_to"] == datetime(2020, 12, 31, 23, 59, 59)

    @pytest.mark.asyncio
    async def test_get_organizations_by_closed_on_date(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by closed date."""
        date_str = "2023-01-15T00:00:00"
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(closed_on=date_str)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert isinstance(call_kwargs["closed_on"], datetime)

    @pytest.mark.asyncio
    async def test_get_organizations_by_exited_on_date(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations by exited date."""
        date_str = "2023-06-20T00:00:00"
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(exited_on=date_str)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert isinstance(call_kwargs["exited_on"], datetime)


class TestGetOrganizationsPagination:
    """Test get_organizations pagination functionality."""

    @pytest.mark.asyncio
    async def test_get_organizations_with_limit(self, mock_organization_model, sample_organization):
        """Test getting organizations with limit."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(limit=10)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_organizations_with_offset(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations with offset."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(offset=20)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["offset"] == 20

    @pytest.mark.asyncio
    async def test_get_organizations_with_limit_and_offset(
        self, mock_organization_model, sample_organization
    ):
        """Test getting organizations with both limit and offset."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(limit=10, offset=20)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs["limit"] == 10
            assert call_kwargs["offset"] == 20


class TestGetOrganizationsParameterConversion:
    """Test get_organizations parameter conversion (UUIDs, dates)."""

    @pytest.mark.asyncio
    async def test_get_organizations_uuid_conversion(
        self, mock_organization_model, sample_organization, sample_org_uuid
    ):
        """Test that string UUIDs are converted to UUID objects."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            uuid_str = str(sample_org_uuid)
            result = await get_organizations(org_uuid=uuid_str)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert isinstance(call_kwargs["org_uuid"], UUID)
            assert call_kwargs["org_uuid"] == sample_org_uuid

    @pytest.mark.asyncio
    async def test_get_organizations_date_conversion(
        self, mock_organization_model, sample_organization
    ):
        """Test that ISO date strings are converted to datetime objects."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            date_str = "2023-06-15T12:30:45"
            result = await get_organizations(founding_date=date_str)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert isinstance(call_kwargs["founding_date"], datetime)
            assert call_kwargs["founding_date"] == datetime(2023, 6, 15, 12, 30, 45)

    @pytest.mark.asyncio
    async def test_get_organizations_multiple_date_conversions(
        self, mock_organization_model, sample_organization
    ):
        """Test that multiple date fields are converted correctly."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(
                founding_date="2015-05-10T00:00:00",
                closed_on="2023-01-15T00:00:00",
                exited_on="2023-06-20T00:00:00",
            )

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert isinstance(call_kwargs["founding_date"], datetime)
            assert isinstance(call_kwargs["closed_on"], datetime)
            assert isinstance(call_kwargs["exited_on"], datetime)

    @pytest.mark.asyncio
    async def test_get_organizations_none_uuid_not_converted(
        self, mock_organization_model, sample_organization
    ):
        """Test that None UUIDs are not converted."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(org_uuid=None)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs.get("org_uuid") is None

    @pytest.mark.asyncio
    async def test_get_organizations_none_date_not_converted(
        self, mock_organization_model, sample_organization
    ):
        """Test that None dates are not converted."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(founding_date=None)

            assert result.success is True
            call_kwargs = mock_organization_model.get.call_args[1]
            assert call_kwargs.get("founding_date") is None


class TestGetOrganizationsErrorHandling:
    """Test get_organizations error handling."""

    @pytest.mark.asyncio
    async def test_get_organizations_invalid_uuid(self, mock_organization_model):
        """Test that invalid UUID strings raise appropriate errors."""
        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(org_uuid="invalid-uuid")

            assert result.success is False
            assert result.error is not None
            assert "invalid" in result.error.lower() or "uuid" in result.error.lower()
            assert result.result is None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_get_organizations_invalid_date(self, mock_organization_model):
        """Test that invalid date strings raise appropriate errors."""
        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations(founding_date="invalid-date")

            assert result.success is False
            assert result.error is not None
            assert result.result is None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_get_organizations_model_initialization_error(self, mock_organization_model):
        """Test handling of model initialization errors."""
        mock_organization_model.initialize.side_effect = Exception("Database connection failed")

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert result.success is False
            assert result.error is not None
            assert "Failed to get organizations" in result.error
            assert result.result is None
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_get_organizations_model_get_error(self, mock_organization_model):
        """Test handling of model.get() errors."""
        mock_organization_model.get.side_effect = Exception("Query failed")

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert result.success is False
            assert result.error is not None
            assert "Failed to get organizations" in result.error
            assert "Query failed" in result.error
            assert result.result is None
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_get_organizations_error_execution_time(self, mock_organization_model):
        """Test that execution time is recorded even on errors."""
        mock_organization_model.get.side_effect = Exception("Query failed")

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert result.success is False
            assert result.execution_time_ms is not None
            assert result.execution_time_ms >= 0


class TestGetOrganizationsEdgeCases:
    """Test get_organizations edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_get_organizations_execution_time_recorded(
        self, mock_organization_model, sample_organization
    ):
        """Test that execution time is properly recorded."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            assert result.execution_time_ms is not None
            assert isinstance(result.execution_time_ms, float)
            assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_get_organizations_tool_output_structure(
        self, mock_organization_model, sample_organization
    ):
        """Test that ToolOutput structure is correct."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            # Verify ToolOutput structure
            assert hasattr(result, "success")
            assert hasattr(result, "result")
            assert hasattr(result, "error")
            assert hasattr(result, "tool_name")
            assert hasattr(result, "execution_time_ms")
            assert hasattr(result, "metadata")
            assert hasattr(result, "timestamp")

    @pytest.mark.asyncio
    async def test_get_organizations_model_initialized_once(
        self, mock_organization_model, sample_organization
    ):
        """Test that model.initialize() is called exactly once."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            await get_organizations()

            assert mock_organization_model.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_get_organizations_model_get_called_once(
        self, mock_organization_model, sample_organization
    ):
        """Test that model.get() is called exactly once per invocation."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            await get_organizations()
            await get_organizations()

            assert mock_organization_model.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_organizations_result_contains_model_dumps(
        self, mock_organization_model, sample_organization
    ):
        """Test that results are properly converted to dictionaries."""
        mock_organization_model.get.return_value = [sample_organization]

        get_organizations_module = sys.modules["src.tools.get_organizations"]
        with patch.object(
            get_organizations_module,
            "OrganizationModel",
            return_value=mock_organization_model,
        ):
            result = await get_organizations()

            # Verify the result is a list of dicts (model_dump format)
            assert isinstance(result.result, list)
            assert len(result.result) == 1
            result_dict = result.result[0]

            # Verify it contains all expected fields from Organization model
            assert "org_uuid" in result_dict
            assert "name" in result_dict
            assert "country" in result_dict
            assert "total_funding_usd" in result_dict

            # Verify UUIDs are present in the dict
            # Note: model_dump() by default keeps UUIDs as UUID objects, not strings
            # To serialize to strings, would need model_dump(mode="json")
            assert isinstance(result_dict["org_uuid"], UUID)
