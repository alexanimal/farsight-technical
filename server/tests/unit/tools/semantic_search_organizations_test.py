"""Unit tests for the semantic_search_organizations tool.

This module tests the semantic_search_organizations function and its various
configurations including semantic search, filtering, parameter conversion,
error handling, and result formatting.
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.contracts.tool_io import ToolOutput
from src.models.pinecone_organizations import PineconeOrganization, PineconeOrganizationModel
from src.tools.semantic_search_organizations import get_tool_metadata, semantic_search_organizations


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_investor_uuid():
    """Create a sample investor UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_pinecone_organization(sample_org_uuid, sample_investor_uuid):
    """Create a sample PineconeOrganization Pydantic model instance."""
    return PineconeOrganization(
        org_uuid=sample_org_uuid,
        name="Test AI Company",
        categories=["Artificial Intelligence", "Machine Learning"],
        org_status="operating",
        total_funding_usd=10000000.0,
        founding_date=datetime(2020, 1, 15),
        last_fundraise_date=datetime(2023, 6, 20),
        employee_count="51-100",
        org_type="company",
        stage="series_a",
        valuation_usd=50000000.0,
        investors=[sample_investor_uuid],
        general_funding_stage="early_stage_venture",
        num_acquisitions=0,
        revenue_range="1M-10M",
        score=0.95,
    )


@pytest.fixture
def mock_pinecone_organization_model():
    """Create a mock PineconeOrganizationModel instance."""
    model = MagicMock(spec=PineconeOrganizationModel)
    model.initialize = AsyncMock()
    model.query = AsyncMock()
    return model


class TestGetToolMetadata:
    """Test get_tool_metadata function."""

    def test_get_tool_metadata_structure(self):
        """Test that get_tool_metadata returns correct structure."""
        metadata = get_tool_metadata()

        assert metadata.name == "semantic_search_organizations"
        assert metadata.description is not None
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.parameters, list)
        assert metadata.returns is not None
        assert metadata.cost_per_call is None
        assert metadata.estimated_latency_ms == 200.0
        assert metadata.timeout_seconds == 30.0
        assert metadata.side_effects is False
        assert metadata.idempotent is True
        assert "pinecone" in metadata.tags
        assert "vector-search" in metadata.tags
        assert "semantic-search" in metadata.tags
        assert "organizations" in metadata.tags
        assert "read-only" in metadata.tags

    def test_get_tool_metadata_parameters(self):
        """Test that get_tool_metadata includes all expected parameters."""
        metadata = get_tool_metadata()
        param_names = {param.name for param in metadata.parameters}

        expected_params = {
            "text",
            "org_uuid",
            "name",
            "categories_contains",
            "org_status",
            "total_funding_usd_min",
            "total_funding_usd_max",
            "founding_date_from",
            "founding_date_to",
            "last_fundraise_date_from",
            "last_fundraise_date_to",
            "employee_count",
            "org_type",
            "stage",
            "valuation_usd_min",
            "valuation_usd_max",
            "investors_contains",
            "general_funding_stage",
            "num_acquisitions_min",
            "num_acquisitions_max",
            "revenue_range",
            "top_k",
            "index_name",
        }

        assert param_names == expected_params

    def test_get_tool_metadata_required_params(self):
        """Test that required parameters are marked correctly."""
        metadata = get_tool_metadata()
        required_params = {p.name for p in metadata.parameters if p.required}

        assert "text" in required_params
        assert "top_k" not in required_params  # Has default
        assert "index_name" not in required_params

    def test_get_tool_metadata_defaults(self):
        """Test that default values are correctly set."""
        metadata = get_tool_metadata()
        top_k_param = next(p for p in metadata.parameters if p.name == "top_k")

        assert top_k_param.default == 10

    def test_get_tool_metadata_returns_schema(self):
        """Test that returns schema is correctly defined."""
        metadata = get_tool_metadata()
        assert metadata.returns["type"] == "array"
        assert "items" in metadata.returns
        assert metadata.returns["items"]["type"] == "object"


class TestSemanticSearchOrganizationsBasic:
    """Test basic semantic_search_organizations functionality."""

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_basic(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test basic semantic search with text query."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert isinstance(result, ToolOutput)
            assert result.success is True
            assert result.tool_name == "semantic_search_organizations"
            assert result.result is not None
            assert isinstance(result.result, list)
            assert len(result.result) == 1
            assert result.error is None
            assert result.execution_time_ms is not None

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_result_structure(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that result has correct structure."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is True
            assert len(result.result) == 1
            result_dict = result.result[0]
            # Should be a dictionary (from model_dump())
            assert isinstance(result_dict, dict)
            assert "org_uuid" in result_dict
            assert "name" in result_dict
            assert "score" in result_dict

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_empty_results(
        self, mock_pinecone_organization_model
    ):
        """Test semantic search when no organizations are found."""
        mock_pinecone_organization_model.query.return_value = []

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="nonexistent sector")

            assert result.success is True
            assert result.result == []
            assert result.metadata["num_results"] == 0

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_multiple_results(
        self,
        mock_pinecone_organization_model,
        sample_pinecone_organization,
        sample_org_uuid,
    ):
        """Test semantic search with multiple results."""
        org2 = PineconeOrganization(
            org_uuid=sample_org_uuid,
            name="Another AI Company",
            categories=["Artificial Intelligence"],
            org_status="operating",
            score=0.90,
        )
        mock_pinecone_organization_model.query.return_value = [
            sample_pinecone_organization,
            org2,
        ]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies", top_k=5)

            assert result.success is True
            assert len(result.result) == 2
            assert result.metadata["num_results"] == 2
            assert result.metadata["top_k"] == 5


class TestSemanticSearchOrganizationsFiltering:
    """Test semantic_search_organizations filtering options."""

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_org_uuid(
        self,
        mock_pinecone_organization_model,
        sample_pinecone_organization,
        sample_org_uuid,
    ):
        """Test filtering by org_uuid."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", org_uuid=str(sample_org_uuid)
            )

            assert result.success is True
            # Verify UUID was converted and passed to query
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["org_uuid"] == sample_org_uuid
            assert isinstance(call_kwargs["org_uuid"], UUID)

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_name(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by name."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", name="Test AI Company"
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["name"] == "Test AI Company"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_categories_contains(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by categories_contains."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", categories_contains="Machine Learning"
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["categories_contains"] == "Machine Learning"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_org_status(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by org_status."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", org_status="operating"
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["org_status"] == "operating"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_funding_range(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by funding range."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies",
                total_funding_usd_min=5000000.0,
                total_funding_usd_max=20000000.0,
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["total_funding_usd_min"] == 5000000.0
            assert call_kwargs["total_funding_usd_max"] == 20000000.0

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_founding_date_range(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by founding date range."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies",
                founding_date_from="2019-01-01T00:00:00",
                founding_date_to="2021-12-31T23:59:59",
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["founding_date_from"] == datetime(2019, 1, 1)
            assert call_kwargs["founding_date_to"] == datetime(2021, 12, 31, 23, 59, 59)
            assert isinstance(call_kwargs["founding_date_from"], datetime)
            assert isinstance(call_kwargs["founding_date_to"], datetime)

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_last_fundraise_date_range(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by last fundraise date range."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies",
                last_fundraise_date_from="2023-01-01T00:00:00",
                last_fundraise_date_to="2023-12-31T23:59:59",
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["last_fundraise_date_from"] == datetime(2023, 1, 1)
            assert call_kwargs["last_fundraise_date_to"] == datetime(2023, 12, 31, 23, 59, 59)
            assert isinstance(call_kwargs["last_fundraise_date_from"], datetime)
            assert isinstance(call_kwargs["last_fundraise_date_to"], datetime)

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_employee_count(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by employee_count."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", employee_count="51-100"
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["employee_count"] == "51-100"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_org_type(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by org_type."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies", org_type="company")

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["org_type"] == "company"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_stage(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by stage."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies", stage="series_a")

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["stage"] == "series_a"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_valuation_range(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by valuation range."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies",
                valuation_usd_min=25000000.0,
                valuation_usd_max=75000000.0,
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["valuation_usd_min"] == 25000000.0
            assert call_kwargs["valuation_usd_max"] == 75000000.0

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_investors_contains(
        self,
        mock_pinecone_organization_model,
        sample_pinecone_organization,
        sample_investor_uuid,
    ):
        """Test filtering by investors_contains."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", investors_contains=str(sample_investor_uuid)
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["investors_contains"] == sample_investor_uuid
            assert isinstance(call_kwargs["investors_contains"], UUID)

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_general_funding_stage(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by general_funding_stage."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", general_funding_stage="early_stage_venture"
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["general_funding_stage"] == "early_stage_venture"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_acquisitions_range(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by acquisitions range."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies",
                num_acquisitions_min=0,
                num_acquisitions_max=5,
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["num_acquisitions_min"] == 0
            assert call_kwargs["num_acquisitions_max"] == 5

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_revenue_range(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test filtering by revenue_range."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", revenue_range="1M-10M"
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["revenue_range"] == "1M-10M"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_top_k(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test custom top_k parameter."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies", top_k=20)

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["top_k"] == 20
            assert result.metadata["top_k"] == 20

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_with_index_name(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test custom index_name parameter."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", index_name="custom-index"
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["index_name"] == "custom-index"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_default_top_k(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that default top_k is 10."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["top_k"] == 10
            assert result.metadata["top_k"] == 10


class TestSemanticSearchOrganizationsParameterConversion:
    """Test semantic_search_organizations parameter conversion."""

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_uuid_conversion(
        self,
        mock_pinecone_organization_model,
        sample_pinecone_organization,
        sample_org_uuid,
    ):
        """Test that string UUIDs are converted to UUID objects."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", org_uuid=str(sample_org_uuid)
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert isinstance(call_kwargs["org_uuid"], UUID)
            assert call_kwargs["org_uuid"] == sample_org_uuid

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_investors_uuid_conversion(
        self,
        mock_pinecone_organization_model,
        sample_pinecone_organization,
        sample_investor_uuid,
    ):
        """Test that investors_contains UUID string is converted to UUID object."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", investors_contains=str(sample_investor_uuid)
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert isinstance(call_kwargs["investors_contains"], UUID)
            assert call_kwargs["investors_contains"] == sample_investor_uuid

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_date_conversion(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that date strings are converted to datetime objects."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies",
                founding_date_from="2020-01-01T00:00:00",
                founding_date_to="2022-12-31T23:59:59",
                last_fundraise_date_from="2023-01-01T00:00:00",
                last_fundraise_date_to="2023-12-31T23:59:59",
            )

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert isinstance(call_kwargs["founding_date_from"], datetime)
            assert isinstance(call_kwargs["founding_date_to"], datetime)
            assert isinstance(call_kwargs["last_fundraise_date_from"], datetime)
            assert isinstance(call_kwargs["last_fundraise_date_to"], datetime)

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_none_uuid_not_converted(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that None UUIDs are not converted."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["org_uuid"] is None
            assert call_kwargs["investors_contains"] is None

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_none_dates_not_converted(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that None dates are not converted."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["founding_date_from"] is None
            assert call_kwargs["founding_date_to"] is None
            assert call_kwargs["last_fundraise_date_from"] is None
            assert call_kwargs["last_fundraise_date_to"] is None


class TestSemanticSearchOrganizationsModelDump:
    """Test semantic_search_organizations model_dump conversion."""

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_result_contains_model_dumps(
        self,
        mock_pinecone_organization_model,
        sample_pinecone_organization,
        sample_org_uuid,
    ):
        """Test that results contain model_dump() output."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is True
            assert len(result.result) == 1
            result_dict = result.result[0]
            # model_dump() by default keeps UUID as UUID object
            assert isinstance(result_dict["org_uuid"], UUID)
            assert result_dict["org_uuid"] == sample_org_uuid
            assert result_dict["name"] == "Test AI Company"
            assert result_dict["score"] == 0.95


class TestSemanticSearchOrganizationsErrorHandling:
    """Test semantic_search_organizations error handling."""

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_invalid_uuid(
        self, mock_pinecone_organization_model
    ):
        """Test that invalid UUID format raises error."""
        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", org_uuid="invalid-uuid"
            )

            assert result.success is False
            assert result.error is not None
            assert "invalid" in result.error.lower() or "uuid" in result.error.lower()
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_invalid_investor_uuid(
        self, mock_pinecone_organization_model
    ):
        """Test that invalid investor UUID format raises error."""
        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", investors_contains="invalid-uuid"
            )

            assert result.success is False
            assert result.error is not None
            assert "invalid" in result.error.lower() or "uuid" in result.error.lower()
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_invalid_date_format(
        self, mock_pinecone_organization_model
    ):
        """Test that invalid date format raises error."""
        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(
                text="AI companies", founding_date_from="invalid-date"
            )

            assert result.success is False
            assert result.error is not None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_model_initialization_error(
        self, mock_pinecone_organization_model
    ):
        """Test handling of model initialization errors."""
        mock_pinecone_organization_model.initialize.side_effect = Exception(
            "Pinecone connection failed"
        )

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is False
            assert result.error is not None
            assert "Failed to semantically search organizations" in result.error
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_model_query_error(
        self, mock_pinecone_organization_model
    ):
        """Test handling of model.query() errors."""
        mock_pinecone_organization_model.query.side_effect = Exception("Query failed")

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is False
            assert result.error is not None
            assert "Failed to semantically search organizations" in result.error
            assert "Query failed" in result.error
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_error_execution_time(
        self, mock_pinecone_organization_model
    ):
        """Test that execution time is recorded even on errors."""
        mock_pinecone_organization_model.query.side_effect = Exception("Query failed")

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is False
            assert result.execution_time_ms is not None
            assert result.execution_time_ms >= 0


class TestSemanticSearchOrganizationsMetadata:
    """Test semantic_search_organizations metadata and result structure."""

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_metadata_structure(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that metadata contains expected information."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies", top_k=5)

            assert result.success is True
            assert "num_results" in result.metadata
            assert "top_k" in result.metadata
            assert result.metadata["num_results"] == 1
            assert result.metadata["top_k"] == 5

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_tool_output_structure(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that ToolOutput structure is correct."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            # Verify ToolOutput structure
            assert hasattr(result, "success")
            assert hasattr(result, "result")
            assert hasattr(result, "error")
            assert hasattr(result, "tool_name")
            assert hasattr(result, "execution_time_ms")
            assert hasattr(result, "metadata")
            assert hasattr(result, "timestamp")

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_execution_time_recorded(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that execution time is properly recorded."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.execution_time_ms is not None
            assert isinstance(result.execution_time_ms, float)
            assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_model_initialization_called(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that model.initialize() is called."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is True
            mock_pinecone_organization_model.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_search_organizations_query_called_with_text(
        self, mock_pinecone_organization_model, sample_pinecone_organization
    ):
        """Test that model.query() is called with text parameter."""
        mock_pinecone_organization_model.query.return_value = [sample_pinecone_organization]

        semantic_search_module = sys.modules["src.tools.semantic_search_organizations"]
        with patch.object(
            semantic_search_module,
            "PineconeOrganizationModel",
            return_value=mock_pinecone_organization_model,
        ):
            result = await semantic_search_organizations(text="AI companies")

            assert result.success is True
            call_kwargs = mock_pinecone_organization_model.query.call_args[1]
            assert call_kwargs["text"] == "AI companies"
