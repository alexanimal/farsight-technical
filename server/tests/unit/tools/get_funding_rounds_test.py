"""Unit tests for the get_funding_rounds tool.

This module tests the get_funding_rounds function and its various
configurations including filtering, parameter conversion, error handling,
and pagination.
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.contracts.tool_io import ToolOutput
from src.models.funding_rounds import FundingRound, FundingRoundModel
from src.tools.get_funding_rounds import get_funding_rounds, get_tool_metadata


@pytest.fixture
def sample_funding_round_uuid():
    """Create a sample funding round UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_funding_round(sample_funding_round_uuid, sample_org_uuid):
    """Create a sample FundingRound Pydantic model instance."""
    return FundingRound(
        funding_round_uuid=sample_funding_round_uuid,
        investment_date=datetime(2023, 6, 15),
        org_uuid=sample_org_uuid,
        general_funding_stage="Series A",
        stage="Series A",
        investors=["Sequoia Capital", "Accel Partners"],
        lead_investors=["Sequoia Capital"],
        fundraise_amount_usd=10000000,
        valuation_usd=50000000,
    )


@pytest.fixture
def mock_funding_round_model():
    """Create a mock FundingRoundModel instance."""
    model = MagicMock(spec=FundingRoundModel)
    model.initialize = AsyncMock()
    model.get = AsyncMock()
    return model


class TestGetToolMetadata:
    """Test get_tool_metadata function."""

    def test_get_tool_metadata_structure(self):
        """Test that get_tool_metadata returns correct structure."""
        metadata = get_tool_metadata()

        assert metadata.name == "get_funding_rounds"
        assert metadata.description is not None
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.parameters, list)
        assert metadata.returns is not None
        assert metadata.cost_per_call is None
        assert metadata.estimated_latency_ms == 100.0
        assert metadata.timeout_seconds == 30.0
        assert metadata.side_effects is False
        assert metadata.idempotent is True
        assert "database" in metadata.tags
        assert "funding" in metadata.tags
        assert "read-only" in metadata.tags

    def test_get_tool_metadata_parameters(self):
        """Test that get_tool_metadata includes all expected parameters."""
        metadata = get_tool_metadata()
        param_names = {param.name for param in metadata.parameters}

        expected_params = {
            "funding_round_uuid",
            "investment_date",
            "investment_date_from",
            "investment_date_to",
            "org_uuid",
            "general_funding_stage",
            "stage",
            "investors_contains",
            "lead_investors_contains",
            "fundraise_amount_usd",
            "fundraise_amount_usd_min",
            "fundraise_amount_usd_max",
            "valuation_usd",
            "valuation_usd_min",
            "valuation_usd_max",
            "limit",
            "offset",
        }

        assert param_names == expected_params

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


class TestGetFundingRoundsBasic:
    """Test basic get_funding_rounds functionality."""

    @pytest.mark.asyncio
    async def test_get_funding_rounds_no_filters(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds with no filters."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        # Patch FundingRoundModel in the module where it's used
        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert isinstance(result, ToolOutput)
            assert result.success is True
            assert result.tool_name == "get_funding_rounds"
            assert result.result is not None
            assert isinstance(result.result, list)
            assert len(result.result) == 1
            assert result.error is None
            assert result.execution_time_ms is not None
            assert result.metadata["num_results"] == 1

            # Verify model was initialized and called
            mock_funding_round_model.initialize.assert_called_once()
            # Note: get() is called with all parameters (even None values)
            # so we just verify it was called, not the exact arguments
            mock_funding_round_model.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_funding_rounds_empty_result(self, mock_funding_round_model):
        """Test getting funding rounds when no results are found."""
        mock_funding_round_model.get.return_value = []

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert result.success is True
            assert result.result == []
            assert result.metadata["num_results"] == 0

    @pytest.mark.asyncio
    async def test_get_funding_rounds_multiple_results(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuid
    ):
        """Test getting multiple funding rounds."""
        round2 = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 7, 20),
            org_uuid=sample_org_uuid,
            general_funding_stage="Seed",
            stage="Seed",
            investors=["Y Combinator"],
            lead_investors=["Y Combinator"],
            fundraise_amount_usd=2000000,
            valuation_usd=10000000,
        )

        mock_funding_round_model.get.return_value = [sample_funding_round, round2]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert result.success is True
            assert len(result.result) == 2
            assert result.metadata["num_results"] == 2

    @pytest.mark.asyncio
    async def test_get_funding_rounds_result_format(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that result contains properly formatted dictionaries."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert isinstance(result.result[0], dict)
            # Verify it's a model dump (contains all fields)
            assert "funding_round_uuid" in result.result[0]
            assert "investment_date" in result.result[0]
            assert "org_uuid" in result.result[0]
            assert "general_funding_stage" in result.result[0]
            assert "fundraise_amount_usd" in result.result[0]
            assert "valuation_usd" in result.result[0]


class TestGetFundingRoundsFiltering:
    """Test get_funding_rounds with various filters."""

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_uuid(
        self, mock_funding_round_model, sample_funding_round, sample_funding_round_uuid
    ):
        """Test getting funding rounds by UUID."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(funding_round_uuid=str(sample_funding_round_uuid))

            assert result.success is True
            mock_funding_round_model.get.assert_called_once()
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["funding_round_uuid"] == sample_funding_round_uuid
            assert isinstance(call_kwargs["funding_round_uuid"], UUID)

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_org_uuid(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuid
    ):
        """Test getting funding rounds by organization UUID."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(org_uuid=str(sample_org_uuid))

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["org_uuid"] == sample_org_uuid
            assert isinstance(call_kwargs["org_uuid"], UUID)

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_general_stage(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by general funding stage."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(general_funding_stage="Series A")

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["general_funding_stage"] == "Series A"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_stage(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by specific stage."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(stage="Series A")

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["stage"] == "Series A"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_investors_contains(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by investor."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(investors_contains="Sequoia Capital")

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["investors_contains"] == "Sequoia Capital"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_lead_investors_contains(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by lead investor."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(lead_investors_contains="Sequoia Capital")

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["lead_investors_contains"] == "Sequoia Capital"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_fundraise_amount_exact(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by exact fundraise amount."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(fundraise_amount_usd=10000000)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["fundraise_amount_usd"] == 10000000

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_fundraise_amount_range(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by fundraise amount range."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(
                fundraise_amount_usd_min=1000000,
                fundraise_amount_usd_max=50000000,
            )

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["fundraise_amount_usd_min"] == 1000000
            assert call_kwargs["fundraise_amount_usd_max"] == 50000000

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_valuation_exact(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by exact valuation."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(valuation_usd=50000000)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["valuation_usd"] == 50000000

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_valuation_range(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by valuation range."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(
                valuation_usd_min=10000000,
                valuation_usd_max=100000000,
            )

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["valuation_usd_min"] == 10000000
            assert call_kwargs["valuation_usd_max"] == 100000000

    @pytest.mark.asyncio
    async def test_get_funding_rounds_with_multiple_filters(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds with multiple filters."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(
                general_funding_stage="Series A",
                fundraise_amount_usd_min=1000000,
                investors_contains="Sequoia Capital",
            )

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["general_funding_stage"] == "Series A"
            assert call_kwargs["fundraise_amount_usd_min"] == 1000000
            assert call_kwargs["investors_contains"] == "Sequoia Capital"


class TestGetFundingRoundsDateFiltering:
    """Test get_funding_rounds with date filters."""

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_investment_date(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by exact investment date."""
        date_str = "2023-06-15T00:00:00"
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(investment_date=date_str)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["investment_date"] == datetime(2023, 6, 15)
            assert isinstance(call_kwargs["investment_date"], datetime)

    @pytest.mark.asyncio
    async def test_get_funding_rounds_by_investment_date_range(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds by investment date range."""
        date_from_str = "2023-01-01T00:00:00"
        date_to_str = "2023-12-31T23:59:59"
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(
                investment_date_from=date_from_str,
                investment_date_to=date_to_str,
            )

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["investment_date_from"] == datetime(2023, 1, 1)
            assert call_kwargs["investment_date_to"] == datetime(2023, 12, 31, 23, 59, 59)


class TestGetFundingRoundsPagination:
    """Test get_funding_rounds pagination functionality."""

    @pytest.mark.asyncio
    async def test_get_funding_rounds_with_limit(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds with limit."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(limit=10)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_funding_rounds_with_offset(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds with offset."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(offset=20)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["offset"] == 20

    @pytest.mark.asyncio
    async def test_get_funding_rounds_with_limit_and_offset(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test getting funding rounds with both limit and offset."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(limit=10, offset=20)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["limit"] == 10
            assert call_kwargs["offset"] == 20


class TestGetFundingRoundsParameterConversion:
    """Test get_funding_rounds parameter conversion (UUIDs, dates)."""

    @pytest.mark.asyncio
    async def test_get_funding_rounds_uuid_conversion(
        self, mock_funding_round_model, sample_funding_round, sample_funding_round_uuid
    ):
        """Test that string UUIDs are converted to UUID objects."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            uuid_str = str(sample_funding_round_uuid)
            result = await get_funding_rounds(funding_round_uuid=uuid_str)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert isinstance(call_kwargs["funding_round_uuid"], UUID)
            assert call_kwargs["funding_round_uuid"] == sample_funding_round_uuid

    @pytest.mark.asyncio
    async def test_get_funding_rounds_org_uuid_conversion(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuid
    ):
        """Test that organization UUID strings are converted to UUID objects."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            uuid_str = str(sample_org_uuid)
            result = await get_funding_rounds(org_uuid=uuid_str)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert isinstance(call_kwargs["org_uuid"], UUID)
            assert call_kwargs["org_uuid"] == sample_org_uuid

    @pytest.mark.asyncio
    async def test_get_funding_rounds_date_conversion(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that ISO date strings are converted to datetime objects."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            date_str = "2023-06-15T12:30:45"
            result = await get_funding_rounds(investment_date=date_str)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert isinstance(call_kwargs["investment_date"], datetime)
            assert call_kwargs["investment_date"] == datetime(2023, 6, 15, 12, 30, 45)

    @pytest.mark.asyncio
    async def test_get_funding_rounds_date_range_conversion(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that date range strings are converted to datetime objects."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            date_from_str = "2023-01-01T00:00:00"
            date_to_str = "2023-12-31T23:59:59"
            result = await get_funding_rounds(
                investment_date_from=date_from_str,
                investment_date_to=date_to_str,
            )

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert isinstance(call_kwargs["investment_date_from"], datetime)
            assert isinstance(call_kwargs["investment_date_to"], datetime)

    @pytest.mark.asyncio
    async def test_get_funding_rounds_none_uuid_not_converted(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that None UUIDs are not converted."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(funding_round_uuid=None)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs.get("funding_round_uuid") is None

    @pytest.mark.asyncio
    async def test_get_funding_rounds_none_date_not_converted(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that None dates are not converted."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(investment_date=None)

            assert result.success is True
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs.get("investment_date") is None


class TestGetFundingRoundsErrorHandling:
    """Test get_funding_rounds error handling."""

    @pytest.mark.asyncio
    async def test_get_funding_rounds_invalid_uuid(self, mock_funding_round_model):
        """Test that invalid UUID strings raise appropriate errors."""
        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(funding_round_uuid="invalid-uuid")

            assert result.success is False
            assert result.error is not None
            assert "invalid" in result.error.lower() or "uuid" in result.error.lower()
            assert result.result is None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_invalid_date(self, mock_funding_round_model):
        """Test that invalid date strings raise appropriate errors."""
        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds(investment_date="invalid-date")

            assert result.success is False
            assert result.error is not None
            assert result.result is None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_model_initialization_error(self, mock_funding_round_model):
        """Test handling of model initialization errors."""
        mock_funding_round_model.initialize.side_effect = Exception("Database connection failed")

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert result.success is False
            assert result.error is not None
            assert "Failed to get funding rounds" in result.error
            assert result.result is None
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_model_get_error(self, mock_funding_round_model):
        """Test handling of model.get() errors."""
        mock_funding_round_model.get.side_effect = Exception("Query failed")

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert result.success is False
            assert result.error is not None
            assert "Failed to get funding rounds" in result.error
            assert "Query failed" in result.error
            assert result.result is None
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_get_funding_rounds_error_execution_time(self, mock_funding_round_model):
        """Test that execution time is recorded even on errors."""
        mock_funding_round_model.get.side_effect = Exception("Query failed")

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert result.success is False
            assert result.execution_time_ms is not None
            assert result.execution_time_ms >= 0


class TestGetFundingRoundsEdgeCases:
    """Test get_funding_rounds edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_get_funding_rounds_execution_time_recorded(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that execution time is properly recorded."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            assert result.execution_time_ms is not None
            assert isinstance(result.execution_time_ms, float)
            assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_get_funding_rounds_tool_output_structure(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that ToolOutput structure is correct."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            # Verify ToolOutput structure
            assert hasattr(result, "success")
            assert hasattr(result, "result")
            assert hasattr(result, "error")
            assert hasattr(result, "tool_name")
            assert hasattr(result, "execution_time_ms")
            assert hasattr(result, "metadata")
            assert hasattr(result, "timestamp")

    @pytest.mark.asyncio
    async def test_get_funding_rounds_model_initialized_once(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that model.initialize() is called exactly once."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            await get_funding_rounds()

            assert mock_funding_round_model.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_get_funding_rounds_model_get_called_once(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that model.get() is called exactly once per invocation."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            await get_funding_rounds()
            await get_funding_rounds()

            assert mock_funding_round_model.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_funding_rounds_result_contains_model_dumps(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that results are properly converted to dictionaries."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        get_funding_rounds_module = sys.modules["src.tools.get_funding_rounds"]
        with patch.object(
            get_funding_rounds_module, "FundingRoundModel", return_value=mock_funding_round_model
        ):
            result = await get_funding_rounds()

            # Verify the result is a list of dicts (model_dump format)
            assert isinstance(result.result, list)
            assert len(result.result) == 1
            result_dict = result.result[0]

            # Verify it contains all expected fields from FundingRound model
            assert "funding_round_uuid" in result_dict
            assert "investment_date" in result_dict
            assert "org_uuid" in result_dict
            assert "general_funding_stage" in result_dict
            assert "fundraise_amount_usd" in result_dict
            assert "valuation_usd" in result_dict

            # Verify UUIDs are present in the dict
            # Note: model_dump() by default keeps UUIDs as UUID objects, not strings
            # To serialize to strings, would need model_dump(mode="json")
            assert isinstance(result_dict["funding_round_uuid"], UUID)

