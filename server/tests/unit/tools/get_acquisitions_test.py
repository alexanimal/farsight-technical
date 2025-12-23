"""Unit tests for the get_acquisitions tool.

This module tests the get_acquisitions function and its various
configurations including filtering, parameter conversion, error handling,
and pagination.
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.contracts.tool_io import ToolOutput
from src.models.acquisitions import Acquisition, AcquisitionModel
from src.tools.get_acquisitions import get_acquisitions, get_tool_metadata


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
def sample_acquisition(sample_acquisition_uuid, sample_acquiree_uuid, sample_acquirer_uuid):
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


@pytest.fixture
def mock_acquisition_model():
    """Create a mock AcquisitionModel instance."""
    model = MagicMock(spec=AcquisitionModel)
    model.initialize = AsyncMock()
    model.get = AsyncMock()
    return model


class TestGetToolMetadata:
    """Test get_tool_metadata function."""

    def test_get_tool_metadata_structure(self):
        """Test that get_tool_metadata returns correct structure."""
        metadata = get_tool_metadata()

        assert metadata.name == "get_acquisitions"
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
        assert "acquisitions" in metadata.tags
        assert "read-only" in metadata.tags

    def test_get_tool_metadata_parameters(self):
        """Test that get_tool_metadata includes all expected parameters."""
        metadata = get_tool_metadata()
        param_names = {param.name for param in metadata.parameters}

        expected_params = {
            "acquisition_uuid",
            "acquiree_uuid",
            "acquirer_uuid",
            "acquisition_type",
            "acquisition_announce_date",
            "acquisition_announce_date_from",
            "acquisition_announce_date_to",
            "acquisition_price_usd",
            "acquisition_price_usd_min",
            "acquisition_price_usd_max",
            "terms",
            "terms_ilike",
            "acquirer_type",
            "limit",
            "offset",
            "include_organizations",
            "order_by",
            "order_direction",
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


class TestGetAcquisitionsBasic:
    """Test basic get_acquisitions functionality."""

    @pytest.mark.asyncio
    async def test_get_acquisitions_no_filters(self, mock_acquisition_model, sample_acquisition):
        """Test getting acquisitions with no filters."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        # Patch AcquisitionModel in the module where it's used
        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert isinstance(result, ToolOutput)
            assert result.success is True
            assert result.tool_name == "get_acquisitions"
            assert result.result is not None
            assert isinstance(result.result, list)
            assert len(result.result) == 1
            assert result.error is None
            assert result.execution_time_ms is not None
            assert result.metadata["num_results"] == 1

            # Verify model was initialized and called
            mock_acquisition_model.initialize.assert_called_once()
            # Note: get() is called with all parameters (even None values)
            # so we just verify it was called, not the exact arguments
            mock_acquisition_model.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_acquisitions_empty_result(self, mock_acquisition_model):
        """Test getting acquisitions when no results are found."""
        mock_acquisition_model.get.return_value = []

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert result.success is True
            assert result.result == []
            assert result.metadata["num_results"] == 0

    @pytest.mark.asyncio
    async def test_get_acquisitions_multiple_results(
        self,
        mock_acquisition_model,
        sample_acquisition,
        sample_acquiree_uuid,
        sample_acquirer_uuid,
    ):
        """Test getting multiple acquisitions."""
        acquisition2 = Acquisition(
            acquisition_uuid=uuid4(),
            acquiree_uuid=sample_acquiree_uuid,
            acquirer_uuid=sample_acquirer_uuid,
            acquisition_type="acquisition",
            acquisition_announce_date=datetime(2023, 7, 20),
            acquisition_price_usd=75000000,
            terms="All cash",
            acquirer_type="private_equity",
        )

        mock_acquisition_model.get.return_value = [sample_acquisition, acquisition2]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert result.success is True
            assert len(result.result) == 2
            assert result.metadata["num_results"] == 2

    @pytest.mark.asyncio
    async def test_get_acquisitions_result_format(self, mock_acquisition_model, sample_acquisition):
        """Test that result contains properly formatted dictionaries."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert isinstance(result.result[0], dict)
            # Verify it's a model dump (contains all fields)
            assert "acquisition_uuid" in result.result[0]
            assert "acquiree_uuid" in result.result[0]
            assert "acquirer_uuid" in result.result[0]
            assert "acquisition_type" in result.result[0]
            assert "acquisition_announce_date" in result.result[0]
            assert "acquisition_price_usd" in result.result[0]
            assert "terms" in result.result[0]
            assert "acquirer_type" in result.result[0]


class TestGetAcquisitionsFiltering:
    """Test get_acquisitions with various filters."""

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_uuid(
        self, mock_acquisition_model, sample_acquisition, sample_acquisition_uuid
    ):
        """Test getting acquisitions by UUID."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_uuid=str(sample_acquisition_uuid))

            assert result.success is True
            mock_acquisition_model.get.assert_called_once()
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquisition_uuid"] == sample_acquisition_uuid
            assert isinstance(call_kwargs["acquisition_uuid"], UUID)

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_acquiree_uuid(
        self, mock_acquisition_model, sample_acquisition, sample_acquiree_uuid
    ):
        """Test getting acquisitions by acquiree UUID."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquiree_uuid=str(sample_acquiree_uuid))

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquiree_uuid"] == sample_acquiree_uuid
            assert isinstance(call_kwargs["acquiree_uuid"], UUID)

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_acquirer_uuid(
        self, mock_acquisition_model, sample_acquisition, sample_acquirer_uuid
    ):
        """Test getting acquisitions by acquirer UUID."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquirer_uuid=str(sample_acquirer_uuid))

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquirer_uuid"] == sample_acquirer_uuid
            assert isinstance(call_kwargs["acquirer_uuid"], UUID)

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_type(self, mock_acquisition_model, sample_acquisition):
        """Test getting acquisitions by type."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_type="merger")

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquisition_type"] == "merger"

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_announce_date(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions by exact announce date."""
        announce_date_str = "2023-06-15T00:00:00"
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_announce_date=announce_date_str)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquisition_announce_date"] == datetime(2023, 6, 15)
            assert isinstance(call_kwargs["acquisition_announce_date"], datetime)

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_date_range(self, mock_acquisition_model, sample_acquisition):
        """Test getting acquisitions by date range."""
        date_from_str = "2023-01-01T00:00:00"
        date_to_str = "2023-12-31T23:59:59"
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(
                acquisition_announce_date_from=date_from_str,
                acquisition_announce_date_to=date_to_str,
            )

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquisition_announce_date_from"] == datetime(2023, 1, 1)
            assert call_kwargs["acquisition_announce_date_to"] == datetime(2023, 12, 31, 23, 59, 59)

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_price_exact(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions by exact price."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_price_usd=50000000)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquisition_price_usd"] == 50000000

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_price_range(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions by price range."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(
                acquisition_price_usd_min=1000000,
                acquisition_price_usd_max=100000000,
            )

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquisition_price_usd_min"] == 1000000
            assert call_kwargs["acquisition_price_usd_max"] == 100000000

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_terms_exact(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions by exact terms match."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(terms="Cash and stock")

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["terms"] == "Cash and stock"

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_terms_ilike(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions by case-insensitive terms search."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(terms_ilike="cash")

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["terms_ilike"] == "cash"

    @pytest.mark.asyncio
    async def test_get_acquisitions_by_acquirer_type(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions by acquirer type."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquirer_type="public_company")

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquirer_type"] == "public_company"

    @pytest.mark.asyncio
    async def test_get_acquisitions_with_multiple_filters(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions with multiple filters."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(
                acquisition_type="merger",
                acquisition_price_usd_min=1000000,
                acquirer_type="public_company",
            )

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["acquisition_type"] == "merger"
            assert call_kwargs["acquisition_price_usd_min"] == 1000000
            assert call_kwargs["acquirer_type"] == "public_company"


class TestGetAcquisitionsPagination:
    """Test get_acquisitions pagination functionality."""

    @pytest.mark.asyncio
    async def test_get_acquisitions_with_limit(self, mock_acquisition_model, sample_acquisition):
        """Test getting acquisitions with limit."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(limit=10)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_acquisitions_with_offset(self, mock_acquisition_model, sample_acquisition):
        """Test getting acquisitions with offset."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(offset=20)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["offset"] == 20

    @pytest.mark.asyncio
    async def test_get_acquisitions_with_limit_and_offset(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test getting acquisitions with both limit and offset."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(limit=10, offset=20)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs["limit"] == 10
            assert call_kwargs["offset"] == 20


class TestGetAcquisitionsParameterConversion:
    """Test get_acquisitions parameter conversion (UUIDs, dates)."""

    @pytest.mark.asyncio
    async def test_get_acquisitions_uuid_conversion(
        self, mock_acquisition_model, sample_acquisition, sample_acquisition_uuid
    ):
        """Test that string UUIDs are converted to UUID objects."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            uuid_str = str(sample_acquisition_uuid)
            result = await get_acquisitions(acquisition_uuid=uuid_str)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert isinstance(call_kwargs["acquisition_uuid"], UUID)
            assert call_kwargs["acquisition_uuid"] == sample_acquisition_uuid

    @pytest.mark.asyncio
    async def test_get_acquisitions_date_conversion(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that ISO date strings are converted to datetime objects."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            date_str = "2023-06-15T12:30:45"
            result = await get_acquisitions(acquisition_announce_date=date_str)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert isinstance(call_kwargs["acquisition_announce_date"], datetime)
            assert call_kwargs["acquisition_announce_date"] == datetime(2023, 6, 15, 12, 30, 45)

    @pytest.mark.asyncio
    async def test_get_acquisitions_date_range_conversion(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that date range strings are converted to datetime objects."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            date_from_str = "2023-01-01T00:00:00"
            date_to_str = "2023-12-31T23:59:59"
            result = await get_acquisitions(
                acquisition_announce_date_from=date_from_str,
                acquisition_announce_date_to=date_to_str,
            )

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert isinstance(call_kwargs["acquisition_announce_date_from"], datetime)
            assert isinstance(call_kwargs["acquisition_announce_date_to"], datetime)

    @pytest.mark.asyncio
    async def test_get_acquisitions_none_uuid_not_converted(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that None UUIDs are not converted."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_uuid=None)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs.get("acquisition_uuid") is None

    @pytest.mark.asyncio
    async def test_get_acquisitions_none_date_not_converted(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that None dates are not converted."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_announce_date=None)

            assert result.success is True
            call_kwargs = mock_acquisition_model.get.call_args[1]
            assert call_kwargs.get("acquisition_announce_date") is None


class TestGetAcquisitionsErrorHandling:
    """Test get_acquisitions error handling."""

    @pytest.mark.asyncio
    async def test_get_acquisitions_invalid_uuid(self, mock_acquisition_model):
        """Test that invalid UUID strings raise appropriate errors."""
        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_uuid="invalid-uuid")

            assert result.success is False
            assert result.error is not None
            assert "invalid" in result.error.lower() or "uuid" in result.error.lower()
            assert result.result is None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_get_acquisitions_invalid_date(self, mock_acquisition_model):
        """Test that invalid date strings raise appropriate errors."""
        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(acquisition_announce_date="invalid-date")

            assert result.success is False
            assert result.error is not None
            assert result.result is None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_get_acquisitions_model_initialization_error(self, mock_acquisition_model):
        """Test handling of model initialization errors."""
        mock_acquisition_model.initialize.side_effect = Exception("Database connection failed")

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert result.success is False
            assert result.error is not None
            assert "Failed to get acquisitions" in result.error
            assert result.result is None
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_get_acquisitions_model_get_error(self, mock_acquisition_model):
        """Test handling of model.get() errors."""
        mock_acquisition_model.get.side_effect = Exception("Query failed")

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert result.success is False
            assert result.error is not None
            assert "Failed to get acquisitions" in result.error
            assert "Query failed" in result.error
            assert result.result is None
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_get_acquisitions_error_execution_time(self, mock_acquisition_model):
        """Test that execution time is recorded even on errors."""
        mock_acquisition_model.get.side_effect = Exception("Query failed")

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert result.success is False
            assert result.execution_time_ms is not None
            assert result.execution_time_ms >= 0


class TestGetAcquisitionsEdgeCases:
    """Test get_acquisitions edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_get_acquisitions_all_optional_params_none(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that all optional parameters can be None."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions(
                acquisition_uuid=None,
                acquiree_uuid=None,
                acquirer_uuid=None,
                acquisition_type=None,
                acquisition_announce_date=None,
                acquisition_announce_date_from=None,
                acquisition_announce_date_to=None,
                acquisition_price_usd=None,
                acquisition_price_usd_min=None,
                acquisition_price_usd_max=None,
                terms=None,
                terms_ilike=None,
                acquirer_type=None,
                limit=None,
                offset=None,
            )

            assert result.success is True
            # Should call get() with no filters
            mock_acquisition_model.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_acquisitions_execution_time_recorded(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that execution time is properly recorded."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            assert result.execution_time_ms is not None
            assert isinstance(result.execution_time_ms, float)
            assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_get_acquisitions_tool_output_structure(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that ToolOutput structure is correct."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            # Verify ToolOutput structure
            assert hasattr(result, "success")
            assert hasattr(result, "result")
            assert hasattr(result, "error")
            assert hasattr(result, "tool_name")
            assert hasattr(result, "execution_time_ms")
            assert hasattr(result, "metadata")
            assert hasattr(result, "timestamp")

    @pytest.mark.asyncio
    async def test_get_acquisitions_model_initialized_once(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that model.initialize() is called exactly once."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            await get_acquisitions()

            assert mock_acquisition_model.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_get_acquisitions_model_get_called_once(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that model.get() is called exactly once per invocation."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            await get_acquisitions()
            await get_acquisitions()

            assert mock_acquisition_model.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_acquisitions_result_contains_model_dumps(
        self, mock_acquisition_model, sample_acquisition
    ):
        """Test that results are properly converted to dictionaries."""
        mock_acquisition_model.get.return_value = [sample_acquisition]

        get_acquisitions_module = sys.modules["src.tools.get_acquisitions"]
        with patch.object(
            get_acquisitions_module,
            "AcquisitionModel",
            return_value=mock_acquisition_model,
        ):
            result = await get_acquisitions()

            # Verify the result is a list of dicts (model_dump format)
            assert isinstance(result.result, list)
            assert len(result.result) == 1
            result_dict = result.result[0]

            # Verify it contains all expected fields from Acquisition model
            assert "acquisition_uuid" in result_dict
            assert "acquiree_uuid" in result_dict
            assert "acquirer_uuid" in result_dict
            assert "acquisition_type" in result_dict
            assert "acquisition_announce_date" in result_dict
            assert "acquisition_price_usd" in result_dict
            assert "terms" in result_dict
            assert "acquirer_type" in result_dict

            # Verify UUIDs are present in the dict
            # Note: model_dump() by default keeps UUIDs as UUID objects, not strings
            # To serialize to strings, would need model_dump(mode="json")
            assert isinstance(result_dict["acquisition_uuid"], UUID)
