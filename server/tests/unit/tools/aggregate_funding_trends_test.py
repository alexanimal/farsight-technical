"""Unit tests for the aggregate_funding_trends tool.

This module tests the aggregate_funding_trends function and its various
configurations including time period aggregation, granularity options,
batch processing, and calculation of aggregate metrics.
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.contracts.tool_io import ToolOutput
from src.models.funding_rounds import FundingRound, FundingRoundModel
from src.tools.aggregate_funding_trends import aggregate_funding_trends, get_tool_metadata


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_org_uuids():
    """Create a list of sample organization UUIDs for testing."""
    return [uuid4(), uuid4(), uuid4()]


@pytest.fixture
def sample_funding_round(sample_org_uuid):
    """Create a sample FundingRound Pydantic model instance."""
    return FundingRound(
        funding_round_uuid=uuid4(),
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

        assert metadata.name == "aggregate_funding_trends"
        assert metadata.description is not None
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.parameters, list)
        assert metadata.returns is not None
        assert metadata.cost_per_call is None
        assert metadata.estimated_latency_ms == 500.0
        assert metadata.timeout_seconds == 60.0
        assert metadata.side_effects is False
        assert metadata.idempotent is True
        assert "aggregation" in metadata.tags
        assert "funding" in metadata.tags
        assert "trends" in metadata.tags
        assert "time-series" in metadata.tags
        assert "read-only" in metadata.tags

    def test_get_tool_metadata_parameters(self):
        """Test that get_tool_metadata includes all expected parameters."""
        metadata = get_tool_metadata()
        param_names = {param.name for param in metadata.parameters}

        expected_params = {
            "org_uuids",
            "time_period_start",
            "time_period_end",
            "granularity",
            "min_funding_amount",
        }

        assert param_names == expected_params

    def test_get_tool_metadata_required_params(self):
        """Test that required parameters are marked correctly."""
        metadata = get_tool_metadata()
        required_params = {p.name for p in metadata.parameters if p.required}

        assert "org_uuids" in required_params
        assert "time_period_start" in required_params
        assert "time_period_end" in required_params
        assert "granularity" not in required_params  # Has default
        assert "min_funding_amount" not in required_params

    def test_get_tool_metadata_granularity_enum(self):
        """Test that granularity parameter has correct enum values."""
        metadata = get_tool_metadata()
        granularity_param = next(p for p in metadata.parameters if p.name == "granularity")

        assert granularity_param.enum == ["monthly", "quarterly", "yearly"]
        assert granularity_param.default == "quarterly"

    def test_get_tool_metadata_returns_schema(self):
        """Test that returns schema is correctly defined."""
        metadata = get_tool_metadata()
        assert metadata.returns["type"] == "object"
        assert "time_period" in metadata.returns["properties"]
        assert "granularity" in metadata.returns["properties"]
        assert "trend_data" in metadata.returns["properties"]
        assert "summary" in metadata.returns["properties"]


class TestAggregateFundingTrendsBasic:
    """Test basic aggregate_funding_trends functionality."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_basic(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test basic aggregation with single round."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert isinstance(result, ToolOutput)
            assert result.success is True
            assert result.tool_name == "aggregate_funding_trends"
            assert result.result is not None
            assert isinstance(result.result, dict)
            assert result.error is None
            assert result.execution_time_ms is not None

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_result_structure(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that result has correct structure."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert "time_period" in result.result
            assert "granularity" in result.result
            assert "trend_data" in result.result
            assert "summary" in result.result
            assert result.result["time_period"]["start"] == "2023-01-01T00:00:00"
            assert result.result["time_period"]["end"] == "2023-12-31T23:59:59"
            assert result.result["granularity"] == "quarterly"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_empty_results(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test aggregation when no funding rounds are found."""
        mock_funding_round_model.get.return_value = []

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            assert result.result["trend_data"] == []
            assert result.result["summary"]["total_funding_usd"] == 0
            assert result.result["summary"]["total_rounds"] == 0
            assert result.result["summary"]["num_periods"] == 0


class TestAggregateFundingTrendsGranularity:
    """Test aggregate_funding_trends with different granularities."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_quarterly(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test aggregation with quarterly granularity."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            assert result.result["granularity"] == "quarterly"
            assert len(result.result["trend_data"]) > 0
            # Check period format (should be like "2023-Q2")
            period = result.result["trend_data"][0]["period"]
            assert "Q" in period or "2023" in period

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_monthly(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test aggregation with monthly granularity."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="monthly",
            )

            assert result.success is True
            assert result.result["granularity"] == "monthly"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_yearly(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test aggregation with yearly granularity."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="yearly",
            )

            assert result.success is True
            assert result.result["granularity"] == "yearly"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_default_granularity(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that default granularity is quarterly."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
            )

            assert result.success is True
            assert result.result["granularity"] == "quarterly"


class TestAggregateFundingTrendsCalculations:
    """Test aggregate_funding_trends calculation logic."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_calculates_metrics(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that aggregation calculates correct metrics."""
        # Create multiple rounds in different periods
        round1 = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 3, 15),  # Q1
            org_uuid=sample_org_uuids[0],
            general_funding_stage="Series A",
            stage="Series A",
            investors=["Sequoia Capital"],
            lead_investors=["Sequoia Capital"],
            fundraise_amount_usd=10000000,
            valuation_usd=50000000,
        )
        round2 = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 6, 20),  # Q2
            org_uuid=sample_org_uuids[0],
            general_funding_stage="Series B",
            stage="Series B",
            investors=["Accel Partners", "Sequoia Capital"],
            lead_investors=["Accel Partners"],
            fundraise_amount_usd=20000000,
            valuation_usd=100000000,
        )

        mock_funding_round_model.get.return_value = [round1, round2]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            assert len(result.result["trend_data"]) >= 2  # At least Q1 and Q2

            # Check summary totals
            summary = result.result["summary"]
            assert summary["total_funding_usd"] == 30000000  # 10M + 20M
            assert summary["total_rounds"] == 2
            assert summary["total_unique_investors"] == 2  # Sequoia, Accel

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_velocity_change(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that velocity change is calculated correctly."""
        round1 = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 3, 15),  # Q1
            org_uuid=sample_org_uuids[0],
            fundraise_amount_usd=10000000,
        )
        round2 = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 6, 20),  # Q2
            org_uuid=sample_org_uuids[0],
            fundraise_amount_usd=20000000,  # 100% increase
        )

        mock_funding_round_model.get.return_value = [round1, round2]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            trend_data = result.result["trend_data"]
            # Find Q2 period
            q2_data = next(
                (d for d in trend_data if "Q2" in d["period"] or "2023-06" in d["period"]),
                None,
            )
            if q2_data:
                # Should have velocity change calculated
                assert q2_data["velocity_change_pct"] is not None
                assert q2_data["velocity_change_pct"] == 100.0  # 100% increase

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_averages_and_medians(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that averages and medians are calculated correctly."""
        rounds = [
            FundingRound(
                funding_round_uuid=uuid4(),
                investment_date=datetime(2023, 6, 15),
                org_uuid=sample_org_uuids[0],
                fundraise_amount_usd=10000000,
            ),
            FundingRound(
                funding_round_uuid=uuid4(),
                investment_date=datetime(2023, 6, 20),
                org_uuid=sample_org_uuids[0],
                fundraise_amount_usd=20000000,
            ),
            FundingRound(
                funding_round_uuid=uuid4(),
                investment_date=datetime(2023, 6, 25),
                org_uuid=sample_org_uuids[0],
                fundraise_amount_usd=30000000,
            ),
        ]

        mock_funding_round_model.get.return_value = rounds

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            trend_data = result.result["trend_data"]
            if trend_data:
                period_data = trend_data[0]
                assert period_data["round_count"] == 3
                assert period_data["avg_round_size_usd"] == 20000000.0  # (10M + 20M + 30M) / 3
                assert period_data["median_round_size_usd"] == 20000000.0  # Middle value


class TestAggregateFundingTrendsBatchProcessing:
    """Test aggregate_funding_trends batch processing and chunking."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_batch_processing(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that batch processing works for multiple organizations."""
        # Create many org UUIDs to trigger chunking
        many_org_uuids = [uuid4() for _ in range(5)]
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in many_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            # Verify model.get was called (may be called multiple times for chunks)
            assert mock_funding_round_model.get.call_count >= 1

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_chunking_large_list(
        self, mock_funding_round_model, sample_funding_round
    ):
        """Test that large org UUID lists are processed in chunks."""
        # Create more than CHUNK_SIZE (1000) org UUIDs
        many_org_uuids = [uuid4() for _ in range(1500)]
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in many_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            # Should have called get() multiple times (at least 2 chunks: 1000 + 500)
            assert mock_funding_round_model.get.call_count >= 2
            assert result.metadata["num_batch_queries"] >= 2


class TestAggregateFundingTrendsFiltering:
    """Test aggregate_funding_trends filtering options."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_with_min_funding_amount(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that min_funding_amount filter is applied."""
        round1 = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 6, 15),
            org_uuid=sample_org_uuids[0],
            fundraise_amount_usd=500000,  # Below threshold
        )
        round2 = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 6, 20),
            org_uuid=sample_org_uuids[0],
            fundraise_amount_usd=2000000,  # Above threshold
        )

        mock_funding_round_model.get.return_value = [round2]  # Only round2 should be returned

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
                min_funding_amount=1000000,
            )

            assert result.success is True
            # Verify min_funding_amount was passed to model.get
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["fundraise_amount_usd_min"] == 1000000

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_date_range_filtering(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that date range filtering is applied."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            # Verify date range was passed to model.get
            call_kwargs = mock_funding_round_model.get.call_args[1]
            assert call_kwargs["investment_date_from"] == datetime(2023, 1, 1)
            assert call_kwargs["investment_date_to"] == datetime(2023, 12, 31, 23, 59, 59)


class TestAggregateFundingTrendsEdgeCases:
    """Test aggregate_funding_trends edge cases."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_rounds_without_dates(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that rounds without investment_date are skipped."""
        round_with_date = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 6, 15),
            org_uuid=sample_org_uuids[0],
            fundraise_amount_usd=10000000,
        )
        round_without_date = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=None,  # Missing date
            org_uuid=sample_org_uuids[0],
            fundraise_amount_usd=20000000,
        )

        mock_funding_round_model.get.return_value = [
            round_with_date,
            round_without_date,
        ]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            # total_rounds counts all rounds from query (including skipped ones)
            # but only rounds with dates are aggregated into trend_data
            assert result.result["summary"]["total_rounds"] == 2
            # Only round_with_date should be included in funding calculations
            assert result.result["summary"]["total_funding_usd"] == 10000000
            # Verify only 1 round appears in trend_data (the one with date)
            total_rounds_in_trends = sum(
                period["round_count"] for period in result.result["trend_data"]
            )
            assert total_rounds_in_trends == 1

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_rounds_without_amounts(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that rounds without funding amounts are handled correctly."""
        round_without_amount = FundingRound(
            funding_round_uuid=uuid4(),
            investment_date=datetime(2023, 6, 15),
            org_uuid=sample_org_uuids[0],
            fundraise_amount_usd=None,  # Missing amount
            investors=["Sequoia Capital"],
        )

        mock_funding_round_model.get.return_value = [round_without_amount]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            # Round should be counted but funding should be 0
            assert result.result["summary"]["total_rounds"] == 1
            assert result.result["summary"]["total_funding_usd"] == 0
            # Investors should still be counted
            assert result.result["summary"]["total_unique_investors"] == 1

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_multiple_periods(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test aggregation across multiple time periods."""
        rounds = [
            FundingRound(
                funding_round_uuid=uuid4(),
                investment_date=datetime(2023, 3, 15),  # Q1
                org_uuid=sample_org_uuids[0],
                fundraise_amount_usd=10000000,
            ),
            FundingRound(
                funding_round_uuid=uuid4(),
                investment_date=datetime(2023, 6, 20),  # Q2
                org_uuid=sample_org_uuids[0],
                fundraise_amount_usd=20000000,
            ),
            FundingRound(
                funding_round_uuid=uuid4(),
                investment_date=datetime(2023, 9, 10),  # Q3
                org_uuid=sample_org_uuids[0],
                fundraise_amount_usd=15000000,
            ),
        ]

        mock_funding_round_model.get.return_value = rounds

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            # Should have multiple periods
            assert len(result.result["trend_data"]) >= 3
            assert result.result["summary"]["num_periods"] >= 3


class TestAggregateFundingTrendsErrorHandling:
    """Test aggregate_funding_trends error handling."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_empty_org_uuids(self, mock_funding_round_model):
        """Test that empty org_uuids list raises error."""
        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is False
            assert result.error is not None
            assert "org_uuids cannot be empty" in result.error
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_invalid_granularity(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that invalid granularity raises error."""
        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="invalid",
            )

            assert result.success is False
            assert result.error is not None
            assert "granularity" in result.error.lower()
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_invalid_date_range(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that invalid date range (start >= end) raises error."""
        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-12-31T23:59:59",
                time_period_end="2023-01-01T00:00:00",  # End before start
                granularity="quarterly",
            )

            assert result.success is False
            assert result.error is not None
            assert "time_period_start" in result.error or "before" in result.error.lower()
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_invalid_date_format(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that invalid date format raises error."""
        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="invalid-date",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is False
            assert result.error is not None
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_invalid_uuid(self, mock_funding_round_model):
        """Test that invalid UUID format raises error."""
        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=["invalid-uuid"],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is False
            assert result.error is not None
            assert "invalid" in result.error.lower() or "uuid" in result.error.lower()
            assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_model_initialization_error(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test handling of model initialization errors."""
        mock_funding_round_model.initialize.side_effect = Exception("Database connection failed")

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is False
            assert result.error is not None
            assert "Failed to aggregate funding trends" in result.error
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_model_get_error(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test handling of model.get() errors."""
        mock_funding_round_model.get.side_effect = Exception("Query failed")

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is False
            assert result.error is not None
            assert "Failed to aggregate funding trends" in result.error
            assert "Query failed" in result.error
            assert result.metadata["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_error_execution_time(
        self, mock_funding_round_model, sample_org_uuids
    ):
        """Test that execution time is recorded even on errors."""
        mock_funding_round_model.get.side_effect = Exception("Query failed")

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is False
            assert result.execution_time_ms is not None
            assert result.execution_time_ms >= 0


class TestAggregateFundingTrendsMetadata:
    """Test aggregate_funding_trends metadata and result structure."""

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_metadata_structure(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that metadata contains expected information."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            assert "num_orgs" in result.metadata
            assert "num_rounds" in result.metadata
            assert "num_periods" in result.metadata
            assert "num_batch_queries" in result.metadata
            assert result.metadata["num_orgs"] == len(sample_org_uuids)
            assert result.metadata["num_rounds"] == 1

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_tool_output_structure(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that ToolOutput structure is correct."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            # Verify ToolOutput structure
            assert hasattr(result, "success")
            assert hasattr(result, "result")
            assert hasattr(result, "error")
            assert hasattr(result, "tool_name")
            assert hasattr(result, "execution_time_ms")
            assert hasattr(result, "metadata")
            assert hasattr(result, "timestamp")

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_execution_time_recorded(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that execution time is properly recorded."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.execution_time_ms is not None
            assert isinstance(result.execution_time_ms, float)
            assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_trend_data_structure(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that trend_data has correct structure."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            trend_data = result.result["trend_data"]
            if trend_data:
                period_data = trend_data[0]
                assert "period" in period_data
                assert "period_key" in period_data
                assert "total_funding_usd" in period_data
                assert "round_count" in period_data
                assert "avg_round_size_usd" in period_data
                assert "median_round_size_usd" in period_data
                assert "unique_investors" in period_data
                assert "velocity_change_pct" in period_data

    @pytest.mark.asyncio
    async def test_aggregate_funding_trends_summary_structure(
        self, mock_funding_round_model, sample_funding_round, sample_org_uuids
    ):
        """Test that summary has correct structure."""
        mock_funding_round_model.get.return_value = [sample_funding_round]

        aggregate_funding_trends_module = sys.modules["src.tools.aggregate_funding_trends"]
        with patch.object(
            aggregate_funding_trends_module,
            "FundingRoundModel",
            return_value=mock_funding_round_model,
        ):
            result = await aggregate_funding_trends(
                org_uuids=[str(uuid) for uuid in sample_org_uuids],
                time_period_start="2023-01-01T00:00:00",
                time_period_end="2023-12-31T23:59:59",
                granularity="quarterly",
            )

            assert result.success is True
            summary = result.result["summary"]
            assert "total_funding_usd" in summary
            assert "total_rounds" in summary
            assert "avg_round_size_usd" in summary
            assert "median_round_size_usd" in summary
            assert "total_unique_investors" in summary
            assert "num_periods" in summary
