"""Unit tests for the calculate_funding_velocity tool.

This module tests the calculate_funding_velocity function and its various
configurations including moving averages, acceleration, momentum scores,
CAGR calculations, and trend direction determination.
"""

import pytest

from src.contracts.tool_io import ToolOutput
from src.tools.calculate_funding_velocity import (
    calculate_funding_velocity,
    get_tool_metadata,
)


@pytest.fixture
def sample_trend_data():
    """Create sample trend data for testing."""
    return [
        {
            "period": "2022-Q1",
            "total_funding_usd": 10000000,
            "velocity_change_pct": None,
        },
        {
            "period": "2022-Q2",
            "total_funding_usd": 15000000,
            "velocity_change_pct": 50.0,
        },
        {
            "period": "2022-Q3",
            "total_funding_usd": 20000000,
            "velocity_change_pct": 33.33,
        },
        {
            "period": "2022-Q4",
            "total_funding_usd": 25000000,
            "velocity_change_pct": 25.0,
        },
    ]


@pytest.fixture
def sample_trend_data_with_decline():
    """Create sample trend data showing decline."""
    return [
        {
            "period": "2022-Q1",
            "total_funding_usd": 30000000,
            "velocity_change_pct": None,
        },
        {
            "period": "2022-Q2",
            "total_funding_usd": 25000000,
            "velocity_change_pct": -16.67,
        },
        {
            "period": "2022-Q3",
            "total_funding_usd": 20000000,
            "velocity_change_pct": -20.0,
        },
        {
            "period": "2022-Q4",
            "total_funding_usd": 15000000,
            "velocity_change_pct": -25.0,
        },
    ]


@pytest.fixture
def sample_trend_data_minimal():
    """Create minimal trend data (2 periods)."""
    return [
        {
            "period": "2022-Q1",
            "total_funding_usd": 10000000,
            "velocity_change_pct": None,
        },
        {
            "period": "2022-Q2",
            "total_funding_usd": 15000000,
            "velocity_change_pct": 50.0,
        },
    ]


class TestGetToolMetadata:
    """Test get_tool_metadata function."""

    def test_get_tool_metadata_structure(self):
        """Test that get_tool_metadata returns correct structure."""
        metadata = get_tool_metadata()

        assert metadata.name == "calculate_funding_velocity"
        assert metadata.description is not None
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.parameters, list)
        assert metadata.returns is not None
        assert metadata.cost_per_call is None
        assert metadata.estimated_latency_ms == 100.0
        assert metadata.timeout_seconds == 30.0
        assert metadata.side_effects is False
        assert metadata.idempotent is True
        assert "calculation" in metadata.tags
        assert "funding" in metadata.tags
        assert "velocity" in metadata.tags
        assert "momentum" in metadata.tags
        assert "trends" in metadata.tags
        assert "read-only" in metadata.tags

    def test_get_tool_metadata_parameters(self):
        """Test that get_tool_metadata includes all expected parameters."""
        metadata = get_tool_metadata()
        param_names = {param.name for param in metadata.parameters}

        expected_params = {
            "trend_data",
            "moving_average_periods",
            "calculate_cagr",
        }

        assert param_names == expected_params

    def test_get_tool_metadata_required_params(self):
        """Test that required parameters are marked correctly."""
        metadata = get_tool_metadata()
        required_params = {p.name for p in metadata.parameters if p.required}

        assert "trend_data" in required_params
        assert "moving_average_periods" not in required_params  # Has default
        assert "calculate_cagr" not in required_params  # Has default

    def test_get_tool_metadata_defaults(self):
        """Test that default values are correctly set."""
        metadata = get_tool_metadata()
        moving_avg_param = next(
            p for p in metadata.parameters if p.name == "moving_average_periods"
        )
        cagr_param = next(p for p in metadata.parameters if p.name == "calculate_cagr")

        assert moving_avg_param.default == 3
        assert cagr_param.default is True

    def test_get_tool_metadata_returns_schema(self):
        """Test that returns schema is correctly defined."""
        metadata = get_tool_metadata()
        assert metadata.returns["type"] == "object"
        assert "velocity_metrics" in metadata.returns["properties"]
        assert "summary" in metadata.returns["properties"]


class TestCalculateFundingVelocityBasic:
    """Test basic calculate_funding_velocity functionality."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_basic(self, sample_trend_data):
        """Test basic velocity calculation."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        assert isinstance(result, ToolOutput)
        assert result.success is True
        assert result.tool_name == "calculate_funding_velocity"
        assert result.result is not None
        assert isinstance(result.result, dict)
        assert result.error is None
        assert result.execution_time_ms is not None

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_result_structure(self, sample_trend_data):
        """Test that result has correct structure."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        assert "velocity_metrics" in result.result
        assert "summary" in result.result
        assert isinstance(result.result["velocity_metrics"], list)
        assert isinstance(result.result["summary"], dict)

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_velocity_metrics_structure(
        self, sample_trend_data
    ):
        """Test that velocity_metrics have correct structure."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        assert len(result.result["velocity_metrics"]) == len(sample_trend_data)
        for metric in result.result["velocity_metrics"]:
            assert "period" in metric
            assert "total_funding_usd" in metric
            assert "velocity_change_pct" in metric
            assert "moving_average_usd" in metric
            assert "acceleration_pct" in metric
            assert "momentum_score" in metric

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_summary_structure(self, sample_trend_data):
        """Test that summary has correct structure."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        summary = result.result["summary"]
        assert "overall_trend" in summary
        assert "average_velocity_pct" in summary
        assert "cagr_pct" in summary
        assert "momentum_direction" in summary
        assert "peak_period" in summary
        assert "trough_period" in summary
        assert "volatility" in summary


class TestCalculateFundingVelocityMovingAverage:
    """Test calculate_funding_velocity moving average calculations."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_moving_average_default(
        self, sample_trend_data
    ):
        """Test moving average with default window (3 periods)."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        metrics = result.result["velocity_metrics"]
        # First period should have None (insufficient data for 3-period MA)
        assert metrics[0]["moving_average_usd"] is None
        # Second period should have None (only 2 periods available)
        assert metrics[1]["moving_average_usd"] is None
        # Third period should have MA (3 periods: Q1, Q2, Q3)
        assert metrics[2]["moving_average_usd"] is not None
        assert metrics[2]["moving_average_usd"] == 15000000.0  # (10M + 15M + 20M) / 3
        # Fourth period should have MA (3 periods: Q2, Q3, Q4)
        assert metrics[3]["moving_average_usd"] is not None
        assert metrics[3]["moving_average_usd"] == 20000000.0  # (15M + 20M + 25M) / 3

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_moving_average_custom_window(
        self, sample_trend_data
    ):
        """Test moving average with custom window size."""
        result = await calculate_funding_velocity(
            trend_data=sample_trend_data, moving_average_periods=2
        )

        metrics = result.result["velocity_metrics"]
        # First period should have None (insufficient data for 2-period MA)
        assert metrics[0]["moving_average_usd"] is None
        # Second period should have MA (2 periods: Q1, Q2)
        assert metrics[1]["moving_average_usd"] is not None
        assert metrics[1]["moving_average_usd"] == 12500000.0  # (10M + 15M) / 2
        # Third period should have MA (2 periods: Q2, Q3)
        assert metrics[2]["moving_average_usd"] is not None
        assert metrics[2]["moving_average_usd"] == 17500000.0  # (15M + 20M) / 2

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_moving_average_window_1(
        self, sample_trend_data
    ):
        """Test moving average with window size of 1."""
        result = await calculate_funding_velocity(
            trend_data=sample_trend_data, moving_average_periods=1
        )

        metrics = result.result["velocity_metrics"]
        # All periods should have MA (window of 1)
        for i, metric in enumerate(metrics):
            assert metric["moving_average_usd"] is not None
            assert metric["moving_average_usd"] == sample_trend_data[i]["total_funding_usd"]


class TestCalculateFundingVelocityAcceleration:
    """Test calculate_funding_velocity acceleration calculations."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_acceleration(self, sample_trend_data):
        """Test acceleration calculation."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        metrics = result.result["velocity_metrics"]
        # First period should have None (no previous velocity)
        assert metrics[0]["acceleration_pct"] is None
        # Second period should have None (previous velocity is None)
        assert metrics[1]["acceleration_pct"] is None
        # Third period: acceleration = 33.33 - 50.0 = -16.67
        assert metrics[2]["acceleration_pct"] is not None
        assert metrics[2]["acceleration_pct"] == pytest.approx(-16.67, abs=0.01)
        # Fourth period: acceleration = 25.0 - 33.33 = -8.33
        assert metrics[3]["acceleration_pct"] is not None
        assert metrics[3]["acceleration_pct"] == pytest.approx(-8.33, abs=0.01)

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_acceleration_with_missing_velocities(
        self, sample_trend_data
    ):
        """Test acceleration when some velocities are missing."""
        # Create data with missing velocity in middle
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 15000000,
                "velocity_change_pct": 50.0,
            },
            {
                "period": "2022-Q3",
                "total_funding_usd": 20000000,
                "velocity_change_pct": None,  # Missing
            },
            {
                "period": "2022-Q4",
                "total_funding_usd": 25000000,
                "velocity_change_pct": 25.0,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        metrics = result.result["velocity_metrics"]
        # Q4 acceleration should be None (previous velocity is None)
        assert metrics[3]["acceleration_pct"] is None


class TestCalculateFundingVelocityMomentum:
    """Test calculate_funding_velocity momentum score calculations."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_momentum_scores(self, sample_trend_data):
        """Test momentum score calculation."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        metrics = result.result["velocity_metrics"]
        # First periods should have None (insufficient data for momentum)
        assert metrics[0]["momentum_score"] is None
        assert metrics[1]["momentum_score"] is None
        # Later periods should have momentum scores (0-100 range)
        assert metrics[2]["momentum_score"] is not None
        assert 0 <= metrics[2]["momentum_score"] <= 100
        assert metrics[3]["momentum_score"] is not None
        assert 0 <= metrics[3]["momentum_score"] <= 100

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_momentum_increasing_trend(
        self, sample_trend_data
    ):
        """Test momentum scores for increasing trend."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        metrics = result.result["velocity_metrics"]
        # With increasing funding and positive velocities, momentum should be positive
        if metrics[2]["momentum_score"] is not None:
            assert metrics[2]["momentum_score"] > 0

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_momentum_decreasing_trend(
        self, sample_trend_data_with_decline
    ):
        """Test momentum scores for decreasing trend."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data_with_decline)

        metrics = result.result["velocity_metrics"]
        # With decreasing funding and negative velocities, momentum should be lower
        # (but still in valid range)
        for metric in metrics:
            if metric["momentum_score"] is not None:
                assert 0 <= metric["momentum_score"] <= 100


class TestCalculateFundingVelocityCAGR:
    """Test calculate_funding_velocity CAGR calculations."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_cagr_enabled(self, sample_trend_data):
        """Test CAGR calculation when enabled."""
        result = await calculate_funding_velocity(
            trend_data=sample_trend_data, calculate_cagr=True
        )

        assert result.success is True
        summary = result.result["summary"]
        # Should have CAGR calculated
        assert summary["cagr_pct"] is not None
        assert isinstance(summary["cagr_pct"], float)
        # CAGR should be positive for increasing trend
        assert summary["cagr_pct"] > 0

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_cagr_disabled(self, sample_trend_data):
        """Test that CAGR is not calculated when disabled."""
        result = await calculate_funding_velocity(
            trend_data=sample_trend_data, calculate_cagr=False
        )

        assert result.success is True
        summary = result.result["summary"]
        assert summary["cagr_pct"] is None

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_cagr_default(self, sample_trend_data):
        """Test that CAGR is calculated by default."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        assert result.success is True
        summary = result.result["summary"]
        assert summary["cagr_pct"] is not None

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_cagr_single_period(self):
        """Test that CAGR is None for single period."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            }
        ]

        result = await calculate_funding_velocity(trend_data=trend_data, calculate_cagr=True)

        assert result.success is True
        summary = result.result["summary"]
        assert summary["cagr_pct"] is None  # Need at least 2 periods

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_cagr_zero_initial(self):
        """Test that CAGR handles zero initial value."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 0,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data, calculate_cagr=True)

        assert result.success is True
        summary = result.result["summary"]
        assert summary["cagr_pct"] is None  # Cannot calculate with zero initial


class TestCalculateFundingVelocityTrendDirection:
    """Test calculate_funding_velocity trend direction determination."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_trend_increasing(self, sample_trend_data):
        """Test trend direction for increasing trend."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        summary = result.result["summary"]
        assert summary["overall_trend"] in ["increasing", "stable", "insufficient_data"]
        assert summary["momentum_direction"] == summary["overall_trend"]

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_trend_decreasing(
        self, sample_trend_data_with_decline
    ):
        """Test trend direction for decreasing trend."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data_with_decline)

        summary = result.result["summary"]
        assert summary["overall_trend"] in ["decreasing", "stable", "insufficient_data"]
        assert summary["momentum_direction"] == summary["overall_trend"]

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_trend_insufficient_data(self):
        """Test trend direction with insufficient data."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            }
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        summary = result.result["summary"]
        assert summary["overall_trend"] == "insufficient_data"


class TestCalculateFundingVelocityPeakTrough:
    """Test calculate_funding_velocity peak and trough identification."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_peak_period(self, sample_trend_data):
        """Test that peak period is identified correctly."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        summary = result.result["summary"]
        assert summary["peak_period"] is not None
        assert summary["peak_period"] == "2022-Q4"  # Highest funding

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_trough_period(self, sample_trend_data):
        """Test that trough period is identified correctly."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        summary = result.result["summary"]
        assert summary["trough_period"] is not None
        assert summary["trough_period"] == "2022-Q1"  # Lowest funding

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_peak_trough_same(self):
        """Test peak and trough when all values are the same."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 10000000,
                "velocity_change_pct": 0.0,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        summary = result.result["summary"]
        # Peak and trough should be set (first occurrence)
        assert summary["peak_period"] is not None
        assert summary["trough_period"] is not None


class TestCalculateFundingVelocityVolatility:
    """Test calculate_funding_velocity volatility calculations."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_volatility(self, sample_trend_data):
        """Test volatility calculation."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        summary = result.result["summary"]
        assert summary["volatility"] is not None
        assert isinstance(summary["volatility"], float)
        assert summary["volatility"] >= 0

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_volatility_single_period(self):
        """Test volatility with single period (should be None)."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            }
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        summary = result.result["summary"]
        assert summary["volatility"] is None  # Need at least 2 periods

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_volatility_zero_mean(self):
        """Test volatility with zero mean (should be None)."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 0,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 0,
                "velocity_change_pct": 0.0,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        summary = result.result["summary"]
        assert summary["volatility"] is None  # Cannot calculate with zero mean


class TestCalculateFundingVelocityAverageVelocity:
    """Test calculate_funding_velocity average velocity calculations."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_average_velocity(self, sample_trend_data):
        """Test average velocity calculation."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        summary = result.result["summary"]
        assert summary["average_velocity_pct"] is not None
        # Average of 50.0, 33.33, 25.0 = 36.11
        assert summary["average_velocity_pct"] == pytest.approx(36.11, abs=0.1)

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_average_velocity_all_none(self):
        """Test average velocity when all velocities are None."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 15000000,
                "velocity_change_pct": None,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        summary = result.result["summary"]
        assert summary["average_velocity_pct"] is None  # No valid velocities

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_average_velocity_some_none(self):
        """Test average velocity when some velocities are None."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 15000000,
                "velocity_change_pct": 50.0,
            },
            {
                "period": "2022-Q3",
                "total_funding_usd": 20000000,
                "velocity_change_pct": None,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        summary = result.result["summary"]
        # Should average only valid velocities (50.0)
        assert summary["average_velocity_pct"] == 50.0


class TestCalculateFundingVelocityEdgeCases:
    """Test calculate_funding_velocity edge cases."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_missing_period_field(self):
        """Test handling of missing period field."""
        trend_data = [
            {
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
                # Missing "period" field
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 15000000,
                "velocity_change_pct": 50.0,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        assert result.success is True
        # Should skip item with missing period
        assert len(result.result["velocity_metrics"]) == 1
        assert result.result["velocity_metrics"][0]["period"] == "2022-Q2"

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_missing_funding_field(self):
        """Test handling of missing total_funding_usd field."""
        trend_data = [
            {
                "period": "2022-Q1",
                "velocity_change_pct": None,
                # Missing "total_funding_usd" field
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 15000000,
                "velocity_change_pct": 50.0,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        assert result.success is True
        # Should use default value of 0 for missing funding
        assert result.result["velocity_metrics"][0]["total_funding_usd"] == 0

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_all_periods_missing(self):
        """Test handling when all periods are missing."""
        trend_data = [
            {
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
                # Missing "period" field
            },
            {
                "total_funding_usd": 15000000,
                "velocity_change_pct": 50.0,
                # Missing "period" field
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        assert result.success is False
        assert "No valid periods found" in result.error
        assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_execution_time_recorded(
        self, sample_trend_data
    ):
        """Test that execution time is properly recorded."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        assert result.execution_time_ms is not None
        assert isinstance(result.execution_time_ms, float)
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_metadata_structure(
        self, sample_trend_data
    ):
        """Test that metadata has correct structure."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        assert "num_periods" in result.metadata
        assert "moving_average_periods" in result.metadata
        assert "cagr_calculated" in result.metadata
        assert result.metadata["num_periods"] == len(sample_trend_data)
        assert result.metadata["moving_average_periods"] == 3  # Default

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_tool_output_structure(
        self, sample_trend_data
    ):
        """Test that ToolOutput structure is correct."""
        result = await calculate_funding_velocity(trend_data=sample_trend_data)

        # Verify ToolOutput structure
        assert hasattr(result, "success")
        assert hasattr(result, "result")
        assert hasattr(result, "error")
        assert hasattr(result, "tool_name")
        assert hasattr(result, "execution_time_ms")
        assert hasattr(result, "metadata")
        assert hasattr(result, "timestamp")


class TestCalculateFundingVelocityErrorHandling:
    """Test calculate_funding_velocity error handling."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_empty_trend_data(self):
        """Test that empty trend_data raises error."""
        result = await calculate_funding_velocity(trend_data=[])

        assert result.success is False
        assert result.error is not None
        assert "trend_data cannot be empty" in result.error
        assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_invalid_moving_average_periods(self):
        """Test that invalid moving_average_periods raises error."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            }
        ]

        result = await calculate_funding_velocity(
            trend_data=trend_data, moving_average_periods=0
        )

        assert result.success is False
        assert result.error is not None
        assert "moving_average_periods" in result.error.lower()
        assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_negative_moving_average_periods(self):
        """Test that negative moving_average_periods raises error."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            }
        ]

        result = await calculate_funding_velocity(
            trend_data=trend_data, moving_average_periods=-1
        )

        assert result.success is False
        assert result.error is not None
        assert "moving_average_periods" in result.error.lower()
        assert result.metadata["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_error_execution_time(self):
        """Test that execution time is recorded even on errors."""
        result = await calculate_funding_velocity(trend_data=[])

        assert result.success is False
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0


class TestCalculateFundingVelocityComplexScenarios:
    """Test calculate_funding_velocity with complex scenarios."""

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_large_dataset(self):
        """Test with larger dataset (many periods)."""
        trend_data = []
        for i in range(12):  # 12 quarters (3 years)
            trend_data.append(
                {
                    "period": f"2022-Q{(i % 4) + 1}",
                    "total_funding_usd": 10000000 + (i * 1000000),
                    "velocity_change_pct": 10.0 if i > 0 else None,
                }
            )

        result = await calculate_funding_velocity(trend_data=trend_data)

        assert result.success is True
        assert len(result.result["velocity_metrics"]) == 12
        assert result.metadata["num_periods"] == 12

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_mixed_velocities(self):
        """Test with mixed positive and negative velocities."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 10000000,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 15000000,
                "velocity_change_pct": 50.0,  # Positive
            },
            {
                "period": "2022-Q3",
                "total_funding_usd": 12000000,
                "velocity_change_pct": -20.0,  # Negative
            },
            {
                "period": "2022-Q4",
                "total_funding_usd": 18000000,
                "velocity_change_pct": 50.0,  # Positive again
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        assert result.success is True
        # Average velocity should account for both positive and negative
        summary = result.result["summary"]
        if summary["average_velocity_pct"] is not None:
            # Average of 50.0, -20.0, 50.0 = 26.67
            assert summary["average_velocity_pct"] == pytest.approx(26.67, abs=0.1)

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_zero_funding_amounts(self):
        """Test with zero funding amounts."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 0,
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 0,
                "velocity_change_pct": 0.0,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        assert result.success is True
        # Should handle zero amounts gracefully
        assert result.result["velocity_metrics"][0]["total_funding_usd"] == 0
        assert result.result["velocity_metrics"][1]["total_funding_usd"] == 0

    @pytest.mark.asyncio
    async def test_calculate_funding_velocity_very_large_amounts(self):
        """Test with very large funding amounts."""
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 1000000000000,  # 1 trillion
                "velocity_change_pct": None,
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 1500000000000,  # 1.5 trillion
                "velocity_change_pct": 50.0,
            },
        ]

        result = await calculate_funding_velocity(trend_data=trend_data)

        assert result.success is True
        assert result.result["velocity_metrics"][0]["total_funding_usd"] == 1000000000000
        assert result.result["velocity_metrics"][1]["total_funding_usd"] == 1500000000000

