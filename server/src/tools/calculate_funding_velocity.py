"""Tool for calculating funding velocity and momentum metrics.

This tool analyzes time-series funding data to calculate velocity metrics including
rate of change, acceleration, momentum indicators, moving averages, and trend direction.
It works with aggregated funding trend data to provide insights into funding momentum.
"""

import logging
import statistics
import time
from typing import Any, Dict, List, Optional

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src.contracts.tool_io import (
    ToolMetadata,
    ToolOutput,
    ToolParameterSchema,
    create_tool_output,
)

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the calculate_funding_velocity tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="calculate_funding_velocity",
        description="Calculate funding velocity and momentum metrics from time-series funding data, including rate of change, acceleration, moving averages, and trend direction.",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="trend_data",
                type="array",
                description="Time-series funding trend data (from aggregate_funding_trends or similar). Each item should have 'period', 'total_funding_usd', 'round_count', and optionally 'velocity_change_pct'. Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="moving_average_periods",
                type="integer",
                description="Number of periods to use for moving average calculations (e.g., 3 for 3-period MA). Default: 3",
                required=False,
                default=3,
            ),
            ToolParameterSchema(
                name="calculate_cagr",
                type="boolean",
                description="Whether to calculate Compound Annual Growth Rate (CAGR). Default: true",
                required=False,
                default=True,
            ),
        ],
        returns={
            "type": "object",
            "description": "Funding velocity metrics and momentum indicators",
            "properties": {
                "velocity_metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "string"},
                            "total_funding_usd": {"type": "integer"},
                            "velocity_change_pct": {"type": "number", "nullable": True},
                            "moving_average_usd": {"type": "number", "nullable": True},
                            "acceleration_pct": {"type": "number", "nullable": True},
                            "momentum_score": {"type": "number", "nullable": True},
                        },
                    },
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "overall_trend": {"type": "string"},
                        "average_velocity_pct": {"type": "number", "nullable": True},
                        "cagr_pct": {"type": "number", "nullable": True},
                        "momentum_direction": {"type": "string"},
                        "peak_period": {"type": "string", "nullable": True},
                        "trough_period": {"type": "string", "nullable": True},
                        "volatility": {"type": "number", "nullable": True},
                    },
                },
            },
        },
        cost_per_call=None,  # Pure computation, no external costs
        estimated_latency_ms=100.0,  # Fast computation
        timeout_seconds=30.0,
        side_effects=False,  # Read-only computation
        idempotent=True,  # Safe to retry
        tags=["calculation", "funding", "velocity", "momentum", "trends", "read-only"],
    )


def _calculate_moving_average(
    values: List[float], window: int, index: int
) -> Optional[float]:
    """Calculate moving average for a given index.

    Args:
        values: List of values to calculate moving average from.
        window: Window size for moving average.
        index: Current index in the values list.

    Returns:
        Moving average value or None if insufficient data.
    """
    if index < window - 1:
        return None

    window_values = values[max(0, index - window + 1) : index + 1]
    if not window_values:
        return None

    return statistics.mean(window_values)


def _calculate_acceleration(
    velocities: List[Optional[float]], index: int
) -> Optional[float]:
    """Calculate acceleration (change in velocity) for a given period.

    Args:
        velocities: List of velocity change percentages.
        index: Current index in the velocities list.

    Returns:
        Acceleration percentage or None if insufficient data.
    """
    if index < 1:
        return None

    current_velocity = velocities[index]
    previous_velocity = velocities[index - 1]

    if current_velocity is None or previous_velocity is None:
        return None

    return current_velocity - previous_velocity


def _calculate_momentum_score(
    funding_amounts: List[int],
    velocities: List[Optional[float]],
    index: int,
    window: int = 3,
) -> Optional[float]:
    """Calculate momentum score based on recent funding and velocity trends.

    Args:
        funding_amounts: List of funding amounts per period.
        velocities: List of velocity change percentages.
        index: Current index.
        window: Window size for momentum calculation.

    Returns:
        Momentum score (0-100 scale) or None if insufficient data.
    """
    if index < window - 1:
        return None

    # Get recent funding amounts and velocities
    recent_funding = funding_amounts[max(0, index - window + 1) : index + 1]
    recent_velocities = [
        v for v in velocities[max(0, index - window + 1) : index + 1] if v is not None
    ]

    if not recent_funding or not recent_velocities:
        return None

    # Calculate momentum based on:
    # 1. Recent funding trend (increasing = positive momentum)
    # 2. Velocity trend (positive velocity = positive momentum)
    # 3. Consistency (stable growth = higher momentum)

    # Funding trend component (0-40 points)
    funding_trend = 0.0
    if len(recent_funding) >= 2:
        funding_growth = (recent_funding[-1] - recent_funding[0]) / max(
            recent_funding[0], 1
        )
        funding_trend = min(40.0, max(0.0, funding_growth * 10))  # Scale to 0-40

    # Velocity component (0-40 points)
    velocity_trend = 0.0
    if recent_velocities:
        avg_velocity = statistics.mean(recent_velocities)
        velocity_trend = min(40.0, max(0.0, 20.0 + avg_velocity))  # Scale around 0

    # Consistency component (0-20 points)
    consistency = 0.0
    if len(recent_velocities) >= 2:
        # Lower variance = higher consistency
        velocity_variance = statistics.variance(recent_velocities) if len(recent_velocities) > 1 else 0
        consistency = max(0.0, 20.0 - min(20.0, velocity_variance / 100))

    momentum_score = funding_trend + velocity_trend + consistency
    return round(min(100.0, max(0.0, momentum_score)), 2)


def _calculate_cagr(
    initial_value: float, final_value: float, num_periods: int
) -> Optional[float]:
    """Calculate Compound Annual Growth Rate (CAGR).

    Args:
        initial_value: Starting funding amount.
        final_value: Ending funding amount.
        num_periods: Number of periods between initial and final.

    Returns:
        CAGR as a percentage or None if calculation not possible.
    """
    if initial_value <= 0 or num_periods <= 0:
        return None

    if final_value <= 0:
        return None

    # CAGR = (Final Value / Initial Value)^(1/Periods) - 1
    growth_factor = final_value / initial_value
    cagr = (growth_factor ** (1.0 / num_periods) - 1.0) * 100

    return round(cagr, 2)


def _determine_trend_direction(
    velocities: List[Optional[float]], window: int = 3
) -> str:
    """Determine overall trend direction from recent velocities.

    Args:
        velocities: List of velocity change percentages.
        window: Number of recent periods to consider.

    Returns:
        Trend direction: "increasing", "decreasing", "stable", or "insufficient_data".
    """
    recent_velocities = [
        v for v in velocities[-window:] if v is not None
    ]

    if len(recent_velocities) < 2:
        return "insufficient_data"

    avg_velocity = statistics.mean(recent_velocities)

    # Thresholds for trend determination
    if avg_velocity > 5.0:
        return "increasing"
    elif avg_velocity < -5.0:
        return "decreasing"
    else:
        return "stable"


@observe(as_type="tool")
async def calculate_funding_velocity(
    trend_data: List[Dict[str, Any]],
    moving_average_periods: int = 3,
    calculate_cagr: bool = True,
) -> ToolOutput:
    """Calculate funding velocity and momentum metrics from time-series funding data.

    This tool:
    1. Calculates moving averages for funding amounts
    2. Computes acceleration (change in velocity)
    3. Generates momentum scores
    4. Determines overall trend direction
    5. Calculates CAGR if requested

    Args:
        trend_data: List of period data dictionaries. Each should have:
            - 'period': Period identifier (string)
            - 'total_funding_usd': Total funding in USD (integer)
            - 'velocity_change_pct': Optional velocity change percentage
        moving_average_periods: Number of periods for moving average (default: 3).
        calculate_cagr: Whether to calculate CAGR (default: True).

    Returns:
        ToolOutput object containing:
        - success: Whether the calculation succeeded
        - result: Dictionary with:
            - velocity_metrics: List of period metrics with velocity calculations
            - summary: Overall summary statistics
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Calculate velocity metrics from aggregated trend data
        trend_data = [
            {
                "period": "2022-Q1",
                "total_funding_usd": 500000000,
                "velocity_change_pct": None
            },
            {
                "period": "2022-Q2",
                "total_funding_usd": 750000000,
                "velocity_change_pct": 50.0
            }
        ]
        result = await calculate_funding_velocity(
            trend_data=trend_data,
            moving_average_periods=3,
            calculate_cagr=True
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate inputs
        if not trend_data:
            raise ValueError("trend_data cannot be empty")

        if moving_average_periods < 1:
            raise ValueError(
                f"moving_average_periods must be >= 1. Got: {moving_average_periods}"
            )

        # Extract data from trend_data
        periods = []
        funding_amounts = []
        velocities = []

        for item in trend_data:
            period = item.get("period")
            total_funding = item.get("total_funding_usd", 0)
            velocity_change = item.get("velocity_change_pct")

            if period is None:
                logger.warning("Skipping item with missing period")
                continue

            periods.append(period)
            funding_amounts.append(int(total_funding))
            velocities.append(velocity_change if velocity_change is not None else None)

        if not periods:
            raise ValueError("No valid periods found in trend_data")

        # Calculate velocity metrics for each period
        velocity_metrics = []
        moving_averages = []
        accelerations = []
        momentum_scores = []

        for i in range(len(periods)):
            # Calculate moving average
            ma = _calculate_moving_average(funding_amounts, moving_average_periods, i)
            moving_averages.append(ma)

            # Calculate acceleration (if we have velocity data)
            acceleration = _calculate_acceleration(velocities, i)
            accelerations.append(acceleration)

            # Calculate momentum score
            momentum = _calculate_momentum_score(
                funding_amounts, velocities, i, window=moving_average_periods
            )
            momentum_scores.append(momentum)

            velocity_metrics.append(
                {
                    "period": periods[i],
                    "total_funding_usd": funding_amounts[i],
                    "velocity_change_pct": (
                        round(velocities[i], 2) if velocities[i] is not None else None
                    ),
                    "moving_average_usd": (
                        round(ma, 2) if ma is not None else None
                    ),
                    "acceleration_pct": (
                        round(acceleration, 2) if acceleration is not None else None
                    ),
                    "momentum_score": momentum,
                }
            )

        # Calculate summary statistics
        valid_velocities = [v for v in velocities if v is not None]
        avg_velocity = (
            round(statistics.mean(valid_velocities), 2) if valid_velocities else None
        )

        # Calculate CAGR if requested
        cagr_pct = None
        if calculate_cagr and len(funding_amounts) >= 2:
            initial_value = float(funding_amounts[0])
            final_value = float(funding_amounts[-1])
            num_periods = len(funding_amounts) - 1
            cagr_pct = _calculate_cagr(initial_value, final_value, num_periods)

        # Determine trend direction
        trend_direction = _determine_trend_direction(velocities, window=moving_average_periods)

        # Find peak and trough periods
        peak_period = None
        trough_period = None
        if funding_amounts:
            peak_index = funding_amounts.index(max(funding_amounts))
            trough_index = funding_amounts.index(min(funding_amounts))
            peak_period = periods[peak_index]
            trough_period = periods[trough_index]

        # Calculate volatility (coefficient of variation)
        volatility = None
        if len(funding_amounts) >= 2 and statistics.mean(funding_amounts) > 0:
            std_dev = statistics.stdev(funding_amounts)
            mean_funding = statistics.mean(funding_amounts)
            volatility = round((std_dev / mean_funding) * 100, 2)

        summary = {
            "overall_trend": trend_direction,
            "average_velocity_pct": avg_velocity,
            "cagr_pct": cagr_pct,
            "momentum_direction": trend_direction,  # Alias for consistency
            "peak_period": peak_period,
            "trough_period": trough_period,
            "volatility": volatility,
        }

        # Build result
        result = {
            "velocity_metrics": velocity_metrics,
            "summary": summary,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "num_periods": len(periods),
            "moving_average_periods": moving_average_periods,
            "cagr_calculated": calculate_cagr and cagr_pct is not None,
        }

        logger.debug(
            f"Calculated funding velocity metrics for {len(periods)} periods "
            f"in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="calculate_funding_velocity",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to calculate funding velocity: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="calculate_funding_velocity",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )

