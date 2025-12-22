"""Tool for identifying patterns in funding data.

This tool analyzes aggregated funding trend data to identify patterns including
peak periods, troughs, unusual spikes, cyclical patterns, and seasonality.
It uses statistical methods to detect anomalies and recurring patterns.
"""

import logging
import statistics
import time
from collections import defaultdict
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
    """Get the ToolMetadata for the identify_funding_patterns tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="identify_funding_patterns",
        description="Identify patterns in aggregated funding data including peak periods, troughs, unusual spikes, cyclical patterns, and seasonality using statistical analysis.",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="trend_data",
                type="array",
                description="Aggregated funding trend data (from aggregate_funding_trends). Each item should have 'period', 'total_funding_usd', 'round_count', and optionally 'velocity_change_pct'. Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="granularity",
                type="string",
                description="Time granularity of the data: 'monthly', 'quarterly', or 'yearly'. Used for seasonality detection. Required.",
                required=True,
                enum=["monthly", "quarterly", "yearly"],
            ),
            ToolParameterSchema(
                name="anomaly_threshold",
                type="number",
                description="Z-score threshold for detecting anomalies (default: 2.0). Values beyond this threshold are considered unusual spikes or drops.",
                required=False,
                default=2.0,
            ),
            ToolParameterSchema(
                name="detect_seasonality",
                type="boolean",
                description="Whether to detect seasonal patterns (e.g., Q4 spikes). Default: true",
                required=False,
                default=True,
            ),
            ToolParameterSchema(
                name="min_periods_for_cycles",
                type="integer",
                description="Minimum number of periods required to detect cyclical patterns. Default: 8",
                required=False,
                default=8,
            ),
        ],
        returns={
            "type": "object",
            "description": "Pattern insights and identified anomalies",
            "properties": {
                "peaks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "string"},
                            "total_funding_usd": {"type": "integer"},
                            "rank": {"type": "integer"},
                        },
                    },
                },
                "troughs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "string"},
                            "total_funding_usd": {"type": "integer"},
                            "rank": {"type": "integer"},
                        },
                    },
                },
                "anomalies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "string"},
                            "total_funding_usd": {"type": "integer"},
                            "z_score": {"type": "number"},
                            "anomaly_type": {"type": "string"},
                            "deviation_pct": {"type": "number"},
                        },
                    },
                },
                "seasonal_patterns": {
                    "type": "object",
                    "properties": {
                        "detected": {"type": "boolean"},
                        "strongest_season": {"type": "string", "nullable": True},
                        "weakest_season": {"type": "string", "nullable": True},
                        "seasonal_variation_pct": {"type": "number", "nullable": True},
                    },
                },
                "cyclical_patterns": {
                    "type": "object",
                    "properties": {
                        "detected": {"type": "boolean"},
                        "cycle_length_periods": {"type": "integer", "nullable": True},
                        "cycle_strength": {"type": "number", "nullable": True},
                        "description": {"type": "string", "nullable": True},
                    },
                },
                "trend_patterns": {
                    "type": "object",
                    "properties": {
                        "overall_direction": {"type": "string"},
                        "growth_consistency": {"type": "string"},
                        "volatility_level": {"type": "string"},
                    },
                },
                "insights": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        cost_per_call=None,  # Pure computation, no external costs
        estimated_latency_ms=200.0,  # Statistical analysis latency
        timeout_seconds=30.0,
        side_effects=False,  # Read-only computation
        idempotent=True,  # Safe to retry
        tags=["analysis", "funding", "patterns", "anomalies", "seasonality", "read-only"],
    )


def _calculate_z_score(value: float, mean: float, std_dev: float) -> Optional[float]:
    """Calculate z-score for a value.

    Args:
        value: The value to calculate z-score for.
        mean: Mean of the distribution.
        std_dev: Standard deviation of the distribution.

    Returns:
        Z-score or None if std_dev is 0.
    """
    if std_dev == 0:
        return None
    return (value - mean) / std_dev


def _detect_anomalies(
    trend_data: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """Detect anomalies in funding data using z-scores.

    Args:
        trend_data: List of period data dictionaries.
        threshold: Z-score threshold for anomaly detection.

    Returns:
        List of anomaly dictionaries with period, funding, z-score, and type.
    """
    if len(trend_data) < 3:
        return []

    # Extract funding amounts
    funding_amounts = [
        item.get("total_funding_usd", 0) for item in trend_data
    ]

    if not funding_amounts:
        return []

    # Calculate mean and standard deviation
    mean_funding = statistics.mean(funding_amounts)
    std_dev = statistics.stdev(funding_amounts) if len(funding_amounts) > 1 else 0

    if std_dev == 0:
        return []

    anomalies = []
    for i, item in enumerate(trend_data):
        funding = item.get("total_funding_usd", 0)
        z_score = _calculate_z_score(float(funding), mean_funding, std_dev)

        if z_score is not None and abs(z_score) >= threshold:
            deviation_pct = ((funding - mean_funding) / mean_funding * 100) if mean_funding > 0 else 0
            anomaly_type = "spike" if z_score > 0 else "drop"

            anomalies.append(
                {
                    "period": item.get("period", "unknown"),
                    "total_funding_usd": funding,
                    "z_score": round(z_score, 2),
                    "anomaly_type": anomaly_type,
                    "deviation_pct": round(deviation_pct, 2),
                }
            )

    # Sort by absolute z-score (most anomalous first)
    anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    return anomalies


def _identify_peaks_and_troughs(
    trend_data: List[Dict[str, Any]], top_n: int = 5
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Identify peak and trough periods.

    Args:
        trend_data: List of period data dictionaries.
        top_n: Number of top peaks/troughs to return.

    Returns:
        Tuple of (peaks, troughs) lists, each sorted by funding amount.
    """
    # Sort by funding amount
    sorted_by_funding = sorted(
        trend_data,
        key=lambda x: x.get("total_funding_usd", 0),
        reverse=True,
    )

    # Get top N peaks
    peaks = []
    for i, item in enumerate(sorted_by_funding[:top_n]):
        peaks.append(
            {
                "period": item.get("period", "unknown"),
                "total_funding_usd": item.get("total_funding_usd", 0),
                "rank": i + 1,
            }
        )

    # Get bottom N troughs
    troughs = []
    for i, item in enumerate(sorted_by_funding[-top_n:]):
        troughs.append(
            {
                "period": item.get("period", "unknown"),
                "total_funding_usd": item.get("total_funding_usd", 0),
                "rank": len(sorted_by_funding) - top_n + i + 1,
            }
        )
    # Reverse to show lowest first
    troughs.reverse()

    return peaks, troughs


def _detect_seasonality(
    trend_data: List[Dict[str, Any]], granularity: str
) -> Dict[str, Any]:
    """Detect seasonal patterns in funding data.

    Args:
        trend_data: List of period data dictionaries.
        granularity: Time granularity ('monthly', 'quarterly', 'yearly').

    Returns:
        Dictionary with seasonality detection results.
    """
    if granularity == "yearly" or len(trend_data) < 4:
        return {
            "detected": False,
            "strongest_season": None,
            "weakest_season": None,
            "seasonal_variation_pct": None,
        }

    # Group by season (month or quarter)
    season_data: Dict[str, List[int]] = defaultdict(list)

    for item in trend_data:
        period = item.get("period", "")
        funding = item.get("total_funding_usd", 0)

        # Extract season from period
        season = None
        if granularity == "monthly":
            # Period format: "YYYY-MM" or "Month YYYY"
            if "-" in period:
                parts = period.split("-")
                if len(parts) >= 2:
                    month = int(parts[1])
                    # Map to quarter
                    season = f"Q{(month - 1) // 3 + 1}"
            elif " " in period:
                # Format: "January 2022"
                month_names = [
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december"
                ]
                period_lower = period.lower()
                for i, month_name in enumerate(month_names):
                    if month_name in period_lower:
                        season = f"Q{(i // 3) + 1}"
                        break
        elif granularity == "quarterly":
            # Period format: "YYYY-Q1" or similar
            if "Q" in period.upper():
                season = period.split("-")[-1] if "-" in period else period

        if season:
            season_data[season].append(funding)

    if len(season_data) < 2:
        return {
            "detected": False,
            "strongest_season": None,
            "weakest_season": None,
            "seasonal_variation_pct": None,
        }

    # Calculate average funding per season
    season_averages = {
        season: statistics.mean(fundings) if fundings else 0
        for season, fundings in season_data.items()
    }

    if not season_averages:
        return {
            "detected": False,
            "strongest_season": None,
            "weakest_season": None,
            "seasonal_variation_pct": None,
        }

    # Find strongest and weakest seasons
    strongest_season = max(season_averages, key=season_averages.get)
    weakest_season = min(season_averages, key=season_averages.get)

    # Calculate seasonal variation (coefficient of variation)
    season_values = list(season_averages.values())
    if len(season_values) > 1 and statistics.mean(season_values) > 0:
        std_dev = statistics.stdev(season_values)
        mean_value = statistics.mean(season_values)
        seasonal_variation_pct = (std_dev / mean_value) * 100
    else:
        seasonal_variation_pct = None

    # Consider seasonality detected if variation is significant (>15%)
    detected = seasonal_variation_pct is not None and seasonal_variation_pct > 15.0

    return {
        "detected": detected,
        "strongest_season": strongest_season if detected else None,
        "weakest_season": weakest_season if detected else None,
        "seasonal_variation_pct": (
            round(seasonal_variation_pct, 2) if seasonal_variation_pct is not None else None
        ),
    }


def _detect_cyclical_patterns(
    trend_data: List[Dict[str, Any]], min_periods: int
) -> Dict[str, Any]:
    """Detect cyclical patterns in funding data using autocorrelation.

    Args:
        trend_data: List of period data dictionaries.
        min_periods: Minimum number of periods required for cycle detection.

    Returns:
        Dictionary with cyclical pattern detection results.
    """
    if len(trend_data) < min_periods:
        return {
            "detected": False,
            "cycle_length_periods": None,
            "cycle_strength": None,
            "description": None,
        }

    # Extract funding amounts
    funding_amounts = [
        item.get("total_funding_usd", 0) for item in trend_data
    ]

    if len(funding_amounts) < min_periods:
        return {
            "detected": False,
            "cycle_length_periods": None,
            "cycle_strength": None,
            "description": None,
        }

    # Calculate autocorrelation for different lags
    mean_funding = statistics.mean(funding_amounts)
    variance = statistics.variance(funding_amounts) if len(funding_amounts) > 1 else 0

    if variance == 0:
        return {
            "detected": False,
            "cycle_length_periods": None,
            "cycle_strength": None,
            "description": None,
        }

    max_lag = min(len(funding_amounts) // 2, 12)  # Limit to reasonable lags
    autocorrelations = []

    for lag in range(1, max_lag + 1):
        # Calculate autocorrelation for this lag
        numerator = 0.0
        for i in range(len(funding_amounts) - lag):
            numerator += (funding_amounts[i] - mean_funding) * (
                funding_amounts[i + lag] - mean_funding
            )

        autocorr = numerator / (variance * (len(funding_amounts) - lag))
        autocorrelations.append((lag, autocorr))

    # Find lag with highest positive autocorrelation
    if not autocorrelations:
        return {
            "detected": False,
            "cycle_length_periods": None,
            "cycle_strength": None,
            "description": None,
        }

    best_lag, best_autocorr = max(autocorrelations, key=lambda x: x[1])

    # Consider a cycle detected if autocorrelation > 0.3
    detected = best_autocorr > 0.3

    description = None
    if detected:
        description = (
            f"Detected cyclical pattern with period of {best_lag} periods "
            f"(autocorrelation: {best_autocorr:.2f})"
        )

    return {
        "detected": detected,
        "cycle_length_periods": best_lag if detected else None,
        "cycle_strength": round(best_autocorr, 3) if detected else None,
        "description": description,
    }


def _analyze_trend_patterns(
    trend_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze overall trend patterns in funding data.

    Args:
        trend_data: List of period data dictionaries.

    Returns:
        Dictionary with trend pattern analysis.
    """
    if len(trend_data) < 2:
        return {
            "overall_direction": "insufficient_data",
            "growth_consistency": "insufficient_data",
            "volatility_level": "insufficient_data",
        }

    # Extract funding amounts
    funding_amounts = [
        item.get("total_funding_usd", 0) for item in trend_data
    ]

    # Determine overall direction
    if len(funding_amounts) >= 2:
        first_half = funding_amounts[: len(funding_amounts) // 2]
        second_half = funding_amounts[len(funding_amounts) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.1:  # 10% increase
            direction = "increasing"
        elif second_avg < first_avg * 0.9:  # 10% decrease
            direction = "decreasing"
        else:
            direction = "stable"
    else:
        direction = "insufficient_data"

    # Calculate growth consistency (coefficient of variation of velocity changes)
    velocity_changes = [
        item.get("velocity_change_pct")
        for item in trend_data
        if item.get("velocity_change_pct") is not None
    ]

    if len(velocity_changes) >= 2:
        mean_velocity = statistics.mean(velocity_changes)
        std_velocity = statistics.stdev(velocity_changes)
        cv = (std_velocity / abs(mean_velocity) * 100) if mean_velocity != 0 else float("inf")

        if cv < 30:
            consistency = "high"
        elif cv < 60:
            consistency = "moderate"
        else:
            consistency = "low"
    else:
        consistency = "insufficient_data"

    # Calculate volatility level
    if len(funding_amounts) >= 2 and statistics.mean(funding_amounts) > 0:
        std_dev = statistics.stdev(funding_amounts)
        mean_funding = statistics.mean(funding_amounts)
        cv = (std_dev / mean_funding) * 100

        if cv < 20:
            volatility = "low"
        elif cv < 50:
            volatility = "moderate"
        else:
            volatility = "high"
    else:
        volatility = "insufficient_data"

    return {
        "overall_direction": direction,
        "growth_consistency": consistency,
        "volatility_level": volatility,
    }


def _generate_insights(
    peaks: List[Dict[str, Any]],
    troughs: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    seasonal_patterns: Dict[str, Any],
    cyclical_patterns: Dict[str, Any],
    trend_patterns: Dict[str, Any],
) -> List[str]:
    """Generate human-readable insights from pattern analysis.

    Args:
        peaks: List of peak periods.
        troughs: List of trough periods.
        anomalies: List of detected anomalies.
        seasonal_patterns: Seasonal pattern detection results.
        cyclical_patterns: Cyclical pattern detection results.
        trend_patterns: Trend pattern analysis results.

    Returns:
        List of insight strings.
    """
    insights = []

    # Peak insights
    if peaks:
        top_peak = peaks[0]
        insights.append(
            f"Peak funding period: {top_peak['period']} with ${top_peak['total_funding_usd']:,}"
        )

    # Trough insights
    if troughs:
        bottom_trough = troughs[0]
        insights.append(
            f"Lowest funding period: {bottom_trough['period']} with ${bottom_trough['total_funding_usd']:,}"
        )

    # Anomaly insights
    if anomalies:
        top_anomaly = anomalies[0]
        insights.append(
            f"Unusual {top_anomaly['anomaly_type']} detected in {top_anomaly['period']} "
            f"({top_anomaly['deviation_pct']:.1f}% deviation from mean)"
        )

    # Seasonal insights
    if seasonal_patterns.get("detected"):
        strongest = seasonal_patterns.get("strongest_season")
        weakest = seasonal_patterns.get("weakest_season")
        variation = seasonal_patterns.get("seasonal_variation_pct")
        if strongest and weakest:
            insights.append(
                f"Seasonal pattern detected: {strongest} is strongest, {weakest} is weakest "
                f"({variation:.1f}% variation)"
            )

    # Cyclical insights
    if cyclical_patterns.get("detected"):
        cycle_length = cyclical_patterns.get("cycle_length_periods")
        strength = cyclical_patterns.get("cycle_strength")
        if cycle_length and strength:
            insights.append(
                f"Cyclical pattern detected: {cycle_length}-period cycle (strength: {strength:.2f})"
            )

    # Trend insights
    direction = trend_patterns.get("overall_direction")
    if direction and direction != "insufficient_data":
        insights.append(f"Overall trend direction: {direction}")

    volatility = trend_patterns.get("volatility_level")
    if volatility and volatility != "insufficient_data":
        insights.append(f"Volatility level: {volatility}")

    return insights


@observe(as_type="tool")
async def identify_funding_patterns(
    trend_data: List[Dict[str, Any]],
    granularity: str,
    anomaly_threshold: float = 2.0,
    detect_seasonality: bool = True,
    min_periods_for_cycles: int = 8,
) -> ToolOutput:
    """Identify patterns in aggregated funding data.

    This tool:
    1. Identifies peak and trough periods
    2. Detects anomalies (unusual spikes or drops)
    3. Detects seasonal patterns (if applicable)
    4. Detects cyclical patterns
    5. Analyzes overall trend patterns
    6. Generates human-readable insights

    Args:
        trend_data: List of period data dictionaries from aggregate_funding_trends.
            Each should have: 'period', 'total_funding_usd', 'round_count',
            and optionally 'velocity_change_pct'.
        granularity: Time granularity - 'monthly', 'quarterly', or 'yearly'.
        anomaly_threshold: Z-score threshold for anomaly detection (default: 2.0).
        detect_seasonality: Whether to detect seasonal patterns (default: True).
        min_periods_for_cycles: Minimum periods for cycle detection (default: 8).

    Returns:
        ToolOutput object containing:
        - success: Whether the analysis succeeded
        - result: Dictionary with:
            - peaks: Top peak periods
            - troughs: Bottom trough periods
            - anomalies: Detected anomalies
            - seasonal_patterns: Seasonal pattern results
            - cyclical_patterns: Cyclical pattern results
            - trend_patterns: Trend analysis results
            - insights: Human-readable insights
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Identify patterns in aggregated funding trends
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
        result = await identify_funding_patterns(
            trend_data=trend_data,
            granularity="quarterly",
            anomaly_threshold=2.0,
            detect_seasonality=True
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate inputs
        if not trend_data:
            raise ValueError("trend_data cannot be empty")

        if granularity not in ["monthly", "quarterly", "yearly"]:
            raise ValueError(
                f"granularity must be one of: monthly, quarterly, yearly. Got: {granularity}"
            )

        if anomaly_threshold < 0:
            raise ValueError(f"anomaly_threshold must be >= 0. Got: {anomaly_threshold}")

        if min_periods_for_cycles < 2:
            raise ValueError(
                f"min_periods_for_cycles must be >= 2. Got: {min_periods_for_cycles}"
            )

        # Identify peaks and troughs
        peaks, troughs = _identify_peaks_and_troughs(trend_data, top_n=5)

        # Detect anomalies
        anomalies = _detect_anomalies(trend_data, anomaly_threshold)

        # Detect seasonality (if requested and applicable)
        seasonal_patterns = {}
        if detect_seasonality:
            seasonal_patterns = _detect_seasonality(trend_data, granularity)
        else:
            seasonal_patterns = {
                "detected": False,
                "strongest_season": None,
                "weakest_season": None,
                "seasonal_variation_pct": None,
            }

        # Detect cyclical patterns
        cyclical_patterns = _detect_cyclical_patterns(trend_data, min_periods_for_cycles)

        # Analyze trend patterns
        trend_patterns = _analyze_trend_patterns(trend_data)

        # Generate insights
        insights = _generate_insights(
            peaks,
            troughs,
            anomalies,
            seasonal_patterns,
            cyclical_patterns,
            trend_patterns,
        )

        # Build result
        result = {
            "peaks": peaks,
            "troughs": troughs,
            "anomalies": anomalies,
            "seasonal_patterns": seasonal_patterns,
            "cyclical_patterns": cyclical_patterns,
            "trend_patterns": trend_patterns,
            "insights": insights,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "num_periods": len(trend_data),
            "num_peaks": len(peaks),
            "num_troughs": len(troughs),
            "num_anomalies": len(anomalies),
            "seasonality_detected": seasonal_patterns.get("detected", False),
            "cyclical_patterns_detected": cyclical_patterns.get("detected", False),
        }

        logger.debug(
            f"Identified funding patterns for {len(trend_data)} periods "
            f"in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="identify_funding_patterns",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to identify funding patterns: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="identify_funding_patterns",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )

