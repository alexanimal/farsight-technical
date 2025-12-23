"""Tool for aggregating funding trends over time.

This tool provides time-series aggregation of funding data for a set of organizations,
grouping funding rounds by time periods (monthly, quarterly, or yearly) and calculating
aggregate metrics like total funding, round counts, averages, medians, and unique investors.
"""

import logging
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from src.contracts.tool_io import ToolMetadata, ToolOutput, ToolParameterSchema, create_tool_output
from src.models.funding_rounds import FundingRound, FundingRoundModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the aggregate_funding_trends tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="aggregate_funding_trends",
        description="Aggregate funding rounds by time period (monthly/quarterly/yearly) for a set of organizations, calculating total funding, round counts, averages, medians, and unique investors per period.",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="org_uuids",
                type="array",
                description="List of organization UUIDs (as strings) to aggregate funding for. Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="time_period_start",
                type="string",
                description="Start date for trend analysis (ISO format string, e.g., '2022-01-01T00:00:00'). Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="time_period_end",
                type="string",
                description="End date for trend analysis (ISO format string, e.g., '2024-12-31T23:59:59'). Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="granularity",
                type="string",
                description="Time granularity for aggregation: 'monthly', 'quarterly', or 'yearly'. Default: 'quarterly'",
                required=False,
                default="quarterly",
                enum=["monthly", "quarterly", "yearly"],
            ),
            ToolParameterSchema(
                name="min_funding_amount",
                type="integer",
                description="Minimum funding amount in USD to include in aggregation (filters out small rounds). Optional.",
                required=False,
            ),
        ],
        returns={
            "type": "object",
            "description": "Aggregated funding trends with time-series data per period",
            "properties": {
                "time_period": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                    },
                },
                "granularity": {"type": "string"},
                "trend_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "string"},
                            "total_funding_usd": {"type": "integer"},
                            "round_count": {"type": "integer"},
                            "avg_round_size_usd": {"type": "number"},
                            "median_round_size_usd": {"type": "number"},
                            "unique_investors": {"type": "integer"},
                            "velocity_change_pct": {"type": "number", "nullable": True},
                        },
                    },
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "total_funding_usd": {"type": "integer"},
                        "total_rounds": {"type": "integer"},
                        "avg_round_size_usd": {"type": "number"},
                        "median_round_size_usd": {"type": "number"},
                        "total_unique_investors": {"type": "integer"},
                        "num_periods": {"type": "integer"},
                    },
                },
            },
        },
        cost_per_call=None,  # Database query and computation, minimal cost
        estimated_latency_ms=500.0,  # Typical aggregation latency
        timeout_seconds=60.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["aggregation", "funding", "trends", "time-series", "read-only"],
    )


def _get_period_key(date: datetime, granularity: str) -> str:
    """Get period key string for a given date and granularity.

    Args:
        date: The date to get period for.
        granularity: One of 'monthly', 'quarterly', 'yearly'.

    Returns:
        Period key string (e.g., '2022-Q1', '2022-03', '2022').
    """
    if granularity == "yearly":
        return str(date.year)
    elif granularity == "monthly":
        return f"{date.year}-{date.month:02d}"
    elif granularity == "quarterly":
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}-Q{quarter}"
    else:
        raise ValueError(f"Invalid granularity: {granularity}")


def _format_period_label(period_key: str, granularity: str) -> str:
    """Format period key into a human-readable label.

    Args:
        period_key: Period key from _get_period_key.
        granularity: One of 'monthly', 'quarterly', 'yearly'.

    Returns:
        Formatted period label.
    """
    if granularity == "yearly":
        return period_key
    elif granularity == "monthly":
        year, month = period_key.split("-")
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        return f"{month_names[int(month) - 1]} {year}"
    elif granularity == "quarterly":
        return period_key
    else:
        return period_key


@observe(as_type="tool")
async def aggregate_funding_trends(
    org_uuids: List[str],
    time_period_start: str,
    time_period_end: str,
    granularity: str = "quarterly",
    min_funding_amount: Optional[int] = None,
) -> ToolOutput:
    """Aggregate funding rounds by time period for a set of organizations.

    This tool:
    1. Queries funding rounds for the specified organizations within the time period
    2. Groups rounds by time period (monthly/quarterly/yearly)
    3. Calculates aggregate metrics per period:
       - Total funding amount
       - Round count
       - Average round size
       - Median round size
       - Unique investors
    4. Calculates period-over-period velocity changes

    Args:
        org_uuids: List of organization UUIDs (as strings) to aggregate funding for.
        time_period_start: Start date for trend analysis (ISO format string).
        time_period_end: End date for trend analysis (ISO format string).
        granularity: Time granularity - 'monthly', 'quarterly', or 'yearly'. Default: 'quarterly'.
        min_funding_amount: Minimum funding amount in USD to include (filters out small rounds).

    Returns:
        ToolOutput object containing:
        - success: Whether the aggregation succeeded
        - result: Dictionary with:
          - time_period: Start and end dates
          - granularity: Granularity used
          - trend_data: List of period aggregations with metrics
          - summary: Overall summary statistics
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Aggregate quarterly funding trends for a set of companies
        result = await aggregate_funding_trends(
            org_uuids=["uuid1", "uuid2", "uuid3"],
            time_period_start="2022-01-01T00:00:00",
            time_period_end="2024-12-31T23:59:59",
            granularity="quarterly",
            min_funding_amount=1000000
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate inputs
        if not org_uuids:
            raise ValueError("org_uuids cannot be empty")

        if granularity not in ["monthly", "quarterly", "yearly"]:
            raise ValueError(
                f"granularity must be one of: monthly, quarterly, yearly. Got: {granularity}"
            )

        # Parse dates - handle 'Z' suffix (UTC) by replacing with '+00:00'
        # Python's fromisoformat doesn't support 'Z' directly
        normalized_start = (
            time_period_start.replace("Z", "+00:00")
            if time_period_start.endswith("Z")
            else time_period_start
        )
        normalized_end = (
            time_period_end.replace("Z", "+00:00")
            if time_period_end.endswith("Z")
            else time_period_end
        )

        period_start = datetime.fromisoformat(normalized_start)
        period_end = datetime.fromisoformat(normalized_end)

        # Convert timezone-aware datetimes to UTC and make them naive
        # PostgreSQL typically stores naive datetimes, so we need to ensure consistency
        if period_start.tzinfo is not None:
            period_start = period_start.astimezone(timezone.utc).replace(tzinfo=None)
        if period_end.tzinfo is not None:
            period_end = period_end.astimezone(timezone.utc).replace(tzinfo=None)

        if period_start >= period_end:
            raise ValueError(
                f"time_period_start ({time_period_start}) must be before time_period_end ({time_period_end})"
            )

        # Convert org UUIDs to UUID objects
        org_uuid_objs = [UUID(uuid_str) for uuid_str in org_uuids]

        # Initialize the model
        model = FundingRoundModel()
        await model.initialize()

        # Query funding rounds for all organizations using batch queries with chunking
        # Chunk size optimized for PostgreSQL IN clause performance
        CHUNK_SIZE = 1000
        result_size_warning_threshold = 1_000_000  # 1M records

        all_rounds: List[FundingRound] = []
        num_batch_queries = 0

        # Process in chunks to avoid PostgreSQL query size limits
        for i in range(0, len(org_uuid_objs), CHUNK_SIZE):
            chunk = org_uuid_objs[i : i + CHUNK_SIZE]
            rounds = await model.get(
                org_uuids=chunk,  # Batch query for chunk
                investment_date_from=period_start,
                investment_date_to=period_end,
                fundraise_amount_usd_min=min_funding_amount,
                order_by="investment_date",
                order_direction="desc",
            )
            all_rounds.extend(rounds)
            num_batch_queries += 1
            logger.debug(
                f"Retrieved {len(rounds)} funding rounds for chunk {num_batch_queries} "
                f"({len(chunk)} organizations)"
            )

        # Result size validation and warnings
        if len(all_rounds) > result_size_warning_threshold:
            logger.warning(
                f"Large result set: {len(all_rounds):,} funding rounds for {len(org_uuids)} organizations. "
                f"This may impact memory usage and processing time."
            )

        logger.info(
            f"Retrieved {len(all_rounds)} funding rounds for {len(org_uuids)} organizations "
            f"across {num_batch_queries} batch queries"
        )

        # Group rounds by period
        period_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "rounds": [],
                "funding_amounts": [],
                "investors": set(),
            }
        )

        for round_obj in all_rounds:
            if round_obj.investment_date is None:
                logger.warning(
                    f"Skipping funding round {round_obj.funding_round_uuid} - missing investment_date"
                )
                continue

            period_key = _get_period_key(round_obj.investment_date, granularity)
            period_data[period_key]["rounds"].append(round_obj)

            # Collect funding amount (if available)
            if round_obj.fundraise_amount_usd is not None:
                period_data[period_key]["funding_amounts"].append(round_obj.fundraise_amount_usd)

            # Collect investors
            if round_obj.investors:
                period_data[period_key]["investors"].update(round_obj.investors)
            if round_obj.lead_investors:
                period_data[period_key]["investors"].update(round_obj.lead_investors)

        # Calculate metrics for each period
        trend_data = []
        sorted_periods = sorted(period_data.keys())

        for i, period_key in enumerate(sorted_periods):
            data = period_data[period_key]
            rounds = data["rounds"]
            funding_amounts = data["funding_amounts"]
            investors = data["investors"]

            # Calculate metrics
            total_funding = sum(funding_amounts) if funding_amounts else 0
            round_count = len(rounds)
            avg_round_size = statistics.mean(funding_amounts) if funding_amounts else 0.0
            median_round_size = statistics.median(funding_amounts) if funding_amounts else 0.0
            unique_investors = len(investors)

            # Calculate velocity change (period-over-period change in total funding)
            velocity_change_pct = None
            if i > 0:
                prev_period_key = sorted_periods[i - 1]
                prev_total_funding = sum(period_data[prev_period_key]["funding_amounts"])
                if prev_total_funding > 0:
                    velocity_change_pct = (
                        (total_funding - prev_total_funding) / prev_total_funding * 100
                    )
                elif total_funding > 0:
                    velocity_change_pct = 100.0  # New funding started

            trend_data.append(
                {
                    "period": _format_period_label(period_key, granularity),
                    "period_key": period_key,  # Keep key for sorting/grouping
                    "total_funding_usd": int(total_funding),
                    "round_count": round_count,
                    "avg_round_size_usd": round(avg_round_size, 2),
                    "median_round_size_usd": round(median_round_size, 2),
                    "unique_investors": unique_investors,
                    "velocity_change_pct": (
                        round(velocity_change_pct, 2) if velocity_change_pct is not None else None
                    ),
                }
            )

        # Calculate summary statistics
        all_funding_amounts = [
            amount
            for period_data in period_data.values()
            for amount in period_data["funding_amounts"]
        ]
        all_investors: set[str] = set()
        for period_data in period_data.values():
            all_investors.update(period_data["investors"])

        summary = {
            "total_funding_usd": (int(sum(all_funding_amounts)) if all_funding_amounts else 0),
            "total_rounds": len(all_rounds),
            "avg_round_size_usd": (
                round(statistics.mean(all_funding_amounts), 2) if all_funding_amounts else 0.0
            ),
            "median_round_size_usd": (
                round(statistics.median(all_funding_amounts), 2) if all_funding_amounts else 0.0
            ),
            "total_unique_investors": len(all_investors),
            "num_periods": len(trend_data),
        }

        # Build result
        result = {
            "time_period": {
                "start": time_period_start,
                "end": time_period_end,
            },
            "granularity": granularity,
            "trend_data": trend_data,
            "summary": summary,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        # Build metadata with result size information
        metadata: Dict[str, Any] = {
            "num_orgs": len(org_uuids),
            "num_rounds": len(all_rounds),
            "num_periods": len(trend_data),
            "num_batch_queries": num_batch_queries,
        }

        # Add warning if result set is large
        if len(all_rounds) > result_size_warning_threshold:
            metadata["warning"] = (
                f"Large result set: {len(all_rounds):,} funding rounds. "
                "Consider filtering by date range or min_funding_amount to reduce size."
            )

        logger.debug(
            f"Aggregated funding trends for {len(org_uuids)} organizations "
            f"across {len(trend_data)} periods in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="aggregate_funding_trends",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to aggregate funding trends: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="aggregate_funding_trends",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
