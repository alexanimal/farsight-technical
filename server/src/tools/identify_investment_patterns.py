"""Tool for identifying investment patterns in an investor's portfolio.

This tool analyzes portfolio data and funding rounds to identify investment patterns
including preferred stages, sector focus, investment velocity, lead investor preferences,
and timing patterns. It works with portfolio data from find_investor_portfolio.
"""

import logging
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from src.contracts.tool_io import ToolMetadata, ToolOutput, ToolParameterSchema, create_tool_output

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the identify_investment_patterns tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="identify_investment_patterns",
        description="Identify investment patterns in an investor's portfolio including preferred stages, sector focus, investment velocity, lead investor preferences, and timing patterns from portfolio data and funding rounds.",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="portfolio_companies",
                type="array",
                description="List of portfolio companies from find_investor_portfolio. Each item should have 'org_uuid', 'funding_rounds' (with investment dates, amounts, stages, and was_lead flag), and optionally company details. Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="time_period_start",
                type="string",
                description="Start date for analysis period (ISO format string, e.g., '2018-01-01T00:00:00'). Optional. Used for filtering investments within a specific timeframe.",
                required=False,
            ),
            ToolParameterSchema(
                name="time_period_end",
                type="string",
                description="End date for analysis period (ISO format string, e.g., '2024-12-31T23:59:59'). Optional. Used for filtering investments within a specific timeframe.",
                required=False,
            ),
        ],
        returns={
            "type": "object",
            "description": "Investment pattern insights",
            "properties": {
                "preferred_stages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of preferred funding stages, ordered by frequency",
                },
                "stage_distribution": {
                    "type": "object",
                    "description": "Distribution of investments by stage",
                },
                "average_round_size_usd": {
                    "type": "number",
                    "nullable": True,
                    "description": "Average round size in USD",
                },
                "median_round_size_usd": {
                    "type": "number",
                    "nullable": True,
                    "description": "Median round size in USD",
                },
                "lead_investor_pct": {
                    "type": "number",
                    "nullable": True,
                    "description": "Percentage of rounds where investor was lead",
                },
                "investment_velocity": {
                    "type": "string",
                    "nullable": True,
                    "description": "Investment velocity description (e.g., '2-3 investments per quarter')",
                },
                "investments_per_quarter": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Number of investments per quarter (if sufficient data)",
                },
                "peak_quarters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Quarters with highest investment activity",
                },
                "round_size_preferences": {
                    "type": "object",
                    "description": "Analysis of round size preferences",
                    "properties": {
                        "small_rounds_pct": {"type": "number", "nullable": True},
                        "medium_rounds_pct": {"type": "number", "nullable": True},
                        "large_rounds_pct": {"type": "number", "nullable": True},
                    },
                },
            },
        },
        cost_per_call=None,  # Pure computation, no external costs
        estimated_latency_ms=300.0,  # Pattern analysis latency
        timeout_seconds=30.0,
        side_effects=False,  # Read-only computation
        idempotent=True,  # Safe to retry
        tags=["analysis", "portfolio", "patterns", "investment", "read-only"],
    )


@observe(as_type="tool")
async def identify_investment_patterns(
    portfolio_companies: List[Dict[str, Any]],
    time_period_start: Optional[str] = None,
    time_period_end: Optional[str] = None,
    # Note: sector analysis is handled by analyze_sector_concentration tool
) -> ToolOutput:
    """Identify investment patterns in an investor's portfolio.

    This tool:
    1. Analyzes stage preferences from funding rounds
    2. Calculates investment velocity and timing patterns
    3. Determines lead investor percentage
    4. Analyzes round size preferences
    5. Identifies seasonal/quarterly patterns

    Args:
        portfolio_companies: List of portfolio company dictionaries from find_investor_portfolio.
            Each should have 'org_uuid', 'funding_rounds', and optionally company details.
        time_period_start: Start date for analysis period (ISO format string).
        time_period_end: End date for analysis period (ISO format string).

    Returns:
        ToolOutput object containing:
        - success: Whether the analysis succeeded
        - result: Dictionary with:
            - preferred_stages: List of preferred stages
            - stage_distribution: Stage distribution counts
            - average_round_size_usd: Average round size
            - median_round_size_usd: Median round size
            - lead_investor_pct: Percentage of lead investments
            - investment_velocity: Velocity description
            - investments_per_quarter: Quarterly investment counts
            - peak_quarters: Quarters with highest activity
            - round_size_preferences: Round size category analysis
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Identify patterns for a portfolio
        portfolio = [
            {
                "org_uuid": "...",
                "funding_rounds": [
                    {
                        "investment_date": "2023-01-15T00:00:00",
                        "fundraise_amount_usd": 5000000,
                        "stage": "Series A",
                        "was_lead": True
                    }
                ]
            }
        ]
        result = await identify_investment_patterns(
            portfolio_companies=portfolio,
            time_period_start="2018-01-01T00:00:00",
            time_period_end="2024-12-31T23:59:59"
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate inputs
        if not portfolio_companies:
            raise ValueError("portfolio_companies cannot be empty")

        # Parse dates if provided
        period_start = None
        period_end = None
        if time_period_start:
            normalized_start = (
                time_period_start.replace("Z", "+00:00")
                if time_period_start.endswith("Z")
                else time_period_start
            )
            period_start = datetime.fromisoformat(normalized_start)
            if period_start.tzinfo is not None:
                period_start = period_start.astimezone(timezone.utc).replace(tzinfo=None)
        if time_period_end:
            normalized_end = (
                time_period_end.replace("Z", "+00:00")
                if time_period_end.endswith("Z")
                else time_period_end
            )
            period_end = datetime.fromisoformat(normalized_end)
            if period_end.tzinfo is not None:
                period_end = period_end.astimezone(timezone.utc).replace(tzinfo=None)

        if period_start and period_end and period_start >= period_end:
            raise ValueError(
                f"time_period_start ({time_period_start}) must be before time_period_end ({time_period_end})"
            )

        logger.info(
            f"Identifying investment patterns for {len(portfolio_companies)} portfolio companies"
        )

        # Collect all funding rounds from portfolio companies
        all_rounds = []
        for company in portfolio_companies:
            funding_rounds = company.get("funding_rounds", [])
            for round_data in funding_rounds:
                # Filter by time period if provided
                if period_start or period_end:
                    investment_date_str = round_data.get("investment_date")
                    if investment_date_str:
                        try:
                            investment_date = datetime.fromisoformat(
                                investment_date_str.replace("Z", "+00:00")
                                if investment_date_str.endswith("Z")
                                else investment_date_str
                            )
                            if investment_date.tzinfo is not None:
                                investment_date = investment_date.astimezone(timezone.utc).replace(
                                    tzinfo=None
                                )

                            if period_start and investment_date < period_start:
                                continue
                            if period_end and investment_date > period_end:
                                continue
                        except (ValueError, TypeError):
                            # Skip rounds with invalid dates
                            continue

                all_rounds.append(round_data)

        if not all_rounds:
            # Return empty patterns if no rounds found
            empty_result: Dict[str, Any] = {
                "preferred_stages": [],
                "stage_distribution": {},
                "average_round_size_usd": None,
                "median_round_size_usd": None,
                "lead_investor_pct": None,
                "investment_velocity": None,
                "investments_per_quarter": [],
                "peak_quarters": [],
                "round_size_preferences": {
                    "small_rounds_pct": None,
                    "medium_rounds_pct": None,
                    "large_rounds_pct": None,
                },
            }

            execution_time_ms = (time.time() - start_time) * 1000
            return create_tool_output(
                tool_name="identify_investment_patterns",
                success=True,
                result=empty_result,
                execution_time_ms=execution_time_ms,
                metadata={"num_rounds": 0},
            )

        logger.info(f"Analyzing {len(all_rounds)} funding rounds")

        # Analyze stage preferences
        stage_counter: Counter[str] = Counter()
        stage_distribution: Dict[str, int] = {}
        for round_data in all_rounds:
            stage = round_data.get("stage") or round_data.get("general_funding_stage")
            if stage:
                stage_counter[stage] += 1

        # Build stage distribution
        total_stages = sum(stage_counter.values())
        for stage, count in stage_counter.items():
            stage_distribution[stage] = count

        # Get preferred stages (top 5, ordered by frequency)
        preferred_stages = [stage for stage, _ in stage_counter.most_common(5)]

        # Analyze round sizes
        round_sizes = [
            round_data.get("fundraise_amount_usd")
            for round_data in all_rounds
            if round_data.get("fundraise_amount_usd") is not None
        ]

        average_round_size_usd = round(statistics.mean(round_sizes)) if round_sizes else None
        median_round_size_usd = round(statistics.median(round_sizes)) if round_sizes else None

        # Analyze round size preferences (small/medium/large)
        # Define thresholds: small < $1M, medium $1M-$10M, large > $10M
        small_rounds = [s for s in round_sizes if s < 1_000_000]
        medium_rounds = [s for s in round_sizes if 1_000_000 <= s <= 10_000_000]
        large_rounds = [s for s in round_sizes if s > 10_000_000]

        total_rounds_with_size = len(round_sizes)
        small_rounds_pct = (
            round((len(small_rounds) / total_rounds_with_size * 100), 1)
            if total_rounds_with_size > 0
            else None
        )
        medium_rounds_pct = (
            round((len(medium_rounds) / total_rounds_with_size * 100), 1)
            if total_rounds_with_size > 0
            else None
        )
        large_rounds_pct = (
            round((len(large_rounds) / total_rounds_with_size * 100), 1)
            if total_rounds_with_size > 0
            else None
        )

        # Analyze lead investor percentage
        lead_investments = [
            round_data for round_data in all_rounds if round_data.get("was_lead", False)
        ]
        lead_investor_pct = (
            round((len(lead_investments) / len(all_rounds) * 100), 1) if all_rounds else None
        )

        # Analyze investment velocity and timing patterns
        investment_dates = []
        for round_data in all_rounds:
            investment_date_str = round_data.get("investment_date")
            if investment_date_str:
                try:
                    investment_date = datetime.fromisoformat(
                        investment_date_str.replace("Z", "+00:00")
                        if investment_date_str.endswith("Z")
                        else investment_date_str
                    )
                    if investment_date.tzinfo is not None:
                        investment_date = investment_date.astimezone(timezone.utc).replace(
                            tzinfo=None
                        )
                    investment_dates.append(investment_date)
                except (ValueError, TypeError):
                    continue

        # Calculate investment velocity
        investment_velocity = None
        investments_per_quarter = []
        peak_quarters = []

        if len(investment_dates) >= 2:
            # Calculate date range
            min_date = min(investment_dates)
            max_date = max(investment_dates)
            date_range_days = (max_date - min_date).days

            if date_range_days > 0:
                # Calculate investments per year
                date_range_years = date_range_days / 365.25
                investments_per_year = len(investment_dates) / date_range_years

                # Format velocity description
                if investments_per_year < 1:
                    investment_velocity = (
                        f"{round(investments_per_year * 12, 1)} investments per year"
                    )
                elif investments_per_year < 4:
                    investments_per_quarter_avg = investments_per_year / 4
                    investment_velocity = (
                        f"{round(investments_per_quarter_avg, 1)} investments per quarter"
                    )
                else:
                    investments_per_month_avg = investments_per_year / 12
                    investment_velocity = (
                        f"{round(investments_per_month_avg, 1)} investments per month"
                    )

            # Analyze quarterly patterns
            if date_range_days >= 90:  # At least 3 months of data
                # Group investments by quarter
                quarterly_counts: Dict[str, int] = defaultdict(int)
                for date in investment_dates:
                    quarter = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
                    quarterly_counts[quarter] += 1

                # Sort quarters chronologically
                sorted_quarters = sorted(quarterly_counts.keys())
                investments_per_quarter = [quarterly_counts[q] for q in sorted_quarters]

                # Find peak quarters (top 3)
                if quarterly_counts:
                    sorted_by_count = sorted(
                        quarterly_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    peak_quarters = [q for q, _ in sorted_by_count[:3]]

        # Build result
        result: Dict[str, Any] = {
            "preferred_stages": preferred_stages,
            "stage_distribution": stage_distribution,
            "average_round_size_usd": average_round_size_usd,
            "median_round_size_usd": median_round_size_usd,
            "lead_investor_pct": lead_investor_pct,
            "investment_velocity": investment_velocity,
            "investments_per_quarter": investments_per_quarter,
            "peak_quarters": peak_quarters,
            "round_size_preferences": {
                "small_rounds_pct": small_rounds_pct,
                "medium_rounds_pct": medium_rounds_pct,
                "large_rounds_pct": large_rounds_pct,
            },
        }

        execution_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "num_rounds": len(all_rounds),
            "num_companies": len(portfolio_companies),
        }

        logger.debug(
            f"Identified investment patterns for {len(all_rounds)} rounds "
            f"in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="identify_investment_patterns",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to identify investment patterns: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="identify_investment_patterns",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
