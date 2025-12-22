"""Tool for calculating portfolio performance metrics.

This tool analyzes an investor's portfolio to calculate performance metrics including
exit rates, time to exit, ROI estimates, and deployment patterns. It works with
portfolio data from find_investor_portfolio and acquisition data.
"""

import logging
import statistics
import time
from datetime import datetime
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

from src.contracts.tool_io import (
    ToolMetadata,
    ToolOutput,
    ToolParameterSchema,
    create_tool_output,
)
from src.models.acquisitions import Acquisition, AcquisitionModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the calculate_portfolio_metrics tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="calculate_portfolio_metrics",
        description="Calculate portfolio performance metrics including exit rates, time to exit, ROI estimates, and deployment patterns from portfolio and acquisition data.",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="portfolio_companies",
                type="array",
                description="List of portfolio companies from find_investor_portfolio. Each item should have 'org_uuid', 'funding_rounds' (with investment dates and amounts), and optionally company details. Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="time_period_start",
                type="string",
                description="Start date for analysis period (ISO format string, e.g., '2018-01-01T00:00:00'). Optional. Used for filtering and calculating metrics within a specific timeframe.",
                required=False,
            ),
            ToolParameterSchema(
                name="time_period_end",
                type="string",
                description="End date for analysis period (ISO format string, e.g., '2024-12-31T23:59:59'). Optional. Used for filtering and calculating metrics within a specific timeframe.",
                required=False,
            ),
            ToolParameterSchema(
                name="include_exits_only",
                type="boolean",
                description="If true, only analyze companies that have exited (been acquired). Default: false (analyzes all portfolio companies).",
                required=False,
                default=False,
            ),
        ],
        returns={
            "type": "object",
            "description": "Portfolio performance metrics",
            "properties": {
                "portfolio_summary": {
                    "type": "object",
                    "properties": {
                        "total_companies": {"type": "integer"},
                        "total_investments": {"type": "integer"},
                        "total_capital_deployed_usd": {"type": "integer", "nullable": True},
                        "exited_companies": {"type": "integer"},
                        "exit_rate_pct": {"type": "number", "nullable": True},
                        "average_time_to_exit_days": {"type": "number", "nullable": True},
                    },
                },
                "performance_metrics": {
                    "type": "object",
                    "properties": {
                        "average_roi_estimate": {"type": "number", "nullable": True},
                        "median_roi_estimate": {"type": "number", "nullable": True},
                        "successful_exits_pct": {"type": "number", "nullable": True},
                        "average_exit_value_usd": {"type": "number", "nullable": True},
                        "median_exit_value_usd": {"type": "number", "nullable": True},
                    },
                },
                "deployment_patterns": {
                    "type": "object",
                    "properties": {
                        "investments_per_year": {"type": "number", "nullable": True},
                        "capital_deployed_per_year_usd": {"type": "number", "nullable": True},
                        "average_round_size_usd": {"type": "number", "nullable": True},
                        "median_round_size_usd": {"type": "number", "nullable": True},
                    },
                },
                "exit_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "org_uuid": {"type": "string"},
                            "name": {"type": "string", "nullable": True},
                            "exit_date": {"type": "string", "nullable": True},
                            "exit_value_usd": {"type": "integer", "nullable": True},
                            "total_invested_usd": {"type": "integer", "nullable": True},
                            "roi_estimate": {"type": "number", "nullable": True},
                            "time_to_exit_days": {"type": "number", "nullable": True},
                        },
                    },
                },
            },
        },
        cost_per_call=None,  # Database query and computation, minimal cost
        estimated_latency_ms=500.0,  # Database query and computation latency
        timeout_seconds=60.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["calculation", "portfolio", "performance", "roi", "read-only"],
    )


@observe(as_type="tool")
async def calculate_portfolio_metrics(
    portfolio_companies: List[Dict[str, Any]],
    time_period_start: Optional[str] = None,
    time_period_end: Optional[str] = None,
    include_exits_only: bool = False,
) -> ToolOutput:
    """Calculate portfolio performance metrics.

    This tool:
    1. Queries acquisitions table for exits of portfolio companies
    2. Matches exits to portfolio companies
    3. Calculates exit rates and time to exit
    4. Estimates ROI (acquisition price vs total invested)
    5. Analyzes deployment patterns

    Args:
        portfolio_companies: List of portfolio company dictionaries from find_investor_portfolio.
            Each should have 'org_uuid', 'funding_rounds', and optionally company details.
        time_period_start: Start date for analysis period (ISO format string).
        time_period_end: End date for analysis period (ISO format string).
        include_exits_only: If true, only analyze exited companies.

    Returns:
        ToolOutput object containing:
        - success: Whether the calculation succeeded
        - result: Dictionary with:
            - portfolio_summary: Summary statistics
            - performance_metrics: ROI and exit metrics
            - deployment_patterns: Investment deployment patterns
            - exit_details: Detailed exit information
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Calculate metrics for a portfolio
        portfolio = [
            {
                "org_uuid": "...",
                "funding_rounds": [...],
                "name": "Company A"
            }
        ]
        result = await calculate_portfolio_metrics(
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
            period_start = datetime.fromisoformat(time_period_start)
        if time_period_end:
            period_end = datetime.fromisoformat(time_period_end)

        if period_start and period_end and period_start >= period_end:
            raise ValueError(
                f"time_period_start ({time_period_start}) must be before time_period_end ({time_period_end})"
            )

        # Initialize acquisition model
        acquisition_model = AcquisitionModel()
        await acquisition_model.initialize()

        # Extract org UUIDs from portfolio
        org_uuids = [
            UUID(company.get("org_uuid"))
            for company in portfolio_companies
            if company.get("org_uuid")
        ]

        if not org_uuids:
            raise ValueError("No valid org_uuids found in portfolio_companies")

        logger.info(f"Calculating metrics for {len(org_uuids)} portfolio companies")

        # Query acquisitions for portfolio companies
        # Note: AcquisitionModel doesn't support batch queries, so we query individually
        # For efficiency, we'll query all acquisitions and filter in memory
        # Order by announce date descending to get most recent acquisitions first
        all_acquisitions = await acquisition_model.get(
            order_by="acquisition_announce_date",
            order_direction="desc",
        )
        
        # Filter to only acquisitions of portfolio companies
        org_uuids_set = set(org_uuids)
        portfolio_acquisitions = [
            acq for acq in all_acquisitions
            if acq.acquiree_uuid and acq.acquiree_uuid in org_uuids_set
        ]

        logger.info(f"Found {len(portfolio_acquisitions)} acquisitions for portfolio companies")

        # Create acquisition lookup by acquiree_uuid
        acquisitions_by_org: Dict[UUID, List[Acquisition]] = {}
        for acquisition in portfolio_acquisitions:
            if acquisition.acquiree_uuid is None:
                continue
            if acquisition.acquiree_uuid not in acquisitions_by_org:
                acquisitions_by_org[acquisition.acquiree_uuid] = []
            acquisitions_by_org[acquisition.acquiree_uuid].append(acquisition)

        # Filter portfolio companies if include_exits_only
        companies_to_analyze = portfolio_companies
        if include_exits_only:
            companies_to_analyze = [
                company
                for company in portfolio_companies
                if UUID(company.get("org_uuid")) in acquisitions_by_org
            ]

        # Calculate metrics
        total_companies = len(companies_to_analyze)
        total_investments = sum(
            len(company.get("funding_rounds", [])) for company in companies_to_analyze
        )
        total_capital_deployed = sum(
            company.get("total_invested_usd") or 0 for company in companies_to_analyze
        )

        # Process exits
        exit_details = []
        times_to_exit = []
        roi_estimates = []
        exit_values = []

        for company in companies_to_analyze:
            org_uuid = UUID(company.get("org_uuid"))
            acquisitions = acquisitions_by_org.get(org_uuid, [])

            if not acquisitions:
                continue

            # Get the most recent acquisition (or first if multiple)
            # Sort by date if available
            acquisition = acquisitions[0]
            if len(acquisitions) > 1:
                acquisitions_with_dates = [
                    (a, a.acquisition_announce_date)
                    for a in acquisitions
                    if a.acquisition_announce_date
                ]
                if acquisitions_with_dates:
                    acquisition = max(acquisitions_with_dates, key=lambda x: x[1])[0]

            # Calculate total invested for this company
            total_invested = company.get("total_invested_usd") or 0

            # Get exit value
            exit_value = acquisition.acquisition_price_usd

            # Calculate ROI estimate (exit_value / total_invested)
            roi_estimate = None
            if exit_value and total_invested > 0:
                roi_estimate = exit_value / total_invested
                roi_estimates.append(roi_estimate)

            if exit_value:
                exit_values.append(exit_value)

            # Calculate time to exit
            # Use first investment date if available
            first_investment_date = None
            funding_rounds = company.get("funding_rounds", [])
            if funding_rounds:
                investment_dates = [
                    datetime.fromisoformat(round_data["investment_date"])
                    for round_data in funding_rounds
                    if round_data.get("investment_date")
                ]
                if investment_dates:
                    first_investment_date = min(investment_dates)

            time_to_exit_days = None
            if (
                first_investment_date
                and acquisition.acquisition_announce_date
            ):
                delta = acquisition.acquisition_announce_date - first_investment_date
                time_to_exit_days = delta.days
                if time_to_exit_days >= 0:
                    times_to_exit.append(time_to_exit_days)

            exit_details.append(
                {
                    "org_uuid": company.get("org_uuid"),
                    "name": company.get("name"),
                    "exit_date": (
                        acquisition.acquisition_announce_date.isoformat()
                        if acquisition.acquisition_announce_date
                        else None
                    ),
                    "exit_value_usd": exit_value,
                    "total_invested_usd": total_invested if total_invested > 0 else None,
                    "roi_estimate": round(roi_estimate, 2) if roi_estimate is not None else None,
                    "time_to_exit_days": time_to_exit_days,
                }
            )

        # Calculate summary metrics
        exited_companies = len(exit_details)
        exit_rate_pct = (
            (exited_companies / total_companies * 100) if total_companies > 0 else None
        )

        average_time_to_exit_days = (
            round(statistics.mean(times_to_exit), 1) if times_to_exit else None
        )

        # Performance metrics
        average_roi_estimate = (
            round(statistics.mean(roi_estimates), 2) if roi_estimates else None
        )
        median_roi_estimate = (
            round(statistics.median(roi_estimates), 2) if roi_estimates else None
        )

        # Successful exits (ROI > 1.0)
        successful_exits = sum(1 for roi in roi_estimates if roi > 1.0)
        successful_exits_pct = (
            (successful_exits / exited_companies * 100) if exited_companies > 0 else None
        )

        average_exit_value_usd = (
            round(statistics.mean(exit_values)) if exit_values else None
        )
        median_exit_value_usd = (
            round(statistics.median(exit_values)) if exit_values else None
        )

        # Deployment patterns
        # Calculate investments per year
        all_investment_dates = []
        all_investment_amounts = []
        for company in companies_to_analyze:
            for round_data in company.get("funding_rounds", []):
                if round_data.get("investment_date"):
                    try:
                        date = datetime.fromisoformat(round_data["investment_date"])
                        all_investment_dates.append(date)
                        if round_data.get("fundraise_amount_usd"):
                            all_investment_amounts.append(round_data["fundraise_amount_usd"])
                    except (ValueError, TypeError):
                        pass

        investments_per_year = None
        capital_deployed_per_year_usd = None
        if all_investment_dates:
            date_range_years = None
            if period_start and period_end:
                delta = period_end - period_start
                date_range_years = delta.days / 365.25
            elif len(all_investment_dates) > 1:
                min_date = min(all_investment_dates)
                max_date = max(all_investment_dates)
                delta = max_date - min_date
                date_range_years = delta.days / 365.25

            if date_range_years and date_range_years > 0:
                investments_per_year = round(len(all_investment_dates) / date_range_years, 2)
                if all_investment_amounts:
                    total_deployed = sum(all_investment_amounts)
                    capital_deployed_per_year_usd = round(total_deployed / date_range_years)

        average_round_size_usd = (
            round(statistics.mean(all_investment_amounts))
            if all_investment_amounts
            else None
        )
        median_round_size_usd = (
            round(statistics.median(all_investment_amounts))
            if all_investment_amounts
            else None
        )

        # Build result
        portfolio_summary = {
            "total_companies": total_companies,
            "total_investments": total_investments,
            "total_capital_deployed_usd": total_capital_deployed if total_capital_deployed > 0 else None,
            "exited_companies": exited_companies,
            "exit_rate_pct": round(exit_rate_pct, 1) if exit_rate_pct is not None else None,
            "average_time_to_exit_days": average_time_to_exit_days,
        }

        performance_metrics = {
            "average_roi_estimate": average_roi_estimate,
            "median_roi_estimate": median_roi_estimate,
            "successful_exits_pct": round(successful_exits_pct, 1) if successful_exits_pct is not None else None,
            "average_exit_value_usd": average_exit_value_usd,
            "median_exit_value_usd": median_exit_value_usd,
        }

        deployment_patterns = {
            "investments_per_year": investments_per_year,
            "capital_deployed_per_year_usd": capital_deployed_per_year_usd,
            "average_round_size_usd": average_round_size_usd,
            "median_round_size_usd": median_round_size_usd,
        }

        result = {
            "portfolio_summary": portfolio_summary,
            "performance_metrics": performance_metrics,
            "deployment_patterns": deployment_patterns,
            "exit_details": exit_details,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "num_companies": total_companies,
            "num_exits": exited_companies,
            "include_exits_only": include_exits_only,
        }

        logger.debug(
            f"Calculated portfolio metrics for {total_companies} companies "
            f"({exited_companies} exits) in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="calculate_portfolio_metrics",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to calculate portfolio metrics: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="calculate_portfolio_metrics",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )

