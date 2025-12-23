"""Tool for comparing portfolio metrics to market benchmarks.

This tool calculates market averages from all companies/investments in the database
and compares a portfolio's performance metrics to these market benchmarks. It works
with portfolio metrics from calculate_portfolio_metrics.
"""

import logging
import statistics
import time
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
from src.models.acquisitions import Acquisition, AcquisitionModel
from src.models.funding_rounds import FundingRound, FundingRoundModel
from src.models.organizations import OrganizationModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the compare_to_market_benchmarks tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="compare_to_market_benchmarks",
        description="Compare portfolio performance metrics to market averages calculated from all companies/investments in the database. Returns relative performance indicators.",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="portfolio_metrics",
                type="object",
                description="Portfolio metrics from calculate_portfolio_metrics. Should include 'portfolio_summary', 'performance_metrics', and 'deployment_patterns'. Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="time_period_start",
                type="string",
                description="Start date for market benchmark calculation (ISO format string, e.g., '2018-01-01T00:00:00'). Should match the portfolio analysis period. Optional but recommended for accurate comparisons.",
                required=False,
            ),
            ToolParameterSchema(
                name="time_period_end",
                type="string",
                description="End date for market benchmark calculation (ISO format string, e.g., '2024-12-31T23:59:59'). Should match the portfolio analysis period. Optional but recommended for accurate comparisons.",
                required=False,
            ),
        ],
        returns={
            "type": "object",
            "description": "Comparison of portfolio metrics to market benchmarks",
            "properties": {
                "market_benchmarks": {
                    "type": "object",
                    "description": "Market average metrics",
                    "properties": {
                        "exit_rate_pct": {"type": "number", "nullable": True},
                        "average_time_to_exit_days": {
                            "type": "number",
                            "nullable": True,
                        },
                        "average_roi_estimate": {"type": "number", "nullable": True},
                        "median_roi_estimate": {"type": "number", "nullable": True},
                        "average_round_size_usd": {"type": "number", "nullable": True},
                        "median_round_size_usd": {"type": "number", "nullable": True},
                    },
                },
                "portfolio_metrics": {
                    "type": "object",
                    "description": "Portfolio metrics (from input)",
                },
                "comparison": {
                    "type": "object",
                    "description": "Relative performance comparison",
                    "properties": {
                        "exit_rate_vs_market": {"type": "string", "nullable": True},
                        "time_to_exit_vs_market": {"type": "string", "nullable": True},
                        "roi_vs_market": {"type": "string", "nullable": True},
                        "round_size_vs_market": {"type": "string", "nullable": True},
                    },
                },
                "relative_performance": {
                    "type": "object",
                    "description": "Relative performance scores",
                    "properties": {
                        "exit_rate_percentile": {"type": "number", "nullable": True},
                        "roi_percentile": {"type": "number", "nullable": True},
                    },
                },
            },
        },
        cost_per_call=None,  # Database queries, minimal cost
        estimated_latency_ms=1000.0,  # Database query latency
        timeout_seconds=60.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["comparison", "benchmark", "portfolio", "market", "read-only"],
    )


@observe(as_type="tool")
async def compare_to_market_benchmarks(
    portfolio_metrics: Dict[str, Any],
    time_period_start: Optional[str] = None,
    time_period_end: Optional[str] = None,
) -> ToolOutput:
    """Compare portfolio metrics to market benchmarks.

    This tool:
    1. Queries the database for all companies/investments in the time period
    2. Calculates market averages for exit rates, ROI, time to exit, round sizes
    3. Compares portfolio metrics to market averages
    4. Returns relative performance indicators

    Args:
        portfolio_metrics: Dictionary from calculate_portfolio_metrics containing:
            - portfolio_summary: Summary statistics
            - performance_metrics: ROI and exit metrics
            - deployment_patterns: Investment deployment patterns
        time_period_start: Start date for market benchmark calculation (ISO format string).
        time_period_end: End date for market benchmark calculation (ISO format string).

    Returns:
        ToolOutput object containing:
        - success: Whether the comparison succeeded
        - result: Dictionary with:
            - market_benchmarks: Market average metrics
            - portfolio_metrics: Portfolio metrics (from input)
            - comparison: Relative performance comparison (e.g., "+5.2%", "-180 days")
            - relative_performance: Percentile scores
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Compare portfolio to market
        portfolio_metrics = {
            "portfolio_summary": {"exit_rate_pct": 26.7, ...},
            "performance_metrics": {"average_roi_estimate": 3.2, ...},
            ...
        }
        result = await compare_to_market_benchmarks(
            portfolio_metrics=portfolio_metrics,
            time_period_start="2018-01-01T00:00:00",
            time_period_end="2024-12-31T23:59:59"
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate inputs
        if not portfolio_metrics:
            raise ValueError("portfolio_metrics cannot be empty")

        portfolio_summary = portfolio_metrics.get("portfolio_summary", {})
        performance_metrics = portfolio_metrics.get("performance_metrics", {})
        deployment_patterns = portfolio_metrics.get("deployment_patterns", {})

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

        logger.info("Calculating market benchmarks from database")

        # Initialize models
        funding_model = FundingRoundModel()
        await funding_model.initialize()

        acquisition_model = AcquisitionModel()
        await acquisition_model.initialize()

        org_model = OrganizationModel()
        await org_model.initialize()

        # Calculate market benchmarks
        # 1. Market exit rate: percentage of all companies that were acquired
        logger.info("Calculating market exit rate")
        all_orgs = await org_model.get()
        all_org_uuids = {org.org_uuid for org in all_orgs if org.org_uuid}

        # Filter by time period if provided (companies founded in period)
        if period_start or period_end:
            filtered_orgs = []
            for org in all_orgs:
                if org.founding_date:
                    if period_start and org.founding_date < period_start:
                        continue
                    if period_end and org.founding_date > period_end:
                        continue
                filtered_orgs.append(org)
            all_org_uuids = {org.org_uuid for org in filtered_orgs if org.org_uuid}

        # Get all acquisitions
        all_acquisitions = await acquisition_model.get(
            acquisition_announce_date_from=period_start,
            acquisition_announce_date_to=period_end,
        )

        # Filter acquisitions to those in our org set
        acquired_org_uuids = {
            acq.acquiree_uuid
            for acq in all_acquisitions
            if acq.acquiree_uuid and acq.acquiree_uuid in all_org_uuids
        }

        market_exit_rate_pct = (
            (len(acquired_org_uuids) / len(all_org_uuids) * 100) if all_org_uuids else None
        )

        # 2. Market time to exit: average time from first funding to acquisition
        # 3. Market ROI: average ROI from acquisitions
        # Optimize: Fetch all funding rounds for acquired companies in batch instead of N+1 queries
        logger.info("Calculating market time to exit and ROI")

        # Get all unique org UUIDs from acquisitions
        acquired_org_uuids_list = [
            acq.acquiree_uuid for acq in all_acquisitions if acq.acquiree_uuid
        ]

        # Build lookup dictionary: org_uuid -> list of funding rounds
        funding_rounds_by_org: Dict[UUID, List[FundingRound]] = {}

        if acquired_org_uuids_list:
            # Fetch all funding rounds for acquired companies in batch
            # Note: We need ALL rounds (not filtered by date) to calculate first funding date and total funding
            all_funding_rounds = await funding_model.get(
                org_uuids=acquired_org_uuids_list,
            )

            # Group by org_uuid
            for round_obj in all_funding_rounds:
                if round_obj.org_uuid:
                    if round_obj.org_uuid not in funding_rounds_by_org:
                        funding_rounds_by_org[round_obj.org_uuid] = []
                    funding_rounds_by_org[round_obj.org_uuid].append(round_obj)

            # Sort rounds by investment_date for each org
            for org_uuid in funding_rounds_by_org:
                funding_rounds_by_org[org_uuid].sort(
                    key=lambda r: r.investment_date or datetime.min.replace(tzinfo=None)
                )

        # Calculate time to exit and ROI using in-memory lookups
        market_times_to_exit = []
        market_roi_estimates = []

        for acquisition in all_acquisitions:
            if not acquisition.acquiree_uuid:
                continue

            company_rounds = funding_rounds_by_org.get(acquisition.acquiree_uuid, [])

            # Calculate time to exit (if we have rounds and acquisition date)
            if (
                company_rounds
                and company_rounds[0].investment_date
                and acquisition.acquisition_announce_date
            ):
                first_funding_date = company_rounds[0].investment_date
                delta = acquisition.acquisition_announce_date - first_funding_date
                if delta.days >= 0:
                    market_times_to_exit.append(delta.days)

            # Calculate ROI (if we have acquisition price and funding)
            if acquisition.acquisition_price_usd and company_rounds:
                total_funding = sum(
                    r.fundraise_amount_usd for r in company_rounds if r.fundraise_amount_usd
                )

                if total_funding > 0:
                    roi = acquisition.acquisition_price_usd / total_funding
                    market_roi_estimates.append(roi)

        market_avg_time_to_exit_days = (
            round(statistics.mean(market_times_to_exit), 1) if market_times_to_exit else None
        )

        market_avg_roi = (
            round(statistics.mean(market_roi_estimates), 2) if market_roi_estimates else None
        )
        market_median_roi = (
            round(statistics.median(market_roi_estimates), 2) if market_roi_estimates else None
        )

        # 4. Market round sizes: average round sizes from all funding rounds
        logger.info("Calculating market round sizes")
        all_market_rounds = await funding_model.get(
            investment_date_from=period_start,
            investment_date_to=period_end,
        )

        market_round_sizes = [
            r.fundraise_amount_usd for r in all_market_rounds if r.fundraise_amount_usd
        ]

        market_avg_round_size = (
            round(statistics.mean(market_round_sizes)) if market_round_sizes else None
        )
        market_median_round_size = (
            round(statistics.median(market_round_sizes)) if market_round_sizes else None
        )

        # Build market benchmarks
        market_benchmarks = {
            "exit_rate_pct": (
                round(market_exit_rate_pct, 1) if market_exit_rate_pct is not None else None
            ),
            "average_time_to_exit_days": market_avg_time_to_exit_days,
            "average_roi_estimate": market_avg_roi,
            "median_roi_estimate": market_median_roi,
            "average_round_size_usd": market_avg_round_size,
            "median_round_size_usd": market_median_round_size,
        }

        # Compare portfolio to market
        portfolio_exit_rate = portfolio_summary.get("exit_rate_pct")
        portfolio_time_to_exit = portfolio_summary.get("average_time_to_exit_days")
        portfolio_avg_roi = performance_metrics.get("average_roi_estimate")
        portfolio_avg_round_size = deployment_patterns.get("average_round_size_usd")

        # Calculate relative differences
        exit_rate_vs_market = None
        if portfolio_exit_rate is not None and market_exit_rate_pct is not None:
            diff = portfolio_exit_rate - market_exit_rate_pct
            exit_rate_vs_market = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"

        time_to_exit_vs_market = None
        if portfolio_time_to_exit is not None and market_avg_time_to_exit_days is not None:
            diff = portfolio_time_to_exit - market_avg_time_to_exit_days
            time_to_exit_vs_market = f"+{diff:.0f} days" if diff >= 0 else f"{diff:.0f} days"

        roi_vs_market = None
        if portfolio_avg_roi is not None and market_avg_roi is not None:
            diff = portfolio_avg_roi - market_avg_roi
            roi_vs_market = f"+{diff:.1f}x" if diff >= 0 else f"{diff:.1f}x"

        round_size_vs_market = None
        if portfolio_avg_round_size is not None and market_avg_round_size is not None:
            diff_pct = (
                (portfolio_avg_round_size - market_avg_round_size) / market_avg_round_size * 100
            )
            round_size_vs_market = f"+{diff_pct:.1f}%" if diff_pct >= 0 else f"{diff_pct:.1f}%"

        comparison = {
            "exit_rate_vs_market": exit_rate_vs_market,
            "time_to_exit_vs_market": time_to_exit_vs_market,
            "roi_vs_market": roi_vs_market,
            "round_size_vs_market": round_size_vs_market,
        }

        # Calculate percentiles (simplified - would need full distribution for accurate percentiles)
        # For now, we'll provide a basic comparison
        exit_rate_percentile = None
        if portfolio_exit_rate is not None and market_exit_rate_pct is not None:
            # Simple percentile estimate: if portfolio is above market, assume >50th percentile
            if portfolio_exit_rate > market_exit_rate_pct:
                exit_rate_percentile = min(
                    99,
                    50 + ((portfolio_exit_rate - market_exit_rate_pct) / market_exit_rate_pct * 50),
                )
            else:
                exit_rate_percentile = max(
                    1,
                    50 - ((market_exit_rate_pct - portfolio_exit_rate) / market_exit_rate_pct * 50),
                )
            exit_rate_percentile = round(exit_rate_percentile, 1)

        roi_percentile = None
        if portfolio_avg_roi is not None and market_avg_roi is not None:
            # Simple percentile estimate
            if portfolio_avg_roi > market_avg_roi:
                roi_percentile = min(
                    99,
                    50 + ((portfolio_avg_roi - market_avg_roi) / market_avg_roi * 50),
                )
            else:
                roi_percentile = max(
                    1, 50 - ((market_avg_roi - portfolio_avg_roi) / market_avg_roi * 50)
                )
            roi_percentile = round(roi_percentile, 1)

        relative_performance = {
            "exit_rate_percentile": exit_rate_percentile,
            "roi_percentile": roi_percentile,
        }

        # Build result
        result = {
            "market_benchmarks": market_benchmarks,
            "portfolio_metrics": portfolio_metrics,
            "comparison": comparison,
            "relative_performance": relative_performance,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "market_companies": len(all_org_uuids),
            "market_acquisitions": len(acquired_org_uuids),
            "market_rounds": len(market_round_sizes),
        }

        logger.debug(f"Compared portfolio to market benchmarks in {execution_time_ms:.2f}ms")

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="compare_to_market_benchmarks",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to compare to market benchmarks: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="compare_to_market_benchmarks",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
