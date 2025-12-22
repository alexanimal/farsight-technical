"""Tool for finding an investor's portfolio of companies.

This tool searches for all companies that an investor has funded by querying
the FundingRounds table for rounds where the investor appears in the investors
or lead_investors fields. It returns a comprehensive portfolio with investment details.
"""

import asyncio
import logging
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
from src.models.funding_rounds import FundingRound, FundingRoundModel
from src.models.organizations import Organization, OrganizationModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the find_investor_portfolio tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="find_investor_portfolio",
        description="Find all companies that an investor has funded by searching funding rounds where the investor appears in investors or lead_investors fields. Returns portfolio companies with investment details.",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="investor_name",
                type="string",
                description="Name of investor to search for (supports partial matching). Required. The tool searches for this name in both investors and lead_investors arrays.",
                required=True,
            ),
            ToolParameterSchema(
                name="time_period_start",
                type="string",
                description="Start date for filtering investments (ISO format string, e.g., '2018-01-01T00:00:00'). Optional. If not provided, includes all investments.",
                required=False,
            ),
            ToolParameterSchema(
                name="time_period_end",
                type="string",
                description="End date for filtering investments (ISO format string, e.g., '2024-12-31T23:59:59'). Optional. If not provided, includes all investments.",
                required=False,
            ),
            ToolParameterSchema(
                name="include_lead_only",
                type="boolean",
                description="If true, only include rounds where investor was a lead investor. Default: false (includes all rounds where investor participated).",
                required=False,
                default=False,
            ),
        ],
        returns={
            "type": "object",
            "description": "Investor portfolio with companies and investment details",
            "properties": {
                "investor_name": {"type": "string"},
                "time_period": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "nullable": True},
                        "end": {"type": "string", "nullable": True},
                    },
                },
                "portfolio_companies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "org_uuid": {"type": "string"},
                            "name": {"type": "string", "nullable": True},
                            "total_funding_usd": {"type": "integer", "nullable": True},
                            "stage": {"type": "string", "nullable": True},
                            "founding_date": {"type": "string", "nullable": True},
                            "investment_count": {"type": "integer"},
                            "lead_investment_count": {"type": "integer"},
                            "total_invested_usd": {"type": "integer", "nullable": True},
                            "first_investment_date": {"type": "string", "nullable": True},
                            "last_investment_date": {"type": "string", "nullable": True},
                            "funding_rounds": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "funding_round_uuid": {"type": "string"},
                                        "investment_date": {"type": "string", "nullable": True},
                                        "fundraise_amount_usd": {"type": "integer", "nullable": True},
                                        "valuation_usd": {"type": "integer", "nullable": True},
                                        "stage": {"type": "string", "nullable": True},
                                        "was_lead": {"type": "boolean"},
                                    },
                                },
                            },
                        },
                    },
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "total_companies": {"type": "integer"},
                        "total_investments": {"type": "integer"},
                        "total_lead_investments": {"type": "integer"},
                        "total_capital_deployed_usd": {"type": "integer", "nullable": True},
                        "unique_companies": {"type": "integer"},
                    },
                },
            },
        },
        cost_per_call=None,  # Database query, minimal cost
        estimated_latency_ms=500.0,  # Database query latency
        timeout_seconds=60.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["database", "investor", "portfolio", "read-only"],
    )


@observe(as_type="tool")
async def find_investor_portfolio(
    investor_name: str,
    time_period_start: Optional[str] = None,
    time_period_end: Optional[str] = None,
    include_lead_only: bool = False,
) -> ToolOutput:
    """Find all companies that an investor has funded.

    This tool:
    1. Searches FundingRounds for rounds where investor appears in investors or lead_investors
    2. Filters by time period if provided
    3. Groups by organization to build portfolio
    4. Fetches organization details for each portfolio company
    5. Aggregates investment statistics per company

    Args:
        investor_name: Name of investor to search for (supports partial matching).
        time_period_start: Start date for filtering investments (ISO format string).
        time_period_end: End date for filtering investments (ISO format string).
        include_lead_only: If true, only include rounds where investor was lead.

    Returns:
        ToolOutput object containing:
        - success: Whether the search succeeded
        - result: Dictionary with:
            - investor_name: Name of investor searched
            - time_period: Start and end dates (if provided)
            - portfolio_companies: List of portfolio companies with investment details
            - summary: Portfolio summary statistics
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Find portfolio for an investor
        result = await find_investor_portfolio(
            investor_name="Sequoia Capital",
            time_period_start="2018-01-01T00:00:00",
            time_period_end="2024-12-31T23:59:59"
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate inputs
        if not investor_name or not investor_name.strip():
            raise ValueError("investor_name cannot be empty")

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

        # Initialize models
        funding_model = FundingRoundModel()
        await funding_model.initialize()

        org_model = OrganizationModel()
        await org_model.initialize()

        # Search for funding rounds where investor participated
        logger.info(f"Searching for funding rounds with investor: {investor_name}")

        # Query funding rounds - search in both investors and lead_investors
        all_rounds = []
        lead_round_uuids = set()  # Track which rounds came from lead_investors query
        
        if include_lead_only:
            # Only search in lead_investors by name
            rounds = await funding_model.get(
                lead_investor_name_contains=investor_name,
                investment_date_from=period_start,
                investment_date_to=period_end,
                order_by="investment_date",
                order_direction="desc",
            )
            all_rounds.extend(rounds)
            # All rounds from this query are lead investments
            lead_round_uuids = {r.funding_round_uuid for r in rounds}
        else:
            # Search in both investors and lead_investors by name
            # Note: We need to query both separately and merge, as the model doesn't support OR queries
            rounds_investors = await funding_model.get(
                investor_name_contains=investor_name,
                investment_date_from=period_start,
                investment_date_to=period_end,
                order_by="investment_date",
                order_direction="desc",
            )
            rounds_lead = await funding_model.get(
                lead_investor_name_contains=investor_name,
                investment_date_from=period_start,
                investment_date_to=period_end,
                order_by="investment_date",
                order_direction="desc",
            )
            
            # Track which rounds came from lead query
            lead_round_uuids = {r.funding_round_uuid for r in rounds_lead}
            
            # Merge and deduplicate by funding_round_uuid
            seen_uuids = set()
            for round_obj in rounds_investors + rounds_lead:
                if round_obj.funding_round_uuid not in seen_uuids:
                    all_rounds.append(round_obj)
                    seen_uuids.add(round_obj.funding_round_uuid)

        logger.info(f"Found {len(all_rounds)} funding rounds for investor {investor_name}")

        if not all_rounds:
            # Return empty portfolio
            result = {
                "investor_name": investor_name,
                "time_period": {
                    "start": time_period_start,
                    "end": time_period_end,
                },
                "portfolio_companies": [],
                "summary": {
                    "total_companies": 0,
                    "total_investments": 0,
                    "total_lead_investments": 0,
                    "total_capital_deployed_usd": None,
                    "unique_companies": 0,
                },
            }

            execution_time_ms = (time.time() - start_time) * 1000
            return create_tool_output(
                tool_name="find_investor_portfolio",
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                metadata={"num_rounds": 0, "num_companies": 0},
            )

        # Group rounds by organization
        org_rounds: Dict[UUID, List[FundingRound]] = {}
        for round_obj in all_rounds:
            if round_obj.org_uuid is None:
                logger.warning(
                    f"Skipping funding round {round_obj.funding_round_uuid} - missing org_uuid"
                )
                continue

            if round_obj.org_uuid not in org_rounds:
                org_rounds[round_obj.org_uuid] = []
            org_rounds[round_obj.org_uuid].append(round_obj)

        # Get organization details for all portfolio companies
        org_uuids = list(org_rounds.keys())
        logger.info(f"Fetching details for {len(org_uuids)} portfolio companies")

        # Query organizations individually (OrganizationModel doesn't support batch queries)
        # Use asyncio.gather for parallel queries to improve performance
        async def fetch_org(uuid: UUID) -> Optional[Organization]:
            """Fetch a single organization by UUID."""
            try:
                orgs = await org_model.get(org_uuid=uuid)
                return orgs[0] if orgs else None
            except Exception as e:
                logger.warning(f"Failed to fetch organization {uuid}: {e}")
                return None
        
        # Query organizations in parallel (batch of 50 at a time to avoid overwhelming the DB)
        BATCH_SIZE = 50
        all_orgs = []
        for i in range(0, len(org_uuids), BATCH_SIZE):
            batch = org_uuids[i : i + BATCH_SIZE]
            batch_results = await asyncio.gather(*[fetch_org(uuid) for uuid in batch])
            all_orgs.extend([org for org in batch_results if org is not None])

        # Create org lookup
        org_lookup = {org.org_uuid: org for org in all_orgs}

        # Build portfolio companies list
        portfolio_companies = []
        total_capital_deployed = 0
        total_lead_investments = 0

        for org_uuid, rounds in org_rounds.items():
            org = org_lookup.get(org_uuid)

            # Count investments and lead investments
            investment_count = 0
            lead_investment_count = 0
            total_invested = 0
            first_investment_date = None
            last_investment_date = None

            funding_rounds = []
            for round_obj in rounds:
                # Check if investor was lead - if round came from lead_investors query, it's a lead investment
                was_lead = round_obj.funding_round_uuid in lead_round_uuids

                if was_lead:
                    lead_investment_count += 1
                investment_count += 1

                # Track investment amounts (if available)
                if round_obj.fundraise_amount_usd:
                    total_invested += round_obj.fundraise_amount_usd

                # Track dates
                if round_obj.investment_date:
                    if first_investment_date is None or round_obj.investment_date < first_investment_date:
                        first_investment_date = round_obj.investment_date
                    if last_investment_date is None or round_obj.investment_date > last_investment_date:
                        last_investment_date = round_obj.investment_date

                funding_rounds.append(
                    {
                        "funding_round_uuid": str(round_obj.funding_round_uuid),
                        "investment_date": (
                            round_obj.investment_date.isoformat()
                            if round_obj.investment_date
                            else None
                        ),
                        "fundraise_amount_usd": round_obj.fundraise_amount_usd,
                        "valuation_usd": round_obj.valuation_usd,
                        "stage": round_obj.stage or round_obj.general_funding_stage,
                        "was_lead": was_lead,
                    }
                )

            total_capital_deployed += total_invested
            total_lead_investments += lead_investment_count

            portfolio_companies.append(
                {
                    "org_uuid": str(org_uuid),
                    "name": org.name if org else None,
                    "total_funding_usd": org.total_funding_usd if org else None,
                    "stage": org.stage or org.general_funding_stage if org else None,
                    "founding_date": (
                        org.founding_date.isoformat() if org and org.founding_date else None
                    ),
                    "investment_count": investment_count,
                    "lead_investment_count": lead_investment_count,
                    "total_invested_usd": total_invested if total_invested > 0 else None,
                    "first_investment_date": (
                        first_investment_date.isoformat() if first_investment_date else None
                    ),
                    "last_investment_date": (
                        last_investment_date.isoformat() if last_investment_date else None
                    ),
                    "funding_rounds": funding_rounds,
                }
            )

        # Sort by total invested (descending)
        portfolio_companies.sort(
            key=lambda x: x.get("total_invested_usd") or 0, reverse=True
        )

        # Build summary
        summary = {
            "total_companies": len(portfolio_companies),
            "total_investments": len(all_rounds),
            "total_lead_investments": total_lead_investments,
            "total_capital_deployed_usd": total_capital_deployed if total_capital_deployed > 0 else None,
            "unique_companies": len(portfolio_companies),
        }

        # Build result
        result = {
            "investor_name": investor_name,
            "time_period": {
                "start": time_period_start,
                "end": time_period_end,
            },
            "portfolio_companies": portfolio_companies,
            "summary": summary,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "num_rounds": len(all_rounds),
            "num_companies": len(portfolio_companies),
            "include_lead_only": include_lead_only,
        }

        logger.debug(
            f"Found portfolio for {investor_name}: {len(portfolio_companies)} companies, "
            f"{len(all_rounds)} investments in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="find_investor_portfolio",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to find investor portfolio: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="find_investor_portfolio",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )

