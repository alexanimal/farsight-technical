"""Tool for retrieving funding round data from the database.

This tool provides a high-level interface for querying funding rounds using
the FundingRoundModel. It can be called by agents to fetch funding round records
based on various filter criteria.
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from src.contracts.tool_io import (ToolMetadata, ToolOutput,
                                   ToolParameterSchema, create_tool_output)
from src.models.funding_rounds import FundingRound, FundingRoundModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the get_funding_rounds tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="get_funding_rounds",
        description="Search for funding rounds by various criteria (UUIDs, dates, amounts, investors, etc.)",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="funding_round_uuid",
                type="string",
                description="Exact match for funding round UUID (as string)",
                required=False,
            ),
            ToolParameterSchema(
                name="investment_date",
                type="string",
                description="Exact match for investment date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="investment_date_from",
                type="string",
                description="Filter funding rounds on or after this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="investment_date_to",
                type="string",
                description="Filter funding rounds on or before this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="org_uuid",
                type="string",
                description="Exact match for organization UUID (as string)",
                required=False,
            ),
            ToolParameterSchema(
                name="org_name_ilike",
                type="string",
                description="Search by organization name (case-insensitive partial match). This performs a JOIN to the organizations table to match by name. Can be combined with org_uuid filter.",
                required=False,
            ),
            ToolParameterSchema(
                name="general_funding_stage",
                type="string",
                description="Exact match for general funding stage",
                required=False,
            ),
            ToolParameterSchema(
                name="stage",
                type="string",
                description="Exact match for specific funding stage",
                required=False,
            ),
            ToolParameterSchema(
                name="investors_contains",
                type="string",
                description="Check if investors array contains this UUID (as string). Note: This expects a UUID, not an investor name. Use investor_name_contains for name-based search.",
                required=False,
            ),
            ToolParameterSchema(
                name="lead_investors_contains",
                type="string",
                description="Check if lead_investors array contains this UUID (as string). Note: This expects a UUID, not an investor name. Use lead_investor_name_contains for name-based search.",
                required=False,
            ),
            ToolParameterSchema(
                name="investor_name_contains",
                type="string",
                description="Search investors by organization name (case-insensitive partial match). This performs a JOIN to the organizations table to match by name.",
                required=False,
            ),
            ToolParameterSchema(
                name="lead_investor_name_contains",
                type="string",
                description="Search lead_investors by organization name (case-insensitive partial match). This performs a JOIN to the organizations table to match by name.",
                required=False,
            ),
            ToolParameterSchema(
                name="fundraise_amount_usd",
                type="integer",
                description="Exact match for fundraise amount in USD",
                required=False,
            ),
            ToolParameterSchema(
                name="fundraise_amount_usd_min",
                type="integer",
                description="Filter funding rounds with amount >= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="fundraise_amount_usd_max",
                type="integer",
                description="Filter funding rounds with amount <= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="valuation_usd",
                type="integer",
                description="Exact match for valuation in USD",
                required=False,
            ),
            ToolParameterSchema(
                name="valuation_usd_min",
                type="integer",
                description="Filter funding rounds with valuation >= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="valuation_usd_max",
                type="integer",
                description="Filter funding rounds with valuation <= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="limit",
                type="integer",
                description="Maximum number of results to return",
                required=False,
            ),
            ToolParameterSchema(
                name="offset",
                type="integer",
                description="Number of results to skip (for pagination)",
                required=False,
            ),
            ToolParameterSchema(
                name="include_organizations",
                type="boolean",
                description="If true, includes nested organization details for org_uuid, investors, and lead_investors. Returns FundingRoundWithOrganizations objects with full organization data.",
                required=False,
            ),
            ToolParameterSchema(
                name="order_by",
                type="string",
                description="Field to order results by. Must be one of: 'investment_date', 'fundraise_amount_usd', 'valuation_usd'. Defaults to 'investment_date' if not specified.",
                required=False,
            ),
            ToolParameterSchema(
                name="order_direction",
                type="string",
                description="Direction to order results. Must be 'asc' or 'desc'. Defaults to 'desc' if not specified.",
                required=False,
            ),
        ],
        returns={
            "type": "array",
            "items": {
                "type": "object",
                "description": "Funding round record with all fields from the FundingRound model. If include_organizations is true, also includes nested organization, investors_organizations, and lead_investors_organizations fields.",
            },
        },
        cost_per_call=None,  # Database query, minimal cost
        estimated_latency_ms=100.0,  # Typical database query latency
        timeout_seconds=30.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["database", "funding", "read-only"],
    )


async def get_funding_rounds(
    funding_round_uuid: Optional[str] = None,
    investment_date: Optional[str] = None,
    investment_date_from: Optional[str] = None,
    investment_date_to: Optional[str] = None,
    org_uuid: Optional[str] = None,
    org_name_ilike: Optional[str] = None,
    general_funding_stage: Optional[str] = None,
    stage: Optional[str] = None,
    investors_contains: Optional[str] = None,
    lead_investors_contains: Optional[str] = None,
    investor_name_contains: Optional[str] = None,
    lead_investor_name_contains: Optional[str] = None,
    fundraise_amount_usd: Optional[int] = None,
    fundraise_amount_usd_min: Optional[int] = None,
    fundraise_amount_usd_max: Optional[int] = None,
    valuation_usd: Optional[int] = None,
    valuation_usd_min: Optional[int] = None,
    valuation_usd_max: Optional[int] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    include_organizations: bool = False,
    order_by: Optional[str] = None,
    order_direction: Optional[str] = None,
) -> ToolOutput:
    """Get funding rounds matching the specified filters.

    This tool queries the fundingrounds table using the FundingRoundModel
    and returns a list of funding round records as dictionaries.

    Args:
        funding_round_uuid: Exact match for funding round UUID (as string).
        investment_date: Exact match for investment date (ISO format string).
        investment_date_from: Filter funding rounds on or after this date (ISO format string).
        investment_date_to: Filter funding rounds on or before this date (ISO format string).
        org_uuid: Exact match for organization UUID (as string).
        org_name_ilike: Search by organization name (case-insensitive partial match).
            This performs a JOIN to the organizations table to match by name.
            Can be combined with org_uuid filter.
        general_funding_stage: Exact match for general funding stage.
        stage: Exact match for specific funding stage.
        investors_contains: Check if investors array contains this UUID (as string).
            Note: This expects a UUID, not an investor name. Use investor_name_contains for name-based search.
        lead_investors_contains: Check if lead_investors array contains this UUID (as string).
            Note: This expects a UUID, not an investor name. Use lead_investor_name_contains for name-based search.
        investor_name_contains: Search investors by organization name (case-insensitive partial match).
            This performs a JOIN to the organizations table to match by name.
        lead_investor_name_contains: Search lead_investors by organization name (case-insensitive partial match).
            This performs a JOIN to the organizations table to match by name.
        fundraise_amount_usd: Exact match for fundraise amount in USD.
        fundraise_amount_usd_min: Filter funding rounds with amount >= this value.
        fundraise_amount_usd_max: Filter funding rounds with amount <= this value.
        valuation_usd: Exact match for valuation in USD.
        valuation_usd_min: Filter funding rounds with valuation >= this value.
        valuation_usd_max: Filter funding rounds with valuation <= this value.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).
        include_organizations: If True, includes nested organization details for org_uuid,
            investors, and lead_investors. Returns FundingRoundWithOrganizations objects
            with full organization data instead of just UUIDs.
        order_by: Field to order results by. Must be one of: "investment_date",
            "fundraise_amount_usd", "valuation_usd". Defaults to "investment_date" if not specified.
        order_direction: Direction to order results. Must be "asc" or "desc".
            Defaults to "desc" if not specified.

    Returns:
        ToolOutput object containing:
        - success: Whether the query succeeded
        - result: List of funding round records as dictionaries (if successful).
            Each dictionary contains all fields from the FundingRound model.
            If include_organizations=True, also includes:
            - organization: Full Organization object for the company receiving funding
            - investors_organizations: List of Organization objects for all investors
            - lead_investors_organizations: List of Organization objects for lead investors
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute the query
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Get all funding rounds
        rounds = await get_funding_rounds()

        # Get by UUID
        round = await get_funding_rounds(funding_round_uuid="123e4567-e89b-12d3-a456-426614174000")

        # Get funding rounds by organization name
        org_rounds = await get_funding_rounds(org_name_ilike="TechCorp")

        # Get funding rounds with amount >= 1M USD
        large_rounds = await get_funding_rounds(fundraise_amount_usd_min=1000000)

        # Get funding rounds in a date range
        recent = await get_funding_rounds(
            investment_date_from="2020-01-01T00:00:00",
            investment_date_to="2023-12-31T23:59:59"
        )

        # Get rounds with specific investor by name
        investor_rounds = await get_funding_rounds(investor_name_contains="Sequoia Capital")
        
        # Get rounds with specific investor by UUID
        investor_rounds_by_uuid = await get_funding_rounds(investors_contains="123e4567-e89b-12d3-a456-426614174000")
        
        # Get rounds with full organization details
        rounds_with_orgs = await get_funding_rounds(
            org_uuid="123e4567-e89b-12d3-a456-426614174000",
            include_organizations=True
        )
        
        # Get rounds ordered by fundraise amount (descending)
        large_rounds = await get_funding_rounds(
            fundraise_amount_usd_min=1000000,
            order_by="fundraise_amount_usd",
            order_direction="desc"
        )
        
        # Get rounds ordered by investment date (ascending)
        oldest_first = await get_funding_rounds(
            order_by="investment_date",
            order_direction="asc"
        )
        ```
    """
    start_time = time.time()
    try:
        # Initialize the model
        model = FundingRoundModel()
        await model.initialize()

        # Convert string UUIDs to UUID objects if provided
        funding_round_uuid_obj: Optional[UUID] = None
        if funding_round_uuid is not None:
            funding_round_uuid_obj = UUID(funding_round_uuid)

        org_uuid_obj: Optional[UUID] = None
        if org_uuid is not None:
            org_uuid_obj = UUID(org_uuid)

        # Convert date strings to datetime objects if provided
        investment_date_obj: Optional[datetime] = None
        if investment_date is not None:
            investment_date_obj = datetime.fromisoformat(investment_date)

        investment_date_from_obj: Optional[datetime] = None
        if investment_date_from is not None:
            investment_date_from_obj = datetime.fromisoformat(investment_date_from)

        investment_date_to_obj: Optional[datetime] = None
        if investment_date_to is not None:
            investment_date_to_obj = datetime.fromisoformat(investment_date_to)

        # Query the model
        funding_rounds = await model.get(
            funding_round_uuid=funding_round_uuid_obj,
            investment_date=investment_date_obj,
            investment_date_from=investment_date_from_obj,
            investment_date_to=investment_date_to_obj,
            org_uuid=org_uuid_obj,
            org_name_ilike=org_name_ilike,
            general_funding_stage=general_funding_stage,
            stage=stage,
            investors_contains=investors_contains,
            lead_investors_contains=lead_investors_contains,
            investor_name_contains=investor_name_contains,
            lead_investor_name_contains=lead_investor_name_contains,
            fundraise_amount_usd=fundraise_amount_usd,
            fundraise_amount_usd_min=fundraise_amount_usd_min,
            fundraise_amount_usd_max=fundraise_amount_usd_max,
            valuation_usd=valuation_usd,
            valuation_usd_min=valuation_usd_min,
            valuation_usd_max=valuation_usd_max,
            limit=limit,
            offset=offset,
            include_organizations=include_organizations,
            order_by=order_by,
            order_direction=order_direction,
        )

        # Convert Pydantic models to dictionaries
        result = [funding_round.model_dump() for funding_round in funding_rounds]
        execution_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Retrieved {len(result)} funding round(s) in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="get_funding_rounds",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata={"num_results": len(result)},
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to get funding rounds: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="get_funding_rounds",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
