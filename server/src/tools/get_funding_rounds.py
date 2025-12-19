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
                description="Check if investors array contains this value",
                required=False,
            ),
            ToolParameterSchema(
                name="lead_investors_contains",
                type="string",
                description="Check if lead_investors array contains this value",
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
        ],
        returns={
            "type": "array",
            "items": {
                "type": "object",
                "description": "Funding round record with all fields from the FundingRound model",
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
    general_funding_stage: Optional[str] = None,
    stage: Optional[str] = None,
    investors_contains: Optional[str] = None,
    lead_investors_contains: Optional[str] = None,
    fundraise_amount_usd: Optional[int] = None,
    fundraise_amount_usd_min: Optional[int] = None,
    fundraise_amount_usd_max: Optional[int] = None,
    valuation_usd: Optional[int] = None,
    valuation_usd_min: Optional[int] = None,
    valuation_usd_max: Optional[int] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
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
        general_funding_stage: Exact match for general funding stage.
        stage: Exact match for specific funding stage.
        investors_contains: Check if investors array contains this value.
        lead_investors_contains: Check if lead_investors array contains this value.
        fundraise_amount_usd: Exact match for fundraise amount in USD.
        fundraise_amount_usd_min: Filter funding rounds with amount >= this value.
        fundraise_amount_usd_max: Filter funding rounds with amount <= this value.
        valuation_usd: Exact match for valuation in USD.
        valuation_usd_min: Filter funding rounds with valuation >= this value.
        valuation_usd_max: Filter funding rounds with valuation <= this value.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).

    Returns:
        ToolOutput object containing:
        - success: Whether the query succeeded
        - result: List of funding round records as dictionaries (if successful).
            Each dictionary contains all fields from the FundingRound model.
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute the query
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Get all funding rounds
        rounds = await get_funding_rounds()

        # Get by UUID
        round = await get_funding_rounds(funding_round_uuid="123e4567-e89b-12d3-a456-426614174000")

        # Get funding rounds with amount >= 1M USD
        large_rounds = await get_funding_rounds(fundraise_amount_usd_min=1000000)

        # Get funding rounds in a date range
        recent = await get_funding_rounds(
            investment_date_from="2020-01-01T00:00:00",
            investment_date_to="2023-12-31T23:59:59"
        )

        # Get rounds with specific investor
        investor_rounds = await get_funding_rounds(investors_contains="Sequoia Capital")
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
            general_funding_stage=general_funding_stage,
            stage=stage,
            investors_contains=investors_contains,
            lead_investors_contains=lead_investors_contains,
            fundraise_amount_usd=fundraise_amount_usd,
            fundraise_amount_usd_min=fundraise_amount_usd_min,
            fundraise_amount_usd_max=fundraise_amount_usd_max,
            valuation_usd=valuation_usd,
            valuation_usd_min=valuation_usd_min,
            valuation_usd_max=valuation_usd_max,
            limit=limit,
            offset=offset,
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
