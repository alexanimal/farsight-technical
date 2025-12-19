"""Tool for retrieving acquisition data from the database.

This tool provides a high-level interface for querying acquisitions using
the AcquisitionModel. It can be called by agents to fetch acquisition records
based on various filter criteria.
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from src.contracts.tool_io import (ToolMetadata, ToolOutput,
                                   ToolParameterSchema, create_tool_output)
from src.models.acquisitions import Acquisition, AcquisitionModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the get_acquisitions tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="get_acquisitions",
        description="Search for company acquisitions by various criteria (UUIDs, dates, prices, etc.)",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="acquisition_uuid",
                type="string",
                description="Exact match for acquisition UUID (as string)",
                required=False,
            ),
            ToolParameterSchema(
                name="acquiree_uuid",
                type="string",
                description="Exact match for acquiree UUID (as string)",
                required=False,
            ),
            ToolParameterSchema(
                name="acquirer_uuid",
                type="string",
                description="Exact match for acquirer UUID (as string)",
                required=False,
            ),
            ToolParameterSchema(
                name="acquisition_type",
                type="string",
                description="Exact match for acquisition type",
                required=False,
            ),
            ToolParameterSchema(
                name="acquisition_announce_date",
                type="string",
                description="Exact match for announce date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="acquisition_announce_date_from",
                type="string",
                description="Filter acquisitions on or after this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="acquisition_announce_date_to",
                type="string",
                description="Filter acquisitions on or before this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="acquisition_price_usd",
                type="integer",
                description="Exact match for acquisition price in USD",
                required=False,
            ),
            ToolParameterSchema(
                name="acquisition_price_usd_min",
                type="integer",
                description="Filter acquisitions with price >= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="acquisition_price_usd_max",
                type="integer",
                description="Filter acquisitions with price <= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="terms",
                type="string",
                description="Exact match for terms",
                required=False,
            ),
            ToolParameterSchema(
                name="terms_ilike",
                type="string",
                description="Case-insensitive partial match for terms",
                required=False,
            ),
            ToolParameterSchema(
                name="acquirer_type",
                type="string",
                description="Exact match for acquirer type",
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
                "description": "Acquisition record with all fields from the Acquisition model",
            },
        },
        cost_per_call=None,  # Database query, minimal cost
        estimated_latency_ms=100.0,  # Typical database query latency
        timeout_seconds=30.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["database", "acquisitions", "read-only"],
    )


async def get_acquisitions(
    acquisition_uuid: Optional[str] = None,
    acquiree_uuid: Optional[str] = None,
    acquirer_uuid: Optional[str] = None,
    acquisition_type: Optional[str] = None,
    acquisition_announce_date: Optional[str] = None,
    acquisition_announce_date_from: Optional[str] = None,
    acquisition_announce_date_to: Optional[str] = None,
    acquisition_price_usd: Optional[int] = None,
    acquisition_price_usd_min: Optional[int] = None,
    acquisition_price_usd_max: Optional[int] = None,
    terms: Optional[str] = None,
    terms_ilike: Optional[str] = None,
    acquirer_type: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> ToolOutput:
    """Get acquisitions matching the specified filters.

    This tool queries the acquisitions table using the AcquisitionModel
    and returns a list of acquisition records as dictionaries.

    Args:
        acquisition_uuid: Exact match for acquisition UUID (as string).
        acquiree_uuid: Exact match for acquiree UUID (as string).
        acquirer_uuid: Exact match for acquirer UUID (as string).
        acquisition_type: Exact match for acquisition type.
        acquisition_announce_date: Exact match for announce date (ISO format string).
        acquisition_announce_date_from: Filter acquisitions on or after this date (ISO format string).
        acquisition_announce_date_to: Filter acquisitions on or before this date (ISO format string).
        acquisition_price_usd: Exact match for acquisition price in USD.
        acquisition_price_usd_min: Filter acquisitions with price >= this value.
        acquisition_price_usd_max: Filter acquisitions with price <= this value.
        terms: Exact match for terms.
        terms_ilike: Case-insensitive partial match for terms.
        acquirer_type: Exact match for acquirer type.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).

    Returns:
        ToolOutput object containing:
        - success: Whether the query succeeded
        - result: List of acquisition records as dictionaries (if successful).
            Each dictionary contains all fields from the Acquisition model.
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute the query
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Get all acquisitions
        acquisitions = await get_acquisitions()

        # Get by UUID
        acquisition = await get_acquisitions(acquisition_uuid="123e4567-e89b-12d3-a456-426614174000")

        # Get acquisitions with price >= 1M USD
        expensive = await get_acquisitions(acquisition_price_usd_min=1000000)

        # Get acquisitions announced in a date range
        recent = await get_acquisitions(
            acquisition_announce_date_from="2020-01-01T00:00:00",
            acquisition_announce_date_to="2023-12-31T23:59:59"
        )
        ```
    """
    start_time = time.time()
    try:
        # Initialize the model
        model = AcquisitionModel()
        await model.initialize()

        # Convert string UUIDs to UUID objects if provided
        acquisition_uuid_obj: Optional[UUID] = None
        if acquisition_uuid is not None:
            acquisition_uuid_obj = UUID(acquisition_uuid)

        acquiree_uuid_obj: Optional[UUID] = None
        if acquiree_uuid is not None:
            acquiree_uuid_obj = UUID(acquiree_uuid)

        acquirer_uuid_obj: Optional[UUID] = None
        if acquirer_uuid is not None:
            acquirer_uuid_obj = UUID(acquirer_uuid)

        # Convert date strings to datetime objects if provided
        announce_date_obj: Optional[datetime] = None
        if acquisition_announce_date is not None:
            announce_date_obj = datetime.fromisoformat(acquisition_announce_date)

        announce_date_from_obj: Optional[datetime] = None
        if acquisition_announce_date_from is not None:
            announce_date_from_obj = datetime.fromisoformat(
                acquisition_announce_date_from
            )

        announce_date_to_obj: Optional[datetime] = None
        if acquisition_announce_date_to is not None:
            announce_date_to_obj = datetime.fromisoformat(acquisition_announce_date_to)

        # Query the model
        acquisitions = await model.get(
            acquisition_uuid=acquisition_uuid_obj,
            acquiree_uuid=acquiree_uuid_obj,
            acquirer_uuid=acquirer_uuid_obj,
            acquisition_type=acquisition_type,
            acquisition_announce_date=announce_date_obj,
            acquisition_announce_date_from=announce_date_from_obj,
            acquisition_announce_date_to=announce_date_to_obj,
            acquisition_price_usd=acquisition_price_usd,
            acquisition_price_usd_min=acquisition_price_usd_min,
            acquisition_price_usd_max=acquisition_price_usd_max,
            terms=terms,
            terms_ilike=terms_ilike,
            acquirer_type=acquirer_type,
            limit=limit,
            offset=offset,
        )

        # Convert Pydantic models to dictionaries
        result = [acquisition.model_dump() for acquisition in acquisitions]
        execution_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Retrieved {len(result)} acquisition(s) in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="get_acquisitions",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata={"num_results": len(result)},
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to get acquisitions: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="get_acquisitions",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
