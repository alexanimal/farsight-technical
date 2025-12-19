"""Tool for semantically searching organizations in Pinecone.

This tool provides a high-level interface for querying organizations using
the PineconeOrganizationModel. It can be called by agents to perform semantic
search on organization data stored in Pinecone vector database.
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from src.contracts.tool_io import (ToolMetadata, ToolOutput,
                                   ToolParameterSchema, create_tool_output)
from src.models.pinecone_organizations import (PineconeOrganization,
                                               PineconeOrganizationModel)

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the semantic_search_organizations tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="semantic_search_organizations",
        description="Semantically search organizations in Pinecone using text query and metadata filters",
        version="1.0.0",
        parameters=[
            ToolParameterSchema(
                name="text",
                type="string",
                description="Text query to embed and search for (required). This will be embedded using text-embedding-3-large and used for semantic similarity search",
                required=True,
            ),
            ToolParameterSchema(
                name="org_uuid",
                type="string",
                description="Exact match for organization UUID (as string)",
                required=False,
            ),
            ToolParameterSchema(
                name="name",
                type="string",
                description="Exact match for organization name",
                required=False,
            ),
            ToolParameterSchema(
                name="categories_contains",
                type="string",
                description="Check if categories array contains this value",
                required=False,
            ),
            ToolParameterSchema(
                name="org_status",
                type="string",
                description="Exact match for organization status (operating, closed, was_acquired, or ipo)",
                required=False,
            ),
            ToolParameterSchema(
                name="total_funding_usd_min",
                type="number",
                description="Filter with total_funding_usd >= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="total_funding_usd_max",
                type="number",
                description="Filter with total_funding_usd <= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="founding_date_from",
                type="string",
                description="Filter organizations founded on or after this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="founding_date_to",
                type="string",
                description="Filter organizations founded on or before this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="last_fundraise_date_from",
                type="string",
                description="Filter with last_fundraise_date >= this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="last_fundraise_date_to",
                type="string",
                description="Filter with last_fundraise_date <= this date (ISO format string)",
                required=False,
            ),
            ToolParameterSchema(
                name="employee_count",
                type="string",
                description="Exact match for employee count range",
                required=False,
            ),
            ToolParameterSchema(
                name="org_type",
                type="string",
                description="Exact match for organization type (investor or company)",
                required=False,
            ),
            ToolParameterSchema(
                name="stage",
                type="string",
                description="Exact match for funding stage",
                required=False,
            ),
            ToolParameterSchema(
                name="valuation_usd_min",
                type="number",
                description="Filter with valuation_usd >= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="valuation_usd_max",
                type="number",
                description="Filter with valuation_usd <= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="investors_contains",
                type="string",
                description="Check if investors array contains this UUID (as string)",
                required=False,
            ),
            ToolParameterSchema(
                name="general_funding_stage",
                type="string",
                description="Exact match for general funding stage",
                required=False,
            ),
            ToolParameterSchema(
                name="num_acquisitions_min",
                type="integer",
                description="Filter with num_acquisitions >= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="num_acquisitions_max",
                type="integer",
                description="Filter with num_acquisitions <= this value",
                required=False,
            ),
            ToolParameterSchema(
                name="revenue_range",
                type="string",
                description="Exact match for revenue range",
                required=False,
            ),
            ToolParameterSchema(
                name="top_k",
                type="integer",
                description="Number of results to return (default: 10)",
                required=False,
                default=10,
            ),
            ToolParameterSchema(
                name="index_name",
                type="string",
                description="Name of the Pinecone index to query. If not provided, uses the default index from settings",
                required=False,
            ),
        ],
        returns={
            "type": "array",
            "items": {
                "type": "object",
                "description": "Organization record with all fields from the PineconeOrganization model, including a 'score' field indicating the similarity score from Pinecone",
            },
        },
        cost_per_call=None,  # Pinecone query, minimal cost (may have API costs)
        estimated_latency_ms=200.0,  # Typical Pinecone query latency (includes embedding generation)
        timeout_seconds=30.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=[
            "pinecone",
            "vector-search",
            "semantic-search",
            "organizations",
            "read-only",
        ],
    )


async def semantic_search_organizations(
    text: str,
    org_uuid: Optional[str] = None,
    name: Optional[str] = None,
    categories_contains: Optional[str] = None,
    org_status: Optional[str] = None,
    total_funding_usd_min: Optional[float] = None,
    total_funding_usd_max: Optional[float] = None,
    founding_date_from: Optional[str] = None,
    founding_date_to: Optional[str] = None,
    last_fundraise_date_from: Optional[str] = None,
    last_fundraise_date_to: Optional[str] = None,
    employee_count: Optional[str] = None,
    org_type: Optional[str] = None,
    stage: Optional[str] = None,
    valuation_usd_min: Optional[float] = None,
    valuation_usd_max: Optional[float] = None,
    investors_contains: Optional[str] = None,
    general_funding_stage: Optional[str] = None,
    num_acquisitions_min: Optional[int] = None,
    num_acquisitions_max: Optional[int] = None,
    revenue_range: Optional[str] = None,
    top_k: int = 10,
    index_name: Optional[str] = None,
) -> ToolOutput:
    """Semantically search organizations in Pinecone using text query and metadata filters.

    This tool embeds the provided text query and searches Pinecone for similar
    organizations, optionally filtered by metadata criteria.

    Args:
        text: Text query to embed and search for (required). This will be embedded
            using text-embedding-3-large and used for semantic similarity search.
        org_uuid: Exact match for organization UUID (as string).
        name: Exact match for organization name.
        categories_contains: Check if categories array contains this value.
        org_status: Exact match for organization status (operating, closed,
            was_acquired, or ipo).
        total_funding_usd_min: Filter with total_funding_usd >= this value.
        total_funding_usd_max: Filter with total_funding_usd <= this value.
        founding_date_from: Filter organizations founded on or after this date (ISO format string).
        founding_date_to: Filter organizations founded on or before this date (ISO format string).
        last_fundraise_date_from: Filter with last_fundraise_date >= this date (ISO format string).
        last_fundraise_date_to: Filter with last_fundraise_date <= this date (ISO format string).
        employee_count: Exact match for employee count range.
        org_type: Exact match for organization type (investor or company).
        stage: Exact match for funding stage.
        valuation_usd_min: Filter with valuation_usd >= this value.
        valuation_usd_max: Filter with valuation_usd <= this value.
        investors_contains: Check if investors array contains this UUID (as string).
        general_funding_stage: Exact match for general funding stage.
        num_acquisitions_min: Filter with num_acquisitions >= this value.
        num_acquisitions_max: Filter with num_acquisitions <= this value.
        revenue_range: Exact match for revenue range.
        top_k: Number of results to return (default: 10).
        index_name: Name of the Pinecone index to query. If not provided,
            uses the default index from settings.

    Returns:
        ToolOutput object containing:
        - success: Whether the query succeeded
        - result: List of organization records as dictionaries (if successful).
            Each dictionary contains all fields from the PineconeOrganization model,
            including a 'score' field indicating the similarity score from Pinecone.
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute the query
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Simple text query
        orgs = await semantic_search_organizations(
            text="AI companies",
            top_k=10
        )

        # Query with metadata filters
        orgs = await semantic_search_organizations(
            text="healthcare startups",
            org_status="operating",
            total_funding_usd_min=1000000,
            general_funding_stage="late_stage_venture",
            top_k=5
        )

        # Query with date range
        orgs = await semantic_search_organizations(
            text="fintech companies",
            founding_date_from="2020-01-01T00:00:00",
            founding_date_to="2023-12-31T23:59:59",
            top_k=20
        )
        ```
    """
    start_time = time.time()
    try:
        # Initialize the model
        model = PineconeOrganizationModel()
        await model.initialize()

        # Convert string UUIDs to UUID objects if provided
        org_uuid_obj: Optional[UUID] = None
        if org_uuid is not None:
            org_uuid_obj = UUID(org_uuid)

        investors_contains_obj: Optional[UUID] = None
        if investors_contains is not None:
            investors_contains_obj = UUID(investors_contains)

        # Convert date strings to datetime objects if provided
        founding_date_from_obj: Optional[datetime] = None
        if founding_date_from is not None:
            founding_date_from_obj = datetime.fromisoformat(founding_date_from)

        founding_date_to_obj: Optional[datetime] = None
        if founding_date_to is not None:
            founding_date_to_obj = datetime.fromisoformat(founding_date_to)

        last_fundraise_date_from_obj: Optional[datetime] = None
        if last_fundraise_date_from is not None:
            last_fundraise_date_from_obj = datetime.fromisoformat(
                last_fundraise_date_from
            )

        last_fundraise_date_to_obj: Optional[datetime] = None
        if last_fundraise_date_to is not None:
            last_fundraise_date_to_obj = datetime.fromisoformat(last_fundraise_date_to)

        # Query the model
        organizations = await model.query(
            text=text,
            org_uuid=org_uuid_obj,
            name=name,
            categories_contains=categories_contains,
            org_status=org_status,
            total_funding_usd_min=total_funding_usd_min,
            total_funding_usd_max=total_funding_usd_max,
            founding_date_from=founding_date_from_obj,
            founding_date_to=founding_date_to_obj,
            last_fundraise_date_from=last_fundraise_date_from_obj,
            last_fundraise_date_to=last_fundraise_date_to_obj,
            employee_count=employee_count,
            org_type=org_type,
            stage=stage,
            valuation_usd_min=valuation_usd_min,
            valuation_usd_max=valuation_usd_max,
            investors_contains=investors_contains_obj,
            general_funding_stage=general_funding_stage,
            num_acquisitions_min=num_acquisitions_min,
            num_acquisitions_max=num_acquisitions_max,
            revenue_range=revenue_range,
            top_k=top_k,
            index_name=index_name,
        )

        # Convert Pydantic models to dictionaries
        result = [org.model_dump() for org in organizations]
        execution_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Retrieved {len(result)} organization(s) from semantic search in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="semantic_search_organizations",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata={"num_results": len(result), "top_k": top_k},
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to semantically search organizations: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="semantic_search_organizations",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
