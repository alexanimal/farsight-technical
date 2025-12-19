"""Tool for retrieving organization data from the database.

This tool provides a high-level interface for querying organizations using
the OrganizationModel. It can be called by agents to fetch organization records
based on various filter criteria.
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from src.contracts.tool_io import (ToolMetadata, ToolOutput,
                                   ToolParameterSchema, create_tool_output)
from src.models.organizations import Organization, OrganizationModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the get_organizations tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    # Build parameter list - this tool has many parameters
    parameters = [
        ToolParameterSchema(
            name="org_uuid",
            type="string",
            description="Exact match for organization UUID (as string)",
            required=False,
        ),
        ToolParameterSchema(
            name="cb_url",
            type="string",
            description="Exact match for Crunchbase URL",
            required=False,
        ),
        ToolParameterSchema(
            name="categories_contains",
            type="string",
            description="Check if categories array contains this value",
            required=False,
        ),
        ToolParameterSchema(
            name="category_groups_contains",
            type="string",
            description="Check if category_groups array contains this value",
            required=False,
        ),
        ToolParameterSchema(
            name="closed_on",
            type="string",
            description="Exact match for closed date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="closed_on_from",
            type="string",
            description="Filter organizations closed on or after this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="closed_on_to",
            type="string",
            description="Filter organizations closed on or before this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="closed_on_precision",
            type="string",
            description="Exact match for closed_on precision",
            required=False,
        ),
        ToolParameterSchema(
            name="company_profit_type",
            type="string",
            description="Exact match for company profit type",
            required=False,
        ),
        ToolParameterSchema(
            name="created_at_from",
            type="string",
            description="Filter organizations created on or after this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="created_at_to",
            type="string",
            description="Filter organizations created on or before this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="raw_description_ilike",
            type="string",
            description="Case-insensitive partial match for raw description",
            required=False,
        ),
        ToolParameterSchema(
            name="rewritten_description_ilike",
            type="string",
            description="Case-insensitive partial match for rewritten description",
            required=False,
        ),
        ToolParameterSchema(
            name="total_funding_native",
            type="integer",
            description="Exact match for total funding native",
            required=False,
        ),
        ToolParameterSchema(
            name="total_funding_native_min",
            type="integer",
            description="Filter with total_funding_native >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="total_funding_native_max",
            type="integer",
            description="Filter with total_funding_native <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="total_funding_currency",
            type="string",
            description="Exact match for funding currency",
            required=False,
        ),
        ToolParameterSchema(
            name="total_funding_usd",
            type="integer",
            description="Exact match for total funding USD",
            required=False,
        ),
        ToolParameterSchema(
            name="total_funding_usd_min",
            type="integer",
            description="Filter with total_funding_usd >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="total_funding_usd_max",
            type="integer",
            description="Filter with total_funding_usd <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="exited_on",
            type="string",
            description="Exact match for exited date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="exited_on_from",
            type="string",
            description="Filter organizations exited on or after this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="exited_on_to",
            type="string",
            description="Filter organizations exited on or before this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="exited_on_precision",
            type="string",
            description="Exact match for exited_on precision",
            required=False,
        ),
        ToolParameterSchema(
            name="founding_date",
            type="string",
            description="Exact match for founding date (ISO format string)",
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
            name="founding_date_precision",
            type="string",
            description="Exact match for founding_date precision",
            required=False,
        ),
        ToolParameterSchema(
            name="general_funding_stage",
            type="string",
            description="Exact match for general funding stage",
            required=False,
        ),
        ToolParameterSchema(
            name="ipo_status",
            type="string",
            description="Exact match for IPO status",
            required=False,
        ),
        ToolParameterSchema(
            name="last_fundraise_date",
            type="string",
            description="Exact match for last fundraise date (ISO format string)",
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
            name="last_funding_total_usd",
            type="integer",
            description="Exact match for last funding total USD",
            required=False,
        ),
        ToolParameterSchema(
            name="last_funding_total_usd_min",
            type="integer",
            description="Filter with last_funding_total_usd >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="last_funding_total_usd_max",
            type="integer",
            description="Filter with last_funding_total_usd <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="stage",
            type="string",
            description="Exact match for stage",
            required=False,
        ),
        ToolParameterSchema(
            name="org_type",
            type="string",
            description="Exact match for organization type",
            required=False,
        ),
        ToolParameterSchema(
            name="city",
            type="string",
            description="Exact match for city",
            required=False,
        ),
        ToolParameterSchema(
            name="state",
            type="string",
            description="Exact match for state",
            required=False,
        ),
        ToolParameterSchema(
            name="country",
            type="string",
            description="Exact match for country",
            required=False,
        ),
        ToolParameterSchema(
            name="continent",
            type="string",
            description="Exact match for continent",
            required=False,
        ),
        ToolParameterSchema(
            name="name",
            type="string",
            description="Exact match for organization name",
            required=False,
        ),
        ToolParameterSchema(
            name="name_ilike",
            type="string",
            description="Case-insensitive partial match for organization name",
            required=False,
        ),
        ToolParameterSchema(
            name="num_acquisitions",
            type="integer",
            description="Exact match for number of acquisitions",
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
            name="employee_count",
            type="string",
            description="Exact match for employee count",
            required=False,
        ),
        ToolParameterSchema(
            name="num_funding_rounds",
            type="integer",
            description="Exact match for number of funding rounds",
            required=False,
        ),
        ToolParameterSchema(
            name="num_funding_rounds_min",
            type="integer",
            description="Filter with num_funding_rounds >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="num_funding_rounds_max",
            type="integer",
            description="Filter with num_funding_rounds <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="num_investments",
            type="integer",
            description="Exact match for number of investments",
            required=False,
        ),
        ToolParameterSchema(
            name="num_investments_min",
            type="integer",
            description="Filter with num_investments >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="num_investments_max",
            type="integer",
            description="Filter with num_investments <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="num_portfolio_organizations",
            type="integer",
            description="Exact match for number of portfolio organizations",
            required=False,
        ),
        ToolParameterSchema(
            name="num_portfolio_organizations_min",
            type="integer",
            description="Filter with num_portfolio_organizations >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="num_portfolio_organizations_max",
            type="integer",
            description="Filter with num_portfolio_organizations <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="operating_status",
            type="string",
            description="Exact match for operating status",
            required=False,
        ),
        ToolParameterSchema(
            name="cb_rank",
            type="integer",
            description="Exact match for Crunchbase rank",
            required=False,
        ),
        ToolParameterSchema(
            name="cb_rank_min",
            type="integer",
            description="Filter with cb_rank >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="cb_rank_max",
            type="integer",
            description="Filter with cb_rank <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="revenue_range",
            type="string",
            description="Exact match for revenue range",
            required=False,
        ),
        ToolParameterSchema(
            name="org_status",
            type="string",
            description="Exact match for organization status",
            required=False,
        ),
        ToolParameterSchema(
            name="updated_at_from",
            type="string",
            description="Filter organizations updated on or after this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="updated_at_to",
            type="string",
            description="Filter organizations updated on or before this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="valuation_usd",
            type="integer",
            description="Exact match for valuation USD",
            required=False,
        ),
        ToolParameterSchema(
            name="valuation_usd_min",
            type="integer",
            description="Filter with valuation_usd >= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="valuation_usd_max",
            type="integer",
            description="Filter with valuation_usd <= this value",
            required=False,
        ),
        ToolParameterSchema(
            name="valuation_date",
            type="string",
            description="Exact match for valuation date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="valuation_date_from",
            type="string",
            description="Filter with valuation_date >= this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="valuation_date_to",
            type="string",
            description="Filter with valuation_date <= this date (ISO format string)",
            required=False,
        ),
        ToolParameterSchema(
            name="org_domain",
            type="string",
            description="Exact match for organization domain",
            required=False,
        ),
        ToolParameterSchema(
            name="org_domain_ilike",
            type="string",
            description="Case-insensitive partial match for organization domain",
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
    ]

    return ToolMetadata(
        name="get_organizations",
        description="Search for organizations by various criteria (UUIDs, names, locations, funding, dates, etc.)",
        version="1.0.0",
        parameters=parameters,
        returns={
            "type": "array",
            "items": {
                "type": "object",
                "description": "Organization record with all fields from the Organization model",
            },
        },
        cost_per_call=None,  # Database query, minimal cost
        estimated_latency_ms=150.0,  # Typical database query latency (may be slower due to many filters)
        timeout_seconds=30.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["database", "organizations", "read-only"],
    )


async def get_organizations(
    org_uuid: Optional[str] = None,
    cb_url: Optional[str] = None,
    categories_contains: Optional[str] = None,
    category_groups_contains: Optional[str] = None,
    closed_on: Optional[str] = None,
    closed_on_from: Optional[str] = None,
    closed_on_to: Optional[str] = None,
    closed_on_precision: Optional[str] = None,
    company_profit_type: Optional[str] = None,
    created_at_from: Optional[str] = None,
    created_at_to: Optional[str] = None,
    raw_description_ilike: Optional[str] = None,
    rewritten_description_ilike: Optional[str] = None,
    total_funding_native: Optional[int] = None,
    total_funding_native_min: Optional[int] = None,
    total_funding_native_max: Optional[int] = None,
    total_funding_currency: Optional[str] = None,
    total_funding_usd: Optional[int] = None,
    total_funding_usd_min: Optional[int] = None,
    total_funding_usd_max: Optional[int] = None,
    exited_on: Optional[str] = None,
    exited_on_from: Optional[str] = None,
    exited_on_to: Optional[str] = None,
    exited_on_precision: Optional[str] = None,
    founding_date: Optional[str] = None,
    founding_date_from: Optional[str] = None,
    founding_date_to: Optional[str] = None,
    founding_date_precision: Optional[str] = None,
    general_funding_stage: Optional[str] = None,
    ipo_status: Optional[str] = None,
    last_fundraise_date: Optional[str] = None,
    last_fundraise_date_from: Optional[str] = None,
    last_fundraise_date_to: Optional[str] = None,
    last_funding_total_usd: Optional[int] = None,
    last_funding_total_usd_min: Optional[int] = None,
    last_funding_total_usd_max: Optional[int] = None,
    stage: Optional[str] = None,
    org_type: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    continent: Optional[str] = None,
    name: Optional[str] = None,
    name_ilike: Optional[str] = None,
    num_acquisitions: Optional[int] = None,
    num_acquisitions_min: Optional[int] = None,
    num_acquisitions_max: Optional[int] = None,
    employee_count: Optional[str] = None,
    num_funding_rounds: Optional[int] = None,
    num_funding_rounds_min: Optional[int] = None,
    num_funding_rounds_max: Optional[int] = None,
    num_investments: Optional[int] = None,
    num_investments_min: Optional[int] = None,
    num_investments_max: Optional[int] = None,
    num_portfolio_organizations: Optional[int] = None,
    num_portfolio_organizations_min: Optional[int] = None,
    num_portfolio_organizations_max: Optional[int] = None,
    operating_status: Optional[str] = None,
    cb_rank: Optional[int] = None,
    cb_rank_min: Optional[int] = None,
    cb_rank_max: Optional[int] = None,
    revenue_range: Optional[str] = None,
    org_status: Optional[str] = None,
    updated_at_from: Optional[str] = None,
    updated_at_to: Optional[str] = None,
    valuation_usd: Optional[int] = None,
    valuation_usd_min: Optional[int] = None,
    valuation_usd_max: Optional[int] = None,
    valuation_date: Optional[str] = None,
    valuation_date_from: Optional[str] = None,
    valuation_date_to: Optional[str] = None,
    org_domain: Optional[str] = None,
    org_domain_ilike: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> ToolOutput:
    """Get organizations matching the specified filters.

    This tool queries the organizations table using the OrganizationModel
    and returns a list of organization records as dictionaries.

    Args:
        org_uuid: Exact match for organization UUID (as string).
        cb_url: Exact match for Crunchbase URL.
        categories_contains: Check if categories array contains this value.
        category_groups_contains: Check if category_groups array contains this value.
        closed_on: Exact match for closed date (ISO format string).
        closed_on_from: Filter organizations closed on or after this date (ISO format string).
        closed_on_to: Filter organizations closed on or before this date (ISO format string).
        closed_on_precision: Exact match for closed_on precision.
        company_profit_type: Exact match for company profit type.
        created_at_from: Filter organizations created on or after this date (ISO format string).
        created_at_to: Filter organizations created on or before this date (ISO format string).
        raw_description_ilike: Case-insensitive partial match for raw description.
        rewritten_description_ilike: Case-insensitive partial match for rewritten description.
        total_funding_native: Exact match for total funding native.
        total_funding_native_min: Filter with total_funding_native >= this value.
        total_funding_native_max: Filter with total_funding_native <= this value.
        total_funding_currency: Exact match for funding currency.
        total_funding_usd: Exact match for total funding USD.
        total_funding_usd_min: Filter with total_funding_usd >= this value.
        total_funding_usd_max: Filter with total_funding_usd <= this value.
        exited_on: Exact match for exited date (ISO format string).
        exited_on_from: Filter organizations exited on or after this date (ISO format string).
        exited_on_to: Filter organizations exited on or before this date (ISO format string).
        exited_on_precision: Exact match for exited_on precision.
        founding_date: Exact match for founding date (ISO format string).
        founding_date_from: Filter organizations founded on or after this date (ISO format string).
        founding_date_to: Filter organizations founded on or before this date (ISO format string).
        founding_date_precision: Exact match for founding_date precision.
        general_funding_stage: Exact match for general funding stage.
        ipo_status: Exact match for IPO status.
        last_fundraise_date: Exact match for last fundraise date (ISO format string).
        last_fundraise_date_from: Filter with last_fundraise_date >= this date (ISO format string).
        last_fundraise_date_to: Filter with last_fundraise_date <= this date (ISO format string).
        last_funding_total_usd: Exact match for last funding total USD.
        last_funding_total_usd_min: Filter with last_funding_total_usd >= this value.
        last_funding_total_usd_max: Filter with last_funding_total_usd <= this value.
        stage: Exact match for stage.
        org_type: Exact match for organization type.
        city: Exact match for city.
        state: Exact match for state.
        country: Exact match for country.
        continent: Exact match for continent.
        name: Exact match for organization name.
        name_ilike: Case-insensitive partial match for organization name.
        num_acquisitions: Exact match for number of acquisitions.
        num_acquisitions_min: Filter with num_acquisitions >= this value.
        num_acquisitions_max: Filter with num_acquisitions <= this value.
        employee_count: Exact match for employee count.
        num_funding_rounds: Exact match for number of funding rounds.
        num_funding_rounds_min: Filter with num_funding_rounds >= this value.
        num_funding_rounds_max: Filter with num_funding_rounds <= this value.
        num_investments: Exact match for number of investments.
        num_investments_min: Filter with num_investments >= this value.
        num_investments_max: Filter with num_investments <= this value.
        num_portfolio_organizations: Exact match for number of portfolio organizations.
        num_portfolio_organizations_min: Filter with num_portfolio_organizations >= this value.
        num_portfolio_organizations_max: Filter with num_portfolio_organizations <= this value.
        operating_status: Exact match for operating status.
        cb_rank: Exact match for Crunchbase rank.
        cb_rank_min: Filter with cb_rank >= this value.
        cb_rank_max: Filter with cb_rank <= this value.
        revenue_range: Exact match for revenue range.
        org_status: Exact match for organization status.
        updated_at_from: Filter organizations updated on or after this date (ISO format string).
        updated_at_to: Filter organizations updated on or before this date (ISO format string).
        valuation_usd: Exact match for valuation USD.
        valuation_usd_min: Filter with valuation_usd >= this value.
        valuation_usd_max: Filter with valuation_usd <= this value.
        valuation_date: Exact match for valuation date (ISO format string).
        valuation_date_from: Filter with valuation_date >= this date (ISO format string).
        valuation_date_to: Filter with valuation_date <= this date (ISO format string).
        org_domain: Exact match for organization domain.
        org_domain_ilike: Case-insensitive partial match for organization domain.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).

    Returns:
        ToolOutput object containing:
        - success: Whether the query succeeded
        - result: List of organization records as dictionaries (if successful).
            Each dictionary contains all fields from the Organization model.
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute the query
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Get all organizations
        orgs = await get_organizations()

        # Get by UUID
        org = await get_organizations(org_uuid="123e4567-e89b-12d3-a456-426614174000")

        # Get organizations in a country with minimum funding
        orgs = await get_organizations(
            country="United States",
            total_funding_usd_min=1000000
        )

        # Get organizations by name search
        orgs = await get_organizations(name_ilike="tech")

        # Get organizations with specific category
        orgs = await get_organizations(categories_contains="Artificial Intelligence")
        ```
    """
    start_time = time.time()
    try:
        # Initialize the model
        model = OrganizationModel()
        await model.initialize()

        # Convert string UUID to UUID object if provided
        org_uuid_obj: Optional[UUID] = None
        if org_uuid is not None:
            org_uuid_obj = UUID(org_uuid)

        # Convert date strings to datetime objects if provided
        def parse_date(date_str: Optional[str]) -> Optional[datetime]:
            if date_str is not None:
                return datetime.fromisoformat(date_str)
            return None

        closed_on_obj = parse_date(closed_on)
        closed_on_from_obj = parse_date(closed_on_from)
        closed_on_to_obj = parse_date(closed_on_to)
        created_at_from_obj = parse_date(created_at_from)
        created_at_to_obj = parse_date(created_at_to)
        exited_on_obj = parse_date(exited_on)
        exited_on_from_obj = parse_date(exited_on_from)
        exited_on_to_obj = parse_date(exited_on_to)
        founding_date_obj = parse_date(founding_date)
        founding_date_from_obj = parse_date(founding_date_from)
        founding_date_to_obj = parse_date(founding_date_to)
        last_fundraise_date_obj = parse_date(last_fundraise_date)
        last_fundraise_date_from_obj = parse_date(last_fundraise_date_from)
        last_fundraise_date_to_obj = parse_date(last_fundraise_date_to)
        updated_at_from_obj = parse_date(updated_at_from)
        updated_at_to_obj = parse_date(updated_at_to)
        valuation_date_obj = parse_date(valuation_date)
        valuation_date_from_obj = parse_date(valuation_date_from)
        valuation_date_to_obj = parse_date(valuation_date_to)

        # Query the model
        organizations = await model.get(
            org_uuid=org_uuid_obj,
            cb_url=cb_url,
            categories_contains=categories_contains,
            category_groups_contains=category_groups_contains,
            closed_on=closed_on_obj,
            closed_on_from=closed_on_from_obj,
            closed_on_to=closed_on_to_obj,
            closed_on_precision=closed_on_precision,
            company_profit_type=company_profit_type,
            created_at_from=created_at_from_obj,
            created_at_to=created_at_to_obj,
            raw_description_ilike=raw_description_ilike,
            rewritten_description_ilike=rewritten_description_ilike,
            total_funding_native=total_funding_native,
            total_funding_native_min=total_funding_native_min,
            total_funding_native_max=total_funding_native_max,
            total_funding_currency=total_funding_currency,
            total_funding_usd=total_funding_usd,
            total_funding_usd_min=total_funding_usd_min,
            total_funding_usd_max=total_funding_usd_max,
            exited_on=exited_on_obj,
            exited_on_from=exited_on_from_obj,
            exited_on_to=exited_on_to_obj,
            exited_on_precision=exited_on_precision,
            founding_date=founding_date_obj,
            founding_date_from=founding_date_from_obj,
            founding_date_to=founding_date_to_obj,
            founding_date_precision=founding_date_precision,
            general_funding_stage=general_funding_stage,
            ipo_status=ipo_status,
            last_fundraise_date=last_fundraise_date_obj,
            last_fundraise_date_from=last_fundraise_date_from_obj,
            last_fundraise_date_to=last_fundraise_date_to_obj,
            last_funding_total_usd=last_funding_total_usd,
            last_funding_total_usd_min=last_funding_total_usd_min,
            last_funding_total_usd_max=last_funding_total_usd_max,
            stage=stage,
            org_type=org_type,
            city=city,
            state=state,
            country=country,
            continent=continent,
            name=name,
            name_ilike=name_ilike,
            num_acquisitions=num_acquisitions,
            num_acquisitions_min=num_acquisitions_min,
            num_acquisitions_max=num_acquisitions_max,
            employee_count=employee_count,
            num_funding_rounds=num_funding_rounds,
            num_funding_rounds_min=num_funding_rounds_min,
            num_funding_rounds_max=num_funding_rounds_max,
            num_investments=num_investments,
            num_investments_min=num_investments_min,
            num_investments_max=num_investments_max,
            num_portfolio_organizations=num_portfolio_organizations,
            num_portfolio_organizations_min=num_portfolio_organizations_min,
            num_portfolio_organizations_max=num_portfolio_organizations_max,
            operating_status=operating_status,
            cb_rank=cb_rank,
            cb_rank_min=cb_rank_min,
            cb_rank_max=cb_rank_max,
            revenue_range=revenue_range,
            org_status=org_status,
            updated_at_from=updated_at_from_obj,
            updated_at_to=updated_at_to_obj,
            valuation_usd=valuation_usd,
            valuation_usd_min=valuation_usd_min,
            valuation_usd_max=valuation_usd_max,
            valuation_date=valuation_date_obj,
            valuation_date_from=valuation_date_from_obj,
            valuation_date_to=valuation_date_to_obj,
            org_domain=org_domain,
            org_domain_ilike=org_domain_ilike,
            limit=limit,
            offset=offset,
        )

        # Convert Pydantic models to dictionaries
        result = [org.model_dump() for org in organizations]
        execution_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Retrieved {len(result)} organization(s) in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="get_organizations",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata={"num_results": len(result)},
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to get organizations: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="get_organizations",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
