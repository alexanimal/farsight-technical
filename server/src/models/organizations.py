"""Organization model for querying the organizations table.

This module provides a Pydantic model and query methods for the organizations table.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.db import PostgresClient, get_postgres_client

logger = logging.getLogger(__name__)


class Organization(BaseModel):
    """Pydantic model for an organization record.

    This model represents a single row from the organizations table with
    proper type validation.
    """

    org_uuid: UUID = Field(..., description="Primary key UUID for the organization")
    cb_url: Optional[str] = Field(None, description="Crunchbase URL")
    categories: Optional[list[str]] = Field(None, description="List of category tags")
    category_groups: Optional[list[str]] = Field(None, description="List of category group tags")
    closed_on: Optional[datetime] = Field(None, description="Date the organization closed")
    closed_on_precision: Optional[str] = Field(None, description="Precision of closed_on date")
    company_profit_type: Optional[str] = Field(None, description="Company profit type")
    created_at: Optional[datetime] = Field(None, description="Record creation timestamp")
    raw_description: Optional[str] = Field(None, description="Raw company description")
    web_scrape: Optional[str] = Field(None, description="Web scraped content")
    rewritten_description: Optional[str] = Field(
        None, description="Rewritten/processed description"
    )
    total_funding_native: Optional[int] = Field(
        None, description="Total funding in native currency"
    )
    total_funding_currency: Optional[str] = Field(None, description="Currency for total funding")
    total_funding_usd: Optional[int] = Field(None, description="Total funding in USD")
    exited_on: Optional[datetime] = Field(None, description="Date the organization exited")
    exited_on_precision: Optional[str] = Field(None, description="Precision of exited_on date")
    founding_date: Optional[datetime] = Field(None, description="Organization founding date")
    founding_date_precision: Optional[str] = Field(None, description="Precision of founding_date")
    general_funding_stage: Optional[str] = Field(None, description="General funding stage")
    logo_url: Optional[str] = Field(None, description="URL to organization logo")
    ipo_status: Optional[str] = Field(None, description="IPO status")
    last_fundraise_date: Optional[datetime] = Field(None, description="Date of last fundraise")
    last_funding_total_native: Optional[int] = Field(
        None, description="Last funding total in native currency"
    )
    last_funding_total_currency: Optional[str] = Field(
        None, description="Currency for last funding"
    )
    last_funding_total_usd: Optional[int] = Field(None, description="Last funding total in USD")
    stage: Optional[str] = Field(None, description="Current stage")
    org_type: Optional[str] = Field(None, description="Organization type")
    city: Optional[str] = Field(None, description="City location")
    state: Optional[str] = Field(None, description="State/Province location")
    country: Optional[str] = Field(None, description="Country location")
    continent: Optional[str] = Field(None, description="Continent location")
    name: Optional[str] = Field(None, description="Organization name")
    num_acquisitions: Optional[int] = Field(None, description="Number of acquisitions")
    employee_count: Optional[str] = Field(None, description="Employee count range")
    num_funding_rounds: Optional[int] = Field(None, description="Number of funding rounds")
    num_investments: Optional[int] = Field(None, description="Number of investments made")
    num_portfolio_organizations: Optional[int] = Field(
        None, description="Number of portfolio organizations"
    )
    operating_status: Optional[str] = Field(None, description="Operating status")
    cb_rank: Optional[int] = Field(None, description="Crunchbase rank")
    revenue_range: Optional[str] = Field(None, description="Revenue range")
    org_status: Optional[str] = Field(None, description="Organization status")
    updated_at: Optional[datetime] = Field(None, description="Record last update timestamp")
    valuation_native: Optional[int] = Field(None, description="Valuation in native currency")
    valuation_currency: Optional[str] = Field(None, description="Currency for valuation")
    valuation_usd: Optional[int] = Field(None, description="Valuation in USD")
    valuation_date: Optional[datetime] = Field(None, description="Date of valuation")
    org_domain: Optional[str] = Field(None, description="Organization domain/website")

    model_config = ConfigDict(from_attributes=True)


class OrganizationModel:
    """Model class for querying the organizations table.

    This class provides methods to query the organizations table using
    the PostgresClient. It includes a generic get method that allows
    filtering by any field(s) in the table.

    Example:
        ```python
        model = OrganizationModel()
        await model.initialize()

        # Get all organizations
        all_orgs = await model.get()

        # Get by specific UUID
        org = await model.get(org_uuid=UUID("..."))

        # Get by multiple filters
        orgs = await model.get(
            country="United States",
            total_funding_usd_min=1000000
        )
        ```
    """

    def __init__(self, client: Optional[PostgresClient] = None):
        """Initialize the OrganizationModel.

        Args:
            client: Optional PostgresClient instance. If not provided, will use
                the default singleton client.
        """
        self._client = client
        self._use_default_client = client is None

    async def initialize(self) -> None:
        """Initialize the database client if using default client."""
        if self._use_default_client and self._client is None:
            self._client = await get_postgres_client()

    async def get(
        self,
        org_uuid: Optional[UUID] = None,
        cb_url: Optional[str] = None,
        categories_contains: Optional[str] = None,
        category_groups_contains: Optional[str] = None,
        closed_on: Optional[datetime] = None,
        closed_on_from: Optional[datetime] = None,
        closed_on_to: Optional[datetime] = None,
        closed_on_precision: Optional[str] = None,
        company_profit_type: Optional[str] = None,
        created_at_from: Optional[datetime] = None,
        created_at_to: Optional[datetime] = None,
        raw_description_ilike: Optional[str] = None,
        rewritten_description_ilike: Optional[str] = None,
        total_funding_native: Optional[int] = None,
        total_funding_native_min: Optional[int] = None,
        total_funding_native_max: Optional[int] = None,
        total_funding_currency: Optional[str] = None,
        total_funding_usd: Optional[int] = None,
        total_funding_usd_min: Optional[int] = None,
        total_funding_usd_max: Optional[int] = None,
        exited_on: Optional[datetime] = None,
        exited_on_from: Optional[datetime] = None,
        exited_on_to: Optional[datetime] = None,
        exited_on_precision: Optional[str] = None,
        founding_date: Optional[datetime] = None,
        founding_date_from: Optional[datetime] = None,
        founding_date_to: Optional[datetime] = None,
        founding_date_precision: Optional[str] = None,
        general_funding_stage: Optional[str] = None,
        ipo_status: Optional[str] = None,
        last_fundraise_date: Optional[datetime] = None,
        last_fundraise_date_from: Optional[datetime] = None,
        last_fundraise_date_to: Optional[datetime] = None,
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
        updated_at_from: Optional[datetime] = None,
        updated_at_to: Optional[datetime] = None,
        valuation_usd: Optional[int] = None,
        valuation_usd_min: Optional[int] = None,
        valuation_usd_max: Optional[int] = None,
        valuation_date: Optional[datetime] = None,
        valuation_date_from: Optional[datetime] = None,
        valuation_date_to: Optional[datetime] = None,
        org_domain: Optional[str] = None,
        org_domain_ilike: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[Organization]:
        """Get organizations matching the specified filters.

        This method builds a dynamic SQL query based on the provided filters.
        All filters are optional and can be combined.

        Args:
            org_uuid: Exact match for organization UUID.
            cb_url: Exact match for Crunchbase URL.
            categories_contains: Check if categories array contains this value.
            category_groups_contains: Check if category_groups array contains this value.
            closed_on: Exact match for closed date.
            closed_on_from: Filter organizations closed on or after this date.
            closed_on_to: Filter organizations closed on or before this date.
            closed_on_precision: Exact match for closed_on precision.
            company_profit_type: Exact match for company profit type.
            created_at_from: Filter organizations created on or after this date.
            created_at_to: Filter organizations created on or before this date.
            raw_description_ilike: Case-insensitive partial match for raw description.
            rewritten_description_ilike: Case-insensitive partial match for rewritten description.
            total_funding_native: Exact match for total funding native.
            total_funding_native_min: Filter with total_funding_native >= this value.
            total_funding_native_max: Filter with total_funding_native <= this value.
            total_funding_currency: Exact match for funding currency.
            total_funding_usd: Exact match for total funding USD.
            total_funding_usd_min: Filter with total_funding_usd >= this value.
            total_funding_usd_max: Filter with total_funding_usd <= this value.
            exited_on: Exact match for exited date.
            exited_on_from: Filter organizations exited on or after this date.
            exited_on_to: Filter organizations exited on or before this date.
            exited_on_precision: Exact match for exited_on precision.
            founding_date: Exact match for founding date.
            founding_date_from: Filter organizations founded on or after this date.
            founding_date_to: Filter organizations founded on or before this date.
            founding_date_precision: Exact match for founding_date precision.
            general_funding_stage: Exact match for general funding stage.
            ipo_status: Exact match for IPO status.
            last_fundraise_date: Exact match for last fundraise date.
            last_fundraise_date_from: Filter with last_fundraise_date >= this date.
            last_fundraise_date_to: Filter with last_fundraise_date <= this date.
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
            updated_at_from: Filter organizations updated on or after this date.
            updated_at_to: Filter organizations updated on or before this date.
            valuation_usd: Exact match for valuation USD.
            valuation_usd_min: Filter with valuation_usd >= this value.
            valuation_usd_max: Filter with valuation_usd <= this value.
            valuation_date: Exact match for valuation date.
            valuation_date_from: Filter with valuation_date >= this date.
            valuation_date_to: Filter with valuation_date <= this date.
            org_domain: Exact match for organization domain.
            org_domain_ilike: Case-insensitive partial match for organization domain.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).

        Returns:
            List of Organization objects matching the filters.

        Raises:
            RuntimeError: If client is not initialized.

        Example:
            ```python
            # Get all organizations
            all_orgs = await model.get()

            # Get by UUID
            org = await model.get(org_uuid=UUID("..."))

            # Get organizations in a country with minimum funding
            orgs = await model.get(
                country="United States",
                total_funding_usd_min=1000000
            )

            # Get organizations by name search
            orgs = await model.get(name_ilike="tech")

            # Get organizations with specific category
            orgs = await model.get(categories_contains="Artificial Intelligence")

            # Get with pagination
            page = await model.get(limit=50, offset=0)
            ```
        """
        if self._client is None:
            raise RuntimeError("PostgresClient not initialized. Call initialize() first.")

        # Build WHERE clause dynamically
        conditions: list[str] = []
        params: list[Any] = []
        param_index = 1

        if org_uuid is not None:
            conditions.append(f"org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if cb_url is not None:
            conditions.append(f"cb_url = ${param_index}")
            params.append(cb_url)
            param_index += 1

        if categories_contains is not None:
            conditions.append(f"${param_index} = ANY(categories)")
            params.append(categories_contains)
            param_index += 1

        if category_groups_contains is not None:
            conditions.append(f"${param_index} = ANY(category_groups)")
            params.append(category_groups_contains)
            param_index += 1

        if closed_on is not None:
            conditions.append(f"closed_on = ${param_index}")
            params.append(closed_on)
            param_index += 1

        if closed_on_from is not None:
            conditions.append(f"closed_on >= ${param_index}")
            params.append(closed_on_from)
            param_index += 1

        if closed_on_to is not None:
            conditions.append(f"closed_on <= ${param_index}")
            params.append(closed_on_to)
            param_index += 1

        if closed_on_precision is not None:
            conditions.append(f"closed_on_precision = ${param_index}")
            params.append(closed_on_precision)
            param_index += 1

        if company_profit_type is not None:
            conditions.append(f"company_profit_type = ${param_index}")
            params.append(company_profit_type)
            param_index += 1

        if created_at_from is not None:
            conditions.append(f"created_at >= ${param_index}")
            params.append(created_at_from)
            param_index += 1

        if created_at_to is not None:
            conditions.append(f"created_at <= ${param_index}")
            params.append(created_at_to)
            param_index += 1

        if raw_description_ilike is not None:
            conditions.append(f"raw_description ILIKE ${param_index}")
            params.append(f"%{raw_description_ilike}%")
            param_index += 1

        if rewritten_description_ilike is not None:
            conditions.append(f"rewritten_description ILIKE ${param_index}")
            params.append(f"%{rewritten_description_ilike}%")
            param_index += 1

        if total_funding_native is not None:
            conditions.append(f"total_funding_native = ${param_index}")
            params.append(total_funding_native)
            param_index += 1

        if total_funding_native_min is not None:
            conditions.append(f"total_funding_native >= ${param_index}")
            params.append(total_funding_native_min)
            param_index += 1

        if total_funding_native_max is not None:
            conditions.append(f"total_funding_native <= ${param_index}")
            params.append(total_funding_native_max)
            param_index += 1

        if total_funding_currency is not None:
            conditions.append(f"total_funding_currency = ${param_index}")
            params.append(total_funding_currency)
            param_index += 1

        if total_funding_usd is not None:
            conditions.append(f"total_funding_usd = ${param_index}")
            params.append(total_funding_usd)
            param_index += 1

        if total_funding_usd_min is not None:
            conditions.append(f"total_funding_usd >= ${param_index}")
            params.append(total_funding_usd_min)
            param_index += 1

        if total_funding_usd_max is not None:
            conditions.append(f"total_funding_usd <= ${param_index}")
            params.append(total_funding_usd_max)
            param_index += 1

        if exited_on is not None:
            conditions.append(f"exited_on = ${param_index}")
            params.append(exited_on)
            param_index += 1

        if exited_on_from is not None:
            conditions.append(f"exited_on >= ${param_index}")
            params.append(exited_on_from)
            param_index += 1

        if exited_on_to is not None:
            conditions.append(f"exited_on <= ${param_index}")
            params.append(exited_on_to)
            param_index += 1

        if exited_on_precision is not None:
            conditions.append(f"exited_on_precision = ${param_index}")
            params.append(exited_on_precision)
            param_index += 1

        if founding_date is not None:
            conditions.append(f"founding_date = ${param_index}")
            params.append(founding_date)
            param_index += 1

        if founding_date_from is not None:
            conditions.append(f"founding_date >= ${param_index}")
            params.append(founding_date_from)
            param_index += 1

        if founding_date_to is not None:
            conditions.append(f"founding_date <= ${param_index}")
            params.append(founding_date_to)
            param_index += 1

        if founding_date_precision is not None:
            conditions.append(f"founding_date_precision = ${param_index}")
            params.append(founding_date_precision)
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if ipo_status is not None:
            conditions.append(f"ipo_status = ${param_index}")
            params.append(ipo_status)
            param_index += 1

        if last_fundraise_date is not None:
            conditions.append(f"last_fundraise_date = ${param_index}")
            params.append(last_fundraise_date)
            param_index += 1

        if last_fundraise_date_from is not None:
            conditions.append(f"last_fundraise_date >= ${param_index}")
            params.append(last_fundraise_date_from)
            param_index += 1

        if last_fundraise_date_to is not None:
            conditions.append(f"last_fundraise_date <= ${param_index}")
            params.append(last_fundraise_date_to)
            param_index += 1

        if last_funding_total_usd is not None:
            conditions.append(f"last_funding_total_usd = ${param_index}")
            params.append(last_funding_total_usd)
            param_index += 1

        if last_funding_total_usd_min is not None:
            conditions.append(f"last_funding_total_usd >= ${param_index}")
            params.append(last_funding_total_usd_min)
            param_index += 1

        if last_funding_total_usd_max is not None:
            conditions.append(f"last_funding_total_usd <= ${param_index}")
            params.append(last_funding_total_usd_max)
            param_index += 1

        if stage is not None:
            conditions.append(f"stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if org_type is not None:
            conditions.append(f"org_type = ${param_index}")
            params.append(org_type)
            param_index += 1

        if city is not None:
            conditions.append(f"city = ${param_index}")
            params.append(city)
            param_index += 1

        if state is not None:
            conditions.append(f"state = ${param_index}")
            params.append(state)
            param_index += 1

        if country is not None:
            conditions.append(f"country = ${param_index}")
            params.append(country)
            param_index += 1

        if continent is not None:
            conditions.append(f"continent = ${param_index}")
            params.append(continent)
            param_index += 1

        if name is not None:
            conditions.append(f"name = ${param_index}")
            params.append(name)
            param_index += 1

        if name_ilike is not None:
            conditions.append(f"name ILIKE ${param_index}")
            params.append(f"%{name_ilike}%")
            param_index += 1

        if num_acquisitions is not None:
            conditions.append(f"num_acquisitions = ${param_index}")
            params.append(num_acquisitions)
            param_index += 1

        if num_acquisitions_min is not None:
            conditions.append(f"num_acquisitions >= ${param_index}")
            params.append(num_acquisitions_min)
            param_index += 1

        if num_acquisitions_max is not None:
            conditions.append(f"num_acquisitions <= ${param_index}")
            params.append(num_acquisitions_max)
            param_index += 1

        if employee_count is not None:
            conditions.append(f"employee_count = ${param_index}")
            params.append(employee_count)
            param_index += 1

        if num_funding_rounds is not None:
            conditions.append(f"num_funding_rounds = ${param_index}")
            params.append(num_funding_rounds)
            param_index += 1

        if num_funding_rounds_min is not None:
            conditions.append(f"num_funding_rounds >= ${param_index}")
            params.append(num_funding_rounds_min)
            param_index += 1

        if num_funding_rounds_max is not None:
            conditions.append(f"num_funding_rounds <= ${param_index}")
            params.append(num_funding_rounds_max)
            param_index += 1

        if num_investments is not None:
            conditions.append(f"num_investments = ${param_index}")
            params.append(num_investments)
            param_index += 1

        if num_investments_min is not None:
            conditions.append(f"num_investments >= ${param_index}")
            params.append(num_investments_min)
            param_index += 1

        if num_investments_max is not None:
            conditions.append(f"num_investments <= ${param_index}")
            params.append(num_investments_max)
            param_index += 1

        if num_portfolio_organizations is not None:
            conditions.append(f"num_portfolio_organizations = ${param_index}")
            params.append(num_portfolio_organizations)
            param_index += 1

        if num_portfolio_organizations_min is not None:
            conditions.append(f"num_portfolio_organizations >= ${param_index}")
            params.append(num_portfolio_organizations_min)
            param_index += 1

        if num_portfolio_organizations_max is not None:
            conditions.append(f"num_portfolio_organizations <= ${param_index}")
            params.append(num_portfolio_organizations_max)
            param_index += 1

        if operating_status is not None:
            conditions.append(f"operating_status = ${param_index}")
            params.append(operating_status)
            param_index += 1

        if cb_rank is not None:
            conditions.append(f"cb_rank = ${param_index}")
            params.append(cb_rank)
            param_index += 1

        if cb_rank_min is not None:
            conditions.append(f"cb_rank >= ${param_index}")
            params.append(cb_rank_min)
            param_index += 1

        if cb_rank_max is not None:
            conditions.append(f"cb_rank <= ${param_index}")
            params.append(cb_rank_max)
            param_index += 1

        if revenue_range is not None:
            conditions.append(f"revenue_range = ${param_index}")
            params.append(revenue_range)
            param_index += 1

        if org_status is not None:
            conditions.append(f"org_status = ${param_index}")
            params.append(org_status)
            param_index += 1

        if updated_at_from is not None:
            conditions.append(f"updated_at >= ${param_index}")
            params.append(updated_at_from)
            param_index += 1

        if updated_at_to is not None:
            conditions.append(f"updated_at <= ${param_index}")
            params.append(updated_at_to)
            param_index += 1

        if valuation_usd is not None:
            conditions.append(f"valuation_usd = ${param_index}")
            params.append(valuation_usd)
            param_index += 1

        if valuation_usd_min is not None:
            conditions.append(f"valuation_usd >= ${param_index}")
            params.append(valuation_usd_min)
            param_index += 1

        if valuation_usd_max is not None:
            conditions.append(f"valuation_usd <= ${param_index}")
            params.append(valuation_usd_max)
            param_index += 1

        if valuation_date is not None:
            conditions.append(f"valuation_date = ${param_index}")
            params.append(valuation_date)
            param_index += 1

        if valuation_date_from is not None:
            conditions.append(f"valuation_date >= ${param_index}")
            params.append(valuation_date_from)
            param_index += 1

        if valuation_date_to is not None:
            conditions.append(f"valuation_date <= ${param_index}")
            params.append(valuation_date_to)
            param_index += 1

        if org_domain is not None:
            conditions.append(f"org_domain = ${param_index}")
            params.append(org_domain)
            param_index += 1

        if org_domain_ilike is not None:
            conditions.append(f"org_domain ILIKE ${param_index}")
            params.append(f"%{org_domain_ilike}%")
            param_index += 1

        # Build the query
        query = "SELECT * FROM organizations"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC NULLS LAST"

        if limit is not None:
            query += f" LIMIT ${param_index}"
            params.append(limit)
            param_index += 1

        if offset is not None:
            query += f" OFFSET ${param_index}"
            params.append(offset)

        # Execute query
        try:
            records = await self._client.query(query, *params)
            organizations = [Organization(**dict(record)) for record in records]
            logger.debug(f"Retrieved {len(organizations)} organization(s)")
            return organizations
        except Exception as e:
            logger.error(f"Failed to query organizations: {e}")
            raise

    async def get_by_uuid(self, org_uuid: UUID) -> Optional[Organization]:
        """Get a single organization by UUID.

        Args:
            org_uuid: The UUID of the organization to retrieve.

        Returns:
            Organization object if found, None otherwise.
        """
        results = await self.get(org_uuid=org_uuid, limit=1)
        return results[0] if results else None

    async def count(
        self,
        org_uuid: Optional[UUID] = None,
        categories_contains: Optional[str] = None,
        category_groups_contains: Optional[str] = None,
        country: Optional[str] = None,
        continent: Optional[str] = None,
        general_funding_stage: Optional[str] = None,
        stage: Optional[str] = None,
        org_type: Optional[str] = None,
        operating_status: Optional[str] = None,
        org_status: Optional[str] = None,
        total_funding_usd_min: Optional[int] = None,
        total_funding_usd_max: Optional[int] = None,
        valuation_usd_min: Optional[int] = None,
        valuation_usd_max: Optional[int] = None,
        name_ilike: Optional[str] = None,
        org_domain_ilike: Optional[str] = None,
    ) -> int:
        """Count organizations matching the specified filters.

        Uses a subset of the most common filters from get() for efficiency.

        Args:
            Common filter parameters from get() method.

        Returns:
            Number of organizations matching the filters.
        """
        if self._client is None:
            raise RuntimeError("PostgresClient not initialized. Call initialize() first.")

        # Build WHERE clause with common filters
        conditions: list[str] = []
        params: list[Any] = []
        param_index = 1

        if org_uuid is not None:
            conditions.append(f"org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if categories_contains is not None:
            conditions.append(f"${param_index} = ANY(categories)")
            params.append(categories_contains)
            param_index += 1

        if category_groups_contains is not None:
            conditions.append(f"${param_index} = ANY(category_groups)")
            params.append(category_groups_contains)
            param_index += 1

        if country is not None:
            conditions.append(f"country = ${param_index}")
            params.append(country)
            param_index += 1

        if continent is not None:
            conditions.append(f"continent = ${param_index}")
            params.append(continent)
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if stage is not None:
            conditions.append(f"stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if org_type is not None:
            conditions.append(f"org_type = ${param_index}")
            params.append(org_type)
            param_index += 1

        if operating_status is not None:
            conditions.append(f"operating_status = ${param_index}")
            params.append(operating_status)
            param_index += 1

        if org_status is not None:
            conditions.append(f"org_status = ${param_index}")
            params.append(org_status)
            param_index += 1

        if total_funding_usd_min is not None:
            conditions.append(f"total_funding_usd >= ${param_index}")
            params.append(total_funding_usd_min)
            param_index += 1

        if total_funding_usd_max is not None:
            conditions.append(f"total_funding_usd <= ${param_index}")
            params.append(total_funding_usd_max)
            param_index += 1

        if valuation_usd_min is not None:
            conditions.append(f"valuation_usd >= ${param_index}")
            params.append(valuation_usd_min)
            param_index += 1

        if valuation_usd_max is not None:
            conditions.append(f"valuation_usd <= ${param_index}")
            params.append(valuation_usd_max)
            param_index += 1

        if name_ilike is not None:
            conditions.append(f"name ILIKE ${param_index}")
            params.append(f"%{name_ilike}%")
            param_index += 1

        if org_domain_ilike is not None:
            conditions.append(f"org_domain ILIKE ${param_index}")
            params.append(f"%{org_domain_ilike}%")

        query = "SELECT COUNT(*) FROM organizations"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            count = await self._client.query_value(query, *params)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.error(f"Failed to count organizations: {e}")
            raise
