"""Funding round model for querying the fundingrounds table.

This module provides a Pydantic model and query methods for the fundingrounds table.
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional, Union
from uuid import UUID

if TYPE_CHECKING:
    from src.models.organizations import Organization

from pydantic import BaseModel, ConfigDict, Field

from src.db import PostgresClient, get_postgres_client

logger = logging.getLogger(__name__)


class FundingRound(BaseModel):
    """Pydantic model for a funding round record.

    This model represents a single row from the fundingrounds table with
    proper type validation.
    """

    funding_round_uuid: UUID = Field(..., description="Primary key UUID for the funding round")
    investment_date: Optional[datetime] = Field(None, description="Date of the investment")
    org_uuid: Optional[UUID] = Field(None, description="UUID of the organization receiving funding")
    general_funding_stage: Optional[str] = Field(
        None, description="General funding stage (e.g., Series A, Seed)"
    )
    stage: Optional[str] = Field(None, description="Specific funding stage")
    investors: Optional[list[str]] = Field(None, description="List of investor names")
    lead_investors: Optional[list[str]] = Field(None, description="List of lead investor names")
    fundraise_amount_usd: Optional[int] = Field(None, description="Total fundraise amount in USD")
    valuation_usd: Optional[int] = Field(None, description="Company valuation in USD")

    model_config = ConfigDict(from_attributes=True)


class FundingRoundWithOrganizations(FundingRound):
    """FundingRound model with nested organization details.

    This extends FundingRound to include full organization objects for:
    - organization: The organization receiving funding (from org_uuid)
    - investors_organizations: List of investor organizations (from investors array)
    - lead_investors_organizations: List of lead investor organizations (from lead_investors array)
    """

    organization: Optional["Organization"] = Field(
        None, description="Organization receiving funding"
    )
    investors_organizations: list["Organization"] = Field(
        default_factory=list, description="List of investor organizations"
    )
    lead_investors_organizations: list["Organization"] = Field(
        default_factory=list, description="List of lead investor organizations"
    )

    model_config = ConfigDict(from_attributes=True)


class FundingRoundModel:
    """Model class for querying the fundingrounds table.

    This class provides methods to query the fundingrounds table using
    the PostgresClient. It includes a generic get method that allows
    filtering by any field(s) in the table.

    Example:
        ```python
        model = FundingRoundModel()
        await model.initialize()

        # Get all funding rounds
        all_rounds = await model.get()

        # Get by specific UUID
        round = await model.get(funding_round_uuid=UUID("..."))

        # Get by multiple filters
        rounds = await model.get(
            general_funding_stage="Series A",
            fundraise_amount_usd_min=1000000
        )
        ```
    """

    def __init__(self, client: Optional[PostgresClient] = None):
        """Initialize the FundingRoundModel.

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
        funding_round_uuid: Optional[UUID] = None,
        investment_date: Optional[datetime] = None,
        investment_date_from: Optional[datetime] = None,
        investment_date_to: Optional[datetime] = None,
        org_uuid: Optional[UUID] = None,
        org_uuids: Optional[list[UUID]] = None,
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
        order_by: Optional[
            Literal["investment_date", "fundraise_amount_usd", "valuation_usd"]
        ] = None,
        order_direction: Optional[Literal["asc", "desc"]] = None,
    ) -> Union[list[FundingRound], list["FundingRoundWithOrganizations"]]:
        """Get funding rounds matching the specified filters.

        This method builds a dynamic SQL query based on the provided filters.
        All filters are optional and can be combined.

        Args:
            funding_round_uuid: Exact match for funding round UUID.
            investment_date: Exact match for investment date.
            investment_date_from: Filter funding rounds on or after this date.
            investment_date_to: Filter funding rounds on or before this date.
            org_uuid: Exact match for organization UUID (single).
            org_uuids: List of organization UUIDs to filter by (batch query).
                If both org_uuid and org_uuids are provided, org_uuids takes precedence.
            org_name_ilike: Search by organization name (case-insensitive partial match).
                This performs a JOIN to the organizations table to match by name.
                Can be combined with org_uuid or org_uuids filters.
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
            fundraise_amount_usd: Exact match for fundraise amount.
            fundraise_amount_usd_min: Filter funding rounds with amount >= this value.
            fundraise_amount_usd_max: Filter funding rounds with amount <= this value.
            valuation_usd: Exact match for valuation.
            valuation_usd_min: Filter funding rounds with valuation >= this value.
            valuation_usd_max: Filter funding rounds with valuation <= this value.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).
            include_organizations: If True, includes nested organization details for
                org_uuid, investors, and lead_investors. Returns FundingRoundWithOrganizations
                objects instead of FundingRound objects.
            order_by: Field to order results by. Must be one of: "investment_date",
                "fundraise_amount_usd", "valuation_usd". Defaults to "investment_date" if not specified.
            order_direction: Direction to order results. Must be "asc" or "desc".
                Defaults to "desc" if not specified.

        Returns:
            List of FundingRound objects matching the filters. If include_organizations=True,
            returns FundingRoundWithOrganizations objects with nested organization data.

        Raises:
            RuntimeError: If client is not initialized.

        Example:
            ```python
            # Get all funding rounds
            all_rounds = await model.get()

            # Get by UUID
            round = await model.get(funding_round_uuid=UUID("..."))

            # Get funding rounds with amount >= 1M USD
            large_rounds = await model.get(fundraise_amount_usd_min=1000000)

            # Get funding rounds in a date range
            recent = await model.get(
                investment_date_from=datetime(2020, 1, 1),
                investment_date_to=datetime(2023, 12, 31)
            )

            # Get rounds with specific investor by UUID
            investor_rounds = await model.get(investors_contains="123e4567-e89b-12d3-a456-426614174000")

            # Get rounds with specific investor by name
            investor_rounds_by_name = await model.get(investor_name_contains="Sequoia Capital")

            # Get rounds for multiple organizations (batch query)
            org_rounds = await model.get(org_uuids=[UUID("..."), UUID("...")])

            # Get rounds by organization name
            org_rounds_by_name = await model.get(org_name_ilike="TechCorp")

            # Get with pagination
            page = await model.get(limit=50, offset=0)
            ```
        """
        if self._client is None:
            raise RuntimeError("PostgresClient not initialized. Call initialize() first.")

        # Determine if we need table alias (for joins or include_organizations)
        needs_alias = (
            include_organizations
            or org_name_ilike is not None
            or investor_name_contains is not None
            or lead_investor_name_contains is not None
        )
        table_prefix = "fr." if needs_alias else ""

        # Build WHERE clause dynamically
        conditions: list[str] = []
        params: list[Any] = []
        param_index = 1

        if funding_round_uuid is not None:
            conditions.append(f"{table_prefix}funding_round_uuid = ${param_index}")
            params.append(str(funding_round_uuid))
            param_index += 1

        if investment_date is not None:
            conditions.append(f"{table_prefix}investment_date = ${param_index}")
            params.append(investment_date)
            param_index += 1

        if investment_date_from is not None:
            conditions.append(f"{table_prefix}investment_date >= ${param_index}")
            params.append(investment_date_from)
            param_index += 1

        if investment_date_to is not None:
            conditions.append(f"{table_prefix}investment_date <= ${param_index}")
            params.append(investment_date_to)
            param_index += 1

        # Handle org_uuid(s) - prioritize org_uuids if both provided
        if org_uuids is not None:
            if len(org_uuids) == 0:
                # Empty list = no results
                return []
            elif len(org_uuids) == 1:
                # Single UUID - use equality for better index usage
                conditions.append(f"{table_prefix}org_uuid = ${param_index}")
                params.append(str(org_uuids[0]))
                param_index += 1
            else:
                # Multiple UUIDs - use IN clause with ANY for array
                conditions.append(f"{table_prefix}org_uuid = ANY(${param_index}::uuid[])")
                params.append([str(uuid) for uuid in org_uuids])
                param_index += 1
        elif org_uuid is not None:
            # Existing single UUID support (backward compatible)
            conditions.append(f"{table_prefix}org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if org_name_ilike is not None:
            # Add condition to filter by organization name
            # The JOIN will be added in the query building section
            conditions.append(f"org.name ILIKE ${param_index}")
            params.append(f"%{org_name_ilike}%")
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"{table_prefix}general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if stage is not None:
            conditions.append(f"{table_prefix}stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if investors_contains is not None:
            conditions.append(f"${param_index} = ANY({table_prefix}investors)")
            params.append(investors_contains)
            param_index += 1

        if lead_investors_contains is not None:
            conditions.append(f"${param_index} = ANY({table_prefix}lead_investors)")
            params.append(lead_investors_contains)
            param_index += 1

        if investor_name_contains is not None:
            # Use EXISTS subquery to join with organizations table and match by name
            conditions.append(
                f"EXISTS ("
                f"SELECT 1 FROM unnest(fr.investors) AS inv_uuid "
                f"JOIN organizations inv_org ON inv_uuid::uuid = inv_org.org_uuid "
                f"WHERE inv_org.name ILIKE ${param_index}"
                f")"
            )
            params.append(f"%{investor_name_contains}%")
            param_index += 1

        if lead_investor_name_contains is not None:
            # Use EXISTS subquery to join with organizations table and match by name
            conditions.append(
                f"EXISTS ("
                f"SELECT 1 FROM unnest(fr.lead_investors) AS lead_inv_uuid "
                f"JOIN organizations lead_inv_org ON lead_inv_uuid::uuid = lead_inv_org.org_uuid "
                f"WHERE lead_inv_org.name ILIKE ${param_index}"
                f")"
            )
            params.append(f"%{lead_investor_name_contains}%")
            param_index += 1

        if fundraise_amount_usd is not None:
            conditions.append(f"{table_prefix}fundraise_amount_usd = ${param_index}")
            params.append(fundraise_amount_usd)
            param_index += 1

        if fundraise_amount_usd_min is not None:
            conditions.append(f"{table_prefix}fundraise_amount_usd >= ${param_index}")
            params.append(fundraise_amount_usd_min)
            param_index += 1

        if fundraise_amount_usd_max is not None:
            conditions.append(f"{table_prefix}fundraise_amount_usd <= ${param_index}")
            params.append(fundraise_amount_usd_max)
            param_index += 1

        if valuation_usd is not None:
            conditions.append(f"{table_prefix}valuation_usd = ${param_index}")
            params.append(valuation_usd)
            param_index += 1

        if valuation_usd_min is not None:
            conditions.append(f"{table_prefix}valuation_usd >= ${param_index}")
            params.append(valuation_usd_min)
            param_index += 1

        if valuation_usd_max is not None:
            conditions.append(f"{table_prefix}valuation_usd <= ${param_index}")
            params.append(valuation_usd_max)
            param_index += 1

        # Build the query
        if include_organizations:
            # Build SELECT with organization joins using JSON aggregation
            query = """
            SELECT 
                fr.*,
                row_to_json(org.*) as organization,
                (
                    SELECT COALESCE(json_agg(row_to_json(inv_org.*)), '[]'::json)
                    FROM unnest(COALESCE(fr.investors, ARRAY[]::text[])) AS inv_uuid
                    LEFT JOIN organizations inv_org ON inv_uuid::uuid = inv_org.org_uuid
                ) as investors_organizations,
                (
                    SELECT COALESCE(json_agg(row_to_json(lead_inv_org.*)), '[]'::json)
                    FROM unnest(COALESCE(fr.lead_investors, ARRAY[]::text[])) AS lead_inv_uuid
                    LEFT JOIN organizations lead_inv_org ON lead_inv_uuid::uuid = lead_inv_org.org_uuid
                ) as lead_investors_organizations
            FROM fundingrounds fr
            LEFT JOIN organizations org ON fr.org_uuid = org.org_uuid
            """
        elif needs_alias:
            # Use alias when org_name_ilike or investor name searches are used
            query = "SELECT fr.* FROM fundingrounds fr"
            if org_name_ilike is not None:
                query += " JOIN organizations org ON fr.org_uuid = org.org_uuid"
        else:
            # No alias needed - simple query
            query = "SELECT * FROM fundingrounds"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Build ORDER BY clause
        # Default to investment_date DESC if not specified
        order_by_field_name = order_by if order_by is not None else "investment_date"
        order_direction_value = order_direction if order_direction is not None else "desc"

        # Validate order_by field
        allowed_order_fields = {
            "investment_date",
            "fundraise_amount_usd",
            "valuation_usd",
        }
        if order_by_field_name not in allowed_order_fields:
            raise ValueError(
                f"order_by must be one of {allowed_order_fields}, got: {order_by_field_name}"
            )

        # Validate order_direction
        if order_direction_value not in {"asc", "desc"}:
            raise ValueError(
                f"order_direction must be 'asc' or 'desc', got: {order_direction_value}"
            )

        # Apply table prefix if needed
        order_by_field = (
            f"{table_prefix}{order_by_field_name}"
            if needs_alias or include_organizations
            else order_by_field_name
        )

        # Use NULLS LAST for descending, NULLS FIRST for ascending (standard SQL behavior)
        nulls_clause = "NULLS LAST" if order_direction_value == "desc" else "NULLS FIRST"
        query += f" ORDER BY {order_by_field} {order_direction_value.upper()} {nulls_clause}"

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

            if include_organizations:
                # Import Organization here to avoid circular imports
                from src.models.organizations import Organization

                funding_rounds_with_orgs = []
                for record in records:
                    record_dict = dict(record)
                    # Extract organization data (JSON strings from PostgreSQL)
                    org_data_raw = record_dict.pop("organization", None)
                    investors_orgs_data_raw = record_dict.pop("investors_organizations", None)
                    lead_investors_orgs_data_raw = record_dict.pop(
                        "lead_investors_organizations", None
                    )

                    # Parse JSON strings if they exist
                    org_data = None
                    if org_data_raw is not None:
                        if isinstance(org_data_raw, str):
                            org_data = json.loads(org_data_raw)
                        else:
                            org_data = org_data_raw

                    investors_orgs_data = []
                    if investors_orgs_data_raw is not None:
                        if isinstance(investors_orgs_data_raw, str):
                            investors_orgs_data = json.loads(investors_orgs_data_raw)
                        elif isinstance(investors_orgs_data_raw, list):
                            investors_orgs_data = investors_orgs_data_raw
                        else:
                            investors_orgs_data = []

                    lead_investors_orgs_data = []
                    if lead_investors_orgs_data_raw is not None:
                        if isinstance(lead_investors_orgs_data_raw, str):
                            lead_investors_orgs_data = json.loads(lead_investors_orgs_data_raw)
                        elif isinstance(lead_investors_orgs_data_raw, list):
                            lead_investors_orgs_data = lead_investors_orgs_data_raw
                        else:
                            lead_investors_orgs_data = []

                    # Create FundingRound from base fields
                    funding_round = FundingRound(**record_dict)

                    # Create nested organization objects
                    organization = Organization(**org_data) if org_data else None
                    investors_organizations = [
                        Organization(**inv_org)
                        for inv_org in investors_orgs_data
                        if inv_org is not None
                    ]
                    lead_investors_organizations = [
                        Organization(**lead_inv_org)
                        for lead_inv_org in lead_investors_orgs_data
                        if lead_inv_org is not None
                    ]

                    # Create FundingRoundWithOrganizations
                    funding_round_with_orgs = FundingRoundWithOrganizations(
                        **funding_round.model_dump(),
                        organization=organization,
                        investors_organizations=investors_organizations,
                        lead_investors_organizations=lead_investors_organizations,
                    )
                    funding_rounds_with_orgs.append(funding_round_with_orgs)

                logger.debug(
                    f"Retrieved {len(funding_rounds_with_orgs)} funding round(s) with organizations"
                )
                return funding_rounds_with_orgs
            else:
                funding_rounds = [FundingRound(**dict(record)) for record in records]
                logger.debug(f"Retrieved {len(funding_rounds)} funding round(s)")
                return funding_rounds
        except Exception as e:
            logger.error(f"Failed to query funding rounds: {e}")
            raise

    async def get_by_uuid(
        self,
        funding_round_uuid: UUID,
        include_organizations: bool = False,
    ) -> Union[Optional[FundingRound], Optional["FundingRoundWithOrganizations"]]:
        """Get a single funding round by UUID.

        Args:
            funding_round_uuid: The UUID of the funding round to retrieve.
            include_organizations: If True, includes nested organization details.

        Returns:
            FundingRound or FundingRoundWithOrganizations object if found, None otherwise.
        """
        results = await self.get(
            funding_round_uuid=funding_round_uuid,
            limit=1,
            include_organizations=include_organizations,
        )
        return results[0] if results else None

    async def count(
        self,
        funding_round_uuid: Optional[UUID] = None,
        investment_date_from: Optional[datetime] = None,
        investment_date_to: Optional[datetime] = None,
        org_uuid: Optional[UUID] = None,
        org_name_ilike: Optional[str] = None,
        general_funding_stage: Optional[str] = None,
        stage: Optional[str] = None,
        investors_contains: Optional[str] = None,
        lead_investors_contains: Optional[str] = None,
        investor_name_contains: Optional[str] = None,
        lead_investor_name_contains: Optional[str] = None,
        fundraise_amount_usd_min: Optional[int] = None,
        fundraise_amount_usd_max: Optional[int] = None,
        valuation_usd_min: Optional[int] = None,
        valuation_usd_max: Optional[int] = None,
    ) -> int:
        """Count funding rounds matching the specified filters.

        Uses the same filter logic as get() but returns only the count.

        Args:
            Same as get() method (excluding limit and offset).

        Returns:
            Number of funding rounds matching the filters.
        """
        if self._client is None:
            raise RuntimeError("PostgresClient not initialized. Call initialize() first.")

        # Determine if we need table alias (for joins)
        needs_alias = (
            org_name_ilike is not None
            or investor_name_contains is not None
            or lead_investor_name_contains is not None
        )
        table_prefix = "fr." if needs_alias else ""

        # Build WHERE clause (reuse same logic as get)
        conditions: list[str] = []
        params: list[Any] = []
        param_index = 1

        if funding_round_uuid is not None:
            conditions.append(f"{table_prefix}funding_round_uuid = ${param_index}")
            params.append(str(funding_round_uuid))
            param_index += 1

        if investment_date_from is not None:
            conditions.append(f"{table_prefix}investment_date >= ${param_index}")
            params.append(investment_date_from)
            param_index += 1

        if investment_date_to is not None:
            conditions.append(f"{table_prefix}investment_date <= ${param_index}")
            params.append(investment_date_to)
            param_index += 1

        if org_uuid is not None:
            conditions.append(f"{table_prefix}org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if org_name_ilike is not None:
            # Add condition to filter by organization name
            # The JOIN will be added in the query building section
            conditions.append(f"org.name ILIKE ${param_index}")
            params.append(f"%{org_name_ilike}%")
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"{table_prefix}general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if stage is not None:
            conditions.append(f"{table_prefix}stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if investors_contains is not None:
            conditions.append(f"${param_index} = ANY({table_prefix}investors)")
            params.append(investors_contains)
            param_index += 1

        if lead_investors_contains is not None:
            conditions.append(f"${param_index} = ANY({table_prefix}lead_investors)")
            params.append(lead_investors_contains)
            param_index += 1

        if investor_name_contains is not None:
            # Use EXISTS subquery to join with organizations table and match by name
            conditions.append(
                f"EXISTS ("
                f"SELECT 1 FROM unnest(fr.investors) AS inv_uuid "
                f"JOIN organizations inv_org ON inv_uuid::uuid = inv_org.org_uuid "
                f"WHERE inv_org.name ILIKE ${param_index}"
                f")"
            )
            params.append(f"%{investor_name_contains}%")
            param_index += 1

        if lead_investor_name_contains is not None:
            # Use EXISTS subquery to join with organizations table and match by name
            conditions.append(
                f"EXISTS ("
                f"SELECT 1 FROM unnest(fr.lead_investors) AS lead_inv_uuid "
                f"JOIN organizations lead_inv_org ON lead_inv_uuid::uuid = lead_inv_org.org_uuid "
                f"WHERE lead_inv_org.name ILIKE ${param_index}"
                f")"
            )
            params.append(f"%{lead_investor_name_contains}%")
            param_index += 1

        if fundraise_amount_usd_min is not None:
            conditions.append(f"{table_prefix}fundraise_amount_usd >= ${param_index}")
            params.append(fundraise_amount_usd_min)
            param_index += 1

        if fundraise_amount_usd_max is not None:
            conditions.append(f"{table_prefix}fundraise_amount_usd <= ${param_index}")
            params.append(fundraise_amount_usd_max)
            param_index += 1

        if valuation_usd_min is not None:
            conditions.append(f"{table_prefix}valuation_usd >= ${param_index}")
            params.append(valuation_usd_min)
            param_index += 1

        if valuation_usd_max is not None:
            conditions.append(f"{table_prefix}valuation_usd <= ${param_index}")
            params.append(valuation_usd_max)

        # Build query - use alias only when needed
        if needs_alias:
            query = "SELECT COUNT(*) FROM fundingrounds fr"
            if org_name_ilike is not None:
                query += " JOIN organizations org ON fr.org_uuid = org.org_uuid"
        else:
            query = "SELECT COUNT(*) FROM fundingrounds"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            count = await self._client.query_value(query, *params)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.error(f"Failed to count funding rounds: {e}")
            raise
