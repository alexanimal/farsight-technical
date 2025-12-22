"""Funding round model for querying the fundingrounds table.

This module provides a Pydantic model and query methods for the fundingrounds table.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.db import PostgresClient, get_postgres_client

logger = logging.getLogger(__name__)


class FundingRound(BaseModel):
    """Pydantic model for a funding round record.

    This model represents a single row from the fundingrounds table with
    proper type validation.
    """

    funding_round_uuid: UUID = Field(
        ..., description="Primary key UUID for the funding round"
    )
    investment_date: Optional[datetime] = Field(
        None, description="Date of the investment"
    )
    org_uuid: Optional[UUID] = Field(
        None, description="UUID of the organization receiving funding"
    )
    general_funding_stage: Optional[str] = Field(
        None, description="General funding stage (e.g., Series A, Seed)"
    )
    stage: Optional[str] = Field(None, description="Specific funding stage")
    investors: Optional[list[str]] = Field(None, description="List of investor names")
    lead_investors: Optional[list[str]] = Field(
        None, description="List of lead investor names"
    )
    fundraise_amount_usd: Optional[int] = Field(
        None, description="Total fundraise amount in USD"
    )
    valuation_usd: Optional[int] = Field(None, description="Company valuation in USD")

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
    ) -> list[FundingRound]:
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

        Returns:
            List of FundingRound objects matching the filters.

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
            raise RuntimeError(
                "PostgresClient not initialized. Call initialize() first."
            )

        # Build WHERE clause dynamically
        conditions: list[str] = []
        params: list[Any] = []
        param_index = 1

        if funding_round_uuid is not None:
            conditions.append(f"fr.funding_round_uuid = ${param_index}")
            params.append(str(funding_round_uuid))
            param_index += 1

        if investment_date is not None:
            conditions.append(f"fr.investment_date = ${param_index}")
            params.append(investment_date)
            param_index += 1

        if investment_date_from is not None:
            conditions.append(f"fr.investment_date >= ${param_index}")
            params.append(investment_date_from)
            param_index += 1

        if investment_date_to is not None:
            conditions.append(f"fr.investment_date <= ${param_index}")
            params.append(investment_date_to)
            param_index += 1

        # Handle org_uuid(s) - prioritize org_uuids if both provided
        if org_uuids is not None:
            if len(org_uuids) == 0:
                # Empty list = no results
                return []
            elif len(org_uuids) == 1:
                # Single UUID - use equality for better index usage
                conditions.append(f"fr.org_uuid = ${param_index}")
                params.append(str(org_uuids[0]))
                param_index += 1
            else:
                # Multiple UUIDs - use IN clause with ANY for array
                conditions.append(f"fr.org_uuid = ANY(${param_index}::uuid[])")
                params.append([str(uuid) for uuid in org_uuids])
                param_index += 1
        elif org_uuid is not None:
            # Existing single UUID support (backward compatible)
            conditions.append(f"fr.org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if org_name_ilike is not None:
            # Add condition to filter by organization name
            # The JOIN will be added in the query building section
            conditions.append(f"org.name ILIKE ${param_index}")
            params.append(f"%{org_name_ilike}%")
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"fr.general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if stage is not None:
            conditions.append(f"fr.stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if investors_contains is not None:
            conditions.append(f"${param_index} = ANY(fr.investors)")
            params.append(investors_contains)
            param_index += 1

        if lead_investors_contains is not None:
            conditions.append(f"${param_index} = ANY(fr.lead_investors)")
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
            conditions.append(f"fr.fundraise_amount_usd = ${param_index}")
            params.append(fundraise_amount_usd)
            param_index += 1

        if fundraise_amount_usd_min is not None:
            conditions.append(f"fr.fundraise_amount_usd >= ${param_index}")
            params.append(fundraise_amount_usd_min)
            param_index += 1

        if fundraise_amount_usd_max is not None:
            conditions.append(f"fr.fundraise_amount_usd <= ${param_index}")
            params.append(fundraise_amount_usd_max)
            param_index += 1

        if valuation_usd is not None:
            conditions.append(f"fr.valuation_usd = ${param_index}")
            params.append(valuation_usd)
            param_index += 1

        if valuation_usd_min is not None:
            conditions.append(f"fr.valuation_usd >= ${param_index}")
            params.append(valuation_usd_min)
            param_index += 1

        if valuation_usd_max is not None:
            conditions.append(f"fr.valuation_usd <= ${param_index}")
            params.append(valuation_usd_max)
            param_index += 1

        # Build the query
        # Use table alias 'fr' for fundingrounds when name-based searches are used
        query = "SELECT fr.* FROM fundingrounds fr"
        
        # Add JOIN to organizations table if org_name_ilike is used
        if org_name_ilike is not None:
            query += " JOIN organizations org ON fr.org_uuid = org.org_uuid"
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY fr.investment_date DESC NULLS LAST"

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
            funding_rounds = [FundingRound(**dict(record)) for record in records]
            logger.debug(f"Retrieved {len(funding_rounds)} funding round(s)")
            return funding_rounds
        except Exception as e:
            logger.error(f"Failed to query funding rounds: {e}")
            raise

    async def get_by_uuid(self, funding_round_uuid: UUID) -> Optional[FundingRound]:
        """Get a single funding round by UUID.

        Args:
            funding_round_uuid: The UUID of the funding round to retrieve.

        Returns:
            FundingRound object if found, None otherwise.
        """
        results = await self.get(funding_round_uuid=funding_round_uuid, limit=1)
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
            raise RuntimeError(
                "PostgresClient not initialized. Call initialize() first."
            )

        # Build WHERE clause (reuse same logic as get)
        conditions: list[str] = []
        params: list[Any] = []
        param_index = 1

        if funding_round_uuid is not None:
            conditions.append(f"fr.funding_round_uuid = ${param_index}")
            params.append(str(funding_round_uuid))
            param_index += 1

        if investment_date_from is not None:
            conditions.append(f"fr.investment_date >= ${param_index}")
            params.append(investment_date_from)
            param_index += 1

        if investment_date_to is not None:
            conditions.append(f"fr.investment_date <= ${param_index}")
            params.append(investment_date_to)
            param_index += 1

        if org_uuid is not None:
            conditions.append(f"fr.org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if org_name_ilike is not None:
            # Add condition to filter by organization name
            # The JOIN will be added in the query building section
            conditions.append(f"org.name ILIKE ${param_index}")
            params.append(f"%{org_name_ilike}%")
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"fr.general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if stage is not None:
            conditions.append(f"fr.stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if investors_contains is not None:
            conditions.append(f"${param_index} = ANY(fr.investors)")
            params.append(investors_contains)
            param_index += 1

        if lead_investors_contains is not None:
            conditions.append(f"${param_index} = ANY(fr.lead_investors)")
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
            conditions.append(f"fr.fundraise_amount_usd >= ${param_index}")
            params.append(fundraise_amount_usd_min)
            param_index += 1

        if fundraise_amount_usd_max is not None:
            conditions.append(f"fr.fundraise_amount_usd <= ${param_index}")
            params.append(fundraise_amount_usd_max)
            param_index += 1

        if valuation_usd_min is not None:
            conditions.append(f"fr.valuation_usd >= ${param_index}")
            params.append(valuation_usd_min)
            param_index += 1

        if valuation_usd_max is not None:
            conditions.append(f"fr.valuation_usd <= ${param_index}")
            params.append(valuation_usd_max)

        # Use table alias 'fr' for fundingrounds when name-based searches are used
        query = "SELECT COUNT(*) FROM fundingrounds fr"
        
        # Add JOIN to organizations table if org_name_ilike is used
        if org_name_ilike is not None:
            query += " JOIN organizations org ON fr.org_uuid = org.org_uuid"
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            count = await self._client.query_value(query, *params)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.error(f"Failed to count funding rounds: {e}")
            raise
