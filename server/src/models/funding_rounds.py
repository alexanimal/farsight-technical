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
    ) -> list[FundingRound]:
        """Get funding rounds matching the specified filters.

        This method builds a dynamic SQL query based on the provided filters.
        All filters are optional and can be combined.

        Args:
            funding_round_uuid: Exact match for funding round UUID.
            investment_date: Exact match for investment date.
            investment_date_from: Filter funding rounds on or after this date.
            investment_date_to: Filter funding rounds on or before this date.
            org_uuid: Exact match for organization UUID.
            general_funding_stage: Exact match for general funding stage.
            stage: Exact match for specific funding stage.
            investors_contains: Check if investors array contains this value.
            lead_investors_contains: Check if lead_investors array contains this value.
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

            # Get rounds with specific investor
            investor_rounds = await model.get(investors_contains="Sequoia Capital")

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
            conditions.append(f"funding_round_uuid = ${param_index}")
            params.append(str(funding_round_uuid))
            param_index += 1

        if investment_date is not None:
            conditions.append(f"investment_date = ${param_index}")
            params.append(investment_date)
            param_index += 1

        if investment_date_from is not None:
            conditions.append(f"investment_date >= ${param_index}")
            params.append(investment_date_from)
            param_index += 1

        if investment_date_to is not None:
            conditions.append(f"investment_date <= ${param_index}")
            params.append(investment_date_to)
            param_index += 1

        if org_uuid is not None:
            conditions.append(f"org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if stage is not None:
            conditions.append(f"stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if investors_contains is not None:
            conditions.append(f"${param_index} = ANY(investors)")
            params.append(investors_contains)
            param_index += 1

        if lead_investors_contains is not None:
            conditions.append(f"${param_index} = ANY(lead_investors)")
            params.append(lead_investors_contains)
            param_index += 1

        if fundraise_amount_usd is not None:
            conditions.append(f"fundraise_amount_usd = ${param_index}")
            params.append(fundraise_amount_usd)
            param_index += 1

        if fundraise_amount_usd_min is not None:
            conditions.append(f"fundraise_amount_usd >= ${param_index}")
            params.append(fundraise_amount_usd_min)
            param_index += 1

        if fundraise_amount_usd_max is not None:
            conditions.append(f"fundraise_amount_usd <= ${param_index}")
            params.append(fundraise_amount_usd_max)
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

        # Build the query
        query = "SELECT * FROM fundingrounds"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY investment_date DESC NULLS LAST"

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
        general_funding_stage: Optional[str] = None,
        stage: Optional[str] = None,
        investors_contains: Optional[str] = None,
        lead_investors_contains: Optional[str] = None,
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
            conditions.append(f"funding_round_uuid = ${param_index}")
            params.append(str(funding_round_uuid))
            param_index += 1

        if investment_date_from is not None:
            conditions.append(f"investment_date >= ${param_index}")
            params.append(investment_date_from)
            param_index += 1

        if investment_date_to is not None:
            conditions.append(f"investment_date <= ${param_index}")
            params.append(investment_date_to)
            param_index += 1

        if org_uuid is not None:
            conditions.append(f"org_uuid = ${param_index}")
            params.append(str(org_uuid))
            param_index += 1

        if general_funding_stage is not None:
            conditions.append(f"general_funding_stage = ${param_index}")
            params.append(general_funding_stage)
            param_index += 1

        if stage is not None:
            conditions.append(f"stage = ${param_index}")
            params.append(stage)
            param_index += 1

        if investors_contains is not None:
            conditions.append(f"${param_index} = ANY(investors)")
            params.append(investors_contains)
            param_index += 1

        if lead_investors_contains is not None:
            conditions.append(f"${param_index} = ANY(lead_investors)")
            params.append(lead_investors_contains)
            param_index += 1

        if fundraise_amount_usd_min is not None:
            conditions.append(f"fundraise_amount_usd >= ${param_index}")
            params.append(fundraise_amount_usd_min)
            param_index += 1

        if fundraise_amount_usd_max is not None:
            conditions.append(f"fundraise_amount_usd <= ${param_index}")
            params.append(fundraise_amount_usd_max)
            param_index += 1

        if valuation_usd_min is not None:
            conditions.append(f"valuation_usd >= ${param_index}")
            params.append(valuation_usd_min)
            param_index += 1

        if valuation_usd_max is not None:
            conditions.append(f"valuation_usd <= ${param_index}")
            params.append(valuation_usd_max)

        query = "SELECT COUNT(*) FROM fundingrounds"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            count = await self._client.query_value(query, *params)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.error(f"Failed to count funding rounds: {e}")
            raise
