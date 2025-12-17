"""Acquisition model for querying the acquisitions table.

This module provides a Pydantic model and query methods for the acquisitions table.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.db import PostgresClient, get_postgres_client

logger = logging.getLogger(__name__)


class Acquisition(BaseModel):
    """Pydantic model for an acquisition record.

    This model represents a single row from the acquisitions table with
    proper type validation.
    """

    acquisition_uuid: UUID = Field(
        ..., description="Primary key UUID for the acquisition"
    )
    acquiree_uuid: Optional[UUID] = Field(
        None, description="UUID of the company being acquired"
    )
    acquirer_uuid: Optional[UUID] = Field(
        None, description="UUID of the acquiring company"
    )
    acquisition_type: Optional[str] = Field(None, description="Type of acquisition")
    acquisition_announce_date: Optional[datetime] = Field(
        None, description="Date the acquisition was announced"
    )
    acquisition_price_usd: Optional[int] = Field(
        None, description="Acquisition price in USD"
    )
    terms: Optional[str] = Field(None, description="Terms of the acquisition")
    acquirer_type: Optional[str] = Field(None, description="Type of acquirer")

    model_config = ConfigDict(from_attributes=True)


class AcquisitionModel:
    """Model class for querying the acquisitions table.

    This class provides methods to query the acquisitions table using
    the PostgresClient. It includes a generic get method that allows
    filtering by any field(s) in the table.

    Example:
        ```python
        model = AcquisitionModel()
        await model.initialize()

        # Get all acquisitions
        all_acquisitions = await model.get()

        # Get by specific UUID
        acquisition = await model.get(acquisition_uuid=UUID("..."))

        # Get by multiple filters
        acquisitions = await model.get(
            acquisition_type="merger",
            acquisition_price_usd_min=1000000
        )
        ```
    """

    def __init__(self, client: Optional[PostgresClient] = None):
        """Initialize the AcquisitionModel.

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
        acquisition_uuid: Optional[UUID] = None,
        acquiree_uuid: Optional[UUID] = None,
        acquirer_uuid: Optional[UUID] = None,
        acquisition_type: Optional[str] = None,
        acquisition_announce_date: Optional[datetime] = None,
        acquisition_announce_date_from: Optional[datetime] = None,
        acquisition_announce_date_to: Optional[datetime] = None,
        acquisition_price_usd: Optional[int] = None,
        acquisition_price_usd_min: Optional[int] = None,
        acquisition_price_usd_max: Optional[int] = None,
        terms: Optional[str] = None,
        terms_ilike: Optional[str] = None,
        acquirer_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[Acquisition]:
        """Get acquisitions matching the specified filters.

        This method builds a dynamic SQL query based on the provided filters.
        All filters are optional and can be combined.

        Args:
            acquisition_uuid: Exact match for acquisition UUID.
            acquiree_uuid: Exact match for acquiree UUID.
            acquirer_uuid: Exact match for acquirer UUID.
            acquisition_type: Exact match for acquisition type.
            acquisition_announce_date: Exact match for announce date.
            acquisition_announce_date_from: Filter acquisitions on or after this date.
            acquisition_announce_date_to: Filter acquisitions on or before this date.
            acquisition_price_usd: Exact match for acquisition price.
            acquisition_price_usd_min: Filter acquisitions with price >= this value.
            acquisition_price_usd_max: Filter acquisitions with price <= this value.
            terms: Exact match for terms.
            terms_ilike: Case-insensitive partial match for terms (uses ILIKE).
            acquirer_type: Exact match for acquirer type.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).

        Returns:
            List of Acquisition objects matching the filters.

        Raises:
            RuntimeError: If client is not initialized.

        Example:
            ```python
            # Get all acquisitions
            all_acquisitions = await model.get()

            # Get by UUID
            acquisition = await model.get(acquisition_uuid=UUID("..."))

            # Get acquisitions with price >= 1M USD
            expensive = await model.get(acquisition_price_usd_min=1000000)

            # Get acquisitions announced in a date range
            recent = await model.get(
                acquisition_announce_date_from=datetime(2020, 1, 1),
                acquisition_announce_date_to=datetime(2023, 12, 31)
            )

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

        if acquisition_uuid is not None:
            conditions.append(f"acquisition_uuid = ${param_index}")
            params.append(str(acquisition_uuid))
            param_index += 1

        if acquiree_uuid is not None:
            conditions.append(f"acquiree_uuid = ${param_index}")
            params.append(str(acquiree_uuid))
            param_index += 1

        if acquirer_uuid is not None:
            conditions.append(f"acquirer_uuid = ${param_index}")
            params.append(str(acquirer_uuid))
            param_index += 1

        if acquisition_type is not None:
            conditions.append(f"acquisition_type = ${param_index}")
            params.append(acquisition_type)
            param_index += 1

        if acquisition_announce_date is not None:
            conditions.append(f"acquisition_announce_date = ${param_index}")
            params.append(acquisition_announce_date)
            param_index += 1

        if acquisition_announce_date_from is not None:
            conditions.append(f"acquisition_announce_date >= ${param_index}")
            params.append(acquisition_announce_date_from)
            param_index += 1

        if acquisition_announce_date_to is not None:
            conditions.append(f"acquisition_announce_date <= ${param_index}")
            params.append(acquisition_announce_date_to)
            param_index += 1

        if acquisition_price_usd is not None:
            conditions.append(f"acquisition_price_usd = ${param_index}")
            params.append(acquisition_price_usd)
            param_index += 1

        if acquisition_price_usd_min is not None:
            conditions.append(f"acquisition_price_usd >= ${param_index}")
            params.append(acquisition_price_usd_min)
            param_index += 1

        if acquisition_price_usd_max is not None:
            conditions.append(f"acquisition_price_usd <= ${param_index}")
            params.append(acquisition_price_usd_max)
            param_index += 1

        if terms is not None:
            conditions.append(f"terms = ${param_index}")
            params.append(terms)
            param_index += 1

        if terms_ilike is not None:
            conditions.append(f"terms ILIKE ${param_index}")
            params.append(f"%{terms_ilike}%")
            param_index += 1

        if acquirer_type is not None:
            conditions.append(f"acquirer_type = ${param_index}")
            params.append(acquirer_type)
            param_index += 1

        # Build the query
        query = "SELECT * FROM acquisitions"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY acquisition_announce_date DESC NULLS LAST"

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
            acquisitions = [Acquisition(**dict(record)) for record in records]
            logger.debug(f"Retrieved {len(acquisitions)} acquisition(s)")
            return acquisitions
        except Exception as e:
            logger.error(f"Failed to query acquisitions: {e}")
            raise

    async def get_by_uuid(self, acquisition_uuid: UUID) -> Optional[Acquisition]:
        """Get a single acquisition by UUID.

        Args:
            acquisition_uuid: The UUID of the acquisition to retrieve.

        Returns:
            Acquisition object if found, None otherwise.
        """
        results = await self.get(acquisition_uuid=acquisition_uuid, limit=1)
        return results[0] if results else None

    async def count(
        self,
        acquisition_uuid: Optional[UUID] = None,
        acquiree_uuid: Optional[UUID] = None,
        acquirer_uuid: Optional[UUID] = None,
        acquisition_type: Optional[str] = None,
        acquisition_announce_date_from: Optional[datetime] = None,
        acquisition_announce_date_to: Optional[datetime] = None,
        acquisition_price_usd_min: Optional[int] = None,
        acquisition_price_usd_max: Optional[int] = None,
        acquirer_type: Optional[str] = None,
    ) -> int:
        """Count acquisitions matching the specified filters.

        Uses the same filter logic as get() but returns only the count.

        Args:
            Same as get() method (excluding limit and offset).

        Returns:
            Number of acquisitions matching the filters.
        """
        if self._client is None:
            raise RuntimeError(
                "PostgresClient not initialized. Call initialize() first."
            )

        # Build WHERE clause (reuse same logic as get)
        conditions: list[str] = []
        params: list[Any] = []
        param_index = 1

        if acquisition_uuid is not None:
            conditions.append(f"acquisition_uuid = ${param_index}")
            params.append(str(acquisition_uuid))
            param_index += 1

        if acquiree_uuid is not None:
            conditions.append(f"acquiree_uuid = ${param_index}")
            params.append(str(acquiree_uuid))
            param_index += 1

        if acquirer_uuid is not None:
            conditions.append(f"acquirer_uuid = ${param_index}")
            params.append(str(acquirer_uuid))
            param_index += 1

        if acquisition_type is not None:
            conditions.append(f"acquisition_type = ${param_index}")
            params.append(acquisition_type)
            param_index += 1

        if acquisition_announce_date_from is not None:
            conditions.append(f"acquisition_announce_date >= ${param_index}")
            params.append(acquisition_announce_date_from)
            param_index += 1

        if acquisition_announce_date_to is not None:
            conditions.append(f"acquisition_announce_date <= ${param_index}")
            params.append(acquisition_announce_date_to)
            param_index += 1

        if acquisition_price_usd_min is not None:
            conditions.append(f"acquisition_price_usd >= ${param_index}")
            params.append(acquisition_price_usd_min)
            param_index += 1

        if acquisition_price_usd_max is not None:
            conditions.append(f"acquisition_price_usd <= ${param_index}")
            params.append(acquisition_price_usd_max)
            param_index += 1

        if acquirer_type is not None:
            conditions.append(f"acquirer_type = ${param_index}")
            params.append(acquirer_type)

        query = "SELECT COUNT(*) FROM acquisitions"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            count = await self._client.query_value(query, *params)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.error(f"Failed to count acquisitions: {e}")
            raise
