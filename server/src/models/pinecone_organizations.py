"""Pinecone organization model for querying the Pinecone vector database.

This module provides a Pydantic model and query methods for organizations
stored in Pinecone with vector embeddings and metadata filters.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.db import PineconeClient, get_pinecone_client
from src.llm import OpenAIClient, get_openai_client

logger = logging.getLogger(__name__)


class PineconeOrganization(BaseModel):
    """Pydantic model for a Pinecone organization record.

    This model represents a single organization document from Pinecone with
    proper type validation for all metadata fields.
    """

    org_uuid: UUID = Field(..., description="Unique identifier for the organization")
    name: Optional[str] = Field(None, description="Name of the organization")
    categories: Optional[list[str]] = Field(
        None, description="Industry categories of the organization"
    )
    org_status: Optional[str] = Field(
        None,
        description="Organization status: operating, closed, was_acquired, or ipo",
    )
    total_funding_usd: Optional[float] = Field(None, description="Total amount of funding raised")
    founding_date: Optional[datetime] = Field(None, description="Date the company was founded")
    last_fundraise_date: Optional[datetime] = Field(
        None, description="Date of the company's last fundraise"
    )
    employee_count: Optional[str] = Field(
        None,
        description="Employee count range: 1-10, 11-50, 51-100, 101-250, 251-500, 501-1000, 1001-5000, 5001-10000, 10000+",
    )
    org_type: Optional[str] = Field(None, description="Organization type: investor or company")
    stage: Optional[str] = Field(
        None,
        description="Funding stage: series_a, series_b, seed, etc.",
    )
    valuation_usd: Optional[float] = Field(None, description="Last valuation amount")
    investors: Optional[list[UUID]] = Field(
        None, description="List of org_uuids of all investors of this company"
    )
    general_funding_stage: Optional[str] = Field(
        None,
        description="General funding stage: early_stage_venture, ipo, late_stage_venture, seed, m_and_a, private_equity",
    )
    num_acquisitions: Optional[int] = Field(
        None, description="Number of acquisitions that this company has made"
    )
    revenue_range: Optional[str] = Field(
        None,
        description="Revenue range: <1M, 1M-10M, 10M-50M, 50M-100M, 100M-500M, 500M-1B, 1B-10B, 10B+",
    )
    score: Optional[float] = Field(None, description="Similarity score from Pinecone query")

    model_config = ConfigDict(from_attributes=True)


class PineconeOrganizationModel:
    """Model class for querying organizations in Pinecone.

    This class provides methods to query the Pinecone vector database using
    text queries (which are embedded) and metadata filters. It uses the
    text-embedding-3-large model for generating embeddings.

    Example:
        ```python
        model = PineconeOrganizationModel()
        await model.initialize()

        # Query by text
        orgs = await model.query(
            text="AI companies",
            top_k=10
        )

        # Query with metadata filters
        orgs = await model.query(
            text="healthcare startups",
            org_status="operating",
            total_funding_usd_min=1000000,
            top_k=5
        )
        ```
    """

    def __init__(
        self,
        pinecone_client: Optional[PineconeClient] = None,
        openai_client: Optional[OpenAIClient] = None,
    ):
        """Initialize the PineconeOrganizationModel.

        Args:
            pinecone_client: Optional PineconeClient instance. If not provided,
                will use the default singleton client.
            openai_client: Optional OpenAIClient instance. If not provided,
                will use the default singleton client.
        """
        self._pinecone_client = pinecone_client
        self._openai_client = openai_client
        self._use_default_pinecone = pinecone_client is None
        self._use_default_openai = openai_client is None

    async def initialize(self) -> None:
        """Initialize the database clients if using default clients."""
        if self._use_default_pinecone and self._pinecone_client is None:
            self._pinecone_client = await get_pinecone_client()
        if self._use_default_openai and self._openai_client is None:
            self._openai_client = await get_openai_client()

    async def query(
        self,
        text: str,
        org_uuid: Optional[UUID] = None,
        name: Optional[str] = None,
        categories_contains: Optional[str] = None,
        org_status: Optional[str] = None,
        total_funding_usd_min: Optional[float] = None,
        total_funding_usd_max: Optional[float] = None,
        founding_date_from: Optional[datetime] = None,
        founding_date_to: Optional[datetime] = None,
        last_fundraise_date_from: Optional[datetime] = None,
        last_fundraise_date_to: Optional[datetime] = None,
        employee_count: Optional[str] = None,
        org_type: Optional[str] = None,
        stage: Optional[str] = None,
        valuation_usd_min: Optional[float] = None,
        valuation_usd_max: Optional[float] = None,
        investors_contains: Optional[UUID] = None,
        general_funding_stage: Optional[str] = None,
        num_acquisitions_min: Optional[int] = None,
        num_acquisitions_max: Optional[int] = None,
        revenue_range: Optional[str] = None,
        top_k: int = 10,
        index_name: Optional[str] = None,
    ) -> list[PineconeOrganization]:
        """Query Pinecone for organizations matching text and metadata filters.

        This method embeds the provided text using text-embedding-3-large,
        then queries Pinecone with the embedding and optional metadata filters.

        Args:
            text: Text query to embed and search for (required).
            org_uuid: Exact match for organization UUID.
            name: Exact match for organization name.
            categories_contains: Check if categories array contains this value.
            org_status: Exact match for organization status (operating, closed,
                was_acquired, or ipo).
            total_funding_usd_min: Filter with total_funding_usd >= this value.
            total_funding_usd_max: Filter with total_funding_usd <= this value.
            founding_date_from: Filter organizations founded on or after this date.
            founding_date_to: Filter organizations founded on or before this date.
            last_fundraise_date_from: Filter with last_fundraise_date >= this date.
            last_fundraise_date_to: Filter with last_fundraise_date <= this date.
            employee_count: Exact match for employee count range.
            org_type: Exact match for organization type (investor or company).
            stage: Exact match for funding stage.
            valuation_usd_min: Filter with valuation_usd >= this value.
            valuation_usd_max: Filter with valuation_usd <= this value.
            investors_contains: Check if investors array contains this UUID.
            general_funding_stage: Exact match for general funding stage.
            num_acquisitions_min: Filter with num_acquisitions >= this value.
            num_acquisitions_max: Filter with num_acquisitions <= this value.
            revenue_range: Exact match for revenue range.
            top_k: Number of results to return (default: 10).
            index_name: Name of the Pinecone index to query. If not provided,
                uses the default index from settings.

        Returns:
            List of PineconeOrganization objects matching the query and filters.

        Raises:
            RuntimeError: If clients are not initialized.
            Exception: If query execution fails.

        Example:
            ```python
            # Simple text query
            orgs = await model.query(
                text="AI companies",
                top_k=10
            )

            # Query with metadata filters
            orgs = await model.query(
                text="healthcare startups",
                org_status="operating",
                total_funding_usd_min=1000000,
                general_funding_stage="late_stage_venture",
                top_k=5
            )

            # Query with date range
            from datetime import datetime
            orgs = await model.query(
                text="fintech companies",
                founding_date_from=datetime(2020, 1, 1),
                founding_date_to=datetime(2023, 12, 31),
                top_k=20
            )
            ```
        """
        if self._pinecone_client is None:
            raise RuntimeError("PineconeClient not initialized. Call initialize() first.")
        if self._openai_client is None:
            raise RuntimeError("OpenAIClient not initialized. Call initialize() first.")

        # Generate embedding for the text query
        try:
            embedding = await self._openai_client.create_embedding(
                text=text, model="text-embedding-3-large"
            )
            logger.debug(f"Generated embedding for text query: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

        # Build metadata filter
        metadata_filter: dict[str, Any] = {}

        if org_uuid is not None:
            metadata_filter["org_uuid"] = {"$eq": str(org_uuid)}

        if name is not None:
            metadata_filter["name"] = {"$eq": name}

        if categories_contains is not None:
            metadata_filter["categories"] = {"$in": [categories_contains]}

        if org_status is not None:
            metadata_filter["org_status"] = {"$eq": org_status}

        if total_funding_usd_min is not None or total_funding_usd_max is not None:
            funding_filter: dict[str, Any] = {}
            if total_funding_usd_min is not None:
                funding_filter["$gte"] = total_funding_usd_min
            if total_funding_usd_max is not None:
                funding_filter["$lte"] = total_funding_usd_max
            metadata_filter["total_funding_usd"] = funding_filter

        if founding_date_from is not None or founding_date_to is not None:
            founding_filter: dict[str, Any] = {}
            if founding_date_from is not None:
                founding_filter["$gte"] = founding_date_from.isoformat()
            if founding_date_to is not None:
                founding_filter["$lte"] = founding_date_to.isoformat()
            metadata_filter["founding_date"] = founding_filter

        if last_fundraise_date_from is not None or last_fundraise_date_to is not None:
            fundraise_filter: dict[str, Any] = {}
            if last_fundraise_date_from is not None:
                fundraise_filter["$gte"] = last_fundraise_date_from.isoformat()
            if last_fundraise_date_to is not None:
                fundraise_filter["$lte"] = last_fundraise_date_to.isoformat()
            metadata_filter["last_fundraise_date"] = fundraise_filter

        if employee_count is not None:
            metadata_filter["employee_count"] = {"$eq": employee_count}

        if org_type is not None:
            metadata_filter["org_type"] = {"$eq": org_type}

        if stage is not None:
            metadata_filter["stage"] = {"$eq": stage}

        if valuation_usd_min is not None or valuation_usd_max is not None:
            valuation_filter: dict[str, Any] = {}
            if valuation_usd_min is not None:
                valuation_filter["$gte"] = valuation_usd_min
            if valuation_usd_max is not None:
                valuation_filter["$lte"] = valuation_usd_max
            metadata_filter["valuation_usd"] = valuation_filter

        if investors_contains is not None:
            metadata_filter["investors"] = {"$in": [str(investors_contains)]}

        if general_funding_stage is not None:
            metadata_filter["general_funding_stage"] = {"$eq": general_funding_stage}

        if num_acquisitions_min is not None or num_acquisitions_max is not None:
            acquisitions_filter: dict[str, Any] = {}
            if num_acquisitions_min is not None:
                acquisitions_filter["$gte"] = num_acquisitions_min
            if num_acquisitions_max is not None:
                acquisitions_filter["$lte"] = num_acquisitions_max
            metadata_filter["num_acquisitions"] = acquisitions_filter

        if revenue_range is not None:
            metadata_filter["revenue_range"] = {"$eq": revenue_range}

        # Query Pinecone
        try:
            filter_dict = metadata_filter if metadata_filter else None
            response = await self._pinecone_client.query(
                query_vector=embedding,
                index_name=index_name,
                top_k=top_k,
                metadata_filter=filter_dict,
                include_metadata=True,
            )

            # Convert results to PineconeOrganization objects
            organizations: list[PineconeOrganization] = []
            for match in response.matches:
                metadata = match.metadata or {}
                # Convert org_uuid from string to UUID if present
                if "org_uuid" in metadata:
                    try:
                        metadata["org_uuid"] = UUID(metadata["org_uuid"])
                    except (ValueError, TypeError):
                        pass

                # Convert investors from list of strings to list of UUIDs
                if "investors" in metadata and isinstance(metadata["investors"], list):
                    try:
                        metadata["investors"] = [
                            UUID(inv) if isinstance(inv, str) else inv
                            for inv in metadata["investors"]
                        ]
                    except (ValueError, TypeError):
                        pass

                # Convert date strings to datetime objects
                for date_field in ["founding_date", "last_fundraise_date"]:
                    if date_field in metadata and isinstance(metadata[date_field], str):
                        try:
                            metadata[date_field] = datetime.fromisoformat(metadata[date_field])
                        except (ValueError, TypeError):
                            pass

                # Add score from match
                metadata["score"] = match.score

                organizations.append(PineconeOrganization(**metadata))

            logger.debug(f"Retrieved {len(organizations)} organization(s) from Pinecone")
            return organizations
        except Exception as e:
            logger.error(f"Failed to query Pinecone: {e}")
            raise
