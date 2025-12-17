"""Pinecone client for managing vector database connections and queries.

This module provides a Pinecone client that manages connections to the vector
database and provides methods for querying documents with metadata filters.
"""

import logging
from typing import Any, Optional

from pinecone import Pinecone

from src.config import settings

logger = logging.getLogger(__name__)


class PineconeClient:
    """Pinecone client for vector database operations.

    This client manages connections to Pinecone and provides methods for
    querying vector embeddings with metadata filters. It supports querying
    documents from specified indexes and listing available indexes.

    Example:
        ```python
        client = PineconeClient()
        await client.initialize()
        results = await client.query(
            index_name="companies",
            query_vector=[0.1, 0.2, ...],
            top_k=10,
            metadata_filter={"sector": "AI"}
        )
        await client.close()
        ```

        Or use as a context manager:
        ```python
        async with PineconeClient() as client:
            results = await client.query(
                index_name="companies",
                query_vector=embedding,
                top_k=5
            )
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_index: Optional[str] = None,
    ):
        """Initialize the Pinecone client.

        Args:
            api_key: Optional Pinecone API key. If not provided, uses
                settings.pinecone_api_key.
            default_index: Optional default index name. If not provided, uses
                settings.pinecone_index. This can be overridden in query calls.
        """
        self._api_key = api_key or settings.pinecone_api_key
        self._default_index = default_index or settings.pinecone_index
        self._client: Optional[Pinecone] = None

    async def initialize(self) -> None:
        """Initialize the Pinecone client connection.

        Raises:
            ValueError: If API key is not provided.
            Exception: If connection to Pinecone fails.
        """
        if self._client is not None:
            logger.warning(
                "Pinecone client already initialized, skipping re-initialization"
            )
            return

        if not self._api_key:
            raise ValueError(
                "Pinecone API key is required. Set PINECONE_API_KEY in environment or pass api_key parameter."
            )

        try:
            self._client = Pinecone(api_key=self._api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise

    async def close(self) -> None:
        """Close the Pinecone client connection."""
        if self._client is not None:
            # Pinecone client doesn't have an explicit close method in v8+
            # but we can clear the reference
            self._client = None
            logger.info("Pinecone client closed")

    async def __aenter__(self) -> "PineconeClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def client(self) -> Pinecone:
        """Get the Pinecone client instance.

        Returns:
            The Pinecone client instance.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if self._client is None:
            raise RuntimeError(
                "Pinecone client not initialized. Call initialize() first or use as context manager."
            )
        return self._client

    def list_indexes(self) -> list[str]:
        """List all available index names in the Pinecone project.

        Returns:
            List of index names.

        Raises:
            RuntimeError: If client is not initialized.
            Exception: If listing indexes fails.

        Example:
            ```python
            indexes = client.list_indexes()
            print(f"Available indexes: {indexes}")
            ```
        """
        if self._client is None:
            raise RuntimeError(
                "Pinecone client not initialized. Call initialize() first or use as context manager."
            )

        try:
            # In Pinecone v8, list_indexes() returns a list of index objects
            indexes = self._client.list_indexes()
            # Extract index names from the response
            if hasattr(indexes, "names"):
                index_names = indexes.names()
            elif isinstance(indexes, list):
                index_names = [
                    idx.name if hasattr(idx, "name") else str(idx) for idx in indexes
                ]
            else:
                # Fallback: try to iterate
                index_names = (
                    [idx.name for idx in indexes]
                    if hasattr(indexes, "__iter__")
                    else []
                )

            logger.debug(f"Found {len(index_names)} indexes: {index_names}")
            return index_names
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            raise

    async def query(
        self,
        query_vector: list[float],
        index_name: Optional[str] = None,
        top_k: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
        namespace: Optional[str] = None,
    ) -> Any:
        """Query the Pinecone index for similar vectors.

        This method queries the specified index with a vector and optional
        metadata filters to find the most similar documents.

        Args:
            query_vector: The query vector (embedding) to search for.
            index_name: Name of the index to query. If not provided, uses
                the default index from settings or initialization.
            top_k: Number of results to return (default: 10).
            metadata_filter: Optional metadata filter dict. Supports Pinecone
                filter expressions. Example: {"sector": "AI", "status": "active"}
            include_metadata: Whether to include metadata in results (default: True).
            include_values: Whether to include vector values in results (default: False).
            namespace: Optional namespace to query within the index.

        Returns:
            Query response object containing matches with scores, metadata, and IDs.
            The response has a 'matches' attribute containing the results.

        Raises:
            RuntimeError: If client is not initialized.
            ValueError: If index_name is not provided and no default is set.
            Exception: If query execution fails.

        Example:
            ```python
            # Simple query
            results = await client.query(
                query_vector=[0.1, 0.2, 0.3, ...],
                top_k=5
            )

            # Query with metadata filter
            results = await client.query(
                query_vector=embedding,
                index_name="companies",
                top_k=10,
                metadata_filter={"sector": "AI", "founded_year": {"$gte": 2020}}
            )

            # Access results
            for match in results.matches:
                print(f"ID: {match.id}, Score: {match.score}")
                if match.metadata:
                    print(f"Metadata: {match.metadata}")
            ```
        """
        if self._client is None:
            raise RuntimeError(
                "Pinecone client not initialized. Call initialize() first or use as context manager."
            )

        index_name = index_name or self._default_index
        if not index_name:
            raise ValueError(
                "index_name must be provided either as parameter or set as default_index."
            )

        try:
            index = self._client.Index(index_name)

            # Build query parameters - Pinecone v8 uses different parameter names
            query_params: dict[str, Any] = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": include_metadata,
                "include_values": include_values,
            }

            if metadata_filter:
                query_params["filter"] = metadata_filter

            if namespace:
                query_params["namespace"] = namespace

            # Note: Pinecone query operations are typically synchronous
            # but we keep the async interface for consistency
            response = index.query(**query_params)
            logger.debug(
                f"Query executed successfully on index '{index_name}': "
                f"top_k={top_k}, filter={metadata_filter is not None}"
            )
            return response
        except Exception as e:
            logger.error(f"Query execution failed on index '{index_name}': {e}")
            raise

    async def fetch(
        self,
        ids: list[str],
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Any:
        """Fetch vectors by their IDs.

        Args:
            ids: List of vector IDs to fetch.
            index_name: Name of the index to query. If not provided, uses
                the default index from settings or initialization.
            namespace: Optional namespace to query within the index.

        Returns:
            Dictionary mapping IDs to their vector data, metadata, and values.

        Raises:
            RuntimeError: If client is not initialized.
            ValueError: If index_name is not provided and no default is set.
            Exception: If fetch execution fails.

        Example:
            ```python
            vectors = await client.fetch(
                ids=["company-1", "company-2"],
                index_name="companies"
            )
            ```
        """
        if self._client is None:
            raise RuntimeError(
                "Pinecone client not initialized. Call initialize() first or use as context manager."
            )

        index_name = index_name or self._default_index
        if not index_name:
            raise ValueError(
                "index_name must be provided either as parameter or set as default_index."
            )

        try:
            index = self._client.Index(index_name)

            fetch_params: dict[str, Any] = {"ids": ids}
            if namespace:
                fetch_params["namespace"] = namespace

            response = index.fetch(**fetch_params)
            logger.debug(
                f"Fetch executed successfully on index '{index_name}': {len(ids)} IDs"
            )
            return response
        except Exception as e:
            logger.error(f"Fetch execution failed on index '{index_name}': {e}")
            raise

    def is_connected(self) -> bool:
        """Check if the Pinecone client is initialized.

        Returns:
            True if client is initialized, False otherwise.
        """
        return self._client is not None

    def get_index_info(self, index_name: Optional[str] = None) -> dict[str, Any]:
        """Get information about a specific index.

        Args:
            index_name: Name of the index. If not provided, uses the default index.

        Returns:
            Dictionary containing index information (dimension, metric, etc.).

        Raises:
            RuntimeError: If client is not initialized.
            ValueError: If index_name is not provided and no default is set.
            Exception: If getting index info fails.

        Example:
            ```python
            info = client.get_index_info("companies")
            print(f"Index dimension: {info['dimension']}")
            ```
        """
        if self._client is None:
            raise RuntimeError(
                "Pinecone client not initialized. Call initialize() first or use as context manager."
            )

        index_name = index_name or self._default_index
        if not index_name:
            raise ValueError(
                "index_name must be provided either as parameter or set as default_index."
            )

        try:
            index = self._client.Index(index_name)
            stats = index.describe_index_stats()
            logger.debug(f"Retrieved index info for '{index_name}'")
            return {
                "name": index_name,
                "dimension": stats.dimension if hasattr(stats, "dimension") else None,
                "index_fullness": (
                    stats.index_fullness if hasattr(stats, "index_fullness") else None
                ),
                "total_vector_count": (
                    stats.total_vector_count
                    if hasattr(stats, "total_vector_count")
                    else None
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get index info for '{index_name}': {e}")
            raise


# Singleton instance for convenience
_default_client: Optional[PineconeClient] = None


async def get_client() -> PineconeClient:
    """Get or create the default singleton PineconeClient instance.

    This is a convenience function for modules that want to use a shared
    client instance without managing it themselves.

    Returns:
        The default PineconeClient instance.

    Example:
        ```python
        client = await get_client()
        results = await client.query(
            query_vector=embedding,
            top_k=10
        )
        ```
    """
    global _default_client
    if _default_client is None:
        _default_client = PineconeClient()
        await _default_client.initialize()
    return _default_client


async def close_default_client() -> None:
    """Close the default singleton client instance."""
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None
