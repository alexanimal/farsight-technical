"""Batch operation router for batch processing API endpoints.

This router implements batch operation endpoints:
- POST /batch - Start a batch operation for multiple items
- GET /batch/{batch_id} - Get batch operation status
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.middleware.auth import verify_api_key
from src.temporal import TemporalClient, get_client
from src.temporal.batch_operations import (
    build_pipeline_batch_items,
    build_sector_analysis_contexts,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class SectorAnalysisBatchRequest(BaseModel):
    """Request model for sector analysis batch operation."""

    sectors: List[str] = Field(..., description="List of sector names to analyze")
    time_period_start: Optional[str] = Field(
        None,
        description="Start date in ISO format. If not provided, uses last 12 months.",
    )
    time_period_end: Optional[str] = Field(
        None, description="End date in ISO format. If not provided, uses current date."
    )
    granularity: str = Field(
        default="quarterly",
        description="Time granularity: 'monthly', 'quarterly', or 'yearly'",
    )
    min_funding_amount: Optional[float] = Field(
        None, description="Optional minimum funding amount filter"
    )


class BatchOperationResponse(BaseModel):
    """Response model for batch operation creation."""

    batch_id: str = Field(..., description="Batch operation identifier")
    total_items: int = Field(..., description="Total number of items in batch")
    workflow_type: str = Field(..., description="Type of workflow being executed")
    message: str = Field(..., description="Status message")


class BatchOperationStatusResponse(BaseModel):
    """Response model for batch operation status."""

    batch_id: str = Field(..., description="Batch operation identifier")
    total_workflows: int = Field(..., description="Total number of workflows")
    completed: int = Field(..., description="Number of completed workflows")
    running: int = Field(..., description="Number of running workflows")
    failed: int = Field(..., description="Number of failed workflows")
    workflow_ids: List[str] = Field(..., description="List of workflow IDs in batch")
    note: Optional[str] = Field(None, description="Additional information")


@router.post(
    "/sector-analysis",
    response_model=BatchOperationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_sector_analysis_batch(
    request: SectorAnalysisBatchRequest,
    api_key: str = Depends(verify_api_key),
) -> BatchOperationResponse:
    """Create a batch operation for sector analysis.

    This endpoint starts a batch operation that runs the sector_analysis pipeline
    for multiple sectors in parallel.

    Args:
        request: Sector analysis batch request.
        api_key: Verified API key (from dependency).

    Returns:
        BatchOperationResponse with batch ID and status.

    Raises:
        HTTPException: If batch operation creation fails.
    """
    try:
        temporal_client: TemporalClient = await get_client()

        # Validate sectors list
        if not request.sectors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="sectors list cannot be empty",
            )

        # Build contexts for each sector
        contexts = build_sector_analysis_contexts(
            sectors=request.sectors,
            time_period_start=request.time_period_start,
            time_period_end=request.time_period_end,
            granularity=request.granularity,
            min_funding_amount=request.min_funding_amount,
        )

        # Build batch items
        items = build_pipeline_batch_items(
            pipeline_type="sector_analysis",
            contexts=contexts,
        )

        # Start batch operation
        batch_id = await temporal_client.start_batch_operation(
            workflow_type="pipeline",
            items=items,
        )

        logger.info(
            f"Created batch operation {batch_id} for {len(request.sectors)} sector(s): "
            f"{', '.join(request.sectors)}"
        )

        return BatchOperationResponse(
            batch_id=batch_id,
            total_items=len(items),
            workflow_type="pipeline",
            message=f"Batch operation started for {len(request.sectors)} sector(s)",
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid request for batch operation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to create batch operation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch operation: {str(e)}",
        )


@router.get(
    "/{batch_id}",
    response_model=BatchOperationStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_batch_operation_status(
    batch_id: str,
    api_key: str = Depends(verify_api_key),
) -> BatchOperationStatusResponse:
    """Get the status of a batch operation.

    Args:
        batch_id: Batch operation identifier.
        api_key: Verified API key (from dependency).

    Returns:
        BatchOperationStatusResponse with batch status.

    Raises:
        HTTPException: If batch operation not found or status query fails.
    """
    try:
        temporal_client: TemporalClient = await get_client()

        # Get batch status
        status_info = await temporal_client.get_batch_operation_status(
            batch_id=batch_id,
            workflow_type="pipeline",
        )

        return BatchOperationStatusResponse(**status_info)

    except RuntimeError as e:
        logger.warning(f"Batch operation not found: {batch_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch operation {batch_id} not found",
        )
    except Exception as e:
        logger.error(f"Failed to get batch operation status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch operation status: {str(e)}",
        )
