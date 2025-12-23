"""Main FastAPI application.

This module sets up the FastAPI application with middleware, routers, and
lifecycle management. The API is stateless and acts as a bridge to Temporal workflows.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import setup_cors, setup_error_handlers
from src.api.routers import tasks
from src.temporal import (
    DEFAULT_TASK_QUEUE,
    DEFAULT_TEMPORAL_ADDRESS,
    DEFAULT_TEMPORAL_NAMESPACE,
    close_client,
    get_client,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown).

    Args:
        app: FastAPI application instance.
    """
    # Startup
    logger.info("Starting API server...")
    try:
        # Initialize Temporal client
        # Read configuration from environment variables (set in docker-compose.yml)
        # Note: This will fail if Temporal server is not running
        # The API will still start, but workflow operations will fail until Temporal is available
        temporal_address = os.getenv("TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)
        temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", DEFAULT_TEMPORAL_NAMESPACE)
        task_queue = os.getenv("TEMPORAL_TASK_QUEUE", DEFAULT_TASK_QUEUE)
        
        await get_client(
            temporal_address=temporal_address,
            temporal_namespace=temporal_namespace,
            task_queue=task_queue,
        )
        logger.info(
            f"Temporal client initialized and connected to {temporal_address} "
            f"(namespace: {temporal_namespace})"
        )
    except Exception as e:
        logger.warning(
            f"Failed to initialize Temporal client: {e}. "
            "API will start but workflow operations will fail. "
            "Make sure Temporal server is running (see README for instructions)."
        )
        # Don't fail startup - client will be initialized on first use
        # This allows the API to start even if Temporal isn't available yet

    yield

    # Shutdown
    logger.info("Shutting down API server...")
    try:
        await close_client()
        logger.info("Temporal client closed")
    except Exception as e:
        logger.error(f"Error closing Temporal client: {e}", exc_info=True)


# Create FastAPI app
app = FastAPI(
    title="Farsight Technical API",
    description="Task-oriented API for agent orchestration via Temporal",
    version="1.0.0",
    lifespan=lifespan,
)

# Set up middleware
setup_cors(app)
setup_error_handlers(app)

# Include routers
app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "status": "ok",
        "service": "Farsight Technical API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Allow running the API directly with: python -m src.api.api
# Note: The RuntimeWarning about module import is harmless and can be ignored.
# For production, use: uvicorn src.api.api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import sys
    import uvicorn
    
    # Suppress the harmless RuntimeWarning about module import
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")
    
    uvicorn.run(
        "src.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )