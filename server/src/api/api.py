"""Main FastAPI application.

This module sets up the FastAPI application with middleware, routers, and
lifecycle management. The API is stateless and acts as a bridge to Temporal workflows.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import setup_cors, setup_error_handlers

logger = logging.getLogger(__name__)

# Import routers with error handling
try:
    from src.api.routers import batch, tasks
    logger.info("Successfully imported batch and tasks routers")
except ImportError as e:
    logger.error(f"Failed to import routers: {e}", exc_info=True)
    raise

# Verify batch router is available
if not hasattr(batch, "router"):
    raise ImportError("batch.router not found - check batch.py for router definition")
from src.db import close_redis_client, get_redis_client
from src.temporal import (
    DEFAULT_TASK_QUEUE,
    DEFAULT_TEMPORAL_ADDRESS,
    DEFAULT_TEMPORAL_NAMESPACE,
    close_client,
    get_client,
)


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

    # Initialize Redis client
    try:
        await get_redis_client()
        logger.info("Redis client initialized and connected")
    except Exception as e:
        logger.warning(
            f"Failed to initialize Redis client: {e}. "
            "API will start but conversation history features will be unavailable. "
            "Make sure Redis server is running."
        )
        # Don't fail startup - graceful degradation
        # Conversation history will return empty lists on error

    yield

    # Shutdown
    logger.info("Shutting down API server...")
    try:
        await close_client()
        logger.info("Temporal client closed")
    except Exception as e:
        logger.error(f"Error closing Temporal client: {e}", exc_info=True)

    try:
        await close_redis_client()
        logger.info("Redis client closed")
    except Exception as e:
        logger.error(f"Error closing Redis client: {e}", exc_info=True)


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
logger.info("Starting router registration...")
logger.info(f"Batch router object: {batch.router}")
logger.info(f"Batch router routes before registration: {len(batch.router.routes)}")
for route in batch.router.routes:
    logger.info(f"  - {route.methods if hasattr(route, 'methods') else 'N/A'} {route.path if hasattr(route, 'path') else 'N/A'}")

try:
    app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
    logger.info(f"Registered tasks router with {len(tasks.router.routes)} route(s)")
except Exception as e:
    logger.error(f"Failed to register tasks router: {e}", exc_info=True)
    raise

try:
    app.include_router(batch.router, prefix="/batch", tags=["batch"])
    logger.info(f"Registered batch router with {len(batch.router.routes)} route(s)")
    # Log all registered routes in the app
    logger.info(f"Total app routes after batch registration: {len(app.routes)}")
    for route in app.routes:
        if hasattr(route, 'path') and '/batch' in route.path:
            logger.info(f"  Batch route: {route.methods if hasattr(route, 'methods') else 'N/A'} {route.path}")
except Exception as e:
    logger.error(f"Failed to register batch router: {e}", exc_info=True)
    raise


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

    # Suppress the harmless RuntimeWarning about module import
    import warnings

    import uvicorn

    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

    uvicorn.run(
        "src.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
