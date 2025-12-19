"""Main FastAPI application.

This module sets up the FastAPI application with middleware, routers, and
lifecycle management. The API is stateless and acts as a bridge to Temporal workflows.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import setup_cors, setup_error_handlers
from src.api.routers import tasks
from src.temporal import close_client, get_client

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
        # Note: This will fail if Temporal server is not running
        # The API will still start, but workflow operations will fail until Temporal is available
        await get_client()
        logger.info("Temporal client initialized and connected")
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
