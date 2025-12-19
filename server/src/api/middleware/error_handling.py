"""Error handling middleware and exception handlers."""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    """Set up global error handlers for the FastAPI app.

    Args:
        app: FastAPI application instance.
    """

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle value errors (validation errors).

        Args:
            request: The request that caused the error.
            exc: The ValueError exception.

        Returns:
            JSON response with error details.
        """
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Validation error",
                "detail": str(exc),
                "type": "ValueError",
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions.

        Args:
            request: The request that caused the error.
            exc: The exception.

        Returns:
            JSON response with error details.
        """
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
                "type": type(exc).__name__,
            },
        )
