"""Authentication middleware for API requests."""

import logging
from typing import Optional

from fastapi import Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Simple API key authentication
# In production, this should be more sophisticated (JWT, OAuth, etc.)
API_KEY_HEADER = "X-API-Key"
API_KEY_ENV = "API_KEY"

# Default API key for development (should be set via environment variable)
DEFAULT_API_KEY = "1234567890"

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias=API_KEY_HEADER),
) -> str:
    """Verify API key from request header.

    Args:
        x_api_key: API key from X-API-Key header.

    Returns:
        The API key if valid.

    Raises:
        HTTPException: If API key is missing or invalid.
    """
    import os

    expected_key = os.getenv(API_KEY_ENV, DEFAULT_API_KEY)

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header.",
        )

    if x_api_key != expected_key:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key
