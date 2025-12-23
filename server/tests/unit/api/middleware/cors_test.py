"""Unit tests for the CORS middleware module.

This module tests the setup_cors function and its configuration,
including middleware registration and CORS settings.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.cors import setup_cors


@pytest.fixture
def mock_app():
    """Create a mock FastAPI application instance."""
    app = MagicMock(spec=FastAPI)
    app.add_middleware = MagicMock()
    return app


class TestSetupCors:
    """Test setup_cors function."""

    def test_setup_cors_calls_add_middleware(self, mock_app):
        """Test that setup_cors calls add_middleware on the app."""
        setup_cors(mock_app)

        mock_app.add_middleware.assert_called_once()

    def test_setup_cors_with_cors_middleware(self, mock_app):
        """Test that setup_cors uses CORSMiddleware."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        assert call_args[0][0] == CORSMiddleware

    def test_setup_cors_allow_origins_configuration(self, mock_app):
        """Test that setup_cors configures allow_origins correctly."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_origins"] == ["*"]

    def test_setup_cors_allow_credentials_configuration(self, mock_app):
        """Test that setup_cors configures allow_credentials correctly."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_credentials"] is True

    def test_setup_cors_allow_methods_configuration(self, mock_app):
        """Test that setup_cors configures allow_methods correctly."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_methods"] == ["*"]

    def test_setup_cors_allow_headers_configuration(self, mock_app):
        """Test that setup_cors configures allow_headers correctly."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_headers"] == ["*"]

    def test_setup_cors_all_configurations_together(self, mock_app):
        """Test that setup_cors applies all configurations correctly."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]

        assert kwargs["allow_origins"] == ["*"]
        assert kwargs["allow_credentials"] is True
        assert kwargs["allow_methods"] == ["*"]
        assert kwargs["allow_headers"] == ["*"]

    def test_setup_cors_called_only_once(self, mock_app):
        """Test that setup_cors only calls add_middleware once."""
        setup_cors(mock_app)

        assert mock_app.add_middleware.call_count == 1

    def test_setup_cors_idempotent(self, mock_app):
        """Test that calling setup_cors multiple times adds middleware multiple times."""
        setup_cors(mock_app)
        setup_cors(mock_app)
        setup_cors(mock_app)

        assert mock_app.add_middleware.call_count == 3

    def test_setup_cors_with_real_fastapi_app(self):
        """Test that setup_cors works with a real FastAPI app instance."""
        app = FastAPI()

        # Should not raise an exception
        setup_cors(app)

        # Verify middleware was added by checking the middleware stack
        # FastAPI stores middleware in app.user_middleware
        assert len(app.user_middleware) > 0

    def test_setup_cors_middleware_order(self, mock_app):
        """Test that setup_cors passes middleware in correct order."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        # First positional argument should be CORSMiddleware
        assert call_args[0][0] == CORSMiddleware
        # Keyword arguments should follow
        assert "allow_origins" in call_args[1]

    def test_setup_cors_no_additional_kwargs(self, mock_app):
        """Test that setup_cors only passes expected keyword arguments."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]

        # Should only have the 4 expected kwargs
        expected_keys = {
            "allow_origins",
            "allow_credentials",
            "allow_methods",
            "allow_headers",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_setup_cors_with_patched_cors_middleware(self, mock_app):
        """Test setup_cors behavior when CORSMiddleware is patched."""
        with patch("src.api.middleware.cors.CORSMiddleware") as mock_cors_class:
            setup_cors(mock_app)

            call_args = mock_app.add_middleware.call_args
            # Should use the patched CORSMiddleware
            assert call_args[0][0] == mock_cors_class

    def test_setup_cors_returns_none(self, mock_app):
        """Test that setup_cors returns None."""
        result = setup_cors(mock_app)

        assert result is None

    def test_setup_cors_with_none_app_raises_error(self):
        """Test that setup_cors raises error when app is None."""
        with pytest.raises(AttributeError):
            setup_cors(None)


class TestSetupCorsConfigurationValues:
    """Test the specific configuration values used in setup_cors."""

    def test_allow_origins_wildcard(self, mock_app):
        """Test that allow_origins uses wildcard for development."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_origins"] == ["*"]
        # In production, this should be specific origins, not wildcard

    def test_allow_credentials_enabled(self, mock_app):
        """Test that allow_credentials is enabled."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_credentials"] is True

    def test_allow_methods_wildcard(self, mock_app):
        """Test that allow_methods uses wildcard."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_methods"] == ["*"]

    def test_allow_headers_wildcard(self, mock_app):
        """Test that allow_headers uses wildcard."""
        setup_cors(mock_app)

        call_args = mock_app.add_middleware.call_args
        kwargs = call_args[1]
        assert kwargs["allow_headers"] == ["*"]


class TestSetupCorsIntegration:
    """Test setup_cors integration with FastAPI app."""

    def test_setup_cors_integration_with_fastapi(self):
        """Test that setup_cors integrates correctly with FastAPI."""
        app = FastAPI(title="Test App")

        # Setup CORS
        setup_cors(app)

        # Verify middleware was added
        assert len(app.user_middleware) > 0

        # Verify the middleware is CORSMiddleware
        # FastAPI stores middleware as (middleware_class, options) tuples
        middleware_classes = [mw.cls for mw in app.user_middleware]
        assert CORSMiddleware in middleware_classes

    def test_setup_cors_with_existing_middleware(self):
        """Test that setup_cors adds to existing middleware."""
        app = FastAPI(title="Test App")

        # Add some other middleware first
        from fastapi.middleware.gzip import GZipMiddleware

        app.add_middleware(GZipMiddleware)

        # Setup CORS
        setup_cors(app)

        # Should have both middlewares
        assert len(app.user_middleware) >= 2

        # Verify CORS middleware is present
        middleware_classes = [mw.cls for mw in app.user_middleware]
        assert CORSMiddleware in middleware_classes
