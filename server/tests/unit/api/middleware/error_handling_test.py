"""Unit tests for the error handling middleware module.

This module tests the setup_error_handlers function and its exception handlers,
including ValueError handler, general exception handler, logging, and response structure.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.api.middleware.error_handling import setup_error_handlers


@pytest.fixture
def mock_app():
    """Create a mock FastAPI application instance."""
    app = MagicMock(spec=FastAPI)
    app.exception_handler = MagicMock()
    return app


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request instance."""
    request = MagicMock(spec=Request)
    return request


class TestSetupErrorHandlers:
    """Test setup_error_handlers function."""

    def test_setup_error_handlers_registers_value_error_handler(self, mock_app):
        """Test that setup_error_handlers registers ValueError handler."""
        setup_error_handlers(mock_app)

        # Should register ValueError handler
        mock_app.exception_handler.assert_any_call(ValueError)

    def test_setup_error_handlers_registers_exception_handler(self, mock_app):
        """Test that setup_error_handlers registers general Exception handler."""
        setup_error_handlers(mock_app)

        # Should register Exception handler
        mock_app.exception_handler.assert_any_call(Exception)

    def test_setup_error_handlers_registers_both_handlers(self, mock_app):
        """Test that setup_error_handlers registers both handlers."""
        setup_error_handlers(mock_app)

        # Should register both handlers
        assert mock_app.exception_handler.call_count == 2
        call_args_list = [call[0][0] for call in mock_app.exception_handler.call_args_list]
        assert ValueError in call_args_list
        assert Exception in call_args_list

    def test_setup_error_handlers_returns_none(self, mock_app):
        """Test that setup_error_handlers returns None."""
        result = setup_error_handlers(mock_app)

        assert result is None

    def test_setup_error_handlers_idempotent(self, mock_app):
        """Test that calling setup_error_handlers multiple times works."""
        setup_error_handlers(mock_app)
        setup_error_handlers(mock_app)

        # Should register handlers each time
        assert mock_app.exception_handler.call_count == 4


class TestValueErrorHandler:
    """Test ValueError exception handler."""

    @pytest.mark.asyncio
    async def test_value_error_handler_logs_warning(self, mock_request):
        """Test that ValueError handler logs a warning."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = ValueError("Test validation error")

        with patch("src.api.middleware.error_handling.logger") as mock_logger:
            # Get the registered handler
            handler = app.exception_handlers[ValueError]
            await handler(mock_request, test_error)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "Validation error" in mock_logger.warning.call_args[0][0]
            assert "Test validation error" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_value_error_handler_returns_json_response(self, mock_request):
        """Test that ValueError handler returns JSONResponse."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = ValueError("Test validation error")
        handler = app.exception_handlers[ValueError]
        response = await handler(mock_request, test_error)

        assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_value_error_handler_status_code(self, mock_request):
        """Test that ValueError handler returns 400 status code."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = ValueError("Test validation error")
        handler = app.exception_handlers[ValueError]
        response = await handler(mock_request, test_error)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_value_error_handler_response_content(self, mock_request):
        """Test that ValueError handler returns correct response content."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = ValueError("Test validation error")
        handler = app.exception_handlers[ValueError]
        response = await handler(mock_request, test_error)

        # Parse response body
        import json

        body = json.loads(response.body.decode())
        assert body["error"] == "Validation error"
        assert body["detail"] == "Test validation error"
        assert body["type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_value_error_handler_with_empty_message(self, mock_request):
        """Test ValueError handler with empty error message."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = ValueError("")
        handler = app.exception_handlers[ValueError]
        response = await handler(mock_request, test_error)

        import json

        body = json.loads(response.body.decode())
        assert body["error"] == "Validation error"
        assert body["detail"] == ""
        assert body["type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_value_error_handler_with_complex_message(self, mock_request):
        """Test ValueError handler with complex error message."""
        app = FastAPI()
        setup_error_handlers(app)

        complex_message = "Field 'email' must be a valid email address: invalid@"
        test_error = ValueError(complex_message)
        handler = app.exception_handlers[ValueError]
        response = await handler(mock_request, test_error)

        import json

        body = json.loads(response.body.decode())
        assert body["detail"] == complex_message


class TestGeneralExceptionHandler:
    """Test general Exception handler."""

    @pytest.mark.asyncio
    async def test_general_exception_handler_logs_error(self, mock_request):
        """Test that general exception handler logs an error with exc_info."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = RuntimeError("Test runtime error")

        with patch("src.api.middleware.error_handling.logger") as mock_logger:
            # Get the registered handler
            handler = app.exception_handlers[Exception]
            await handler(mock_request, test_error)

            # Verify error was logged with exc_info
            mock_logger.error.assert_called_once()
            assert "Unhandled exception" in mock_logger.error.call_args[0][0]
            assert "Test runtime error" in mock_logger.error.call_args[0][0]
            # Verify exc_info=True was passed
            assert mock_logger.error.call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    async def test_general_exception_handler_returns_json_response(self, mock_request):
        """Test that general exception handler returns JSONResponse."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = RuntimeError("Test runtime error")
        handler = app.exception_handlers[Exception]
        response = await handler(mock_request, test_error)

        assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_general_exception_handler_status_code(self, mock_request):
        """Test that general exception handler returns 500 status code."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = RuntimeError("Test runtime error")
        handler = app.exception_handlers[Exception]
        response = await handler(mock_request, test_error)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_general_exception_handler_response_content(self, mock_request):
        """Test that general exception handler returns correct response content."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = RuntimeError("Test runtime error")
        handler = app.exception_handlers[Exception]
        response = await handler(mock_request, test_error)

        import json

        body = json.loads(response.body.decode())
        assert body["error"] == "Internal server error"
        assert body["detail"] == "An unexpected error occurred"
        assert body["type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_general_exception_handler_with_different_exception_types(
        self, mock_request
    ):
        """Test general exception handler with different exception types."""
        app = FastAPI()
        setup_error_handlers(app)

        handler = app.exception_handlers[Exception]

        # Test with different exception types
        exception_types = [
            KeyError("key not found"),
            TypeError("type error"),
            AttributeError("attribute error"),
            IndexError("index out of range"),
        ]

        for exc in exception_types:
            response = await handler(mock_request, exc)

            import json

            body = json.loads(response.body.decode())
            assert body["type"] == type(exc).__name__
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_general_exception_handler_with_custom_exception(self, mock_request):
        """Test general exception handler with custom exception class."""
        app = FastAPI()
        setup_error_handlers(app)

        class CustomException(Exception):
            pass

        test_error = CustomException("Custom error message")
        handler = app.exception_handlers[Exception]
        response = await handler(mock_request, test_error)

        import json

        body = json.loads(response.body.decode())
        assert body["type"] == "CustomException"
        assert body["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_general_exception_handler_with_empty_message(self, mock_request):
        """Test general exception handler with empty error message."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = RuntimeError("")
        handler = app.exception_handlers[Exception]
        response = await handler(mock_request, test_error)

        import json

        body = json.loads(response.body.decode())
        assert body["error"] == "Internal server error"
        assert body["detail"] == "An unexpected error occurred"
        assert body["type"] == "RuntimeError"


class TestExceptionHandlerPriority:
    """Test exception handler priority and specificity."""

    @pytest.mark.asyncio
    async def test_value_error_handler_takes_precedence(self, mock_request):
        """Test that ValueError handler is used for ValueError, not general handler."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = ValueError("Value error")

        # ValueError should use specific handler
        value_error_handler = app.exception_handlers[ValueError]
        response = await value_error_handler(mock_request, test_error)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

        import json

        body = json.loads(response.body.decode())
        assert body["type"] == "ValueError"
        assert body["error"] == "Validation error"

    @pytest.mark.asyncio
    async def test_general_handler_for_non_value_errors(self, mock_request):
        """Test that general handler is used for non-ValueError exceptions."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = RuntimeError("Runtime error")

        # RuntimeError should use general handler
        general_handler = app.exception_handlers[Exception]
        response = await general_handler(mock_request, test_error)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        import json

        body = json.loads(response.body.decode())
        assert body["type"] == "RuntimeError"
        assert body["error"] == "Internal server error"


class TestErrorHandlingIntegration:
    """Test error handling integration with FastAPI app."""

    def test_setup_error_handlers_with_real_fastapi_app(self):
        """Test that setup_error_handlers works with real FastAPI app."""
        app = FastAPI()

        # Should not raise an exception
        setup_error_handlers(app)

        # Verify handlers were registered
        assert ValueError in app.exception_handlers
        assert Exception in app.exception_handlers

    @pytest.mark.asyncio
    async def test_error_handlers_with_mocked_request(self, mock_request):
        """Test error handlers with mocked Request object."""
        app = FastAPI()
        setup_error_handlers(app)

        # Test ValueError handler directly
        value_error = ValueError("Test value error")
        value_error_handler = app.exception_handlers[ValueError]
        response = await value_error_handler(mock_request, value_error)

        assert response.status_code == 400
        import json

        body = json.loads(response.body.decode())
        assert body["error"] == "Validation error"
        assert body["type"] == "ValueError"
        assert body["detail"] == "Test value error"

        # Test general exception handler directly
        runtime_error = RuntimeError("Test runtime error")
        general_handler = app.exception_handlers[Exception]
        response = await general_handler(mock_request, runtime_error)

        assert response.status_code == 500
        body = json.loads(response.body.decode())
        assert body["error"] == "Internal server error"
        assert body["type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_error_handlers_with_multiple_setups(self, mock_request):
        """Test that multiple calls to setup_error_handlers work correctly."""
        app = FastAPI()

        setup_error_handlers(app)
        setup_error_handlers(app)

        # Should still have both handlers
        assert ValueError in app.exception_handlers
        assert Exception in app.exception_handlers

        # Handlers should still work after multiple setups
        test_error = ValueError("Test error")
        handler = app.exception_handlers[ValueError]
        response = await handler(mock_request, test_error)

        assert response.status_code == 400
        import json

        body = json.loads(response.body.decode())
        assert body["type"] == "ValueError"
        assert body["error"] == "Validation error"


class TestErrorHandlingEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_value_error_handler_with_none_request(self):
        """Test ValueError handler handles None request gracefully."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = ValueError("Test error")
        handler = app.exception_handlers[ValueError]

        # Handler should still work even with None request
        # (though this shouldn't happen in practice)
        response = await handler(None, test_error)
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_general_handler_with_none_request(self):
        """Test general handler handles None request gracefully."""
        app = FastAPI()
        setup_error_handlers(app)

        test_error = RuntimeError("Test error")
        handler = app.exception_handlers[Exception]

        response = await handler(None, test_error)
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_value_error_handler_with_unicode_message(self, mock_request):
        """Test ValueError handler with unicode characters in message."""
        app = FastAPI()
        setup_error_handlers(app)

        unicode_message = "验证错误: 测试-🚀-αβγ"
        test_error = ValueError(unicode_message)
        handler = app.exception_handlers[ValueError]
        response = await handler(mock_request, test_error)

        import json

        body = json.loads(response.body.decode())
        assert body["detail"] == unicode_message

    @pytest.mark.asyncio
    async def test_general_handler_with_unicode_message(self, mock_request):
        """Test general handler with unicode characters in message."""
        app = FastAPI()
        setup_error_handlers(app)

        unicode_message = "运行时错误: 测试-🚀-αβγ"
        test_error = RuntimeError(unicode_message)
        handler = app.exception_handlers[Exception]
        response = await handler(mock_request, test_error)

        import json

        body = json.loads(response.body.decode())
        # Note: detail is fixed, but type should be correct
        assert body["type"] == "RuntimeError"

