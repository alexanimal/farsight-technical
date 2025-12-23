"""Unit tests for the authentication middleware module.

This module tests the verify_api_key function and its various behaviors,
including API key validation, environment variable handling, and error cases.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException, status

from src.api.middleware.auth import API_KEY_ENV, API_KEY_HEADER, DEFAULT_API_KEY, verify_api_key


class TestVerifyApiKeySuccess:
    """Test successful API key verification."""

    @pytest.mark.asyncio
    async def test_verify_api_key_with_valid_key_from_env(self, monkeypatch):
        """Test verification with valid API key from environment variable."""
        test_key = "test-api-key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        result = await verify_api_key(x_api_key=test_key)

        assert result == test_key

    @pytest.mark.asyncio
    async def test_verify_api_key_with_valid_key_from_default(self, monkeypatch):
        """Test verification with valid API key using default value."""
        # Remove environment variable to use default
        monkeypatch.delenv(API_KEY_ENV, raising=False)

        result = await verify_api_key(x_api_key=DEFAULT_API_KEY)

        assert result == DEFAULT_API_KEY

    @pytest.mark.asyncio
    async def test_verify_api_key_case_sensitive(self, monkeypatch):
        """Test that API key verification is case-sensitive."""
        test_key = "Test-Key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        # Different case should fail
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="test-key-12345")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid API key" in exc_info.value.detail

        # Correct case should succeed
        result = await verify_api_key(x_api_key=test_key)
        assert result == test_key

    @pytest.mark.asyncio
    async def test_verify_api_key_with_whitespace(self, monkeypatch):
        """Test that API key verification handles whitespace correctly."""
        test_key = "test-key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        # Key with leading/trailing whitespace should fail
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key=" test-key-12345 ")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

        # Correct key should succeed
        result = await verify_api_key(x_api_key=test_key)
        assert result == test_key


class TestVerifyApiKeyMissingKey:
    """Test API key verification when key is missing."""

    @pytest.mark.asyncio
    async def test_verify_api_key_missing_header(self, monkeypatch):
        """Test verification when X-API-Key header is missing."""
        test_key = "test-api-key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key=None)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "API key required" in exc_info.value.detail
        assert "X-API-Key header" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_api_key_empty_string(self, monkeypatch):
        """Test verification when API key is empty string."""
        test_key = "test-api-key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="")

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "API key required" in exc_info.value.detail


class TestVerifyApiKeyInvalidKey:
    """Test API key verification with invalid keys."""

    @pytest.mark.asyncio
    async def test_verify_api_key_invalid_key(self, monkeypatch):
        """Test verification with invalid API key."""
        test_key = "test-api-key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="wrong-key")

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_api_key_invalid_key_logs_warning(self, monkeypatch):
        """Test that invalid API key attempts are logged."""
        test_key = "test-api-key-12345"
        invalid_key = "wrong-key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        with patch("src.api.middleware.auth.logger") as mock_logger:
            with pytest.raises(HTTPException):
                await verify_api_key(x_api_key=invalid_key)

            # Verify warning was logged with truncated key
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Invalid API key attempt" in call_args
            assert invalid_key[:10] in call_args

    @pytest.mark.asyncio
    async def test_verify_api_key_partial_match_fails(self, monkeypatch):
        """Test that partial key matches fail."""
        test_key = "test-api-key-12345"
        monkeypatch.setenv(API_KEY_ENV, test_key)

        # Partial match should fail
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="test-api-key-1234")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

        # Exact match should succeed
        result = await verify_api_key(x_api_key=test_key)
        assert result == test_key


class TestVerifyApiKeyEnvironmentVariables:
    """Test API key verification with different environment variable configurations."""

    @pytest.mark.asyncio
    async def test_verify_api_key_env_var_takes_precedence(self, monkeypatch):
        """Test that environment variable takes precedence over default."""
        env_key = "env-api-key-12345"
        monkeypatch.setenv(API_KEY_ENV, env_key)

        # Default key should fail
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key=DEFAULT_API_KEY)
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

        # Env key should succeed
        result = await verify_api_key(x_api_key=env_key)
        assert result == env_key

    @pytest.mark.asyncio
    async def test_verify_api_key_empty_env_var_uses_default(self, monkeypatch):
        """Test that unset env var uses default."""
        # Delete env var to use default (not setting to empty string)
        monkeypatch.delenv(API_KEY_ENV, raising=False)

        # Unset env var should use default
        result = await verify_api_key(x_api_key=DEFAULT_API_KEY)
        assert result == DEFAULT_API_KEY

        # Wrong key should fail
        with pytest.raises(HTTPException):
            await verify_api_key(x_api_key="wrong-key")

    @pytest.mark.asyncio
    async def test_verify_api_key_empty_string_env_var(self, monkeypatch):
        """Test that empty string env var is treated as empty string (not default)."""
        # Set env var to empty string explicitly
        monkeypatch.setenv(API_KEY_ENV, "")

        # Empty string env var means expected_key is "", not default
        # So DEFAULT_API_KEY should fail
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key=DEFAULT_API_KEY)
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

        # Empty string should also fail (because x_api_key cannot be empty)
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="")
        assert "API key required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_api_key_env_var_changes_dynamically(self, monkeypatch):
        """Test that environment variable changes are reflected."""
        key1 = "key-1-12345"
        key2 = "key-2-67890"

        # Set first key
        monkeypatch.setenv(API_KEY_ENV, key1)
        result1 = await verify_api_key(x_api_key=key1)
        assert result1 == key1

        # Change to second key
        monkeypatch.setenv(API_KEY_ENV, key2)
        result2 = await verify_api_key(x_api_key=key2)
        assert result2 == key2

        # First key should now fail
        with pytest.raises(HTTPException):
            await verify_api_key(x_api_key=key1)


class TestVerifyApiKeyEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_verify_api_key_very_long_key(self, monkeypatch):
        """Test verification with very long API key."""
        long_key = "a" * 1000
        monkeypatch.setenv(API_KEY_ENV, long_key)

        result = await verify_api_key(x_api_key=long_key)
        assert result == long_key

        # Wrong long key should fail
        wrong_long_key = "b" * 1000
        with pytest.raises(HTTPException):
            await verify_api_key(x_api_key=wrong_long_key)

    @pytest.mark.asyncio
    async def test_verify_api_key_special_characters(self, monkeypatch):
        """Test verification with special characters in key."""
        special_key = "test-key-!@#$%^&*()_+-=[]{}|;:,.<>?"
        monkeypatch.setenv(API_KEY_ENV, special_key)

        result = await verify_api_key(x_api_key=special_key)
        assert result == special_key

    @pytest.mark.asyncio
    async def test_verify_api_key_unicode_characters(self, monkeypatch):
        """Test verification with unicode characters in key."""
        unicode_key = "test-key-测试-🚀-αβγ"
        monkeypatch.setenv(API_KEY_ENV, unicode_key)

        result = await verify_api_key(x_api_key=unicode_key)
        assert result == unicode_key

    @pytest.mark.asyncio
    async def test_verify_api_key_single_character(self, monkeypatch):
        """Test verification with single character key."""
        single_char_key = "a"
        monkeypatch.setenv(API_KEY_ENV, single_char_key)

        result = await verify_api_key(x_api_key=single_char_key)
        assert result == single_char_key

    @pytest.mark.asyncio
    async def test_verify_api_key_numeric_key(self, monkeypatch):
        """Test verification with numeric key."""
        numeric_key = "1234567890"
        monkeypatch.setenv(API_KEY_ENV, numeric_key)

        result = await verify_api_key(x_api_key=numeric_key)
        assert result == numeric_key


class TestVerifyApiKeyConstants:
    """Test that constants are correctly defined."""

    def test_api_key_header_constant(self):
        """Test that API_KEY_HEADER constant is correct."""
        assert API_KEY_HEADER == "X-API-Key"

    def test_api_key_env_constant(self):
        """Test that API_KEY_ENV constant is correct."""
        assert API_KEY_ENV == "API_KEY"

    def test_default_api_key_constant(self):
        """Test that DEFAULT_API_KEY constant is defined."""
        assert DEFAULT_API_KEY == "1234567890"
        assert isinstance(DEFAULT_API_KEY, str)
        assert len(DEFAULT_API_KEY) > 0
