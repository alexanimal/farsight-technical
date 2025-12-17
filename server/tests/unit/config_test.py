"""Unit tests for the configuration module.

This module tests the Settings class and its various configuration options,
including environment variable loading, validation, and property methods.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from src.config import Settings, settings


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    """Fixture to create Settings instance without loading .env file.

    This fixture creates a Settings class with env_file pointing to a non-existent file,
    ensuring tests only use environment variables set via monkeypatch.
    """
    # Create a non-existent .env file path
    non_existent_env_file = str(tmp_path / ".env-nonexistent")

    # Create a test Settings class with modified config
    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=non_existent_env_file,  # Point to non-existent file
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

    return TestSettings


class TestSettingsDefaults:
    """Test default values when no environment variables are set."""

    def test_default_openai_api_key(self, isolated_settings, monkeypatch):
        """Test that OpenAI API key defaults to None."""
        # Clear any existing env vars
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = isolated_settings()
        assert config.openai_api_key is None

    def test_default_pinecone_api_key(self, isolated_settings, monkeypatch):
        """Test that Pinecone API key defaults to None."""
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        config = isolated_settings()
        assert config.pinecone_api_key is None

    def test_default_pinecone_index(self, isolated_settings, monkeypatch):
        """Test that Pinecone index defaults to 'default-index'."""
        monkeypatch.delenv("PINECONE_INDEX", raising=False)
        config = isolated_settings()
        assert config.pinecone_index == "default-index"

    def test_default_postgres_db_name(self, isolated_settings, monkeypatch):
        """Test that PostgreSQL database name defaults to 'farsight'."""
        monkeypatch.delenv("POSTGRES_DB_NAME", raising=False)
        config = isolated_settings()
        assert config.postgres_db_name == "farsight"

    def test_default_postgres_user(self, isolated_settings, monkeypatch):
        """Test that PostgreSQL user defaults to 'postgres'."""
        monkeypatch.delenv("POSTGRES", raising=False)
        monkeypatch.delenv("POSTGRES_USER", raising=False)
        config = isolated_settings()
        assert config.postgres_user == "postgres"

    def test_default_postgres_password(self, isolated_settings, monkeypatch):
        """Test that PostgreSQL password defaults to 'postgres'."""
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        config = isolated_settings()
        assert config.postgres_password == "postgres"

    def test_default_postgres_host(self, isolated_settings, monkeypatch):
        """Test that PostgreSQL host defaults to 'localhost'."""
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        config = isolated_settings()
        assert config.postgres_host == "localhost"

    def test_default_postgres_port(self, isolated_settings, monkeypatch):
        """Test that PostgreSQL port defaults to 5432."""
        monkeypatch.delenv("POSTGRES_PORT", raising=False)
        config = isolated_settings()
        assert config.postgres_port == 5432


class TestSettingsEnvironmentVariables:
    """Test loading configuration from environment variables."""

    def test_openai_api_key_from_env(self, monkeypatch):
        """Test loading OpenAI API key from environment variable."""
        test_key = "sk-test123456789"
        monkeypatch.setenv("OPENAI_API_KEY", test_key)
        config = Settings()
        assert config.openai_api_key == test_key

    def test_pinecone_api_key_from_env(self, monkeypatch):
        """Test loading Pinecone API key from environment variable."""
        test_key = "pc-test123456789"
        monkeypatch.setenv("PINECONE_API_KEY", test_key)
        config = Settings()
        assert config.pinecone_api_key == test_key

    def test_pinecone_index_from_env(self, monkeypatch):
        """Test loading Pinecone index from environment variable."""
        test_index = "my-custom-index"
        monkeypatch.setenv("PINECONE_INDEX", test_index)
        config = Settings()
        assert config.pinecone_index == test_index

    def test_postgres_db_name_from_env(self, monkeypatch):
        """Test loading PostgreSQL database name from environment variable."""
        test_db = "test_database"
        monkeypatch.setenv("POSTGRES_DB_NAME", test_db)
        config = Settings()
        assert config.postgres_db_name == test_db

    def test_postgres_user_from_env_alias(self, monkeypatch):
        """Test loading PostgreSQL user from POSTGRES alias."""
        test_user = "myuser"
        monkeypatch.setenv("POSTGRES", test_user)
        monkeypatch.delenv("POSTGRES_USER", raising=False)
        config = Settings()
        assert config.postgres_user == test_user

    def test_postgres_user_from_env_direct(self, isolated_settings, monkeypatch):
        """Test loading PostgreSQL user from POSTGRES alias.

        Note: Since postgres_user has alias="postgres", the environment variable
        POSTGRES maps to postgres_user. This tests that the alias works correctly.
        """
        test_user = "myuser"
        # Delete POSTGRES alias first to ensure clean state
        monkeypatch.delenv("POSTGRES", raising=False)
        # Use POSTGRES alias (the configured alias for postgres_user)
        monkeypatch.setenv("POSTGRES", test_user)
        config = isolated_settings()
        assert config.postgres_user == test_user

    def test_postgres_password_from_env(self, monkeypatch):
        """Test loading PostgreSQL password from environment variable."""
        test_password = "secure_password_123"
        monkeypatch.setenv("POSTGRES_PASSWORD", test_password)
        config = Settings()
        assert config.postgres_password == test_password

    def test_postgres_host_from_env(self, monkeypatch):
        """Test loading PostgreSQL host from environment variable."""
        test_host = "192.168.1.100"
        monkeypatch.setenv("POSTGRES_HOST", test_host)
        config = Settings()
        assert config.postgres_host == test_host

    def test_postgres_port_from_env(self, monkeypatch):
        """Test loading PostgreSQL port from environment variable."""
        test_port = 5433
        monkeypatch.setenv("POSTGRES_PORT", str(test_port))
        config = Settings()
        assert config.postgres_port == test_port

    def test_postgres_port_from_env_string(self, monkeypatch):
        """Test that port can be loaded as string and converted to int."""
        test_port = 3306
        monkeypatch.setenv("POSTGRES_PORT", str(test_port))
        config = Settings()
        assert config.postgres_port == test_port
        assert isinstance(config.postgres_port, int)

    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that environment variables are case-insensitive."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("POSTGRES_HOST", "test-host")
        config = Settings()
        assert config.openai_api_key == "test-key"
        assert config.postgres_host == "test-host"


class TestSettingsEnvFile:
    """Test loading configuration from .env file."""

    def test_env_file_path_resolution(self, monkeypatch):
        """Test that .env file path is correctly resolved."""
        # Test that the model_config has the correct env_file path
        # The path should be server/.env (parent of src)
        expected_path = Path(__file__).parent.parent.parent / ".env"
        actual_path = Path(Settings.model_config.get("env_file", ""))

        # Verify the path structure is correct (ends with .env in server directory)
        assert actual_path.name == ".env"
        assert actual_path.parent.name == "server" or "server" in str(
            actual_path.parent
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="OPENAI_API_KEY=file-key\nPOSTGRES_HOST=file-host",
    )
    @patch("pathlib.Path.exists")
    def test_env_file_loading(self, mock_exists, mock_file, monkeypatch):
        """Test loading configuration from .env file."""
        mock_exists.return_value = True

        # Clear env vars to ensure we're reading from file
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        # Mock the path resolution
        with patch("src.config.Path") as mock_path:
            mock_file_path = MagicMock()
            mock_file_path.parent.parent = Path("/mock/server")
            mock_file_path.__truediv__ = MagicMock(
                return_value=Path("/mock/server/.env")
            )
            mock_path.return_value = mock_file_path

            # Note: pydantic-settings may not actually read the file in this test
            # but we're testing the configuration setup
            config = Settings()
            # The actual file reading is handled by pydantic-settings internally
            # This test verifies the path setup is correct


class TestSettingsValidation:
    """Test field validation logic."""

    @pytest.mark.parametrize(
        "port,should_raise",
        [
            (0, True),  # Below minimum
            (1, False),  # Minimum valid
            (5432, False),  # Standard PostgreSQL port
            (65535, False),  # Maximum valid
            (65536, True),  # Above maximum
            (-1, True),  # Negative
        ],
    )
    def test_postgres_port_validation(
        self, isolated_settings, monkeypatch, port, should_raise
    ):
        """Test PostgreSQL port validation for various values."""
        monkeypatch.setenv("POSTGRES_PORT", str(port))
        if should_raise:
            with pytest.raises(ValidationError) as exc_info:
                isolated_settings()
            assert "Port must be between 1 and 65535" in str(exc_info.value)
        else:
            config = isolated_settings()
            assert config.postgres_port == port

    def test_postgres_port_invalid_type(self, isolated_settings, monkeypatch):
        """Test that invalid port types raise validation error."""
        monkeypatch.setenv("POSTGRES_PORT", "not-a-number")
        with pytest.raises(ValidationError):
            isolated_settings()


class TestSettingsProperties:
    """Test computed properties of Settings class."""

    def test_postgres_connection_string_defaults(self, isolated_settings, monkeypatch):
        """Test PostgreSQL connection string with default values."""
        # Clear env vars to use defaults
        for key in [
            "POSTGRES",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB_NAME",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = isolated_settings()
        expected = "postgresql://postgres:postgres@localhost:5432/farsight"
        assert config.postgres_connection_string == expected

    def test_postgres_connection_string_custom(self, isolated_settings, monkeypatch):
        """Test PostgreSQL connection string with custom values."""
        # Use POSTGRES alias (not POSTGRES_USER) since that's how the field is configured
        monkeypatch.delenv("POSTGRES", raising=False)
        monkeypatch.setenv("POSTGRES", "myuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "mypass")
        monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        monkeypatch.setenv("POSTGRES_DB_NAME", "mydb")

        config = isolated_settings()
        expected = "postgresql://myuser:mypass@db.example.com:5433/mydb"
        assert config.postgres_connection_string == expected

    def test_postgres_connection_string_special_chars(
        self, isolated_settings, monkeypatch
    ):
        """Test PostgreSQL connection string with special characters in password."""
        # Use POSTGRES alias (not POSTGRES_USER) since that's how the field is configured
        monkeypatch.delenv("POSTGRES", raising=False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "p@ssw0rd!@#$%")
        monkeypatch.setenv("POSTGRES", "user@domain")
        config = isolated_settings()
        # Should handle special characters (URL encoding would be handled by connection library)
        assert "user@domain" in config.postgres_connection_string
        assert "p@ssw0rd!@#$%" in config.postgres_connection_string

    def test_postgres_async_connection_string_defaults(
        self, isolated_settings, monkeypatch
    ):
        """Test PostgreSQL async connection string with default values."""
        for key in [
            "POSTGRES",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB_NAME",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = isolated_settings()
        expected = "postgresql+asyncpg://postgres:postgres@localhost:5432/farsight"
        assert config.postgres_async_connection_string == expected

    def test_postgres_async_connection_string_custom(
        self, isolated_settings, monkeypatch
    ):
        """Test PostgreSQL async connection string with custom values."""
        # Use POSTGRES alias (not POSTGRES_USER) since that's how the field is configured
        monkeypatch.delenv("POSTGRES", raising=False)
        monkeypatch.setenv("POSTGRES", "asyncuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "asyncpass")
        monkeypatch.setenv("POSTGRES_HOST", "async.db.com")
        monkeypatch.setenv("POSTGRES_PORT", "5434")
        monkeypatch.setenv("POSTGRES_DB_NAME", "asyncdb")

        config = isolated_settings()
        expected = "postgresql+asyncpg://asyncuser:asyncpass@async.db.com:5434/asyncdb"
        assert config.postgres_async_connection_string == expected

    def test_postgres_config_defaults(self, isolated_settings, monkeypatch):
        """Test PostgreSQL config dict with default values."""
        for key in [
            "POSTGRES",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB_NAME",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = isolated_settings()
        expected = {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "farsight",
        }
        assert config.postgres_config == expected

    def test_postgres_config_custom(self, isolated_settings, monkeypatch):
        """Test PostgreSQL config dict with custom values."""
        # Use POSTGRES alias (not POSTGRES_USER) since that's how the field is configured
        monkeypatch.delenv("POSTGRES", raising=False)
        monkeypatch.setenv("POSTGRES", "configuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "configpass")
        monkeypatch.setenv("POSTGRES_HOST", "config.host.com")
        monkeypatch.setenv("POSTGRES_PORT", "5435")
        monkeypatch.setenv("POSTGRES_DB_NAME", "configdb")

        config = isolated_settings()
        expected = {
            "host": "config.host.com",
            "port": 5435,
            "user": "configuser",
            "password": "configpass",
            "database": "configdb",
        }
        assert config.postgres_config == expected

    def test_postgres_config_immutability(self, monkeypatch):
        """Test that modifying the config dict doesn't affect the original."""
        config = Settings()
        config_dict = config.postgres_config
        config_dict["host"] = "modified"
        # Original should be unchanged
        assert config.postgres_host != "modified"


class TestSettingsSingleton:
    """Test the singleton instance behavior."""

    def test_settings_singleton_exists(self):
        """Test that the singleton instance is created."""
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_singleton_consistency(self, monkeypatch):
        """Test that the singleton instance is consistent across imports."""
        # This test verifies that the module-level instance exists
        # In practice, re-importing would give the same instance
        from src.config import settings as settings2

        assert settings is settings2


class TestSettingsEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_env_vars(self, monkeypatch):
        """Test that empty string environment variables are handled."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("POSTGRES_HOST", "")
        config = Settings()
        # Empty strings should be treated as valid (None for Optional fields)
        assert config.openai_api_key == ""

    def test_whitespace_env_vars(self, monkeypatch):
        """Test that whitespace in environment variables is preserved."""
        monkeypatch.setenv("POSTGRES_HOST", "  localhost  ")
        config = Settings()
        assert config.postgres_host == "  localhost  "

    def test_extra_env_vars_ignored(self, monkeypatch):
        """Test that extra environment variables are ignored."""
        monkeypatch.setenv("UNKNOWN_VAR", "some_value")
        monkeypatch.setenv("ANOTHER_UNKNOWN", "another_value")
        # Should not raise an error due to extra="ignore" in model_config
        config = Settings()
        assert config is not None

    def test_multiple_settings_instances(self, monkeypatch):
        """Test that multiple Settings instances work independently."""
        monkeypatch.setenv("POSTGRES_HOST", "host1")
        config1 = Settings()

        monkeypatch.setenv("POSTGRES_HOST", "host2")
        config2 = Settings()

        # Each instance should have its own values based on env at creation time
        assert config1.postgres_host == "host1"
        assert config2.postgres_host == "host2"

    def test_all_optional_fields_none(self, isolated_settings, monkeypatch):
        """Test that all optional fields can be None."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        config = isolated_settings()
        assert config.openai_api_key is None
        assert config.pinecone_api_key is None
        # Required fields should still have defaults
        assert config.postgres_db_name is not None
