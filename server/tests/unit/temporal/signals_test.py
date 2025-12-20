"""Unit tests for the Temporal signals module.

This module tests all signal schemas and models in signals.py, including
validation, serialization, edge cases, and error handling.
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.temporal.signals import (
    SIGNAL_CANCELLATION,
    SIGNAL_CONFIGURATION_CHANGE,
    SIGNAL_STATUS_UPDATE,
    SIGNAL_USER_INPUT,
    CancellationSignal,
    ConfigurationChangeSignal,
    SignalType,
    StatusUpdateSignal,
    UserInputSignal,
)


class TestSignalType:
    """Test SignalType enum."""

    def test_enum_values(self):
        """Test that all enum values are defined correctly."""
        assert SignalType.CANCELLATION == "cancellation"
        assert SignalType.USER_INPUT == "user_input"
        assert SignalType.STATUS_UPDATE == "status_update"
        assert SignalType.CONFIGURATION_CHANGE == "configuration_change"

    def test_enum_string_representation(self):
        """Test that enum values are strings."""
        assert isinstance(SignalType.CANCELLATION, str)
        assert isinstance(SignalType.USER_INPUT, str)

    def test_enum_membership(self):
        """Test that enum values can be checked for membership."""
        assert "cancellation" in SignalType.__members__.values()
        assert "user_input" in SignalType.__members__.values()
        assert "invalid" not in SignalType.__members__.values()


class TestCancellationSignal:
    """Test CancellationSignal model."""

    def test_create_minimal(self):
        """Test creating with no fields (all optional with defaults)."""
        signal = CancellationSignal()
        assert signal.reason is None
        assert signal.requested_by is None
        assert isinstance(signal.timestamp, datetime)

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = CancellationSignal(
            reason="User requested cancellation",
            requested_by="user-123",
            timestamp=custom_timestamp,
        )
        assert signal.reason == "User requested cancellation"
        assert signal.requested_by == "user-123"
        assert signal.timestamp == custom_timestamp

    def test_create_with_reason_only(self):
        """Test creating with only reason field."""
        signal = CancellationSignal(reason="Test cancellation")
        assert signal.reason == "Test cancellation"
        assert signal.requested_by is None
        assert isinstance(signal.timestamp, datetime)

    def test_create_with_requested_by_only(self):
        """Test creating with only requested_by field."""
        signal = CancellationSignal(requested_by="user-456")
        assert signal.reason is None
        assert signal.requested_by == "user-456"
        assert isinstance(signal.timestamp, datetime)

    def test_default_timestamp(self):
        """Test that timestamp gets a default value when not provided."""
        with patch("src.temporal.signals.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            signal = CancellationSignal()
            assert signal.timestamp == mock_now
            mock_datetime.utcnow.assert_called_once()

    def test_custom_timestamp(self):
        """Test that custom timestamp is used when provided."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = CancellationSignal(timestamp=custom_timestamp)
        assert signal.timestamp == custom_timestamp

    def test_all_fields_optional(self):
        """Test that all fields are optional."""
        signal = CancellationSignal()
        assert signal.reason is None
        assert signal.requested_by is None
        assert isinstance(signal.timestamp, datetime)

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = CancellationSignal(
            reason="Test reason",
            requested_by="user-123",
            timestamp=custom_timestamp,
        )
        data = signal.model_dump()
        assert isinstance(data, dict)
        assert data["reason"] == "Test reason"
        assert data["requested_by"] == "user-123"
        assert "timestamp" in data

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "reason": "Test reason",
            "requested_by": "user-123",
            "timestamp": "2024-01-01T12:00:00",
        }
        signal = CancellationSignal(**data)
        assert signal.reason == "Test reason"
        assert signal.requested_by == "user-123"


class TestUserInputSignal:
    """Test UserInputSignal model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        signal = UserInputSignal(input_text="Hello, world!")
        assert signal.input_text == "Hello, world!"
        assert signal.input_type is None
        assert signal.user_id is None
        assert signal.conversation_id is None
        assert signal.metadata == {}
        assert isinstance(signal.timestamp, datetime)

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = UserInputSignal(
            input_text="What is the weather?",
            input_type="question",
            user_id="user-123",
            conversation_id="conv-456",
            metadata={"source": "web"},
            timestamp=custom_timestamp,
        )
        assert signal.input_text == "What is the weather?"
        assert signal.input_type == "question"
        assert signal.user_id == "user-123"
        assert signal.conversation_id == "conv-456"
        assert signal.metadata == {"source": "web"}
        assert signal.timestamp == custom_timestamp

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        signal = UserInputSignal(input_text="test")
        assert signal.input_type is None
        assert signal.user_id is None
        assert signal.conversation_id is None
        assert signal.metadata == {}
        assert isinstance(signal.timestamp, datetime)

    def test_default_timestamp(self):
        """Test that timestamp gets a default value when not provided."""
        with patch("src.temporal.signals.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            signal = UserInputSignal(input_text="test")
            assert signal.timestamp == mock_now
            mock_datetime.utcnow.assert_called_once()

    @pytest.mark.parametrize(
        "field_name",
        ["input_text"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "input_text": "test input",
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            UserInputSignal(**kwargs)

    def test_empty_input_text(self):
        """Test that empty string is allowed for input_text."""
        signal = UserInputSignal(input_text="")
        assert signal.input_text == ""

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = UserInputSignal(
            input_text="test input",
            input_type="question",
            metadata={"key": "value"},
            timestamp=custom_timestamp,
        )
        data = signal.model_dump()
        assert isinstance(data, dict)
        assert data["input_text"] == "test input"
        assert data["input_type"] == "question"
        assert data["metadata"] == {"key": "value"}

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "input_text": "test input",
            "input_type": "question",
            "user_id": "user-123",
            "timestamp": "2024-01-01T12:00:00",
        }
        signal = UserInputSignal(**data)
        assert signal.input_text == "test input"
        assert signal.input_type == "question"
        assert signal.user_id == "user-123"


class TestStatusUpdateSignal:
    """Test StatusUpdateSignal model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        signal = StatusUpdateSignal(status="paused")
        assert signal.status == "paused"
        assert signal.status_code is None
        assert signal.message is None
        assert signal.source is None
        assert signal.metadata == {}
        assert isinstance(signal.timestamp, datetime)

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = StatusUpdateSignal(
            status="paused",
            status_code="PAUSED",
            message="Workflow paused by user",
            source="api",
            metadata={"reason": "user_request"},
            timestamp=custom_timestamp,
        )
        assert signal.status == "paused"
        assert signal.status_code == "PAUSED"
        assert signal.message == "Workflow paused by user"
        assert signal.source == "api"
        assert signal.metadata == {"reason": "user_request"}
        assert signal.timestamp == custom_timestamp

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        signal = StatusUpdateSignal(status="running")
        assert signal.status_code is None
        assert signal.message is None
        assert signal.source is None
        assert signal.metadata == {}
        assert isinstance(signal.timestamp, datetime)

    def test_default_timestamp(self):
        """Test that timestamp gets a default value when not provided."""
        with patch("src.temporal.signals.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            signal = StatusUpdateSignal(status="running")
            assert signal.timestamp == mock_now
            mock_datetime.utcnow.assert_called_once()

    @pytest.mark.parametrize(
        "field_name",
        ["status"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "status": "running",
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            StatusUpdateSignal(**kwargs)

    def test_empty_status(self):
        """Test that empty string is allowed for status."""
        signal = StatusUpdateSignal(status="")
        assert signal.status == ""

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = StatusUpdateSignal(
            status="paused",
            status_code="PAUSED",
            message="Test message",
            metadata={"key": "value"},
            timestamp=custom_timestamp,
        )
        data = signal.model_dump()
        assert isinstance(data, dict)
        assert data["status"] == "paused"
        assert data["status_code"] == "PAUSED"
        assert data["message"] == "Test message"
        assert data["metadata"] == {"key": "value"}

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "status": "paused",
            "status_code": "PAUSED",
            "message": "Test message",
            "source": "api",
            "timestamp": "2024-01-01T12:00:00",
        }
        signal = StatusUpdateSignal(**data)
        assert signal.status == "paused"
        assert signal.status_code == "PAUSED"
        assert signal.message == "Test message"
        assert signal.source == "api"


class TestConfigurationChangeSignal:
    """Test ConfigurationChangeSignal model."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        signal = ConfigurationChangeSignal(
            config_key="max_retries", config_value=5
        )
        assert signal.config_key == "max_retries"
        assert signal.config_value == 5
        assert signal.config_type is None
        assert signal.metadata == {}
        assert isinstance(signal.timestamp, datetime)

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = ConfigurationChangeSignal(
            config_key="timeout",
            config_value=300,
            config_type="execution",
            metadata={"source": "admin"},
            timestamp=custom_timestamp,
        )
        assert signal.config_key == "timeout"
        assert signal.config_value == 300
        assert signal.config_type == "execution"
        assert signal.metadata == {"source": "admin"}
        assert signal.timestamp == custom_timestamp

    def test_create_with_string_value(self):
        """Test creating with string config_value."""
        signal = ConfigurationChangeSignal(
            config_key="api_key", config_value="secret-key-123"
        )
        assert signal.config_key == "api_key"
        assert signal.config_value == "secret-key-123"

    def test_create_with_dict_value(self):
        """Test creating with dict config_value."""
        config_dict = {"key1": "value1", "key2": "value2"}
        signal = ConfigurationChangeSignal(
            config_key="settings", config_value=config_dict
        )
        assert signal.config_key == "settings"
        assert signal.config_value == config_dict

    def test_create_with_list_value(self):
        """Test creating with list config_value."""
        config_list = ["item1", "item2", "item3"]
        signal = ConfigurationChangeSignal(
            config_key="allowed_domains", config_value=config_list
        )
        assert signal.config_key == "allowed_domains"
        assert signal.config_value == config_list

    def test_create_with_none_value(self):
        """Test creating with None config_value."""
        signal = ConfigurationChangeSignal(config_key="optional_setting", config_value=None)
        assert signal.config_key == "optional_setting"
        assert signal.config_value is None

    def test_defaults(self):
        """Test that optional fields have correct defaults."""
        signal = ConfigurationChangeSignal(
            config_key="test_key", config_value="test_value"
        )
        assert signal.config_type is None
        assert signal.metadata == {}
        assert isinstance(signal.timestamp, datetime)

    def test_default_timestamp(self):
        """Test that timestamp gets a default value when not provided."""
        with patch("src.temporal.signals.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            signal = ConfigurationChangeSignal(
                config_key="test", config_value="value"
            )
            assert signal.timestamp == mock_now
            mock_datetime.utcnow.assert_called_once()

    @pytest.mark.parametrize(
        "field_name",
        ["config_key", "config_value"],
    )
    def test_required_fields(self, field_name):
        """Test that required fields cannot be missing."""
        kwargs = {
            "config_key": "test_key",
            "config_value": "test_value",
        }
        del kwargs[field_name]
        with pytest.raises(ValidationError):
            ConfigurationChangeSignal(**kwargs)

    def test_empty_config_key(self):
        """Test that empty string is allowed for config_key."""
        signal = ConfigurationChangeSignal(config_key="", config_value="value")
        assert signal.config_key == ""

    def test_serialization(self):
        """Test that model can be serialized to dict."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = ConfigurationChangeSignal(
            config_key="timeout",
            config_value=300,
            config_type="execution",
            metadata={"key": "value"},
            timestamp=custom_timestamp,
        )
        data = signal.model_dump()
        assert isinstance(data, dict)
        assert data["config_key"] == "timeout"
        assert data["config_value"] == 300
        assert data["config_type"] == "execution"
        assert data["metadata"] == {"key": "value"}

    def test_deserialization(self):
        """Test that model can be created from dict."""
        data = {
            "config_key": "timeout",
            "config_value": 300,
            "config_type": "execution",
            "metadata": {"key": "value"},
            "timestamp": "2024-01-01T12:00:00",
        }
        signal = ConfigurationChangeSignal(**data)
        assert signal.config_key == "timeout"
        assert signal.config_value == 300
        assert signal.config_type == "execution"
        assert signal.metadata == {"key": "value"}

    def test_deserialization_with_complex_value(self):
        """Test deserialization with complex config_value."""
        data = {
            "config_key": "settings",
            "config_value": {"nested": {"key": "value"}},
            "timestamp": "2024-01-01T12:00:00",
        }
        signal = ConfigurationChangeSignal(**data)
        assert signal.config_key == "settings"
        assert signal.config_value == {"nested": {"key": "value"}}


class TestSignalConstants:
    """Test signal name constants."""

    def test_signal_cancellation_constant(self):
        """Test SIGNAL_CANCELLATION constant."""
        assert SIGNAL_CANCELLATION == "cancellation"
        assert isinstance(SIGNAL_CANCELLATION, str)

    def test_signal_user_input_constant(self):
        """Test SIGNAL_USER_INPUT constant."""
        assert SIGNAL_USER_INPUT == "user_input"
        assert isinstance(SIGNAL_USER_INPUT, str)

    def test_signal_status_update_constant(self):
        """Test SIGNAL_STATUS_UPDATE constant."""
        assert SIGNAL_STATUS_UPDATE == "status_update"
        assert isinstance(SIGNAL_STATUS_UPDATE, str)

    def test_signal_configuration_change_constant(self):
        """Test SIGNAL_CONFIGURATION_CHANGE constant."""
        assert SIGNAL_CONFIGURATION_CHANGE == "configuration_change"
        assert isinstance(SIGNAL_CONFIGURATION_CHANGE, str)

    def test_all_constants_are_strings(self):
        """Test that all signal constants are strings."""
        assert isinstance(SIGNAL_CANCELLATION, str)
        assert isinstance(SIGNAL_USER_INPUT, str)
        assert isinstance(SIGNAL_STATUS_UPDATE, str)
        assert isinstance(SIGNAL_CONFIGURATION_CHANGE, str)

    def test_constants_match_enum_values(self):
        """Test that constants match SignalType enum values."""
        assert SIGNAL_CANCELLATION == SignalType.CANCELLATION
        assert SIGNAL_USER_INPUT == SignalType.USER_INPUT
        assert SIGNAL_STATUS_UPDATE == SignalType.STATUS_UPDATE
        assert SIGNAL_CONFIGURATION_CHANGE == SignalType.CONFIGURATION_CHANGE


class TestSignalModelsEdgeCases:
    """Test edge cases and special scenarios for signal models."""

    def test_cancellation_signal_with_empty_reason(self):
        """Test CancellationSignal with empty string reason."""
        signal = CancellationSignal(reason="")
        assert signal.reason == ""

    def test_user_input_signal_with_long_text(self):
        """Test UserInputSignal with very long input text."""
        long_text = "a" * 10000
        signal = UserInputSignal(input_text=long_text)
        assert len(signal.input_text) == 10000
        assert signal.input_text == long_text

    def test_status_update_signal_with_various_status_values(self):
        """Test StatusUpdateSignal with various status string values."""
        statuses = ["running", "paused", "completed", "failed", "cancelled"]
        for status in statuses:
            signal = StatusUpdateSignal(status=status)
            assert signal.status == status

    def test_configuration_change_signal_with_boolean_value(self):
        """Test ConfigurationChangeSignal with boolean config_value."""
        signal = ConfigurationChangeSignal(
            config_key="enabled", config_value=True
        )
        assert signal.config_value is True

    def test_configuration_change_signal_with_float_value(self):
        """Test ConfigurationChangeSignal with float config_value."""
        signal = ConfigurationChangeSignal(
            config_key="threshold", config_value=0.95
        )
        assert signal.config_value == 0.95

    def test_metadata_preservation(self):
        """Test that metadata dictionaries are preserved correctly."""
        complex_metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "string": "test",
        }
        signal = UserInputSignal(
            input_text="test", metadata=complex_metadata
        )
        assert signal.metadata == complex_metadata
        data = signal.model_dump()
        assert data["metadata"] == complex_metadata

    def test_timestamp_serialization(self):
        """Test that timestamp fields serialize correctly."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal = CancellationSignal(timestamp=custom_timestamp)
        data = signal.model_dump()
        assert "timestamp" in data
        # Pydantic serializes datetime to ISO format string
        assert isinstance(data["timestamp"], str) or isinstance(
            data["timestamp"], datetime
        )

    def test_multiple_signals_with_same_timestamp(self):
        """Test creating multiple signals with the same timestamp."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        signal1 = CancellationSignal(timestamp=custom_timestamp)
        signal2 = UserInputSignal(
            input_text="test", timestamp=custom_timestamp
        )
        assert signal1.timestamp == signal2.timestamp == custom_timestamp

    def test_user_input_signal_with_all_optional_fields(self):
        """Test UserInputSignal with all optional fields populated."""
        signal = UserInputSignal(
            input_text="test",
            input_type="clarification",
            user_id="user-123",
            conversation_id="conv-456",
            metadata={"key": "value"},
        )
        assert signal.input_text == "test"
        assert signal.input_type == "clarification"
        assert signal.user_id == "user-123"
        assert signal.conversation_id == "conv-456"
        assert signal.metadata == {"key": "value"}

    def test_status_update_signal_with_all_optional_fields(self):
        """Test StatusUpdateSignal with all optional fields populated."""
        signal = StatusUpdateSignal(
            status="paused",
            status_code="PAUSED",
            message="Workflow paused",
            source="admin",
            metadata={"reason": "maintenance"},
        )
        assert signal.status == "paused"
        assert signal.status_code == "PAUSED"
        assert signal.message == "Workflow paused"
        assert signal.source == "admin"
        assert signal.metadata == {"reason": "maintenance"}

    def test_configuration_change_signal_with_nested_dict_value(self):
        """Test ConfigurationChangeSignal with nested dict config_value."""
        nested_dict = {
            "level1": {
                "level2": {
                    "level3": "value",
                },
            },
        }
        signal = ConfigurationChangeSignal(
            config_key="nested_config", config_value=nested_dict
        )
        assert signal.config_value == nested_dict
        assert signal.config_value["level1"]["level2"]["level3"] == "value"

