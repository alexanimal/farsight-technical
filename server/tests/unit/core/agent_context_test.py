"""Unit tests for the agent context module.

This module tests the AgentContext class and its various methods,
including data manipulation, metadata management, and history tracking.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.core.agent_context import AgentContext


@pytest.fixture
def sample_context_data():
    """Create sample context data for testing."""
    return {
        "query": "What are the latest funding rounds?",
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "metadata": {"source": "api", "priority": "high"},
        "shared_data": {"org_id": "org-789"},
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    }


class TestAgentContextCreation:
    """Test AgentContext creation."""

    def test_create_with_all_fields(self, sample_context_data):
        """Test creating AgentContext with all fields."""
        context = AgentContext(**sample_context_data)
        assert context.query == "What are the latest funding rounds?"
        assert context.conversation_id == "conv-123"
        assert context.user_id == "user-456"
        assert context.metadata == {"source": "api", "priority": "high"}
        assert context.shared_data == {"org_id": "org-789"}
        assert len(context.conversation_history) == 2

    def test_create_with_minimal_fields(self):
        """Test creating AgentContext with only required fields."""
        context = AgentContext(query="Test query")
        assert context.query == "Test query"
        assert context.conversation_id is None
        assert context.user_id is None
        assert context.metadata == {}
        assert context.shared_data == {}
        assert context.conversation_history is None
        assert isinstance(context.timestamp, datetime)

    def test_create_with_default_timestamp(self):
        """Test that timestamp is automatically set."""
        with patch("src.core.agent_context.datetime") as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            context = AgentContext(query="test")
            assert context.timestamp == mock_now

    def test_create_missing_required_field(self):
        """Test AgentContext validation with missing required field."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AgentContext()


class TestAgentContextSharedData:
    """Test AgentContext shared data methods."""

    def test_add_to_shared_data(self):
        """Test adding data to shared_data."""
        context = AgentContext(query="test")
        context.add_to_shared_data("key1", "value1")
        assert context.shared_data["key1"] == "value1"

    def test_add_to_shared_data_overwrite(self):
        """Test overwriting existing shared_data value."""
        context = AgentContext(query="test")
        context.add_to_shared_data("key1", "value1")
        context.add_to_shared_data("key1", "value2")
        assert context.shared_data["key1"] == "value2"

    def test_get_from_shared_data_existing(self):
        """Test retrieving existing shared_data value."""
        context = AgentContext(query="test")
        context.add_to_shared_data("key1", "value1")
        result = context.get_from_shared_data("key1")
        assert result == "value1"

    def test_get_from_shared_data_missing(self):
        """Test retrieving missing shared_data value with default."""
        context = AgentContext(query="test")
        result = context.get_from_shared_data("missing_key")
        assert result is None

    def test_get_from_shared_data_with_default(self):
        """Test retrieving missing shared_data value with custom default."""
        context = AgentContext(query="test")
        result = context.get_from_shared_data("missing_key", default="default_value")
        assert result == "default_value"

    def test_multiple_shared_data_items(self):
        """Test managing multiple shared_data items."""
        context = AgentContext(query="test")
        context.add_to_shared_data("key1", "value1")
        context.add_to_shared_data("key2", "value2")
        context.add_to_shared_data("key3", {"nested": "data"})
        assert len(context.shared_data) == 3
        assert context.get_from_shared_data("key3") == {"nested": "data"}


class TestAgentContextMetadata:
    """Test AgentContext metadata methods."""

    def test_add_metadata(self):
        """Test adding metadata."""
        context = AgentContext(query="test")
        context.add_metadata("source", "api")
        assert context.metadata["source"] == "api"

    def test_add_metadata_overwrite(self):
        """Test overwriting existing metadata value."""
        context = AgentContext(query="test")
        context.add_metadata("source", "api")
        context.add_metadata("source", "webhook")
        assert context.metadata["source"] == "webhook"

    def test_get_metadata_existing(self):
        """Test retrieving existing metadata value."""
        context = AgentContext(query="test")
        context.add_metadata("priority", "high")
        result = context.get_metadata("priority")
        assert result == "high"

    def test_get_metadata_missing(self):
        """Test retrieving missing metadata value with default."""
        context = AgentContext(query="test")
        result = context.get_metadata("missing_key")
        assert result is None

    def test_get_metadata_with_default(self):
        """Test retrieving missing metadata value with custom default."""
        context = AgentContext(query="test")
        result = context.get_metadata("missing_key", default="default_value")
        assert result == "default_value"

    def test_metadata_from_initialization(self, sample_context_data):
        """Test metadata set during initialization."""
        context = AgentContext(**sample_context_data)
        assert context.get_metadata("source") == "api"
        assert context.get_metadata("priority") == "high"


class TestAgentContextHistory:
    """Test AgentContext conversation history methods."""

    def test_add_to_history_first_message(self):
        """Test adding first message to history."""
        context = AgentContext(query="test")
        context.add_to_history("user", "Hello")
        assert context.conversation_history is not None
        assert len(context.conversation_history) == 1
        assert context.conversation_history[0]["role"] == "user"
        assert context.conversation_history[0]["content"] == "Hello"

    def test_add_to_history_multiple_messages(self):
        """Test adding multiple messages to history."""
        context = AgentContext(query="test")
        context.add_to_history("user", "Hello")
        context.add_to_history("assistant", "Hi there!")
        context.add_to_history("user", "How are you?")
        assert len(context.conversation_history) == 3
        assert context.conversation_history[1]["role"] == "assistant"
        assert context.conversation_history[1]["content"] == "Hi there!"

    def test_add_to_history_with_existing_history(self, sample_context_data):
        """Test adding to existing conversation history."""
        context = AgentContext(**sample_context_data)
        initial_count = len(context.conversation_history)
        context.add_to_history("user", "New message")
        assert len(context.conversation_history) == initial_count + 1
        assert context.conversation_history[-1]["content"] == "New message"

    def test_add_to_history_different_roles(self):
        """Test adding messages with different roles."""
        context = AgentContext(query="test")
        context.add_to_history("user", "User message")
        context.add_to_history("assistant", "Assistant message")
        context.add_to_history("system", "System message")
        assert len(context.conversation_history) == 3
        roles = [msg["role"] for msg in context.conversation_history]
        assert "user" in roles
        assert "assistant" in roles
        assert "system" in roles


class TestAgentContextModelDump:
    """Test AgentContext model_dump() method."""

    def test_model_dump_basic(self):
        """Test basic model_dump() functionality."""
        context = AgentContext(query="test query")
        data = context.model_dump()
        assert data["query"] == "test query"
        assert data["conversation_id"] is None
        assert isinstance(data["timestamp"], str)  # Should be ISO format string

    def test_model_dump_timestamp_serialization(self):
        """Test that timestamp is serialized as ISO format string."""
        # Create context and verify timestamp is datetime object
        context = AgentContext(query="test")
        assert isinstance(context.timestamp, datetime)
        original_timestamp = context.timestamp
        
        # Call model_dump and verify timestamp is serialized to ISO string
        data = context.model_dump()
        assert isinstance(data["timestamp"], str)
        assert data["timestamp"] == original_timestamp.isoformat()
        
        # Verify internal timestamp is still a datetime object
        assert isinstance(context.timestamp, datetime)
        assert context.timestamp == original_timestamp

    def test_model_dump_with_all_fields(self, sample_context_data):
        """Test model_dump() with all fields populated."""
        context = AgentContext(**sample_context_data)
        data = context.model_dump()
        assert data["query"] == sample_context_data["query"]
        assert data["conversation_id"] == sample_context_data["conversation_id"]
        assert data["user_id"] == sample_context_data["user_id"]
        assert data["metadata"] == sample_context_data["metadata"]
        assert data["shared_data"] == sample_context_data["shared_data"]
        assert data["conversation_history"] == sample_context_data[
            "conversation_history"
        ]
        assert isinstance(data["timestamp"], str)

    def test_model_dump_preserves_datetime_object_internally(self):
        """Test that model_dump doesn't modify internal timestamp."""
        context = AgentContext(query="test")
        original_timestamp = context.timestamp
        data = context.model_dump()
        # Internal timestamp should still be datetime object
        assert isinstance(context.timestamp, datetime)
        assert context.timestamp == original_timestamp
        # But serialized version should be string
        assert isinstance(data["timestamp"], str)

