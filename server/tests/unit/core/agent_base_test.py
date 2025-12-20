"""Unit tests for the agent base module.

This module tests the AgentBase class and AgentConfig model, including
YAML configuration loading, validation, and property access.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from src.core.agent_base import AgentBase, AgentConfig


@pytest.fixture
def sample_config_data():
    """Create sample configuration data for testing."""
    return {
        "name": "test_agent",
        "description": "A test agent for unit testing",
        "category": "testing",
        "tools": [
            {"name": "tool1", "type": "function"},
            {"name": "tool2", "type": "query"},
        ],
        "metadata": {"version": "1.0", "author": "test"},
    }


@pytest.fixture
def sample_config_yaml(sample_config_data):
    """Create sample YAML configuration string."""
    return yaml.dump(sample_config_data)


@pytest.fixture
def temp_config_file(tmp_path, sample_config_yaml):
    """Create a temporary config file for testing."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(sample_config_yaml, encoding="utf-8")
    return config_file


class TestAgentConfig:
    """Test AgentConfig Pydantic model."""

    def test_agent_config_creation(self, sample_config_data):
        """Test creating AgentConfig with all fields."""
        config = AgentConfig(**sample_config_data)
        assert config.name == "test_agent"
        assert config.description == "A test agent for unit testing"
        assert config.category == "testing"
        assert len(config.tools) == 2
        assert config.metadata == {"version": "1.0", "author": "test"}

    def test_agent_config_minimal(self):
        """Test creating AgentConfig with only required fields."""
        config = AgentConfig(
            name="minimal_agent",
            description="Minimal agent",
            category="test",
        )
        assert config.name == "minimal_agent"
        assert config.description == "Minimal agent"
        assert config.category == "test"
        assert config.tools == []
        assert config.metadata == {}

    def test_agent_config_missing_required_field(self):
        """Test AgentConfig validation with missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(description="Missing name", category="test")
        assert "name" in str(exc_info.value).lower()

    def test_agent_config_defaults(self):
        """Test AgentConfig default values."""
        config = AgentConfig(
            name="test", description="test", category="test"
        )
        assert config.tools == []
        assert config.metadata == {}


class TestAgentBaseInitialization:
    """Test AgentBase initialization."""

    def test_init_with_config_path(self, temp_config_file, sample_config_data):
        """Test initialization with explicit config path."""
        agent = AgentBase(config_path=temp_config_file)
        assert agent.config_path == temp_config_file
        assert agent.config.name == sample_config_data["name"]
        assert agent.config.description == sample_config_data["description"]

    def test_init_with_nonexistent_file(self, tmp_path):
        """Test initialization with nonexistent config file."""
        nonexistent_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError) as exc_info:
            AgentBase(config_path=nonexistent_file)
        assert "not found" in str(exc_info.value).lower()

    def test_init_without_config_path_auto_discovery(
        self, temp_config_file, sample_config_data
    ):
        """Test initialization without config path (auto-discovery)."""
        # Patch __file__ to point to the directory containing our temp config file
        with patch("src.core.agent_base.__file__", str(temp_config_file.parent / "agent_base.py")):
            # The config.yaml should be in the same directory
            agent = AgentBase(config_path=None)
            assert agent.config.name == sample_config_data["name"]

    @patch("src.core.agent_base.Path")
    def test_init_without_config_path_not_found(self, mock_path):
        """Test initialization without config path when file doesn't exist."""
        # Mock Path(__file__) to return a path
        mock_file_path = MagicMock()
        mock_file_path.parent = Path("/nonexistent/path")
        mock_path.return_value = mock_file_path

        # Mock exists() to return False
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError) as exc_info:
                AgentBase(config_path=None)
            assert "config.yaml" in str(exc_info.value).lower()


class TestAgentBaseLoadConfig:
    """Test AgentBase._load_config() method."""

    def test_load_config_valid(self, temp_config_file, sample_config_data):
        """Test loading valid YAML configuration."""
        agent = AgentBase(config_path=temp_config_file)
        assert agent.config.name == sample_config_data["name"]
        assert agent.config.description == sample_config_data["description"]
        assert agent.config.category == sample_config_data["category"]

    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            AgentBase(config_path=empty_file)
        assert "empty" in str(exc_info.value).lower()

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML file."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            AgentBase(config_path=invalid_file)
        assert "invalid yaml" in str(exc_info.value).lower() or "yaml" in str(
            exc_info.value
        ).lower()

    def test_load_config_missing_required_field(self, tmp_path):
        """Test loading YAML with missing required field."""
        incomplete_file = tmp_path / "incomplete.yaml"
        incomplete_file.write_text(
            "description: test\ndescription: test", encoding="utf-8"
        )

        with pytest.raises(ValueError):
            AgentBase(config_path=incomplete_file)

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_load_config_file_read_error(self, mock_open_func, tmp_path):
        """Test handling file read errors."""
        config_file = tmp_path / "config.yaml"
        config_file.touch()

        with pytest.raises(ValueError) as exc_info:
            AgentBase(config_path=config_file)
        assert "error loading config" in str(exc_info.value).lower()


class TestAgentBaseProperties:
    """Test AgentBase property accessors."""

    def test_name_property(self, temp_config_file):
        """Test name property."""
        agent = AgentBase(config_path=temp_config_file)
        assert agent.name == "test_agent"

    def test_description_property(self, temp_config_file):
        """Test description property."""
        agent = AgentBase(config_path=temp_config_file)
        assert agent.description == "A test agent for unit testing"

    def test_category_property(self, temp_config_file):
        """Test category property."""
        agent = AgentBase(config_path=temp_config_file)
        assert agent.category == "testing"

    def test_tools_property(self, temp_config_file):
        """Test tools property."""
        agent = AgentBase(config_path=temp_config_file)
        assert len(agent.tools) == 2
        assert agent.tools[0]["name"] == "tool1"
        assert agent.tools[1]["name"] == "tool2"

    def test_metadata_property(self, temp_config_file):
        """Test metadata property."""
        agent = AgentBase(config_path=temp_config_file)
        assert agent.metadata["version"] == "1.0"
        assert agent.metadata["author"] == "test"


class TestAgentBaseMethods:
    """Test AgentBase methods."""

    def test_get_tool_names(self, temp_config_file):
        """Test get_tool_names() method."""
        agent = AgentBase(config_path=temp_config_file)
        tool_names = agent.get_tool_names()
        assert len(tool_names) == 2
        assert "tool1" in tool_names
        assert "tool2" in tool_names

    def test_get_tool_names_empty(self, tmp_path):
        """Test get_tool_names() with no tools."""
        config_data = {
            "name": "empty_agent",
            "description": "Agent with no tools",
            "category": "test",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        agent = AgentBase(config_path=config_file)
        tool_names = agent.get_tool_names()
        assert tool_names == []

    def test_get_tool_names_missing_name(self, tmp_path):
        """Test get_tool_names() with tools missing name field."""
        config_data = {
            "name": "agent",
            "description": "test",
            "category": "test",
            "tools": [{"type": "function"}, {"name": "named_tool"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        agent = AgentBase(config_path=config_file)
        tool_names = agent.get_tool_names()
        assert len(tool_names) == 1
        assert "named_tool" in tool_names

    def test_repr(self, temp_config_file):
        """Test __repr__ method."""
        agent = AgentBase(config_path=temp_config_file)
        repr_str = repr(agent)
        assert "AgentBase" in repr_str
        assert "test_agent" in repr_str
        assert "testing" in repr_str

