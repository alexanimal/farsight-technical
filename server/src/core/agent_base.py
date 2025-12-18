"""Base class for all agents in the system.

This module provides the AgentBase class that serves as the foundation for
all specialized agents. It handles loading and parsing YAML configuration
files and provides a common interface for agent operations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration model for an agent loaded from YAML."""

    name: str = Field(..., description="Unique identifier for the agent")
    description: str = Field(..., description="Human-readable description of what the agent does")
    category: str = Field(..., description="Category/type of agent (e.g., 'acquisition', 'orchestration')")
    tools: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of tools/capabilities this agent provides",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the agent",
    )


class AgentBase:
    """Base class for all agents in the system.

    This class provides the foundation for agent functionality, including
    loading configuration from YAML files and storing agent metadata.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the agent with configuration from a YAML file.

        Args:
            config_path: Path to the agent's config.yaml file. If None,
                attempts to find config.yaml in the agent's directory.
        """
        if config_path is None:
            # Try to find config.yaml in the same directory as the agent class
            agent_dir = Path(__file__).parent
            config_path = agent_dir / "config.yaml"
            if not config_path.exists():
                # If not found, try to infer from class name
                raise ValueError(
                    f"Could not find config.yaml. Please provide config_path or "
                    f"ensure config.yaml exists in {agent_dir}"
                )

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config = self._load_config()

    def _load_config(self) -> AgentConfig:
        """Load and parse the agent's YAML configuration file.

        Returns:
            AgentConfig: Parsed configuration object

        Raises:
            ValueError: If the config file is invalid or missing required fields
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                raise ValueError(f"Config file {self.config_path} is empty")

            return AgentConfig(**config_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {self.config_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading config from {self.config_path}: {e}") from e

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get the agent's description."""
        return self.config.description

    @property
    def category(self) -> str:
        """Get the agent's category."""
        return self.config.category

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Get the list of tools this agent provides."""
        return self.config.tools

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the agent's metadata."""
        return self.config.metadata

    def get_tool_names(self) -> List[str]:
        """Get a list of tool names this agent provides.

        Returns:
            List of tool names (extracted from the 'name' field of each tool)
        """
        return [tool.get("name", "") for tool in self.tools if tool.get("name")]

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"
