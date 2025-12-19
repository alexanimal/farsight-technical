"""Agent registry for discovering available agents without Temporal dependency.

This module provides agent discovery and registration functionality that can be
used by both agents (like orchestration) and Temporal activities. It has no
dependency on Temporal, ensuring agents remain isolated and testable.

According to the architecture plan:
- Agents must not import Temporal
- Agent discovery should be available to both agents and activities
- Agents should be runnable in isolation, unit tests, scripts, or batch jobs
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from src.core.agent_base import AgentBase

logger = logging.getLogger(__name__)

# Agent registry mapping agent names/categories to their classes and config paths
# Format: {agent_name: (agent_class, config_path)}
_AGENT_REGISTRY: Dict[str, Tuple[Type[AgentBase], Path]] = {}


def register_agent(name: str, agent_class: Type[AgentBase], config_path: Path) -> None:
    """Register an agent class in the agent registry.

    Args:
        name: The name of the agent (must match config file name).
        agent_class: The agent class (must inherit from AgentBase).
        config_path: Path to the agent's YAML config file.
    """
    if not issubclass(agent_class, AgentBase):
        raise ValueError(f"Agent {name} must inherit from AgentBase")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _AGENT_REGISTRY[name] = (agent_class, config_path)
    logger.info(f"Registered agent: {name} at {config_path}")


def get_agent(name: str) -> Optional[Tuple[Type[AgentBase], Path]]:
    """Get an agent class and config path by name.

    Args:
        name: The name of the agent.

    Returns:
        Tuple of (agent_class, config_path) if found, None otherwise.
    """
    return _AGENT_REGISTRY.get(name)


def list_available_agents() -> list[str]:
    """List all available agent names.

    Returns:
        List of available agent names.
    """
    return list(_AGENT_REGISTRY.keys())


def get_agent_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for an agent by name.

    This is useful for orchestration agents that need to know about
    available agents without instantiating them.

    Args:
        name: The name of the agent.

    Returns:
        Dictionary with agent metadata (name, description, category, etc.)
        or None if agent not found.
    """
    agent_info = get_agent(name)
    if agent_info is None:
        return None

    agent_class, config_path = agent_info

    # Try to load config file to get metadata
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return {
            "name": config_data.get("name", name),
            "description": config_data.get("description", ""),
            "category": config_data.get("category", "unknown"),
            "metadata": config_data.get("metadata", {}),
        }
    except Exception as e:
        logger.warning(f"Failed to load metadata for agent {name}: {e}")
        return {
            "name": name,
            "description": f"Agent for handling {name} queries",
            "category": "unknown",
            "metadata": {},
        }


def discover_agents() -> None:
    """Discover and register agents from the agents directory.

    This function attempts to auto-discover agents by looking in the
    src/agents directory structure. Agents should be in:
    src/agents/{category}/agent.py with configs in src/configs/agents/{category}_agent.yaml

    This function is defensive and will skip agents that can't be imported
    or don't have the expected structure.

    This function can be called multiple times safely (idempotent).
    """
    # Base paths
    # File is at: server/src/agents/registry.py
    # So src_base = server/src
    src_base = Path(__file__).parent.parent
    agents_base = src_base / "agents"
    configs_base = src_base / "configs" / "agents"

    # Known agent mappings (name -> config_filename)
    # Agents can be registered manually via register_agent() if auto-discovery fails
    agent_mappings = [
        ("acquisition", "acquisition_agent.yaml"),
        ("orchestration", "orchestration_agent.yaml"),
    ]

    for agent_name, config_filename in agent_mappings:
        # Skip if already registered
        if agent_name in _AGENT_REGISTRY:
            logger.debug(f"Agent {agent_name} already registered, skipping")
            continue

        try:
            # Try to import the agent module
            module_path = f"src.agents.{agent_name}.agent"
            try:
                agent_module = importlib.import_module(module_path)
            except (ImportError, ModuleNotFoundError) as e:
                logger.debug(
                    f"Could not import agent module {module_path}: {e}. "
                    "Agent may not be implemented yet."
                )
                continue

            # Try to find the agent class
            agent_class = None
            for attr_name in dir(agent_module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(agent_module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, AgentBase)
                    and attr != AgentBase
                ):
                    agent_class = attr
                    break

            if agent_class is None:
                logger.debug(
                    f"No AgentBase subclass found in {module_path}. "
                    "Agent may not be implemented yet."
                )
                continue

            # Check for config file
            config_path = configs_base / config_filename
            if not config_path.exists():
                logger.warning(
                    f"Config file not found for {agent_name}: {config_path}. "
                    "Agent will not be registered."
                )
                continue

            # Register the agent
            register_agent(agent_name, agent_class, config_path)

        except Exception as e:
            logger.warning(
                f"Failed to register agent {agent_name} during discovery: {e}"
            )


# Initialize agent registry on module load
try:
    discover_agents()
except Exception as e:
    logger.warning(f"Failed to auto-discover agents: {e}")

