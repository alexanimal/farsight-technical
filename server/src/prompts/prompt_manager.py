"""Prompt management system for agents.

This module provides a centralized prompt manager that applies organizational
standards and formatting to agent prompts. It supports temporal context,
personas, organizational defaults, and markdown formatting instructions.

The prompt manager is designed to be:
- Modular and testable
- Agent-agnostic (no business logic)
- Configurable and extensible
- Compatible with the system's separation of concerns
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PromptOptions(BaseModel):
    """Options for building prompts with the prompt manager.

    Attributes:
        add_temporal_context: Whether to add current date/time to the prompt.
        persona: Optional persona description to append as a section.
        add_markdown_instructions: Whether to add markdown formatting instructions.
        custom_sections: Optional dictionary of custom sections to add.
            Keys are section names, values are section content.
    """

    add_temporal_context: bool = Field(
        default=True, description="Add current date and time to the prompt"
    )
    persona: Optional[str] = Field(
        default=None, description="Persona description to append as a section"
    )
    add_markdown_instructions: bool = Field(
        default=False, description="Add markdown output formatting instructions"
    )
    custom_sections: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom sections to add to the prompt (section_name: content)",
    )


class PromptManager:
    """Centralized prompt manager for applying organizational standards.

    This class provides a unified interface for building prompts with:
    - Organizational defaults (always included)
    - Temporal context (optional)
    - Persona definitions (optional)
    - Markdown formatting instructions (optional)
    - Custom sections (optional)

    The manager is designed to be stateless and thread-safe, allowing
    agents to use it without coupling to execution or orchestration logic.

    Example:
        ```python
        manager = PromptManager()

        # Register an agent's system prompt
        manager.register_agent_prompt(
            agent_name="acquisition",
            system_prompt="You are an expert in M&A transactions..."
        )

        # Build a prompt with options
        full_prompt = manager.build_system_prompt(
            agent_name="acquisition",
            options=PromptOptions(
                add_temporal_context=True,
                persona="You are a senior financial analyst...",
                add_markdown_instructions=True
            )
        )
        ```
    """

    # Organizational defaults - always included in prompts
    ORGANIZATIONAL_DEFAULTS = """## Organizational Standards

You are part of the Farsight technical system. Follow these standards:

- **Accuracy**: Provide accurate, fact-based information. If uncertain, state your uncertainty.
- **Clarity**: Communicate clearly and concisely. Avoid unnecessary jargon.
- **Completeness**: Address all aspects of the query thoroughly.
- **Professionalism**: Maintain a professional, helpful tone.
- **Ethics**: Do not provide misleading information or make unfounded claims.
"""

    # Markdown formatting instructions template
    MARKDOWN_INSTRUCTIONS = """## Output Formatting

Format your responses using Markdown:

- Use **bold** for emphasis and important terms
- Use *italics* for subtle emphasis
- Use `code blocks` for technical terms, code, or identifiers
- Use lists (bulleted or numbered) for multiple items
- Use headers (##, ###) to structure longer responses
- Use tables when presenting structured data
- Use blockquotes (>) for important notes or warnings

Ensure your output is well-formatted and easy to read.
"""

    def __init__(self, organizational_defaults: Optional[str] = None):
        """Initialize the prompt manager.

        Args:
            organizational_defaults: Optional custom organizational defaults.
                If None, uses the default organizational standards.
        """
        self._agent_prompts: Dict[str, str] = {}
        self._organizational_defaults = organizational_defaults or self.ORGANIZATIONAL_DEFAULTS

    def register_agent_prompt(
        self, agent_name: str, system_prompt: str, overwrite: bool = False
    ) -> None:
        """Register a system prompt for an agent.

        Agents can register their prompts during initialization or at runtime.
        This allows the prompt manager to build complete prompts with
        organizational standards applied.

        Args:
            agent_name: Unique identifier for the agent (e.g., "acquisition", "orchestration").
            system_prompt: The agent's base system prompt.
            overwrite: If True, overwrites existing prompt. If False, raises error if exists.

        Raises:
            ValueError: If agent_name already registered and overwrite=False.
        """
        if agent_name in self._agent_prompts and not overwrite:
            raise ValueError(
                f"Agent '{agent_name}' already has a registered prompt. "
                "Set overwrite=True to replace it."
            )

        self._agent_prompts[agent_name] = system_prompt
        logger.debug(f"Registered prompt for agent: {agent_name}")

    def get_agent_prompt(self, agent_name: str) -> Optional[str]:
        """Get the registered prompt for an agent.

        Args:
            agent_name: The agent's unique identifier.

        Returns:
            The agent's system prompt if registered, None otherwise.
        """
        return self._agent_prompts.get(agent_name)

    def build_system_prompt(
        self,
        agent_name: Optional[str] = None,
        base_prompt: Optional[str] = None,
        options: Optional[PromptOptions] = None,
    ) -> str:
        """Build a complete system prompt with organizational standards applied.

        This method combines:
        1. Agent's base prompt (from registration or provided)
        2. Organizational defaults (always included)
        3. Temporal context (if requested)
        4. Persona (if provided)
        5. Markdown instructions (if requested)
        6. Custom sections (if provided)

        Args:
            agent_name: Name of the agent to use registered prompt.
                Ignored if base_prompt is provided.
            base_prompt: Direct prompt to use instead of registered one.
                Takes precedence over agent_name.
            options: PromptOptions instance with formatting options.
                If None, uses default options (no temporal context, no persona, etc.).

        Returns:
            Complete system prompt with all requested sections.

        Raises:
            ValueError: If neither agent_name nor base_prompt is provided,
                or if agent_name is provided but not registered.
        """
        if options is None:
            options = PromptOptions()

        # Get base prompt
        if base_prompt:
            prompt_parts = [base_prompt]
        elif agent_name:
            if agent_name not in self._agent_prompts:
                raise ValueError(
                    f"Agent '{agent_name}' not registered. "
                    "Register it first with register_agent_prompt() or provide base_prompt."
                )
            prompt_parts = [self._agent_prompts[agent_name]]
        else:
            raise ValueError("Either agent_name or base_prompt must be provided.")

        # Add organizational defaults (always included)
        prompt_parts.append("\n" + self._organizational_defaults)

        # Add temporal context if requested
        if options.add_temporal_context:
            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            prompt_parts.append(f"\n## Temporal Context\n\nCurrent date and time: {current_time}")

        # Add persona if provided
        if options.persona:
            prompt_parts.append(f"\n## Persona\n\n{options.persona}")

        # Add custom sections if provided
        if options.custom_sections:
            for section_name, section_content in options.custom_sections.items():
                prompt_parts.append(f"\n## {section_name}\n\n{section_content}")

        # Add markdown instructions if requested
        if options.add_markdown_instructions:
            prompt_parts.append("\n" + self.MARKDOWN_INSTRUCTIONS)

        # Combine all parts
        return "\n".join(prompt_parts)

    def build_user_prompt(
        self,
        user_query: str,
        options: Optional[PromptOptions] = None,
    ) -> str:
        """Build a user prompt with optional enhancements.

        This method can add temporal context or other enhancements to user prompts.
        Currently, it primarily serves as a placeholder for future enhancements
        and ensures consistency with the system prompt building pattern.

        Args:
            user_query: The user's query or message.
            options: PromptOptions instance. Currently only add_temporal_context
                affects user prompts.

        Returns:
            Enhanced user prompt.
        """
        if options is None:
            options = PromptOptions()

        prompt_parts = [user_query]

        # Add temporal context if requested (less common for user prompts, but available)
        if options.add_temporal_context:
            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            prompt_parts.append(f"\n\n[Context: Current time is {current_time}]")

        return "\n".join(prompt_parts)

    def update_organizational_defaults(self, new_defaults: str) -> None:
        """Update the organizational defaults used in all prompts.

        Args:
            new_defaults: New organizational standards text.
        """
        self._organizational_defaults = new_defaults
        logger.info("Organizational defaults updated")

    def get_organizational_defaults(self) -> str:
        """Get the current organizational defaults.

        Returns:
            The current organizational defaults text.
        """
        return self._organizational_defaults

    def list_registered_agents(self) -> list[str]:
        """Get a list of all registered agent names.

        Returns:
            List of agent names that have registered prompts.
        """
        return list(self._agent_prompts.keys())

    def clear_agent_prompt(self, agent_name: str) -> None:
        """Remove a registered agent prompt.

        Args:
            agent_name: The agent's unique identifier.

        Raises:
            KeyError: If agent_name is not registered.
        """
        if agent_name not in self._agent_prompts:
            raise KeyError(f"Agent '{agent_name}' is not registered.")
        del self._agent_prompts[agent_name]
        logger.debug(f"Cleared prompt for agent: {agent_name}")


# Singleton instance - import this in other modules
_prompt_manager_instance: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the singleton PromptManager instance.

    This function provides a global instance of the prompt manager,
    allowing agents to use it without passing it around.

    Returns:
        The singleton PromptManager instance.
    """
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance


def reset_prompt_manager() -> None:
    """Reset the singleton PromptManager instance.

    This is primarily useful for testing to ensure clean state.
    """
    global _prompt_manager_instance
    _prompt_manager_instance = None
