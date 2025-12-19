"""Unit tests for the prompt manager module.

This module tests the PromptManager class and its various methods,
including prompt registration, building prompts with options, and
organizational standards application.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.prompts.prompt_manager import (
    PromptManager,
    PromptOptions,
    get_prompt_manager,
    reset_prompt_manager,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the prompt manager singleton before and after each test."""
    reset_prompt_manager()
    yield
    reset_prompt_manager()


@pytest.fixture
def prompt_manager():
    """Create a fresh PromptManager instance for testing."""
    return PromptManager()


@pytest.fixture
def sample_agent_prompt():
    """Create a sample agent system prompt for testing."""
    return "You are an expert in M&A transactions and company acquisitions."


@pytest.fixture
def custom_organizational_defaults():
    """Create custom organizational defaults for testing."""
    return "## Custom Standards\n\nFollow these custom rules."


class TestPromptManagerInitialization:
    """Test PromptManager initialization."""

    def test_default_initialization(self, prompt_manager):
        """Test that PromptManager initializes with default organizational defaults."""
        assert prompt_manager is not None
        assert prompt_manager._agent_prompts == {}
        assert "Organizational Standards" in prompt_manager._organizational_defaults
        assert "Farsight technical system" in prompt_manager._organizational_defaults

    def test_custom_organizational_defaults(
        self, custom_organizational_defaults
    ):
        """Test initialization with custom organizational defaults."""
        manager = PromptManager(
            organizational_defaults=custom_organizational_defaults
        )
        assert manager._organizational_defaults == custom_organizational_defaults

    def test_organizational_defaults_class_constant(self):
        """Test that ORGANIZATIONAL_DEFAULTS class constant exists."""
        assert hasattr(PromptManager, "ORGANIZATIONAL_DEFAULTS")
        assert isinstance(PromptManager.ORGANIZATIONAL_DEFAULTS, str)
        assert len(PromptManager.ORGANIZATIONAL_DEFAULTS) > 0

    def test_markdown_instructions_class_constant(self):
        """Test that MARKDOWN_INSTRUCTIONS class constant exists."""
        assert hasattr(PromptManager, "MARKDOWN_INSTRUCTIONS")
        assert isinstance(PromptManager.MARKDOWN_INSTRUCTIONS, str)
        assert "Markdown" in PromptManager.MARKDOWN_INSTRUCTIONS


class TestPromptManagerAgentRegistration:
    """Test agent prompt registration functionality."""

    def test_register_agent_prompt(self, prompt_manager, sample_agent_prompt):
        """Test registering an agent prompt."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        assert "acquisition" in prompt_manager._agent_prompts
        assert prompt_manager._agent_prompts["acquisition"] == sample_agent_prompt

    def test_register_multiple_agents(
        self, prompt_manager, sample_agent_prompt
    ):
        """Test registering multiple agent prompts."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        prompt_manager.register_agent_prompt(
            agent_name="orchestration",
            system_prompt="You are an orchestration agent.",
        )
        assert len(prompt_manager._agent_prompts) == 2
        assert "acquisition" in prompt_manager._agent_prompts
        assert "orchestration" in prompt_manager._agent_prompts

    def test_register_agent_prompt_overwrite_false(
        self, prompt_manager, sample_agent_prompt
    ):
        """Test that registering duplicate agent raises error when overwrite=False."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        with pytest.raises(ValueError) as exc_info:
            prompt_manager.register_agent_prompt(
                agent_name="acquisition", system_prompt="Different prompt"
            )
        assert "already has a registered prompt" in str(exc_info.value)

    def test_register_agent_prompt_overwrite_true(
        self, prompt_manager, sample_agent_prompt
    ):
        """Test that registering duplicate agent overwrites when overwrite=True."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        new_prompt = "Updated prompt"
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=new_prompt, overwrite=True
        )
        assert prompt_manager._agent_prompts["acquisition"] == new_prompt

    def test_get_agent_prompt_exists(
        self, prompt_manager, sample_agent_prompt
    ):
        """Test getting a registered agent prompt."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        result = prompt_manager.get_agent_prompt("acquisition")
        assert result == sample_agent_prompt

    def test_get_agent_prompt_not_exists(self, prompt_manager):
        """Test getting a non-existent agent prompt returns None."""
        result = prompt_manager.get_agent_prompt("nonexistent")
        assert result is None

    def test_clear_agent_prompt(self, prompt_manager, sample_agent_prompt):
        """Test clearing a registered agent prompt."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        prompt_manager.clear_agent_prompt("acquisition")
        assert "acquisition" not in prompt_manager._agent_prompts
        assert prompt_manager.get_agent_prompt("acquisition") is None

    def test_clear_agent_prompt_not_exists(self, prompt_manager):
        """Test clearing a non-existent agent prompt raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            prompt_manager.clear_agent_prompt("nonexistent")
        assert "not registered" in str(exc_info.value)

    def test_list_registered_agents(self, prompt_manager, sample_agent_prompt):
        """Test listing all registered agents."""
        assert prompt_manager.list_registered_agents() == []
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        prompt_manager.register_agent_prompt(
            agent_name="orchestration",
            system_prompt="Orchestration prompt",
        )
        agents = prompt_manager.list_registered_agents()
        assert len(agents) == 2
        assert "acquisition" in agents
        assert "orchestration" in agents


class TestPromptManagerBuildSystemPrompt:
    """Test building system prompts with various options."""

    def test_build_system_prompt_with_base_prompt(self, prompt_manager):
        """Test building a system prompt with a direct base prompt."""
        base_prompt = "You are a helpful assistant."
        result = prompt_manager.build_system_prompt(base_prompt=base_prompt)
        assert base_prompt in result
        assert "Organizational Standards" in result

    def test_build_system_prompt_with_registered_agent(
        self, prompt_manager, sample_agent_prompt
    ):
        """Test building a system prompt using a registered agent."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        result = prompt_manager.build_system_prompt(agent_name="acquisition")
        assert sample_agent_prompt in result
        assert "Organizational Standards" in result

    def test_build_system_prompt_base_prompt_takes_precedence(
        self, prompt_manager, sample_agent_prompt
    ):
        """Test that base_prompt takes precedence over agent_name."""
        prompt_manager.register_agent_prompt(
            agent_name="acquisition", system_prompt=sample_agent_prompt
        )
        base_prompt = "This is the base prompt."
        result = prompt_manager.build_system_prompt(
            agent_name="acquisition", base_prompt=base_prompt
        )
        assert base_prompt in result
        assert sample_agent_prompt not in result

    def test_build_system_prompt_no_agent_or_base(self, prompt_manager):
        """Test that building without agent_name or base_prompt raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            prompt_manager.build_system_prompt()
        assert "Either agent_name or base_prompt must be provided" in str(
            exc_info.value
        )

    def test_build_system_prompt_unregistered_agent(self, prompt_manager):
        """Test that building with unregistered agent_name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            prompt_manager.build_system_prompt(agent_name="nonexistent")
        assert "not registered" in str(exc_info.value)

    def test_build_system_prompt_always_includes_organizational_defaults(
        self, prompt_manager
    ):
        """Test that organizational defaults are always included."""
        result = prompt_manager.build_system_prompt(
            base_prompt="Test prompt"
        )
        assert "Organizational Standards" in result
        assert "Farsight technical system" in result

    def test_build_system_prompt_with_temporal_context(
        self, prompt_manager
    ):
        """Test building a system prompt with temporal context."""
        with patch("src.prompts.prompt_manager.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = (
                "2024-01-15 10:30:00 UTC"
            )
            options = PromptOptions(add_temporal_context=True)
            result = prompt_manager.build_system_prompt(
                base_prompt="Test prompt", options=options
            )
            assert "Temporal Context" in result
            assert "2024-01-15 10:30:00 UTC" in result

    def test_build_system_prompt_without_temporal_context(
        self, prompt_manager
    ):
        """Test that temporal context is not included when not requested."""
        options = PromptOptions(add_temporal_context=False)
        result = prompt_manager.build_system_prompt(
            base_prompt="Test prompt", options=options
        )
        assert "Temporal Context" not in result

    def test_build_system_prompt_with_persona(self, prompt_manager):
        """Test building a system prompt with persona."""
        persona = "You are a senior financial analyst with 20 years of experience."
        options = PromptOptions(persona=persona)
        result = prompt_manager.build_system_prompt(
            base_prompt="Test prompt", options=options
        )
        assert "Persona" in result
        assert persona in result

    def test_build_system_prompt_without_persona(self, prompt_manager):
        """Test that persona is not included when not provided."""
        options = PromptOptions(persona=None)
        result = prompt_manager.build_system_prompt(
            base_prompt="Test prompt", options=options
        )
        assert "Persona" not in result

    def test_build_system_prompt_with_markdown_instructions(
        self, prompt_manager
    ):
        """Test building a system prompt with markdown instructions."""
        options = PromptOptions(add_markdown_instructions=True)
        result = prompt_manager.build_system_prompt(
            base_prompt="Test prompt", options=options
        )
        assert "Output Formatting" in result
        assert "Markdown" in result
        assert "**bold**" in result

    def test_build_system_prompt_without_markdown_instructions(
        self, prompt_manager
    ):
        """Test that markdown instructions are not included when not requested."""
        options = PromptOptions(add_markdown_instructions=False)
        result = prompt_manager.build_system_prompt(
            base_prompt="Test prompt", options=options
        )
        assert "Output Formatting" not in result

    def test_build_system_prompt_with_custom_sections(self, prompt_manager):
        """Test building a system prompt with custom sections."""
        custom_sections = {
            "Additional Context": "This is additional context.",
            "Constraints": "Follow these constraints.",
        }
        options = PromptOptions(custom_sections=custom_sections)
        result = prompt_manager.build_system_prompt(
            base_prompt="Test prompt", options=options
        )
        assert "Additional Context" in result
        assert "This is additional context." in result
        assert "Constraints" in result
        assert "Follow these constraints." in result

    def test_build_system_prompt_with_all_options(self, prompt_manager):
        """Test building a system prompt with all options enabled."""
        with patch("src.prompts.prompt_manager.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = (
                "2024-01-15 10:30:00 UTC"
            )
            options = PromptOptions(
                add_temporal_context=True,
                persona="You are an expert.",
                add_markdown_instructions=True,
                custom_sections={"Custom": "Custom content"},
            )
            result = prompt_manager.build_system_prompt(
                base_prompt="Base prompt", options=options
            )
            # Check all sections are present
            assert "Base prompt" in result
            assert "Organizational Standards" in result
            assert "Temporal Context" in result
            assert "2024-01-15 10:30:00 UTC" in result
            assert "Persona" in result
            assert "You are an expert." in result
            assert "Custom" in result
            assert "Custom content" in result
            assert "Output Formatting" in result

    def test_build_system_prompt_default_options(self, prompt_manager):
        """Test that default options are used when None is provided."""
        result = prompt_manager.build_system_prompt(base_prompt="Test prompt")
        # Should include organizational defaults but not optional sections
        assert "Organizational Standards" in result
        assert "Temporal Context" not in result
        assert "Persona" not in result
        assert "Output Formatting" not in result

    def test_build_system_prompt_section_order(self, prompt_manager):
        """Test that sections appear in the correct order."""
        options = PromptOptions(
            add_temporal_context=True,
            persona="Test persona",
            add_markdown_instructions=True,
        )
        result = prompt_manager.build_system_prompt(
            base_prompt="Base", options=options
        )
        # Check order: base -> org defaults -> temporal -> persona -> markdown
        base_idx = result.find("Base")
        org_idx = result.find("Organizational Standards")
        temporal_idx = result.find("Temporal Context")
        persona_idx = result.find("Persona")
        markdown_idx = result.find("Output Formatting")
        assert base_idx < org_idx < temporal_idx < persona_idx < markdown_idx


class TestPromptManagerBuildUserPrompt:
    """Test building user prompts."""

    def test_build_user_prompt_basic(self, prompt_manager):
        """Test building a basic user prompt."""
        user_query = "What is the capital of France?"
        result = prompt_manager.build_user_prompt(user_query)
        assert result == user_query

    def test_build_user_prompt_with_temporal_context(self, prompt_manager):
        """Test building a user prompt with temporal context."""
        with patch("src.prompts.prompt_manager.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = (
                "2024-01-15 10:30:00 UTC"
            )
            options = PromptOptions(add_temporal_context=True)
            user_query = "What is the weather?"
            result = prompt_manager.build_user_prompt(user_query, options=options)
            assert user_query in result
            assert "2024-01-15 10:30:00 UTC" in result
            assert "[Context: Current time is" in result

    def test_build_user_prompt_without_temporal_context(self, prompt_manager):
        """Test that temporal context is not added when not requested."""
        options = PromptOptions(add_temporal_context=False)
        user_query = "What is the weather?"
        result = prompt_manager.build_user_prompt(user_query, options=options)
        assert result == user_query
        assert "Context: Current time" not in result

    def test_build_user_prompt_default_options(self, prompt_manager):
        """Test that default options are used when None is provided."""
        user_query = "Test query"
        result = prompt_manager.build_user_prompt(user_query)
        assert result == user_query


class TestPromptManagerOrganizationalDefaults:
    """Test organizational defaults management."""

    def test_get_organizational_defaults(self, prompt_manager):
        """Test getting organizational defaults."""
        defaults = prompt_manager.get_organizational_defaults()
        assert isinstance(defaults, str)
        assert len(defaults) > 0
        assert "Organizational Standards" in defaults

    def test_update_organizational_defaults(
        self, prompt_manager, custom_organizational_defaults
    ):
        """Test updating organizational defaults."""
        prompt_manager.update_organizational_defaults(
            custom_organizational_defaults
        )
        assert (
            prompt_manager.get_organizational_defaults()
            == custom_organizational_defaults
        )

    def test_updated_defaults_affect_new_prompts(
        self, prompt_manager, custom_organizational_defaults
    ):
        """Test that updated defaults are used in new prompts."""
        prompt_manager.update_organizational_defaults(
            custom_organizational_defaults
        )
        result = prompt_manager.build_system_prompt(base_prompt="Test")
        assert custom_organizational_defaults in result
        assert "Custom Standards" in result


class TestPromptOptions:
    """Test PromptOptions model."""

    def test_prompt_options_defaults(self):
        """Test that PromptOptions has correct defaults."""
        options = PromptOptions()
        assert options.add_temporal_context is False
        assert options.persona is None
        assert options.add_markdown_instructions is False
        assert options.custom_sections is None

    def test_prompt_options_with_all_fields(self):
        """Test PromptOptions with all fields set."""
        custom_sections = {"Section1": "Content1", "Section2": "Content2"}
        options = PromptOptions(
            add_temporal_context=True,
            persona="Test persona",
            add_markdown_instructions=True,
            custom_sections=custom_sections,
        )
        assert options.add_temporal_context is True
        assert options.persona == "Test persona"
        assert options.add_markdown_instructions is True
        assert options.custom_sections == custom_sections

    def test_prompt_options_partial(self):
        """Test PromptOptions with partial fields."""
        options = PromptOptions(add_temporal_context=True)
        assert options.add_temporal_context is True
        assert options.persona is None
        assert options.add_markdown_instructions is False


class TestPromptManagerSingleton:
    """Test singleton pattern for PromptManager."""

    def test_get_prompt_manager_returns_instance(self):
        """Test that get_prompt_manager returns a PromptManager instance."""
        manager = get_prompt_manager()
        assert isinstance(manager, PromptManager)

    def test_get_prompt_manager_singleton(self):
        """Test that get_prompt_manager returns the same instance."""
        manager1 = get_prompt_manager()
        manager2 = get_prompt_manager()
        assert manager1 is manager2

    def test_reset_prompt_manager(self):
        """Test that reset_prompt_manager creates a new instance."""
        manager1 = get_prompt_manager()
        manager1.register_agent_prompt("test", "Test prompt")
        reset_prompt_manager()
        manager2 = get_prompt_manager()
        assert manager1 is not manager2
        assert manager2.get_agent_prompt("test") is None

    def test_singleton_independent_from_direct_instantiation(self):
        """Test that direct instantiation doesn't affect singleton."""
        direct_manager = PromptManager()
        singleton_manager = get_prompt_manager()
        assert direct_manager is not singleton_manager
        # They should be independent
        direct_manager.register_agent_prompt("direct", "Direct prompt")
        assert singleton_manager.get_agent_prompt("direct") is None


class TestPromptManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_base_prompt(self, prompt_manager):
        """Test that empty base prompt raises ValueError."""
        # Empty string is falsy, so it's treated as "not provided"
        with pytest.raises(ValueError) as exc_info:
            prompt_manager.build_system_prompt(base_prompt="")
        assert "Either agent_name or base_prompt must be provided" in str(
            exc_info.value
        )

    def test_empty_persona(self, prompt_manager):
        """Test that empty persona string is not added (falsy check)."""
        options = PromptOptions(persona="")
        result = prompt_manager.build_system_prompt(
            base_prompt="Test", options=options
        )
        # Empty string is falsy, so Persona section should not be added
        assert "Persona" not in result
        assert "Test" in result
        assert "Organizational Standards" in result

    def test_empty_custom_sections(self, prompt_manager):
        """Test that empty custom sections dict is handled."""
        options = PromptOptions(custom_sections={})
        result = prompt_manager.build_system_prompt(
            base_prompt="Test", options=options
        )
        # Should not raise error, but no custom sections added
        assert "Test" in result

    def test_custom_sections_with_empty_values(self, prompt_manager):
        """Test custom sections with empty string values."""
        options = PromptOptions(custom_sections={"Section": ""})
        result = prompt_manager.build_system_prompt(
            base_prompt="Test", options=options
        )
        assert "Section" in result

    def test_multiple_custom_sections_order(self, prompt_manager):
        """Test that multiple custom sections are added in order."""
        custom_sections = {
            "First": "First content",
            "Second": "Second content",
            "Third": "Third content",
        }
        options = PromptOptions(custom_sections=custom_sections)
        result = prompt_manager.build_system_prompt(
            base_prompt="Test", options=options
        )
        # Check all sections are present
        assert "First" in result
        assert "Second" in result
        assert "Third" in result

    def test_very_long_prompt(self, prompt_manager):
        """Test building with a very long base prompt."""
        long_prompt = "A" * 10000
        result = prompt_manager.build_system_prompt(base_prompt=long_prompt)
        assert long_prompt in result
        assert len(result) > 10000

    def test_special_characters_in_prompt(self, prompt_manager):
        """Test building with special characters in prompt."""
        special_prompt = "Test with special chars: !@#$%^&*()[]{}|\\:;\"'<>?,./"
        result = prompt_manager.build_system_prompt(base_prompt=special_prompt)
        assert special_prompt in result

    def test_unicode_characters_in_prompt(self, prompt_manager):
        """Test building with unicode characters in prompt."""
        unicode_prompt = "Test with unicode: 你好世界 🌍 émojis"
        result = prompt_manager.build_system_prompt(base_prompt=unicode_prompt)
        assert unicode_prompt in result

    def test_newlines_in_prompt(self, prompt_manager):
        """Test building with newlines in prompt."""
        multiline_prompt = "Line 1\nLine 2\nLine 3"
        result = prompt_manager.build_system_prompt(base_prompt=multiline_prompt)
        assert multiline_prompt in result

    def test_temporal_context_uses_utcnow(self, prompt_manager):
        """Test that temporal context uses datetime.utcnow()."""
        with patch("src.prompts.prompt_manager.datetime") as mock_datetime:
            # Create a mock datetime object with strftime method
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15 10:30:00 UTC"
            mock_datetime.utcnow.return_value = mock_now

            options = PromptOptions(add_temporal_context=True)
            result = prompt_manager.build_system_prompt(
                base_prompt="Test", options=options
            )
            mock_datetime.utcnow.assert_called_once()
            mock_now.strftime.assert_called_once_with("%Y-%m-%d %H:%M:%S UTC")
            assert "2024-01-15 10:30:00 UTC" in result

    def test_register_agent_with_empty_name(self, prompt_manager):
        """Test registering agent with empty name."""
        # Empty string is technically valid, though not recommended
        prompt_manager.register_agent_prompt("", "Empty name prompt")
        assert "" in prompt_manager._agent_prompts

    def test_register_agent_with_empty_prompt(self, prompt_manager):
        """Test registering agent with empty prompt."""
        prompt_manager.register_agent_prompt("test", "")
        assert prompt_manager._agent_prompts["test"] == ""

    def test_build_prompt_with_none_options(self, prompt_manager):
        """Test that None options uses defaults."""
        result = prompt_manager.build_system_prompt(
            base_prompt="Test", options=None
        )
        assert "Test" in result
        assert "Organizational Standards" in result

