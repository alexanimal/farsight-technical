"""Unit tests for the orchestration agent module.

This module tests the OrchestrationAgent class and its various methods,
including execution planning, consolidation, validation, and error handling.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from uuid import uuid4

import pytest
import yaml

from src.agents.orchestration.agent import (
    _FALLBACK_AVAILABLE_AGENTS,
    OrchestrationAgent,
    _get_available_agents,
)
from src.contracts.agent_io import AgentOutput, create_agent_output
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentInsight, ResponseStatus


@pytest.fixture
def sample_config_data():
    """Create sample configuration data for testing."""
    return {
        "name": "orchestration",
        "description": "Agent specialized in planning and routing queries",
        "category": "orchestration",
        "tools": [],
        "metadata": {"version": "1.0"},
    }


@pytest.fixture
def sample_config_yaml(sample_config_data):
    """Create sample YAML configuration string."""
    return yaml.dump(sample_config_data)


@pytest.fixture
def temp_config_file(tmp_path, sample_config_yaml):
    """Create a temporary config file for testing."""
    config_file = tmp_path / "orchestration_agent.yaml"
    config_file.write_text(sample_config_yaml, encoding="utf-8")
    return config_file


@pytest.fixture
def sample_agent_context():
    """Create a sample AgentContext for testing."""
    return AgentContext(query="What acquisitions did Google make?")


@pytest.fixture
def sample_available_agents():
    """Create sample available agents dictionary."""
    return {
        "acquisition": {
            "name": "acquisition",
            "description": "Handles queries about company acquisitions",
            "keywords": ["acquisition", "acquired", "merger"],
        },
        "funding_rounds": {
            "name": "funding_rounds",
            "description": "Handles queries about funding rounds",
            "keywords": ["funding", "investment", "round"],
        },
        "organizations": {
            "name": "organizations",
            "description": "Handles queries about organizations",
            "keywords": ["company", "organization", "startup"],
        },
    }


@pytest.fixture
def sample_agent_output():
    """Create a sample AgentOutput for testing."""
    return create_agent_output(
        content=AgentInsight(
            summary="Test insight summary",
            key_findings=["Finding 1", "Finding 2"],
            confidence=0.9,
        ),
        agent_name="acquisition",
        agent_category="acquisition",
        status=ResponseStatus.SUCCESS,
    )


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    manager = MagicMock()
    manager.register_agent_prompt = MagicMock()
    manager.build_system_prompt = MagicMock(return_value="System prompt")
    manager.build_user_prompt = MagicMock(return_value="User prompt")
    return manager


class TestOrchestrationAgentInitialization:
    """Test OrchestrationAgent initialization."""

    def test_init_with_config_path(self, temp_config_file, sample_config_data):
        """Test initialization with explicit config path."""
        agent = OrchestrationAgent(config_path=temp_config_file)
        assert agent.config_path == temp_config_file
        assert agent.name == sample_config_data["name"]
        assert agent.category == sample_config_data["category"]

    def test_init_without_config_path_auto_discovery(self, tmp_path, sample_config_yaml):
        """Test initialization without config path (auto-discovery)."""
        # The agent calculates: Path(__file__).parent.parent.parent.parent / "configs" / "agents" / "orchestration_agent.yaml"
        # If __file__ is tmp_path / "src" / "agents" / "orchestration" / "agent.py"
        # Then parent.parent.parent.parent = tmp_path
        # So config should be at: tmp_path / "configs" / "agents" / "orchestration_agent.yaml"
        configs_dir = tmp_path / "configs" / "agents"
        configs_dir.mkdir(parents=True)
        config_file = configs_dir / "orchestration_agent.yaml"
        config_file.write_text(sample_config_yaml, encoding="utf-8")

        # Mock __file__ to point to expected location
        with patch(
            "src.agents.orchestration.agent.__file__",
            str(tmp_path / "src" / "agents" / "orchestration" / "agent.py"),
        ):
            agent = OrchestrationAgent(config_path=None)
            assert agent.name == "orchestration"

    def test_init_registers_prompt(self, temp_config_file, mock_prompt_manager):
        """Test that initialization registers prompt with prompt manager."""
        with patch(
            "src.agents.orchestration.agent.get_prompt_manager",
            return_value=mock_prompt_manager,
        ):
            agent = OrchestrationAgent(config_path=temp_config_file)

            # Should register one prompt
            assert mock_prompt_manager.register_agent_prompt.call_count == 1
            call_args = mock_prompt_manager.register_agent_prompt.call_args
            assert call_args[1]["agent_name"] == "orchestration"


class TestOrchestrationAgentExecute:
    """Test OrchestrationAgent.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_planning_mode_success(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test successful execution in planning mode."""
        # Mock LLM response
        plan_result = {
            "function_name": "create_execution_plan",
            "arguments": {
                "agents": ["acquisition"],
                "execution_mode": "sequential",
                "reasoning": "Query is about acquisitions",
                "confidence": 0.9,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent._get_available_agents",
            return_value=sample_available_agents,
        ):
            with patch(
                "src.agents.orchestration.agent.generate_llm_function_response",
                return_value=plan_result,
            ):
                result = await agent.execute(sample_agent_context)

                assert isinstance(result, AgentOutput)
                assert result.status == ResponseStatus.SUCCESS
                assert result.agent_name == "orchestration"
                assert isinstance(result.content, AgentInsight)
                assert result.metadata["plan_created"] is True
                assert result.metadata["num_agents"] == 1
                assert "execution_plan" in result.metadata
                assert result.metadata["execution_plan"]["agents"] == ["acquisition"]

    @pytest.mark.asyncio
    async def test_execute_planning_mode_multiple_agents(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test planning mode with multiple agents selected."""
        # Mock LLM response
        plan_result = {
            "function_name": "create_execution_plan",
            "arguments": {
                "agents": ["acquisition", "organizations"],
                "execution_mode": "parallel",
                "reasoning": "Query needs both acquisition and organization data",
                "confidence": 0.85,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent._get_available_agents",
            return_value=sample_available_agents,
        ):
            with patch(
                "src.agents.orchestration.agent.generate_llm_function_response",
                return_value=plan_result,
            ):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.SUCCESS
                assert result.metadata["num_agents"] == 2
                assert len(result.metadata["execution_plan"]["agents"]) == 2
                assert result.metadata["execution_plan"]["execution_mode"] == "parallel"

    @pytest.mark.asyncio
    async def test_execute_planning_mode_no_agents(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test planning mode when no agents are selected."""
        # Mock LLM response
        plan_result = {
            "function_name": "create_execution_plan",
            "arguments": {
                "agents": [],
                "execution_mode": "sequential",
                "reasoning": "No relevant agents found",
                "confidence": 0.3,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent._get_available_agents",
            return_value=sample_available_agents,
        ):
            with patch(
                "src.agents.orchestration.agent.generate_llm_function_response",
                return_value=plan_result,
            ):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.SUCCESS
                assert result.metadata["num_agents"] == 0
                assert len(result.metadata["execution_plan"]["agents"]) == 0

    @pytest.mark.asyncio
    async def test_execute_planning_mode_uses_metadata_available_agents(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test that planning mode uses available_agents from metadata if provided."""
        # Set available_agents in metadata
        sample_agent_context.metadata["available_agents"] = sample_available_agents

        plan_result = {
            "function_name": "create_execution_plan",
            "arguments": {
                "agents": ["acquisition"],
                "execution_mode": "sequential",
                "reasoning": "Test",
                "confidence": 0.9,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch("src.agents.orchestration.agent._get_available_agents") as mock_get_agents:
            with patch(
                "src.agents.orchestration.agent.generate_llm_function_response",
                return_value=plan_result,
            ):
                result = await agent.execute(sample_agent_context)

                # Should not call _get_available_agents since it's in metadata
                mock_get_agents.assert_not_called()
                assert result.status == ResponseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_consolidation_mode_success(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test successful execution in consolidation mode."""
        # Set mode and agent_responses in metadata
        sample_agent_context.metadata["mode"] = "consolidation"
        sample_agent_context.metadata["agent_responses"] = [sample_agent_output]

        # Mock LLM response
        consolidate_result = {
            "function_name": "consolidate_insights",
            "arguments": {
                "summary": "Consolidated answer from multiple agents",
                "key_findings": ["Consolidated finding 1"],
                "confidence": 0.9,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=consolidate_result,
        ):
            result = await agent.execute(sample_agent_context)

            assert isinstance(result, AgentOutput)
            assert result.status == ResponseStatus.SUCCESS
            assert isinstance(result.content, AgentInsight)
            assert result.content.summary == "Consolidated answer from multiple agents"
            assert result.metadata["mode"] == "consolidation"
            assert result.metadata["num_agents_consolidated"] == 1

    @pytest.mark.asyncio
    async def test_execute_consolidation_mode_no_responses(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test consolidation mode when no agent responses provided."""
        sample_agent_context.metadata["mode"] = "consolidation"
        sample_agent_context.metadata["agent_responses"] = []

        agent = OrchestrationAgent(config_path=temp_config_file)

        result = await agent.execute(sample_agent_context)

        assert result.status == ResponseStatus.ERROR
        assert "No agent responses provided" in result.error

    @pytest.mark.asyncio
    async def test_execute_consolidation_mode_multiple_responses(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test consolidation mode with multiple agent responses."""
        sample_agent_context.metadata["mode"] = "consolidation"

        # Create second agent output
        agent_output2 = create_agent_output(
            content=AgentInsight(
                summary="Second insight",
                confidence=0.8,
            ),
            agent_name="organizations",
            agent_category="organizations",
            status=ResponseStatus.SUCCESS,
        )

        sample_agent_context.metadata["agent_responses"] = [
            sample_agent_output,
            agent_output2,
        ]

        consolidate_result = {
            "function_name": "consolidate_insights",
            "arguments": {
                "summary": "Consolidated from 2 agents",
                "confidence": 0.85,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=consolidate_result,
        ):
            result = await agent.execute(sample_agent_context)

            assert result.status == ResponseStatus.SUCCESS
            assert result.metadata["num_agents_consolidated"] == 2

    @pytest.mark.asyncio
    async def test_execute_exception_handling(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute handles exceptions gracefully."""
        agent = OrchestrationAgent(config_path=temp_config_file)

        # Mock _get_available_agents to raise exception
        with patch(
            "src.agents.orchestration.agent._get_available_agents",
            side_effect=Exception("Unexpected error"),
        ):
            result = await agent.execute(sample_agent_context)

            assert result.status == ResponseStatus.ERROR
            assert "failed" in result.error.lower()
            assert result.content == ""


class TestOrchestrationAgentCreateExecutionPlan:
    """Test OrchestrationAgent._create_execution_plan() method."""

    @pytest.mark.asyncio
    async def test_create_execution_plan_success(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test successful execution plan creation."""
        plan_result = {
            "function_name": "create_execution_plan",
            "arguments": {
                "agents": ["acquisition"],
                "execution_mode": "sequential",
                "reasoning": "Query is about acquisitions",
                "confidence": 0.9,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=plan_result,
        ):
            result = await agent._create_execution_plan(
                sample_agent_context, sample_available_agents
            )

            assert result["agents"] == ["acquisition"]
            assert result["execution_mode"] == "sequential"
            assert result["reasoning"] == "Query is about acquisitions"
            assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_create_execution_plan_multiple_agents(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test execution plan with multiple agents."""
        plan_result = {
            "function_name": "create_execution_plan",
            "arguments": {
                "agents": ["acquisition", "organizations"],
                "execution_mode": "parallel",
                "reasoning": "Need both agents",
                "confidence": 0.8,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=plan_result,
        ):
            result = await agent._create_execution_plan(
                sample_agent_context, sample_available_agents
            )

            assert len(result["agents"]) == 2
            assert "acquisition" in result["agents"]
            assert "organizations" in result["agents"]

    @pytest.mark.asyncio
    async def test_create_execution_plan_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test execution plan creation when LLM call fails."""
        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            side_effect=Exception("LLM error"),
        ):
            with patch("src.agents.orchestration.agent.logger") as mock_logger:
                result = await agent._create_execution_plan(
                    sample_agent_context, sample_available_agents
                )

                # Should use fallback plan
                assert "agents" in result
                assert "execution_mode" in result
                assert "reasoning" in result
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_create_execution_plan_unexpected_function_call(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test execution plan creation when LLM returns unexpected function."""
        wrong_result = {
            "function_name": "wrong_function",
            "arguments": {},
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=wrong_result,
        ):
            with patch("src.agents.orchestration.agent.logger") as mock_logger:
                result = await agent._create_execution_plan(
                    sample_agent_context, sample_available_agents
                )

                # Should use fallback plan
                assert "agents" in result
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_create_execution_plan_no_function_call(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test execution plan creation when LLM doesn't make function call."""
        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value="not a dict",
        ):
            with patch("src.agents.orchestration.agent.logger") as mock_logger:
                result = await agent._create_execution_plan(
                    sample_agent_context, sample_available_agents
                )

                # Should use fallback plan
                assert "agents" in result
                mock_logger.warning.assert_called()


class TestOrchestrationAgentValidateAndNormalizePlan:
    """Test OrchestrationAgent._validate_and_normalize_plan() method."""

    def test_validate_and_normalize_plan_valid(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test validation of valid plan."""
        plan_data = {
            "agents": ["acquisition"],
            "execution_mode": "sequential",
            "reasoning": "Test reasoning",
            "confidence": 0.9,
        }

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._validate_and_normalize_plan(plan_data, sample_available_agents)

        assert result["agents"] == ["acquisition"]
        assert result["execution_mode"] == "sequential"
        assert result["reasoning"] == "Test reasoning"
        assert result["confidence"] == 0.9

    def test_validate_and_normalize_plan_filters_invalid_agents(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test that invalid agent names are filtered out."""
        plan_data = {
            "agents": ["acquisition", "invalid_agent", "organizations"],
            "execution_mode": "sequential",
            "reasoning": "Test",
            "confidence": 0.9,
        }

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._validate_and_normalize_plan(plan_data, sample_available_agents)

        assert "acquisition" in result["agents"]
        assert "organizations" in result["agents"]
        assert "invalid_agent" not in result["agents"]

    def test_validate_and_normalize_plan_fixes_invalid_execution_mode(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test that invalid execution mode is fixed."""
        plan_data = {
            "agents": ["acquisition"],
            "execution_mode": "invalid_mode",
            "reasoning": "Test",
            "confidence": 0.9,
        }

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._validate_and_normalize_plan(plan_data, sample_available_agents)

        assert result["execution_mode"] == "sequential"  # Default

    def test_validate_and_normalize_plan_fixes_invalid_confidence(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test that invalid confidence is fixed."""
        plan_data = {
            "agents": ["acquisition"],
            "execution_mode": "sequential",
            "reasoning": "Test",
            "confidence": 1.5,  # Invalid: > 1.0
        }

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._validate_and_normalize_plan(plan_data, sample_available_agents)

        assert result["confidence"] == 0.5  # Default

    def test_validate_and_normalize_plan_adds_agent_from_reasoning(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test that agent mentioned in reasoning is added if not in agents list."""
        plan_data = {
            "agents": [],  # Empty but reasoning mentions agent
            "execution_mode": "sequential",
            "reasoning": "The acquisition agent would be relevant for this query",
            "confidence": 0.9,
        }

        agent = OrchestrationAgent(config_path=temp_config_file)
        with patch("src.agents.orchestration.agent.logger") as mock_logger:
            result = agent._validate_and_normalize_plan(plan_data, sample_available_agents)

            # Should add acquisition agent since it's mentioned in reasoning
            assert "acquisition" in result["agents"]
            mock_logger.info.assert_called()

    def test_validate_and_normalize_plan_handles_non_list_agents(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test that non-list agents value is handled."""
        plan_data = {
            "agents": "not a list",  # Invalid type
            "execution_mode": "sequential",
            "reasoning": "Test",
            "confidence": 0.9,
        }

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._validate_and_normalize_plan(plan_data, sample_available_agents)

        assert isinstance(result["agents"], list)
        assert len(result["agents"]) == 0

    def test_validate_and_normalize_plan_handles_non_string_reasoning(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test that non-string reasoning is handled."""
        plan_data = {
            "agents": ["acquisition"],
            "execution_mode": "sequential",
            "reasoning": 123,  # Invalid type
            "confidence": 0.9,
        }

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._validate_and_normalize_plan(plan_data, sample_available_agents)

        assert result["reasoning"] == ""  # Default


class TestOrchestrationAgentCreateFallbackPlan:
    """Test OrchestrationAgent._create_fallback_plan() method."""

    def test_create_fallback_plan_keyword_matching(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test fallback plan creation with keyword matching."""
        context = AgentContext(query="What acquisitions did Google make?")

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._create_fallback_plan(context, sample_available_agents)

        assert "acquisition" in result["agents"]  # "acquisition" keyword matches
        assert result["execution_mode"] == "sequential"
        assert "reasoning" in result
        assert result["confidence"] == 0.6

    def test_create_fallback_plan_no_keywords_match(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test fallback plan when no keywords match."""
        context = AgentContext(query="What is the weather today?")

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._create_fallback_plan(context, sample_available_agents)

        assert len(result["agents"]) == 0
        assert result["execution_mode"] == "sequential"

    def test_create_fallback_plan_multiple_keywords_match(
        self,
        temp_config_file,
        sample_available_agents,
    ):
        """Test fallback plan when multiple agents match."""
        context = AgentContext(query="Show me funding rounds for companies")

        agent = OrchestrationAgent(config_path=temp_config_file)
        result = agent._create_fallback_plan(context, sample_available_agents)

        # "funding" matches funding_rounds, "companies" matches organizations
        assert len(result["agents"]) >= 1
        assert result["execution_mode"] == "sequential"


class TestOrchestrationAgentConsolidateResponses:
    """Test OrchestrationAgent._consolidate_responses() method."""

    @pytest.mark.asyncio
    async def test_consolidate_responses_success(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test successful consolidation of responses."""
        consolidate_result = {
            "function_name": "consolidate_insights",
            "arguments": {
                "summary": "Consolidated answer",
                "key_findings": ["Finding 1", "Finding 2"],
                "confidence": 0.9,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=consolidate_result,
        ):
            result = await agent._consolidate_responses(sample_agent_context, [sample_agent_output])

            assert isinstance(result, AgentInsight)
            assert result.summary == "Consolidated answer"
            assert len(result.key_findings) == 2
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_consolidate_responses_multiple_insights(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test consolidation with multiple agent insights."""
        agent_output2 = create_agent_output(
            content=AgentInsight(
                summary="Second insight",
                confidence=0.8,
            ),
            agent_name="organizations",
            agent_category="organizations",
            status=ResponseStatus.SUCCESS,
        )

        consolidate_result = {
            "function_name": "consolidate_insights",
            "arguments": {
                "summary": "Combined insights from 2 agents",
                "confidence": 0.85,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=consolidate_result,
        ):
            result = await agent._consolidate_responses(
                sample_agent_context, [sample_agent_output, agent_output2]
            )

            assert isinstance(result, AgentInsight)
            assert "2" in result.summary or "Combined" in result.summary

    @pytest.mark.asyncio
    async def test_consolidate_responses_no_insights(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test consolidation when no insights are available."""
        # Create response without AgentInsight content
        empty_output = create_agent_output(
            content="",
            agent_name="acquisition",
            agent_category="acquisition",
            status=ResponseStatus.SUCCESS,
        )

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch("src.agents.orchestration.agent.logger") as mock_logger:
            result = await agent._consolidate_responses(sample_agent_context, [empty_output])

            assert isinstance(result, AgentInsight)
            assert "No insights" in result.summary or "not available" in result.summary.lower()
            assert result.confidence == 0.0
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_consolidate_responses_handles_dict_responses(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test consolidation handles dict responses."""
        # Convert AgentOutput to dict
        agent_output_dict = sample_agent_output.model_dump()

        consolidate_result = {
            "function_name": "consolidate_insights",
            "arguments": {
                "summary": "Consolidated from dict",
                "confidence": 0.9,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=consolidate_result,
        ):
            result = await agent._consolidate_responses(sample_agent_context, [agent_output_dict])

            assert isinstance(result, AgentInsight)
            assert result.summary == "Consolidated from dict"

    @pytest.mark.asyncio
    async def test_consolidate_responses_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test consolidation when LLM call fails."""
        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            side_effect=Exception("LLM error"),
        ):
            with patch("src.agents.orchestration.agent.logger") as mock_logger:
                result = await agent._consolidate_responses(
                    sample_agent_context, [sample_agent_output]
                )

                assert isinstance(result, AgentInsight)
                assert "failed" in result.summary.lower() or "error" in result.summary.lower()
                assert result.confidence == 0.0
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_consolidate_responses_no_function_call(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test consolidation when LLM doesn't make function call."""
        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value="not a dict",
        ):
            result = await agent._consolidate_responses(sample_agent_context, [sample_agent_output])

            assert isinstance(result, AgentInsight)
            assert "failed" in result.summary.lower()


class TestGetAvailableAgents:
    """Test _get_available_agents() module-level function."""

    def test_get_available_agents_registry_available(
        self,
        sample_available_agents,
    ):
        """Test getting available agents from registry."""
        with patch("src.agents.orchestration.agent._AGENT_REGISTRY_AVAILABLE", True):
            with patch(
                "src.agents.orchestration.agent.list_available_agents",
                return_value=["acquisition", "organizations"],
            ):
                with patch(
                    "src.agents.orchestration.agent.get_agent_metadata"
                ) as mock_get_metadata:
                    # Mock metadata for each agent
                    def metadata_side_effect(agent_name):
                        if agent_name == "acquisition":
                            return {
                                "description": "Handles acquisitions",
                                "metadata": {"keywords": ["acquisition", "merger"]},
                            }
                        elif agent_name == "organizations":
                            return {
                                "description": "Handles organizations",
                                "metadata": {"keywords": ["company", "organization"]},
                            }
                        return None

                    mock_get_metadata.side_effect = metadata_side_effect

                    result = _get_available_agents()

                    assert "acquisition" in result
                    assert "organizations" in result
                    assert "orchestration" not in result  # Should be filtered out

    def test_get_available_agents_registry_unavailable(
        self,
    ):
        """Test getting available agents when registry is unavailable."""
        with patch("src.agents.orchestration.agent._AGENT_REGISTRY_AVAILABLE", False):
            result = _get_available_agents()

            # Should return fallback
            assert result == _FALLBACK_AVAILABLE_AGENTS

    def test_get_available_agents_registry_error(
        self,
    ):
        """Test getting available agents when registry raises error."""
        with patch("src.agents.orchestration.agent._AGENT_REGISTRY_AVAILABLE", True):
            with patch(
                "src.agents.orchestration.agent.list_available_agents",
                side_effect=Exception("Registry error"),
            ):
                with patch("src.agents.orchestration.agent.logger") as mock_logger:
                    result = _get_available_agents()

                    # Should return fallback
                    assert result == _FALLBACK_AVAILABLE_AGENTS
                    mock_logger.warning.assert_called()

    def test_get_available_agents_loads_config_file(
        self,
        tmp_path,
    ):
        """Test that _get_available_agents loads config file when metadata incomplete."""
        config_file = tmp_path / "acquisition_agent.yaml"
        config_data = {
            "name": "acquisition",
            "description": "Handles acquisitions",
            "metadata": {"keywords": ["acquisition"]},
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        with patch("src.agents.orchestration.agent._AGENT_REGISTRY_AVAILABLE", True):
            with patch(
                "src.agents.orchestration.agent.list_available_agents",
                return_value=["acquisition"],
            ):
                # Mock get_agent_metadata to return default description (triggers config file loading)
                with patch(
                    "src.agents.orchestration.agent.get_agent_metadata"
                ) as mock_get_metadata:
                    mock_get_metadata.return_value = {
                        "description": "Agent for handling acquisition queries",  # Default description
                        "metadata": {},
                    }

                    with patch("src.agents.orchestration.agent.get_agent") as mock_get_agent:
                        # Mock get_agent to return config path
                        mock_agent_class = MagicMock()
                        mock_get_agent.return_value = (mock_agent_class, config_file)

                        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
                            result = _get_available_agents()

                            assert "acquisition" in result
                            assert result["acquisition"]["description"] == "Handles acquisitions"

    def test_get_available_agents_generates_keywords(
        self,
    ):
        """Test that keywords are generated when not in metadata."""
        with patch("src.agents.orchestration.agent._AGENT_REGISTRY_AVAILABLE", True):
            with patch(
                "src.agents.orchestration.agent.list_available_agents",
                return_value=["acquisition"],
            ):
                with patch(
                    "src.agents.orchestration.agent.get_agent_metadata"
                ) as mock_get_metadata:
                    mock_get_metadata.return_value = {
                        "description": "Handles queries about company acquisitions and mergers",
                        "metadata": {},  # No keywords
                    }

                    result = _get_available_agents()

                    assert "acquisition" in result
                    assert len(result["acquisition"]["keywords"]) > 0
                    assert "acquisition" in result["acquisition"]["keywords"]

    def test_get_available_agents_filters_orchestration(
        self,
    ):
        """Test that orchestration agent is filtered out."""
        with patch("src.agents.orchestration.agent._AGENT_REGISTRY_AVAILABLE", True):
            with patch(
                "src.agents.orchestration.agent.list_available_agents",
                return_value=["acquisition", "orchestration", "organizations"],
            ):
                with patch(
                    "src.agents.orchestration.agent.get_agent_metadata"
                ) as mock_get_metadata:

                    def metadata_side_effect(agent_name):
                        return {
                            "description": f"Handles {agent_name}",
                            "metadata": {},
                        }

                    mock_get_metadata.side_effect = metadata_side_effect

                    result = _get_available_agents()

                    assert "acquisition" in result
                    assert "organizations" in result
                    assert "orchestration" not in result


class TestOrchestrationAgentEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_execute_planning_mode_fallback_plan(
        self,
        temp_config_file,
        sample_agent_context,
        sample_available_agents,
    ):
        """Test that execute uses fallback plan when LLM fails."""
        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent._get_available_agents",
            return_value=sample_available_agents,
        ):
            with patch(
                "src.agents.orchestration.agent.generate_llm_function_response",
                side_effect=Exception("LLM error"),
            ):
                with patch("src.agents.orchestration.agent.logger") as mock_logger:
                    result = await agent.execute(sample_agent_context)

                    # Should still succeed with fallback plan
                    assert result.status == ResponseStatus.SUCCESS
                    assert "execution_plan" in result.metadata
                    mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_consolidate_responses_skips_non_insight_content(
        self,
        temp_config_file,
        sample_agent_context,
        sample_agent_output,
    ):
        """Test that consolidation skips non-insight content."""
        # Create response with string content (not AgentInsight)
        string_output = create_agent_output(
            content="Just a string",
            agent_name="acquisition",
            agent_category="acquisition",
            status=ResponseStatus.SUCCESS,
        )

        consolidate_result = {
            "function_name": "consolidate_insights",
            "arguments": {
                "summary": "Consolidated",
                "confidence": 0.9,
            },
        }

        agent = OrchestrationAgent(config_path=temp_config_file)

        with patch(
            "src.agents.orchestration.agent.generate_llm_function_response",
            return_value=consolidate_result,
        ):
            with patch("src.agents.orchestration.agent.logger") as mock_logger:
                result = await agent._consolidate_responses(
                    sample_agent_context, [sample_agent_output, string_output]
                )

                # Should only consolidate the valid insight
                assert isinstance(result, AgentInsight)
                mock_logger.debug.assert_called()
