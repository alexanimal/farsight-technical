"""Unit tests for the organizations agent module.

This module tests the OrganizationsAgent class and its various methods,
including query execution, search strategy determination, parameter extraction,
response formatting, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import yaml

from src.contracts.agent_io import AgentOutput, create_agent_output
from src.contracts.tool_io import ToolOutput
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentInsight, ResponseStatus
from src.agents.organizations.agent import OrganizationsAgent


@pytest.fixture
def sample_config_data():
    """Create sample configuration data for testing."""
    return {
        "name": "organizations",
        "description": "Agent specialized in handling organization-related queries",
        "category": "organizations",
        "tools": [
            {
                "name": "get_organizations",
                "description": "Search for organizations",
            },
            {
                "name": "semantic_search_organizations",
                "description": "Semantic search for organizations",
            },
        ],
        "metadata": {"version": "1.0"},
    }


@pytest.fixture
def sample_config_yaml(sample_config_data):
    """Create sample YAML configuration string."""
    return yaml.dump(sample_config_data)


@pytest.fixture
def temp_config_file(tmp_path, sample_config_yaml):
    """Create a temporary config file for testing."""
    config_file = tmp_path / "organizations_agent.yaml"
    config_file.write_text(sample_config_yaml, encoding="utf-8")
    return config_file


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_organization(sample_org_uuid):
    """Create a sample organization dict (as returned from get_organizations)."""
    return {
        "org_uuid": sample_org_uuid,
        "name": "Test Company",
        "org_domain": "test.com",
        "city": "San Francisco",
        "state": "California",
        "country": "United States",
        "total_funding_usd": 10000000,
        "stage": "Series A",
    }


@pytest.fixture
def sample_agent_context():
    """Create a sample AgentContext for testing."""
    return AgentContext(query="Show me Google")


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    manager = MagicMock()
    manager.register_agent_prompt = MagicMock()
    manager.build_system_prompt = MagicMock(return_value="System prompt")
    manager.build_user_prompt = MagicMock(return_value="User prompt")
    return manager


class TestOrganizationsAgentInitialization:
    """Test OrganizationsAgent initialization."""

    def test_init_with_config_path(self, temp_config_file, sample_config_data):
        """Test initialization with explicit config path."""
        agent = OrganizationsAgent(config_path=temp_config_file)
        assert agent.config_path == temp_config_file
        assert agent.name == sample_config_data["name"]
        assert agent.category == sample_config_data["category"]
        assert hasattr(agent, "_determine_strategy_prompt")
        assert hasattr(agent, "_extract_params_prompt")

    def test_init_without_config_path_auto_discovery(self, tmp_path, sample_config_yaml):
        """Test initialization without config path (auto-discovery)."""
        # The agent calculates: Path(__file__).parent.parent.parent.parent / "configs" / "agents" / "organizations_agent.yaml"
        # If __file__ is tmp_path / "src" / "agents" / "organizations" / "agent.py"
        # Then parent.parent.parent.parent = tmp_path
        # So config should be at: tmp_path / "configs" / "agents" / "organizations_agent.yaml"
        configs_dir = tmp_path / "configs" / "agents"
        configs_dir.mkdir(parents=True)
        config_file = configs_dir / "organizations_agent.yaml"
        config_file.write_text(sample_config_yaml, encoding="utf-8")

        # Mock __file__ to point to expected location
        with patch("src.agents.organizations.agent.__file__", str(tmp_path / "src" / "agents" / "organizations" / "agent.py")):
            agent = OrganizationsAgent(config_path=None)
            assert agent.name == "organizations"

    def test_init_registers_prompts(self, temp_config_file, mock_prompt_manager):
        """Test that initialization registers prompts with prompt manager."""
        with patch("src.agents.organizations.agent.get_prompt_manager", return_value=mock_prompt_manager):
            agent = OrganizationsAgent(config_path=temp_config_file)
            
            # Should register two prompts
            assert mock_prompt_manager.register_agent_prompt.call_count == 2
            call_args_list = mock_prompt_manager.register_agent_prompt.call_args_list
            
            # Check that prompts are registered with correct agent names
            registered_names = [call[1]["agent_name"] for call in call_args_list]
            assert any("determine_strategy" in name for name in registered_names)
            assert any("extract_params" in name for name in registered_names)


class TestOrganizationsAgentExecute:
    """Test OrganizationsAgent.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_semantic_search_flow(
        self,
        temp_config_file,
        sample_agent_context,
        sample_organization,
    ):
        """Test successful execution flow with semantic search."""
        # Mock LLM responses
        strategy_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": True,
                "sector_name": "AI companies",
                "top_k": 10,
            },
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "Found AI companies.",
                "key_findings": ["Finding 1", "Finding 2"],
                "confidence": 0.9,
            },
        }

        # Mock semantic search output
        mock_semantic_output = ToolOutput(
            tool_name="semantic_search_organizations",
            success=True,
            result=[sample_organization],
            execution_time_ms=100,
        )

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "determine_search_strategy" in str(kwargs.get("tools", [])):
                    return strategy_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.organizations.agent.semantic_search_organizations", return_value=mock_semantic_output):
                result = await agent.execute(sample_agent_context)

                assert isinstance(result, AgentOutput)
                assert result.status == ResponseStatus.SUCCESS
                assert result.agent_name == "organizations"
                assert result.agent_category == "organizations"
                assert isinstance(result.content, AgentInsight)
                assert len(result.tool_calls) == 1
                assert result.tool_calls[0]["name"] == "semantic_search_organizations"
                assert result.metadata["num_results"] == 1
                assert result.metadata["search_strategy"] == "semantic"

    @pytest.mark.asyncio
    async def test_execute_structured_search_flow(
        self,
        temp_config_file,
        sample_agent_context,
        sample_organization,
    ):
        """Test successful execution flow with structured search."""
        # Mock LLM responses
        strategy_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": False,
                "company_name": "Google",
            },
        }
        
        extract_params_result = {
            "function_name": "get_organizations",
            "arguments": {"name_ilike": "Google", "limit": 10},
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "Found Google.",
                "key_findings": ["Finding 1"],
                "confidence": 0.9,
            },
        }

        # Mock get_organizations output
        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[sample_organization],
            execution_time_ms=100,
        )

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "determine_search_strategy" in str(kwargs.get("tools", [])):
                    return strategy_result
                elif "get_organizations" in str(kwargs.get("tools", [])):
                    return extract_params_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.organizations.agent.get_organizations", return_value=mock_orgs_output):
                result = await agent.execute(sample_agent_context)

                assert isinstance(result, AgentOutput)
                assert result.status == ResponseStatus.SUCCESS
                assert isinstance(result.content, AgentInsight)
                assert len(result.tool_calls) == 1
                assert result.tool_calls[0]["name"] == "get_organizations"
                assert result.metadata["search_strategy"] == "structured"

    @pytest.mark.asyncio
    async def test_execute_semantic_search_failure(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute when semantic search tool fails."""
        # Mock LLM response
        strategy_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": True,
                "sector_name": "AI companies",
            },
        }

        # Mock failed semantic search output
        mock_semantic_output = ToolOutput(
            tool_name="semantic_search_organizations",
            success=False,
            error="Database connection failed",
            execution_time_ms=100,
        )

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=strategy_result):
            with patch("src.agents.organizations.agent.semantic_search_organizations", return_value=mock_semantic_output):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.ERROR
                assert "Failed to retrieve organizations" in result.error
                assert result.content == ""

    @pytest.mark.asyncio
    async def test_execute_structured_search_failure(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute when get_organizations tool fails."""
        # Mock LLM responses
        strategy_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": False,
                "company_name": "Google",
            },
        }
        
        extract_params_result = {
            "function_name": "get_organizations",
            "arguments": {"limit": 10},
        }

        # Mock failed get_organizations output
        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=False,
            error="Database connection failed",
            execution_time_ms=100,
        )

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "determine_search_strategy" in str(kwargs.get("tools", [])):
                    return strategy_result
                elif "get_organizations" in str(kwargs.get("tools", [])):
                    return extract_params_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.organizations.agent.get_organizations", return_value=mock_orgs_output):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.ERROR
                assert "Failed to retrieve organizations" in result.error
                assert result.content == ""

    @pytest.mark.asyncio
    async def test_execute_exception_handling(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute handles exceptions gracefully."""
        agent = OrganizationsAgent(config_path=temp_config_file)
        
        # Mock get_organizations to raise an exception that propagates to execute's try/except
        with patch("src.agents.organizations.agent.get_organizations", side_effect=Exception("Unexpected error")):
            # Mock the internal methods to return valid data so we reach get_organizations
            with patch("src.agents.organizations.agent.generate_llm_function_response") as mock_llm:
                # Strategy call: structured search
                strategy_result = {
                    "function_name": "determine_search_strategy",
                    "arguments": {
                        "use_semantic_search": False,
                    },
                }
                # Extract params call
                extract_params_result = {
                    "function_name": "get_organizations",
                    "arguments": {"limit": 10},
                }
                
                async def llm_side_effect(*args, **kwargs):
                    if "determine_search_strategy" in str(kwargs.get("tools", [])):
                        return strategy_result
                    elif "get_organizations" in str(kwargs.get("tools", [])):
                        return extract_params_result
                    return None
                
                mock_llm.side_effect = llm_side_effect
                
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.ERROR
                assert "failed to process query" in result.error.lower()
                assert result.content == ""

    @pytest.mark.asyncio
    async def test_execute_tracks_tool_calls(
        self,
        temp_config_file,
        sample_agent_context,
        sample_organization,
    ):
        """Test that execute properly tracks all tool calls."""
        # Mock LLM responses for structured search
        strategy_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": False,
                "company_name": "Google",
            },
        }
        
        extract_params_result = {
            "function_name": "get_organizations",
            "arguments": {"name_ilike": "Google", "limit": 10},
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {"summary": "Test summary", "confidence": 0.8},
        }

        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[sample_organization],
            execution_time_ms=100,
        )

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "determine_search_strategy" in str(kwargs.get("tools", [])):
                    return strategy_result
                elif "get_organizations" in str(kwargs.get("tools", [])):
                    return extract_params_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.organizations.agent.get_organizations", return_value=mock_orgs_output):
                result = await agent.execute(sample_agent_context)

                assert len(result.tool_calls) == 1
                assert result.tool_calls[0]["name"] == "get_organizations"


class TestOrganizationsAgentDetermineSearchStrategy:
    """Test OrganizationsAgent._determine_search_strategy() method."""

    @pytest.mark.asyncio
    async def test_determine_strategy_semantic_search(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test determining semantic search strategy."""
        # Mock LLM response
        llm_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": True,
                "sector_name": "AI companies",
                "top_k": 10,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._determine_search_strategy(sample_agent_context)

            assert result["use_semantic_search"] is True
            assert result["strategy"] == "semantic"
            assert "semantic_params" in result
            assert result["semantic_params"]["text"] == "AI companies"
            assert result["semantic_params"]["top_k"] == 10

    @pytest.mark.asyncio
    async def test_determine_strategy_structured_search(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test determining structured search strategy."""
        # Mock LLM response
        llm_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": False,
                "company_name": "Google",
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._determine_search_strategy(sample_agent_context)

            assert result["use_semantic_search"] is False
            assert result["strategy"] == "structured"
            assert result["company_name"] == "Google"

    @pytest.mark.asyncio
    async def test_determine_strategy_structured_search_no_company_name(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test determining structured search strategy without company name."""
        # Mock LLM response
        llm_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": False,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._determine_search_strategy(sample_agent_context)

            assert result["use_semantic_search"] is False
            assert result["strategy"] == "structured"
            assert "company_name" not in result

    @pytest.mark.asyncio
    async def test_determine_strategy_semantic_search_fallback_to_query(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test semantic search falls back to query when sector_name not provided."""
        # Mock LLM response without sector_name
        llm_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": True,
                "top_k": 10,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._determine_search_strategy(sample_agent_context)

            assert result["use_semantic_search"] is True
            assert result["semantic_params"]["text"] == sample_agent_context.query

    @pytest.mark.asyncio
    async def test_determine_strategy_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test when LLM call fails (defaults to structured search)."""
        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", side_effect=Exception("LLM error")):
            with patch("src.agents.organizations.agent.logger") as mock_logger:
                result = await agent._determine_search_strategy(sample_agent_context)

                # Should default to structured search
                assert result["use_semantic_search"] is False
                assert result["strategy"] == "structured"
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_determine_strategy_unexpected_function_call(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test when LLM returns unexpected function."""
        # Mock LLM response with wrong function name
        llm_result = {
            "function_name": "wrong_function",
            "arguments": {},
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.organizations.agent.logger") as mock_logger:
                result = await agent._determine_search_strategy(sample_agent_context)

                # Should default to structured search
                assert result["use_semantic_search"] is False
                assert result["strategy"] == "structured"


class TestOrganizationsAgentExtractSearchParameters:
    """Test OrganizationsAgent._extract_search_parameters() method."""

    @pytest.mark.asyncio
    async def test_extract_search_parameters_success(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test successful parameter extraction."""
        # Mock LLM response
        llm_result = {
            "function_name": "get_organizations",
            "arguments": {
                "name_ilike": "Google",
                "city": "San Francisco",
                "limit": 10,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context)

            assert result["name_ilike"] == "Google"
            assert result["city"] == "San Francisco"
            assert result["limit"] == 10

    @pytest.mark.asyncio
    async def test_extract_search_parameters_with_company_name(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test parameter extraction when company_name is provided."""
        # Mock LLM response (without name/name_ilike)
        llm_result = {
            "function_name": "get_organizations",
            "arguments": {
                "limit": 10,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context, company_name="Google")

            # Should add company_name as name_ilike
            assert result["name_ilike"] == "Google"
            assert result["limit"] == 10

    @pytest.mark.asyncio
    async def test_extract_search_parameters_company_name_not_overridden(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test that company_name is not added if name/name_ilike already exists."""
        # Mock LLM response with name_ilike already present
        llm_result = {
            "function_name": "get_organizations",
            "arguments": {
                "name_ilike": "Microsoft",
                "limit": 10,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context, company_name="Google")

            # Should not override existing name_ilike
            assert result["name_ilike"] == "Microsoft"
            assert "Google" not in str(result)

    @pytest.mark.asyncio
    async def test_extract_search_parameters_default_limit(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test that default limit is set when not provided."""
        # Mock LLM response without limit
        llm_result = {
            "function_name": "get_organizations",
            "arguments": {},
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context)

            assert result["limit"] == 10

    @pytest.mark.asyncio
    async def test_extract_search_parameters_filters_none_values(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test that None values are filtered out."""
        # Mock LLM response with None values
        llm_result = {
            "function_name": "get_organizations",
            "arguments": {
                "city": None,
                "state": None,
                "limit": 10,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context)

            assert "city" not in result
            assert "state" not in result
            assert "limit" in result

    @pytest.mark.asyncio
    async def test_extract_search_parameters_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test parameter extraction when LLM call fails."""
        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", side_effect=Exception("LLM error")):
            with patch("src.agents.organizations.agent.logger") as mock_logger:
                result = await agent._extract_search_parameters(sample_agent_context)

                # Should return default parameters
                assert result["limit"] == 10
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_extract_search_parameters_unexpected_function_call(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test parameter extraction when LLM returns unexpected function."""
        # Mock LLM response with wrong function name
        llm_result = {
            "function_name": "wrong_function",
            "arguments": {},
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.organizations.agent.logger") as mock_logger:
                result = await agent._extract_search_parameters(sample_agent_context)

                # Should return default parameters
                assert result["limit"] == 10
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_extract_search_parameters_no_function_call(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test parameter extraction when LLM doesn't make function call."""
        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value="not a dict"):
            with patch("src.agents.organizations.agent.logger") as mock_logger:
                result = await agent._extract_search_parameters(sample_agent_context)

                # Should return default parameters
                assert result["limit"] == 10
                mock_logger.warning.assert_called()


class TestOrganizationsAgentFormatResponse:
    """Test OrganizationsAgent._format_response() method."""

    @pytest.mark.asyncio
    async def test_format_response_success(
        self,
        temp_config_file,
        sample_agent_context,
        sample_organization,
    ):
        """Test successful response formatting."""
        strategy = {"strategy": "structured", "use_semantic_search": False}

        # Mock LLM response
        llm_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "Found 1 organization matching your criteria.",
                "key_findings": ["Key finding 1", "Key finding 2"],
                "evidence": {"num_results": 1},
                "confidence": 0.95,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._format_response(
                sample_agent_context, [sample_organization], strategy
            )

            assert isinstance(result, AgentInsight)
            assert result.summary == "Found 1 organization matching your criteria."
            assert len(result.key_findings) == 2
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_format_response_empty_results(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test response formatting with no results."""
        strategy = {"strategy": "structured", "use_semantic_search": False}

        # Mock LLM response
        llm_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "No organizations found.",
                "confidence": 0.0,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._format_response(
                sample_agent_context, [], strategy
            )

            assert isinstance(result, AgentInsight)
            assert "No organizations found" in result.summary or "couldn't find" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_format_response_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
        sample_organization,
    ):
        """Test response formatting when LLM call fails."""
        strategy = {"strategy": "structured", "use_semantic_search": False}

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", side_effect=Exception("LLM error")):
            with patch("src.agents.organizations.agent.logger") as mock_logger:
                result = await agent._format_response(
                    sample_agent_context, [sample_organization], strategy
                )

                assert isinstance(result, AgentInsight)
                assert "Found" in result.summary or "error" in result.summary.lower()
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_format_response_no_function_call(
        self,
        temp_config_file,
        sample_agent_context,
        sample_organization,
    ):
        """Test response formatting when LLM doesn't make function call."""
        strategy = {"strategy": "structured", "use_semantic_search": False}

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value="not a dict"):
            result = await agent._format_response(
                sample_agent_context, [sample_organization], strategy
            )

            assert isinstance(result, AgentInsight)
            assert "Found" in result.summary
            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_format_response_multiple_organizations(
        self,
        temp_config_file,
        sample_agent_context,
        sample_organization,
    ):
        """Test response formatting with multiple organizations."""
        strategy = {"strategy": "semantic", "use_semantic_search": True}
        
        organization2 = {
            "org_uuid": uuid4(),
            "name": "Another Company",
            "city": "New York",
        }

        # Mock LLM response
        llm_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "Found 2 organizations.",
                "key_findings": ["Multiple organizations found"],
                "confidence": 0.9,
            },
        }

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._format_response(
                sample_agent_context, [sample_organization, organization2], strategy
            )

            assert isinstance(result, AgentInsight)
            assert "2" in result.summary or "multiple" in result.summary.lower()


class TestOrganizationsAgentEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_execute_empty_organizations_result_semantic(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute when semantic search returns empty result."""
        # Mock LLM responses
        strategy_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": True,
                "sector_name": "AI companies",
            },
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "No organizations found.",
                "confidence": 0.0,
            },
        }

        mock_semantic_output = ToolOutput(
            tool_name="semantic_search_organizations",
            success=True,
            result=[],
            execution_time_ms=100,
        )

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "determine_search_strategy" in str(kwargs.get("tools", [])):
                    return strategy_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.organizations.agent.semantic_search_organizations", return_value=mock_semantic_output):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.SUCCESS
                assert result.metadata["num_results"] == 0
                assert isinstance(result.content, AgentInsight)

    @pytest.mark.asyncio
    async def test_execute_empty_organizations_result_structured(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute when get_organizations returns empty result."""
        # Mock LLM responses
        strategy_result = {
            "function_name": "determine_search_strategy",
            "arguments": {
                "use_semantic_search": False,
            },
        }
        
        extract_params_result = {
            "function_name": "get_organizations",
            "arguments": {"limit": 10},
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "No organizations found.",
                "confidence": 0.0,
            },
        }

        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[],
            execution_time_ms=100,
        )

        agent = OrganizationsAgent(config_path=temp_config_file)
        
        with patch("src.agents.organizations.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "determine_search_strategy" in str(kwargs.get("tools", [])):
                    return strategy_result
                elif "get_organizations" in str(kwargs.get("tools", [])):
                    return extract_params_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.organizations.agent.get_organizations", return_value=mock_orgs_output):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.SUCCESS
                assert result.metadata["num_results"] == 0
                assert isinstance(result.content, AgentInsight)

