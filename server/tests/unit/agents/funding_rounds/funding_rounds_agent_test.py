"""Unit tests for the funding rounds agent module.

This module tests the FundingRoundsAgent class and its various methods,
including query execution, company name resolution, parameter extraction,
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
from src.agents.funding_rounds.agent import FundingRoundsAgent


@pytest.fixture
def sample_config_data():
    """Create sample configuration data for testing."""
    return {
        "name": "funding_rounds",
        "description": "Agent specialized in handling funding round-related queries",
        "category": "funding_rounds",
        "tools": [
            {
                "name": "get_funding_rounds",
                "description": "Search for funding rounds",
            }
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
    config_file = tmp_path / "funding_rounds_agent.yaml"
    config_file.write_text(sample_config_yaml, encoding="utf-8")
    return config_file


@pytest.fixture
def sample_org_uuid():
    """Create a sample organization UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_funding_round_uuid():
    """Create a sample funding round UUID for testing."""
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
    }


@pytest.fixture
def sample_funding_round(sample_funding_round_uuid, sample_org_uuid):
    """Create a sample funding round dict (as returned from get_funding_rounds)."""
    return {
        "funding_round_uuid": sample_funding_round_uuid,
        "org_uuid": sample_org_uuid,
        "investment_date": "2023-06-15T00:00:00",
        "general_funding_stage": "series_a",
        "fundraise_amount_usd": 10000000,
        "valuation_usd": 50000000,
        "investors": ["Investor 1", "Investor 2"],
        "lead_investors": ["Investor 1"],
    }


@pytest.fixture
def sample_agent_context():
    """Create a sample AgentContext for testing."""
    return AgentContext(query="What funding rounds did Google raise?")


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    manager = MagicMock()
    manager.register_agent_prompt = MagicMock()
    manager.build_system_prompt = MagicMock(return_value="System prompt")
    manager.build_user_prompt = MagicMock(return_value="User prompt")
    return manager


class TestFundingRoundsAgentInitialization:
    """Test FundingRoundsAgent initialization."""

    def test_init_with_config_path(self, temp_config_file, sample_config_data):
        """Test initialization with explicit config path."""
        agent = FundingRoundsAgent(config_path=temp_config_file)
        assert agent.config_path == temp_config_file
        assert agent.name == sample_config_data["name"]
        assert agent.category == sample_config_data["category"]
        assert hasattr(agent, "_identify_companies_prompt")
        assert hasattr(agent, "_extract_params_prompt")

    def test_init_without_config_path_auto_discovery(self, tmp_path, sample_config_yaml):
        """Test initialization without config path (auto-discovery)."""
        # The agent calculates: Path(__file__).parent.parent.parent.parent / "configs" / "agents" / "funding_rounds_agent.yaml"
        # If __file__ is tmp_path / "src" / "agents" / "funding_rounds" / "agent.py"
        # Then parent.parent.parent.parent = tmp_path
        # So config should be at: tmp_path / "configs" / "agents" / "funding_rounds_agent.yaml"
        configs_dir = tmp_path / "configs" / "agents"
        configs_dir.mkdir(parents=True)
        config_file = configs_dir / "funding_rounds_agent.yaml"
        config_file.write_text(sample_config_yaml, encoding="utf-8")

        # Mock __file__ to point to expected location
        with patch("src.agents.funding_rounds.agent.__file__", str(tmp_path / "src" / "agents" / "funding_rounds" / "agent.py")):
            agent = FundingRoundsAgent(config_path=None)
            assert agent.name == "funding_rounds"

    def test_init_registers_prompts(self, temp_config_file, mock_prompt_manager):
        """Test that initialization registers prompts with prompt manager."""
        with patch("src.agents.funding_rounds.agent.get_prompt_manager", return_value=mock_prompt_manager):
            agent = FundingRoundsAgent(config_path=temp_config_file)
            
            # Should register two prompts
            assert mock_prompt_manager.register_agent_prompt.call_count == 2
            call_args_list = mock_prompt_manager.register_agent_prompt.call_args_list
            
            # Check that prompts are registered with correct agent names
            registered_names = [call[1]["agent_name"] for call in call_args_list]
            assert any("identify_companies" in name for name in registered_names)
            assert any("extract_params" in name for name in registered_names)


class TestFundingRoundsAgentExecute:
    """Test FundingRoundsAgent.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_successful_flow(
        self,
        temp_config_file,
        sample_agent_context,
        sample_funding_round,
        sample_org_uuid,
    ):
        """Test successful execution flow with all steps."""
        # Mock LLM responses
        identify_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": "Google", "sector_name": None},
        }
        
        extract_params_result = {
            "function_name": "get_funding_rounds",
            "arguments": {"limit": 10},
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "Google has raised several funding rounds.",
                "key_findings": ["Finding 1", "Finding 2"],
                "confidence": 0.9,
            },
        }

        # Mock tool outputs
        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[{"org_uuid": sample_org_uuid, "name": "Google"}],
            execution_time_ms=50,
        )
        
        mock_funding_rounds_output = ToolOutput(
            tool_name="get_funding_rounds",
            success=True,
            result=[sample_funding_round],
            execution_time_ms=100,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response") as mock_llm:
            # Configure LLM mock to return different results for different calls
            async def llm_side_effect(*args, **kwargs):
                # First call: identify companies
                if "identify_company_names" in str(kwargs.get("tools", [])):
                    return identify_result
                # Second call: extract parameters
                elif "get_funding_rounds" in str(kwargs.get("tools", [])):
                    return extract_params_result
                # Third call: format response
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.funding_rounds.agent.get_organizations", return_value=mock_orgs_output):
                with patch("src.agents.funding_rounds.agent.get_funding_rounds", return_value=mock_funding_rounds_output):
                    result = await agent.execute(sample_agent_context)

                    assert isinstance(result, AgentOutput)
                    assert result.status == ResponseStatus.SUCCESS
                    assert result.agent_name == "funding_rounds"
                    assert result.agent_category == "funding_rounds"
                    assert isinstance(result.content, AgentInsight)
                    assert result.content.summary == "Google has raised several funding rounds."
                    assert len(result.tool_calls) >= 1
                    assert result.metadata["num_results"] == 1

    @pytest.mark.asyncio
    async def test_execute_get_funding_rounds_failure(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute when get_funding_rounds tool fails."""
        # Mock LLM responses
        identify_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": None, "sector_name": None},
        }
        
        extract_params_result = {
            "function_name": "get_funding_rounds",
            "arguments": {"limit": 10},
        }

        # Mock failed tool output
        mock_funding_rounds_output = ToolOutput(
            tool_name="get_funding_rounds",
            success=False,
            error="Database connection failed",
            execution_time_ms=100,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "identify_company_names" in str(kwargs.get("tools", [])):
                    return identify_result
                elif "get_funding_rounds" in str(kwargs.get("tools", [])):
                    return extract_params_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.funding_rounds.agent.get_funding_rounds", return_value=mock_funding_rounds_output):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.ERROR
                assert "Failed to retrieve funding rounds" in result.error
                assert result.content == ""

    @pytest.mark.asyncio
    async def test_execute_exception_handling(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute handles exceptions gracefully."""
        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        # Mock get_funding_rounds to raise an exception that propagates to execute's try/except
        # This will actually trigger the exception handler in execute()
        with patch("src.agents.funding_rounds.agent.get_funding_rounds", side_effect=Exception("Unexpected error")):
            # Mock the internal methods to return valid data so we reach get_funding_rounds
            with patch("src.agents.funding_rounds.agent.generate_llm_function_response") as mock_llm:
                # First call: identify companies (returns empty)
                identify_result = {
                    "function_name": "identify_company_names",
                    "arguments": {"company_name": None, "sector_name": None},
                }
                # Second call: extract parameters
                extract_params_result = {
                    "function_name": "get_funding_rounds",
                    "arguments": {"limit": 10},
                }
                
                async def llm_side_effect(*args, **kwargs):
                    if "identify_company_names" in str(kwargs.get("tools", [])):
                        return identify_result
                    elif "get_funding_rounds" in str(kwargs.get("tools", [])):
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
        sample_funding_round,
        sample_org_uuid,
    ):
        """Test that execute properly tracks all tool calls."""
        # Mock LLM responses
        identify_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": "Google", "sector_name": None},
        }
        
        extract_params_result = {
            "function_name": "get_funding_rounds",
            "arguments": {"limit": 10},
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {"summary": "Test summary", "confidence": 0.8},
        }

        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[{"org_uuid": sample_org_uuid, "name": "Google"}],
            execution_time_ms=50,
        )
        
        mock_funding_rounds_output = ToolOutput(
            tool_name="get_funding_rounds",
            success=True,
            result=[sample_funding_round],
            execution_time_ms=100,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "identify_company_names" in str(kwargs.get("tools", [])):
                    return identify_result
                elif "get_funding_rounds" in str(kwargs.get("tools", [])):
                    return extract_params_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.funding_rounds.agent.get_organizations", return_value=mock_orgs_output):
                with patch("src.agents.funding_rounds.agent.get_funding_rounds", return_value=mock_funding_rounds_output):
                    result = await agent.execute(sample_agent_context)

                    assert len(result.tool_calls) >= 2  # get_organizations + get_funding_rounds
                    # Check that get_funding_rounds is in tool calls
                    tool_names = [call["name"] for call in result.tool_calls]
                    assert "get_funding_rounds" in tool_names
                    assert "get_organizations" in tool_names


class TestFundingRoundsAgentResolveCompanyNames:
    """Test FundingRoundsAgent._resolve_company_names() method."""

    @pytest.mark.asyncio
    async def test_resolve_company_names_with_company(
        self,
        temp_config_file,
        sample_agent_context,
        sample_org_uuid,
    ):
        """Test resolving company name to UUID."""
        # Mock LLM response
        llm_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": "Google", "sector_name": None},
        }

        # Mock get_organizations output
        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[{"org_uuid": sample_org_uuid, "name": "Google Inc."}],
            execution_time_ms=50,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.funding_rounds.agent.get_organizations", return_value=mock_orgs_output):
                result = await agent._resolve_company_names(sample_agent_context)

                assert result["org_uuid"] == str(sample_org_uuid)
                assert len(result["get_organizations_calls"]) == 1
                assert result["get_organizations_calls"][0]["parameters"]["name_ilike"] == "Google"

    @pytest.mark.asyncio
    async def test_resolve_company_names_with_sector(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test resolving sector name (uses semantic search)."""
        # Mock LLM response
        llm_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": None, "sector_name": "AI startups"},
        }

        # Mock semantic search output
        mock_semantic_output = ToolOutput(
            tool_name="semantic_search_organizations",
            success=True,
            result=[
                {"org_uuid": uuid4(), "name": "AI Company 1"},
                {"org_uuid": uuid4(), "name": "AI Company 2"},
            ],
            execution_time_ms=100,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.funding_rounds.agent.semantic_search_organizations", return_value=mock_semantic_output):
                result = await agent._resolve_company_names(sample_agent_context)

                assert result["org_uuid"] is None
                assert len(result["semantic_search_calls"]) == 1
                assert result["semantic_search_calls"][0]["parameters"]["text"] == "AI startups"

    @pytest.mark.asyncio
    async def test_resolve_company_names_with_both(
        self,
        temp_config_file,
        sample_agent_context,
        sample_org_uuid,
    ):
        """Test resolving both company name and sector."""
        # Mock LLM response
        llm_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": "Google", "sector_name": "AI companies"},
        }

        # Mock get_organizations output
        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[{"org_uuid": sample_org_uuid, "name": "Google Inc."}],
            execution_time_ms=50,
        )

        # Mock semantic search output
        mock_semantic_output = ToolOutput(
            tool_name="semantic_search_organizations",
            success=True,
            result=[
                {"org_uuid": uuid4(), "name": "AI Company 1"},
            ],
            execution_time_ms=100,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.funding_rounds.agent.get_organizations", return_value=mock_orgs_output):
                with patch("src.agents.funding_rounds.agent.semantic_search_organizations", return_value=mock_semantic_output):
                    result = await agent._resolve_company_names(sample_agent_context)

                    assert result["org_uuid"] == str(sample_org_uuid)
                    assert len(result["get_organizations_calls"]) == 1
                    assert len(result["semantic_search_calls"]) == 1

    @pytest.mark.asyncio
    async def test_resolve_company_names_no_companies_found(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test when no company names are identified."""
        # Mock LLM response with no companies
        llm_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": None, "sector_name": None},
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._resolve_company_names(sample_agent_context)

            assert result["org_uuid"] is None
            assert len(result["get_organizations_calls"]) == 0
            assert len(result["semantic_search_calls"]) == 0

    @pytest.mark.asyncio
    async def test_resolve_company_names_get_organizations_fails(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test when get_organizations fails."""
        # Mock LLM response
        llm_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": "Google", "sector_name": None},
        }

        # Mock failed get_organizations output
        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=False,
            error="Database error",
            execution_time_ms=50,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.funding_rounds.agent.get_organizations", return_value=mock_orgs_output):
                with patch("src.agents.funding_rounds.agent.logger") as mock_logger:
                    result = await agent._resolve_company_names(sample_agent_context)

                    assert result["org_uuid"] is None
                    # Should log warning
                    mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_resolve_company_names_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test when LLM call fails."""
        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", side_effect=Exception("LLM error")):
            with patch("src.agents.funding_rounds.agent.logger") as mock_logger:
                result = await agent._resolve_company_names(sample_agent_context)

                assert result["org_uuid"] is None
                # Should log warning
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_resolve_company_names_empty_orgs_result(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test when get_organizations returns empty result."""
        # Mock LLM response
        llm_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": "Nonexistent", "sector_name": None},
        }

        # Mock empty get_organizations output
        mock_orgs_output = ToolOutput(
            tool_name="get_organizations",
            success=True,
            result=[],
            execution_time_ms=50,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.funding_rounds.agent.get_organizations", return_value=mock_orgs_output):
                result = await agent._resolve_company_names(sample_agent_context)

                assert result["org_uuid"] is None


class TestFundingRoundsAgentExtractSearchParameters:
    """Test FundingRoundsAgent._extract_search_parameters() method."""

    @pytest.mark.asyncio
    async def test_extract_search_parameters_success(
        self,
        temp_config_file,
        sample_agent_context,
        sample_org_uuid,
    ):
        """Test successful parameter extraction."""
        resolved_uuids = {
            "org_uuid": str(sample_org_uuid),
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        # Mock LLM response
        llm_result = {
            "function_name": "get_funding_rounds",
            "arguments": {
                "org_uuid": str(sample_org_uuid),
                "limit": 10,
                "general_funding_stage": "series_a",
            },
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context, resolved_uuids)

            assert result["org_uuid"] == str(sample_org_uuid)
            assert result["limit"] == 10
            assert result["general_funding_stage"] == "series_a"

    @pytest.mark.asyncio
    async def test_extract_search_parameters_with_resolved_uuids(
        self,
        temp_config_file,
        sample_agent_context,
        sample_org_uuid,
    ):
        """Test parameter extraction with resolved UUIDs (should override LLM)."""
        resolved_uuids = {
            "org_uuid": str(sample_org_uuid),
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        # Mock LLM response (UUID should be overridden)
        llm_result = {
            "function_name": "get_funding_rounds",
            "arguments": {
                "org_uuid": "wrong-uuid",
                "limit": 10,
            },
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context, resolved_uuids)

            # Resolved UUID should take precedence
            assert result["org_uuid"] == str(sample_org_uuid)

    @pytest.mark.asyncio
    async def test_extract_search_parameters_default_limit(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test that default limit is set when not provided."""
        resolved_uuids = {
            "org_uuid": None,
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        # Mock LLM response without limit
        llm_result = {
            "function_name": "get_funding_rounds",
            "arguments": {},
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context, resolved_uuids)

            assert result["limit"] == 10

    @pytest.mark.asyncio
    async def test_extract_search_parameters_filters_none_values(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test that None values are filtered out."""
        resolved_uuids = {
            "org_uuid": None,
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        # Mock LLM response with None values
        llm_result = {
            "function_name": "get_funding_rounds",
            "arguments": {
                "general_funding_stage": None,
                "fundraise_amount_usd": None,
                "limit": 10,
            },
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._extract_search_parameters(sample_agent_context, resolved_uuids)

            assert "general_funding_stage" not in result
            assert "fundraise_amount_usd" not in result
            assert "limit" in result

    @pytest.mark.asyncio
    async def test_extract_search_parameters_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test parameter extraction when LLM call fails."""
        resolved_uuids = {
            "org_uuid": None,
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", side_effect=Exception("LLM error")):
            with patch("src.agents.funding_rounds.agent.logger") as mock_logger:
                result = await agent._extract_search_parameters(sample_agent_context, resolved_uuids)

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
        resolved_uuids = {
            "org_uuid": None,
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        # Mock LLM response with wrong function name
        llm_result = {
            "function_name": "wrong_function",
            "arguments": {},
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            with patch("src.agents.funding_rounds.agent.logger") as mock_logger:
                result = await agent._extract_search_parameters(sample_agent_context, resolved_uuids)

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
        resolved_uuids = {
            "org_uuid": None,
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value="not a dict"):
            with patch("src.agents.funding_rounds.agent.logger") as mock_logger:
                result = await agent._extract_search_parameters(sample_agent_context, resolved_uuids)

                # Should return default parameters
                assert result["limit"] == 10
                mock_logger.warning.assert_called()


class TestFundingRoundsAgentFormatResponse:
    """Test FundingRoundsAgent._format_response() method."""

    @pytest.mark.asyncio
    async def test_format_response_success(
        self,
        temp_config_file,
        sample_agent_context,
        sample_funding_round,
    ):
        """Test successful response formatting."""
        search_params = {"limit": 10}

        # Mock LLM response
        llm_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "Found 1 funding round matching your criteria.",
                "key_findings": ["Key finding 1", "Key finding 2"],
                "evidence": {"num_results": 1},
                "confidence": 0.95,
            },
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._format_response(
                sample_agent_context, [sample_funding_round], search_params
            )

            assert isinstance(result, AgentInsight)
            assert result.summary == "Found 1 funding round matching your criteria."
            assert len(result.key_findings) == 2
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_format_response_empty_results(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test response formatting with no results."""
        search_params = {"limit": 10}

        # Mock LLM response
        llm_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "No funding rounds found.",
                "confidence": 0.0,
            },
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._format_response(
                sample_agent_context, [], search_params
            )

            assert isinstance(result, AgentInsight)
            assert "No funding rounds found" in result.summary or "couldn't find" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_format_response_llm_fails(
        self,
        temp_config_file,
        sample_agent_context,
        sample_funding_round,
    ):
        """Test response formatting when LLM call fails."""
        search_params = {"limit": 10}

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", side_effect=Exception("LLM error")):
            with patch("src.agents.funding_rounds.agent.logger") as mock_logger:
                result = await agent._format_response(
                    sample_agent_context, [sample_funding_round], search_params
                )

                assert isinstance(result, AgentInsight)
                assert "Found" in result.summary or "error" in result.summary.lower()
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_format_response_no_function_call(
        self,
        temp_config_file,
        sample_agent_context,
        sample_funding_round,
    ):
        """Test response formatting when LLM doesn't make function call."""
        search_params = {"limit": 10}

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value="not a dict"):
            result = await agent._format_response(
                sample_agent_context, [sample_funding_round], search_params
            )

            assert isinstance(result, AgentInsight)
            assert "Found" in result.summary
            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_format_response_multiple_funding_rounds(
        self,
        temp_config_file,
        sample_agent_context,
        sample_funding_round,
    ):
        """Test response formatting with multiple funding rounds."""
        search_params = {"limit": 10}
        
        funding_round2 = {
            "funding_round_uuid": uuid4(),
            "org_uuid": uuid4(),
            "investment_date": "2023-07-20T00:00:00",
            "general_funding_stage": "series_b",
            "fundraise_amount_usd": 20000000,
        }

        # Mock LLM response
        llm_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "Found 2 funding rounds.",
                "key_findings": ["Multiple rounds found"],
                "confidence": 0.9,
            },
        }

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response", return_value=llm_result):
            result = await agent._format_response(
                sample_agent_context, [sample_funding_round, funding_round2], search_params
            )

            assert isinstance(result, AgentInsight)
            assert "2" in result.summary or "multiple" in result.summary.lower()


class TestFundingRoundsAgentEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_execute_with_semantic_search_calls(
        self,
        temp_config_file,
        sample_agent_context,
        sample_funding_round,
    ):
        """Test execute when semantic search is used."""
        # Mock LLM responses
        identify_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": None, "sector_name": "AI startups"},
        }
        
        extract_params_result = {
            "function_name": "get_funding_rounds",
            "arguments": {"limit": 10},
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {"summary": "Test summary", "confidence": 0.8},
        }

        mock_semantic_output = ToolOutput(
            tool_name="semantic_search_organizations",
            success=True,
            result=[{"org_uuid": uuid4(), "name": "AI Company"}],
            execution_time_ms=100,
        )
        
        mock_funding_rounds_output = ToolOutput(
            tool_name="get_funding_rounds",
            success=True,
            result=[sample_funding_round],
            execution_time_ms=100,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "identify_company_names" in str(kwargs.get("tools", [])):
                    return identify_result
                elif "get_funding_rounds" in str(kwargs.get("tools", [])):
                    return extract_params_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.funding_rounds.agent.semantic_search_organizations", return_value=mock_semantic_output):
                with patch("src.agents.funding_rounds.agent.get_funding_rounds", return_value=mock_funding_rounds_output):
                    result = await agent.execute(sample_agent_context)

                    assert result.status == ResponseStatus.SUCCESS
                    # Check that semantic search call is tracked
                    tool_names = [call["name"] for call in result.tool_calls]
                    assert "semantic_search_organizations" in tool_names

    @pytest.mark.asyncio
    async def test_execute_empty_funding_rounds_result(
        self,
        temp_config_file,
        sample_agent_context,
    ):
        """Test execute when get_funding_rounds returns empty result."""
        # Mock LLM responses
        identify_result = {
            "function_name": "identify_company_names",
            "arguments": {"company_name": None, "sector_name": None},
        }
        
        extract_params_result = {
            "function_name": "get_funding_rounds",
            "arguments": {"limit": 10},
        }
        
        format_result = {
            "function_name": "generate_insight",
            "arguments": {
                "summary": "No funding rounds found.",
                "confidence": 0.0,
            },
        }

        mock_funding_rounds_output = ToolOutput(
            tool_name="get_funding_rounds",
            success=True,
            result=[],
            execution_time_ms=100,
        )

        agent = FundingRoundsAgent(config_path=temp_config_file)
        
        with patch("src.agents.funding_rounds.agent.generate_llm_function_response") as mock_llm:
            async def llm_side_effect(*args, **kwargs):
                if "identify_company_names" in str(kwargs.get("tools", [])):
                    return identify_result
                elif "get_funding_rounds" in str(kwargs.get("tools", [])):
                    return extract_params_result
                elif "generate_insight" in str(kwargs.get("tools", [])):
                    return format_result
                return None
            
            mock_llm.side_effect = llm_side_effect

            with patch("src.agents.funding_rounds.agent.get_funding_rounds", return_value=mock_funding_rounds_output):
                result = await agent.execute(sample_agent_context)

                assert result.status == ResponseStatus.SUCCESS
                assert result.metadata["num_results"] == 0
                assert isinstance(result.content, AgentInsight)

