"""Funding rounds agent for handling funding round-related queries.

This agent is specialized in:
- Finding funding rounds for companies
- Getting funding round details (amounts, investors, stages, dates)
- Analyzing funding trends
- Searching funding rounds by various criteria (company names, dates, amounts, investors, stages, etc.)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src.contracts.agent_io import AgentOutput, create_agent_output
from src.core.agent_response import AgentInsight
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus
from src.prompts.prompt_manager import PromptOptions, get_prompt_manager
from src.tools.generate_llm_function_response import \
    generate_llm_function_response
from src.tools.get_funding_rounds import get_funding_rounds
from src.tools.get_organizations import get_organizations
from src.tools.semantic_search_organizations import \
    semantic_search_organizations

logger = logging.getLogger(__name__)


class FundingRoundsAgent(AgentBase):
    """Agent specialized in handling funding round-related queries and data retrieval.

    This agent uses LLM reasoning to understand user queries about funding rounds
    and calls the get_funding_rounds tool to retrieve relevant data from the database.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the funding rounds agent.

        Args:
            config_path: Path to the agent's YAML config file.
                If None, will attempt to find it automatically.
        """
        if config_path is None:
            # Default to funding_rounds_agent.yaml in configs/agents
            src_base = Path(__file__).parent.parent.parent.parent
            config_path = src_base / "configs" / "agents" / "funding_rounds_agent.yaml"

        super().__init__(config_path=config_path)

        # Store base prompts as instance variables for reuse
        # Base prompt for company name and sector identification
        self._identify_companies_prompt = """You are analyzing a query about funding rounds. Your job is to identify:
1. Specific company names mentioned in the query
2. Sector/industry names mentioned in the query (for semantic search)

IMPORTANT: The semantic_search_organizations tool is designed to query by SECTOR/INDUSTRY names, not specific company names. For specific company names, use get_organizations with name/name_ilike parameter.

Examples:
- "What funding rounds did Google raise?" → company_name: "Google", sector_name: None
- "Show me funding rounds for Microsoft" → company_name: "Microsoft", sector_name: None
- "Tell me about funding rounds for AI startups" → company_name: None, sector_name: "AI startups"
- "Funding rounds for GitHub" → company_name: "GitHub", sector_name: None
- "Funding rounds for healthcare companies" → company_name: None, sector_name: "healthcare companies"

Only identify names that are clearly company names or sector/industry names. Leave fields as null if not mentioned."""

        # Base prompt for parameter extraction
        self._extract_params_prompt = """You are a funding rounds search assistant. Your job is to analyze user queries about funding rounds and extract relevant search parameters.

Rules:
- Only extract parameters that are explicitly mentioned or clearly implied in the query
- For date ranges, convert relative terms (e.g., "last year", "2023") to ISO format dates
- For amounts, convert mentions like "over $1M" to fundraise_amount_usd_min
- Company names have already been resolved to UUIDs - use the provided UUIDs if available
- Set a reasonable limit (default 10) if the user doesn't specify
- Leave parameters as None/null if not mentioned in the query
- Be conservative - only extract what you're confident about"""

        # Register prompts with prompt manager for consistency and potential reuse
        prompt_manager = get_prompt_manager()
        prompt_manager.register_agent_prompt(
            agent_name=f"{self.name}_identify_companies",
            system_prompt=self._identify_companies_prompt,
            overwrite=True,
        )
        prompt_manager.register_agent_prompt(
            agent_name=f"{self.name}_extract_params",
            system_prompt=self._extract_params_prompt,
            overwrite=True,
        )

    @observe(as_type="agent")
    async def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the funding rounds agent to handle funding round-related queries.

        This method:
        1. Analyzes the user query to identify company names mentioned
        2. Resolves company names to UUIDs using semantic_search_organizations
        3. Extracts other search parameters (dates, amounts, investors, stages, etc.)
        4. Calls the get_funding_rounds tool with resolved UUIDs and parameters
        5. Formats the results into a natural language response
        6. Returns a structured response with tool calls tracked

        Args:
            context: The agent context containing the user query and metadata.

        Returns:
            AgentOutput containing:
            - content: Natural language response with funding round information
            - status: SUCCESS if query processed successfully
            - metadata: Information about the search and results
            - tool_calls: List of tool calls made during execution
        """
        try:
            logger.info(f"Funding rounds agent processing query: {context.query[:100]}")

            # Step 1: Identify and resolve company names to UUIDs
            resolved_uuids = await self._resolve_company_names(context)

            # Step 2: Extract other search parameters from the query
            search_params = await self._extract_search_parameters(
                context, resolved_uuids
            )

            # Step 3: Call get_funding_rounds tool with extracted parameters
            # Always include organizations for better insights
            search_params["include_organizations"] = True
            funding_rounds_output = await get_funding_rounds(**search_params)

            # Check if tool execution was successful
            if not funding_rounds_output.success:
                error_msg = (
                    funding_rounds_output.error or "Failed to retrieve funding rounds"
                )
                logger.error(f"get_funding_rounds tool failed: {error_msg}")
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"Failed to retrieve funding rounds: {error_msg}",
                )

            # Extract result from ToolOutput
            funding_rounds = funding_rounds_output.result or []

            # Step 4: Format results into natural language response
            response_content = await self._format_response(
                context, funding_rounds, search_params
            )

            # Build tool calls list
            tool_calls = []

            # Track all tool calls
            if resolved_uuids.get("semantic_search_calls"):
                for call in resolved_uuids["semantic_search_calls"]:
                    tool_calls.append(
                        {
                            "name": "semantic_search_organizations",
                            "parameters": call["parameters"],
                            "result": call["result"],
                        }
                    )
            if resolved_uuids.get("get_organizations_calls"):
                for call in resolved_uuids["get_organizations_calls"]:
                    tool_calls.append(
                        {
                            "name": "get_organizations",
                            "parameters": call["parameters"],
                            "result": call["result"],
                        }
                    )

            tool_calls.append(
                {
                    "name": "get_funding_rounds",
                    "parameters": search_params,
                    "result": {
                        "num_results": len(funding_rounds),
                        "sample_ids": (
                            [fr.get("funding_round_uuid") for fr in funding_rounds[:3]]
                            if funding_rounds
                            else []
                        ),
                        "execution_time_ms": funding_rounds_output.execution_time_ms,
                        "success": funding_rounds_output.success,
                    },
                }
            )

            logger.info(
                f"Funding rounds agent completed: found {len(funding_rounds)} funding round(s)"
            )

            # Return AgentOutput using contract helper
            return create_agent_output(
                content=response_content,
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.SUCCESS,
                tool_calls=tool_calls,
                metadata={
                    "query": context.query,
                    "num_results": len(funding_rounds),
                    "search_parameters": search_params,
                    "resolved_companies": resolved_uuids,
                },
            )

        except Exception as e:
            error_msg = f"Funding rounds agent failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
            )

    async def _resolve_company_names(self, context: AgentContext) -> Dict[str, Any]:
        """Identify company names in the query and resolve them to UUIDs.

        This method uses LLM to identify company names mentioned in the query,
        then uses semantic_search_organizations to find their UUIDs.

        Args:
            context: The agent context containing the user query.

        Returns:
            Dictionary containing:
            - org_uuid: UUID of the organization (if found)
            - semantic_search_calls: List of semantic search tool calls made
        """
        # Use LLM to identify company names and sectors in the query
        identify_companies_tool = {
            "type": "function",
            "function": {
                "name": "identify_company_names",
                "description": "Identify company names and sector/industry names mentioned in the query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company_name": {
                            "type": "string",
                            "description": "Name of a specific company/organization mentioned in the query (e.g., 'Google', 'Microsoft')",
                        },
                        "sector_name": {
                            "type": "string",
                            "description": "Sector or industry name mentioned in the query (e.g., 'AI startups', 'healthcare companies', 'fintech')",
                        },
                    },
                    "required": [],
                },
            },
        }

        # Build system prompt using prompt manager
        prompt_manager = get_prompt_manager()
        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=self._identify_companies_prompt,
            options=PromptOptions(
                add_temporal_context=False, add_markdown_instructions=False
            ),
        )

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Identify any company names mentioned in this query."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        resolved_uuids: Dict[str, Any] = {
            "org_uuid": None,
            "semantic_search_calls": [],
            "get_organizations_calls": [],
        }

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[identify_companies_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "identify_company_names"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "identify_company_names":
                    args = result["arguments"]
                    company_name = args.get("company_name")
                    sector_name = args.get("sector_name")

                    # If specific company name provided, use get_organizations
                    if company_name:
                        try:
                            orgs_output = await get_organizations(
                                name_ilike=company_name,
                                limit=3,  # Get top 3 matches
                            )
                            if orgs_output.success and orgs_output.result:
                                orgs = orgs_output.result
                                if orgs:
                                    # Use the top match - convert UUID to string
                                    org_uuid = orgs[0].get("org_uuid")
                                    resolved_uuids["org_uuid"] = (
                                        str(org_uuid) if org_uuid else None
                                    )
                                    resolved_uuids["get_organizations_calls"].append(
                                        {
                                            "parameters": {
                                                "name_ilike": company_name,
                                                "limit": 3,
                                            },
                                            "result": {
                                                "num_results": len(orgs),
                                                "matched_uuid": resolved_uuids[
                                                    "org_uuid"
                                                ],
                                                "matched_name": orgs[0].get("name"),
                                                "execution_time_ms": orgs_output.execution_time_ms,
                                            },
                                        }
                                    )
                                    logger.info(
                                        f"Resolved organization '{company_name}' to UUID: {resolved_uuids['org_uuid']}"
                                    )
                            else:
                                logger.warning(
                                    f"get_organizations failed for organization '{company_name}': {orgs_output.error}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to resolve organization name '{company_name}': {e}"
                            )

                    # If sector name provided, use semantic search (but don't resolve to UUID - this is for sector queries)
                    if sector_name:
                        try:
                            orgs_output = await semantic_search_organizations(
                                text=sector_name,
                                top_k=10,  # Get more results for sector queries
                            )
                            if orgs_output.success and orgs_output.result:
                                orgs = orgs_output.result
                                resolved_uuids["semantic_search_calls"].append(
                                    {
                                        "parameters": {
                                            "text": sector_name,
                                            "top_k": 10,
                                        },
                                        "result": {
                                            "num_results": len(orgs),
                                            "execution_time_ms": orgs_output.execution_time_ms,
                                        },
                                    }
                                )
                                logger.info(
                                    f"Found {len(orgs)} organizations in sector '{sector_name}'"
                                )
                            else:
                                logger.warning(
                                    f"Semantic search failed for sector '{sector_name}': {orgs_output.error}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to search sector '{sector_name}': {e}"
                            )

        except Exception as e:
            logger.warning(
                f"Failed to identify/resolve company names: {e}", exc_info=True
            )
            # Continue without resolved UUIDs - agent can still search by other criteria

        return resolved_uuids

    async def _extract_search_parameters(
        self, context: AgentContext, resolved_uuids: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract search parameters from user query using LLM function calling.

        Args:
            context: The agent context containing the user query.
            resolved_uuids: Dictionary with resolved organization UUIDs.

        Returns:
            Dictionary of parameters to pass to get_funding_rounds tool.
        """
        # Define the function/tool schema for get_funding_rounds
        get_funding_rounds_tool = {
            "type": "function",
            "function": {
                "name": "get_funding_rounds",
                "description": "Search for funding rounds by various criteria. Extract parameters from the user query to search the funding rounds database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "funding_round_uuid": {
                            "type": "string",
                            "description": "Exact UUID of a specific funding round (if mentioned)",
                        },
                        "org_uuid": {
                            "type": "string",
                            "description": "UUID of the organization that raised the funding",
                        },
                        "investment_date": {
                            "type": "string",
                            "description": "Exact investment date in ISO format (YYYY-MM-DDTHH:MM:SS)",
                        },
                        "investment_date_from": {
                            "type": "string",
                            "description": "Filter funding rounds on or after this date (ISO format)",
                        },
                        "investment_date_to": {
                            "type": "string",
                            "description": "Filter funding rounds on or before this date (ISO format)",
                        },
                        "general_funding_stage": {
                            "type": "string",
                            "description": "General funding stage (e.g., 'seed', 'series_a', 'series_b', 'late_stage_venture', 'ipo')",
                        },
                        "stage": {
                            "type": "string",
                            "description": "Specific funding stage",
                        },
                        "investors_contains": {
                            "type": "string",
                            "description": "Check if investors array contains this value (investor name or UUID)",
                        },
                        "lead_investors_contains": {
                            "type": "string",
                            "description": "Check if lead_investors array contains this value (investor name or UUID)",
                        },
                        "fundraise_amount_usd": {
                            "type": "integer",
                            "description": "Exact fundraise amount in USD",
                        },
                        "fundraise_amount_usd_min": {
                            "type": "integer",
                            "description": "Minimum fundraise amount in USD",
                        },
                        "fundraise_amount_usd_max": {
                            "type": "integer",
                            "description": "Maximum fundraise amount in USD",
                        },
                        "valuation_usd": {
                            "type": "integer",
                            "description": "Exact valuation in USD",
                        },
                        "valuation_usd_min": {
                            "type": "integer",
                            "description": "Minimum valuation in USD",
                        },
                        "valuation_usd_max": {
                            "type": "integer",
                            "description": "Maximum valuation in USD",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10 if not specified)",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Number of results to skip for pagination",
                        },
                        "include_organizations": {
                            "type": "boolean",
                            "description": "Include nested organization details for the company, investors, and lead investors. Should be set to true to get full organization information for generating insights.",
                        },
                        "order_by": {
                            "type": "string",
                            "description": "Field to order results by. Must be one of: 'investment_date', 'fundraise_amount_usd', 'valuation_usd'. Use 'investment_date' for chronological ordering, 'fundraise_amount_usd' for amount-based ordering, 'valuation_usd' for valuation-based ordering. Defaults to 'investment_date' if not specified.",
                            "enum": ["investment_date", "fundraise_amount_usd", "valuation_usd"],
                        },
                        "order_direction": {
                            "type": "string",
                            "description": "Direction to order results. Use 'desc' for descending (newest/highest first), 'asc' for ascending (oldest/lowest first). Defaults to 'desc' if not specified.",
                            "enum": ["asc", "desc"],
                        },
                    },
                    "required": [],
                },
            },
        }

        # Build system prompt using prompt manager
        prompt_manager = get_prompt_manager()
        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=self._extract_params_prompt,
            options=PromptOptions(
                add_temporal_context=False, add_markdown_instructions=False
            ),
        )

        # Build prompt with resolved UUIDs if available
        uuid_info = []
        if resolved_uuids.get("org_uuid"):
            uuid_info.append(
                f"Organization UUID (already resolved): {resolved_uuids['org_uuid']}"
            )

        uuid_context = (
            "\n".join(uuid_info)
            if uuid_info
            else "No company names were identified in the query."
        )

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Resolved Organization UUIDs:
{uuid_context}

Analyze this query and extract the relevant search parameters for finding funding rounds. Use the resolved UUIDs above if they are available. Only include parameters that are clearly mentioned or implied in the query.

Call the get_funding_rounds function with the extracted parameters."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[get_funding_rounds_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,  # Lower temperature for more consistent parameter extraction
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_funding_rounds"},
                },
            )

            # Check if we got a function call result
            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "get_funding_rounds":
                    params = result["arguments"]
                    # Filter out None values and set default limit
                    filtered_params = {k: v for k, v in params.items() if v is not None}

                    # Override with resolved UUIDs if available (they take precedence)
                    if resolved_uuids.get("org_uuid"):
                        filtered_params["org_uuid"] = resolved_uuids["org_uuid"]

                    if "limit" not in filtered_params:
                        filtered_params["limit"] = 10
                    # Always include organizations for better insights
                    filtered_params["include_organizations"] = True
                    return filtered_params
                else:
                    logger.warning(
                        f"Unexpected function call: {result.get('function_name')}, using defaults"
                    )
                    return {"limit": 10}
            else:
                logger.warning(
                    f"LLM did not make expected function call. Got: {type(result)}, using defaults"
                )
                return {"limit": 10}

        except Exception as e:
            logger.warning(
                f"LLM parameter extraction failed: {e}, using defaults", exc_info=True
            )
            return {"limit": 10}

    async def _format_response(
        self,
        context: AgentContext,
        funding_rounds: list[Dict[str, Any]],
        search_params: Dict[str, Any],
    ) -> AgentInsight:
        """Generate domain insight from funding round data using LLM.

        Args:
            context: The agent context.
            funding_rounds: List of funding round records from the database.
            search_params: Parameters used for the search.

        Returns:
            AgentInsight object with domain interpretation and reasoning.
        """
        # Define function schema matching AgentInsight structure
        generate_insight_tool = {
            "type": "function",
            "function": {
                "name": "generate_insight",
                "description": "Generate domain insight from funding round data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Human-readable insight summary answering the user's query",
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Bullet points of key findings",
                        },
                        "evidence": {
                            "type": "object",
                            "description": "Supporting data from tool calls",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in the insight",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }

        # Build system prompt using prompt_manager
        prompt_manager = get_prompt_manager()
        base_prompt = """You are a funding rounds analysis expert. Analyze the funding round data and generate insights that directly answer the user's query.

Your task:
- Identify funding trends over time
- Highlight notable rounds (size, investors, stage)
- Identify patterns (funding velocity, stage progression)
- Compare funding across companies/sectors if multiple results
- State uncertainty where data is missing
- Directly answer the user's query in the summary

If no funding rounds are found, explain why (e.g., search criteria too narrow, no data available for the specified parameters)."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(
                add_temporal_context=False, add_markdown_instructions=False
            ),
        )

        # Build user prompt with data
        user_prompt_content = f"""User Query: {context.query}

Funding Rounds Data (JSON):
{json.dumps(funding_rounds, indent=2, default=str)}

Search Parameters: {json.dumps(search_params, indent=2, default=str)}

Analyze this data and generate insights that directly answer the user's query. Call the generate_insight function with your analysis."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        try:
            # Call LLM with function calling
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[generate_insight_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "generate_insight"},
                },
            )

            # Extract and return AgentInsight
            if isinstance(result, dict) and "function_name" in result:
                args = result["arguments"]
                return AgentInsight(
                    summary=args["summary"],
                    key_findings=args.get("key_findings"),
                    evidence=args.get("evidence"),
                    confidence=args.get("confidence"),
                )
            else:
                # Fallback if LLM doesn't call function
                if funding_rounds:
                    return AgentInsight(
                        summary=f"Found {len(funding_rounds)} funding round(s) but failed to generate insight.",
                        confidence=0.0,
                    )
                else:
                    return AgentInsight(
                        summary=f"I couldn't find any funding rounds matching your criteria. You asked about: {context.query}. Try adjusting your search parameters or broadening your criteria.",
                        confidence=0.0,
                    )
        except Exception as e:
            logger.warning(
                f"Failed to generate insight from LLM: {e}", exc_info=True
            )
            # Fallback insight
            if funding_rounds:
                return AgentInsight(
                    summary=f"Found {len(funding_rounds)} funding round(s) but encountered an error generating insights.",
                    confidence=0.0,
                )
            else:
                return AgentInsight(
                    summary=f"I couldn't find any funding rounds matching your criteria. You asked about: {context.query}. Try adjusting your search parameters or broadening your criteria.",
                    confidence=0.0,
                )

