"""Organizations agent for handling organization-related queries.

This agent is specialized in:
- Finding organizations by name, domain, location, or other criteria
- Getting organization details (funding, stage, status, etc.)
- Analyzing organization trends
- Searching organizations by various criteria (names, locations, funding, dates, categories, etc.)
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
from src.tools.get_organizations import get_organizations
from src.tools.semantic_search_organizations import \
    semantic_search_organizations

logger = logging.getLogger(__name__)


class OrganizationsAgent(AgentBase):
    """Agent specialized in handling organization-related queries and data retrieval.

    This agent uses LLM reasoning to understand user queries about organizations
    and calls the get_organizations or semantic_search_organizations tools to retrieve
    relevant data from the database.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the organizations agent.

        Args:
            config_path: Path to the agent's YAML config file.
                If None, will attempt to find it automatically.
        """
        if config_path is None:
            # Default to organizations_agent.yaml in configs/agents
            src_base = Path(__file__).parent.parent.parent.parent
            config_path = src_base / "configs" / "agents" / "organizations_agent.yaml"

        super().__init__(config_path=config_path)

        # Store base prompts as instance variables for reuse
        # Base prompt for determining search strategy
        self._determine_strategy_prompt = """You are analyzing a query about organizations. Your job is to determine the best search strategy:

IMPORTANT: The semantic_search_organizations tool is designed to query by SECTOR/INDUSTRY names (e.g., "AI companies", "healthcare startups", "fintech"). It should NOT be used for specific company name searches.

- Use semantic search ONLY if the query asks about companies in a specific SECTOR or INDUSTRY (e.g., "AI companies", "healthcare startups", "fintech companies", "closed-source LLM providers")
- Use structured search (get_organizations) if:
  - The query mentions a specific company name (use name or name_ilike parameter)
  - The query contains specific filters (exact names, dates, amounts, locations, etc.)
  - The query is about finding a specific organization by name

Examples:
- "Find AI companies" → semantic search (sector query)
- "Find companies in the healthcare sector" → semantic search (sector query)
- "Show me Google" → structured search (specific company name)
- "Show me companies in San Francisco" → structured search (location filter)
- "Companies with funding over $10M" → structured search (amount filter)
- "Tech startups in healthcare" → semantic search (sector query)
- "What is OpenAI?" → structured search (specific company name)
 
Return the search strategy, sector name (if applicable), and any company names mentioned."""

        # Base prompt for parameter extraction
        self._extract_params_prompt = """You are an organization search assistant. Your job is to analyze user queries about organizations and extract relevant search parameters.

Rules:
- Only extract parameters that are explicitly mentioned or clearly implied in the query
- For date ranges, convert relative terms (e.g., "last year", "2023") to ISO format dates
- For amounts, convert mentions like "over $1M" to total_funding_usd_min
- Company names can be used for semantic search or exact name matching
- Set a reasonable limit (default 10) if the user doesn't specify
- Leave parameters as None/null if not mentioned in the query
- Be conservative - only extract what you're confident about"""

        # Register prompts with prompt manager for consistency and potential reuse
        prompt_manager = get_prompt_manager()
        prompt_manager.register_agent_prompt(
            agent_name=f"{self.name}_determine_strategy",
            system_prompt=self._determine_strategy_prompt,
            overwrite=True,
        )
        prompt_manager.register_agent_prompt(
            agent_name=f"{self.name}_extract_params",
            system_prompt=self._extract_params_prompt,
            overwrite=True,
        )

    @observe(as_type="agent")
    async def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the organizations agent to handle organization-related queries.

        This method:
        1. Determines whether to use semantic search or structured search
        2. If semantic search: uses semantic_search_organizations with the query text
        3. If structured search: extracts parameters and calls get_organizations
        4. Formats the results into a natural language response
        5. Returns a structured response with tool calls tracked

        Args:
            context: The agent context containing the user query and metadata.

        Returns:
            AgentOutput containing:
            - content: Natural language response with organization information
            - status: SUCCESS if query processed successfully
            - metadata: Information about the search and results
            - tool_calls: List of tool calls made during execution
        """
        try:
            logger.info(f"Organizations agent processing query: {context.query[:100]}")

            # Step 1: Determine search strategy
            strategy = await self._determine_search_strategy(context)

            tool_calls = []
            organizations = []

            if strategy.get("use_semantic_search"):
                # Step 2a: Use semantic search
                semantic_params = strategy.get("semantic_params", {})
                orgs_output = await semantic_search_organizations(**semantic_params)

                if not orgs_output.success:
                    error_msg = (
                        orgs_output.error or "Failed to retrieve organizations"
                    )
                    logger.error(f"semantic_search_organizations tool failed: {error_msg}")
                    return create_agent_output(
                        content="",
                        agent_name=self.name,
                        agent_category=self.category,
                        status=ResponseStatus.ERROR,
                        error=f"Failed to retrieve organizations: {error_msg}",
                    )

                organizations = orgs_output.result or []
                tool_calls.append(
                    {
                        "name": "semantic_search_organizations",
                        "parameters": semantic_params,
                        "result": {
                            "num_results": len(organizations),
                            "execution_time_ms": orgs_output.execution_time_ms,
                            "success": orgs_output.success,
                        },
                    }
                )
            else:
                # Step 2b: Use structured search
                # If company name was identified, include it in search params
                company_name = strategy.get("company_name")
                search_params = await self._extract_search_parameters(context, company_name)
                orgs_output = await get_organizations(**search_params)

                if not orgs_output.success:
                    error_msg = (
                        orgs_output.error or "Failed to retrieve organizations"
                    )
                    logger.error(f"get_organizations tool failed: {error_msg}")
                    return create_agent_output(
                        content="",
                        agent_name=self.name,
                        agent_category=self.category,
                        status=ResponseStatus.ERROR,
                        error=f"Failed to retrieve organizations: {error_msg}",
                    )

                organizations = orgs_output.result or []
                tool_calls.append(
                    {
                        "name": "get_organizations",
                        "parameters": search_params,
                        "result": {
                            "num_results": len(organizations),
                            "sample_ids": (
                                [org.get("org_uuid") for org in organizations[:3]]
                                if organizations
                                else []
                            ),
                            "execution_time_ms": orgs_output.execution_time_ms,
                            "success": orgs_output.success,
                        },
                    }
                )

            # Step 3: Format results into natural language response
            response_content = await self._format_response(
                context, organizations, strategy
            )

            logger.info(
                f"Organizations agent completed: found {len(organizations)} organization(s)"
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
                    "num_results": len(organizations),
                    "search_strategy": strategy.get("strategy", "unknown"),
                },
            )

        except Exception as e:
            error_msg = f"Organizations agent failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
            )

    async def _determine_search_strategy(self, context: AgentContext) -> Dict[str, Any]:
        """Determine whether to use semantic search or structured search.

        Args:
            context: The agent context containing the user query.

        Returns:
            Dictionary containing:
            - use_semantic_search: Boolean indicating if semantic search should be used
            - strategy: String describing the strategy
            - semantic_params: Parameters for semantic search (if applicable)
        """
        # Use pre-extracted metadata when available
        extracted = context.get_metadata("extracted_entities", {})
        sectors = extracted.get("sectors", [])
        companies = extracted.get("companies", {}).get("names", [])

        if sectors:
            return {
                "use_semantic_search": True,
                "strategy": "semantic",
                "semantic_params": {"text": sectors[0], "top_k": 10},
            }

        if companies:
            return {
                "use_semantic_search": False,
                "strategy": "structured",
                "company_name": companies[0],
            }

        # Use LLM to determine search strategy
        determine_strategy_tool = {
            "type": "function",
            "function": {
                "name": "determine_search_strategy",
                "description": "Determine whether to use semantic search (for sector/industry queries) or structured search (for company names or specific filters).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "use_semantic_search": {
                            "type": "boolean",
                            "description": "True if semantic search should be used (for sector/industry queries), False for structured search (for company names or specific filters)",
                        },
                        "sector_name": {
                            "type": "string",
                            "description": "Sector or industry name for semantic search (e.g., 'AI companies', 'healthcare startups', 'fintech'). Only provide if use_semantic_search is true.",
                        },
                        "company_name": {
                            "type": "string",
                            "description": "Specific company name mentioned in the query. Only provide if a specific company name is mentioned.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results for semantic search (default: 10)",
                        },
                    },
                    "required": ["use_semantic_search"],
                },
            },
        }

        # Build system prompt using prompt manager
        prompt_manager = get_prompt_manager()
        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=self._determine_strategy_prompt,
            options=PromptOptions(
                add_temporal_context=False, add_markdown_instructions=False
            ),
        )

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Determine the best search strategy for this query."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[determine_strategy_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "determine_search_strategy"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "determine_search_strategy":
                    args = result["arguments"]
                    use_semantic = args.get("use_semantic_search", False)
                    sector_name = args.get("sector_name")
                    company_name = args.get("company_name")
                    top_k = args.get("top_k", 10)

                    if use_semantic:
                        # Use sector name if provided, otherwise fall back to query
                        search_text = sector_name or context.query
                        return {
                            "use_semantic_search": True,
                            "strategy": "semantic",
                            "semantic_params": {
                                "text": search_text,
                                "top_k": top_k,
                            },
                        }
                    else:
                        # For structured search, include company name if identified
                        strategy_info = {
                            "use_semantic_search": False,
                            "strategy": "structured",
                        }
                        if company_name:
                            strategy_info["company_name"] = company_name
                        return strategy_info
        except Exception as e:
            logger.warning(
                f"Failed to determine search strategy: {e}, defaulting to structured search",
                exc_info=True,
            )

        # Default to structured search
        return {
            "use_semantic_search": False,
            "strategy": "structured",
        }

    async def _extract_search_parameters(
        self, context: AgentContext, company_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract search parameters from user query using LLM function calling.

        Args:
            context: The agent context containing the user query.

        Returns:
            Dictionary of parameters to pass to get_organizations tool.
        """
        # Use pre-extracted metadata when available
        extracted = context.get_metadata("extracted_entities", {})
        search_params: Dict[str, Any] = {}

        # Company name / sectors
        companies = extracted.get("companies", {}).get("names", [])
        sectors = extracted.get("sectors", [])

        if company_name:
            search_params["name_ilike"] = company_name
        elif companies:
            search_params["name_ilike"] = companies[0]

        if sectors:
            search_params["categories_contains"] = sectors[0]

        # Time period (founding dates) and funding amounts
        time_period = extracted.get("time_period", {})
        if time_period.get("start"):
            search_params["founding_date_from"] = time_period["start"]
        if time_period.get("end"):
            search_params["founding_date_to"] = time_period["end"]

        amounts = extracted.get("amounts", {})
        if amounts.get("fundraise_min") is not None:
            search_params["total_funding_usd_min"] = amounts["fundraise_min"]
        if amounts.get("fundraise_max") is not None:
            search_params["total_funding_usd_max"] = amounts["fundraise_max"]

        if search_params:
            if "limit" not in search_params:
                search_params["limit"] = 10
            return search_params

        # Define the function/tool schema for get_organizations (simplified version)
        # Note: get_organizations has many parameters, so we'll include the most common ones
        get_organizations_tool = {
            "type": "function",
            "function": {
                "name": "get_organizations",
                "description": "Search for organizations by various criteria. Extract parameters from the user query to search the organizations database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "org_uuid": {
                            "type": "string",
                            "description": "Exact UUID of a specific organization (if mentioned)",
                        },
                        "name": {
                            "type": "string",
                            "description": "Exact match for organization name",
                        },
                        "name_ilike": {
                            "type": "string",
                            "description": "Case-insensitive partial match for organization name",
                        },
                        "org_domain": {
                            "type": "string",
                            "description": "Exact match for organization domain",
                        },
                        "org_domain_ilike": {
                            "type": "string",
                            "description": "Case-insensitive partial match for organization domain",
                        },
                        "city": {
                            "type": "string",
                            "description": "Exact match for city",
                        },
                        "state": {
                            "type": "string",
                            "description": "Exact match for state",
                        },
                        "country": {
                            "type": "string",
                            "description": "Exact match for country",
                        },
                        "continent": {
                            "type": "string",
                            "description": "Exact match for continent",
                        },
                        "categories_contains": {
                            "type": "string",
                            "description": "Check if categories array contains this value",
                        },
                        "total_funding_usd_min": {
                            "type": "integer",
                            "description": "Filter with total_funding_usd >= this value",
                        },
                        "total_funding_usd_max": {
                            "type": "integer",
                            "description": "Filter with total_funding_usd <= this value",
                        },
                        "founding_date_from": {
                            "type": "string",
                            "description": "Filter organizations founded on or after this date (ISO format)",
                        },
                        "founding_date_to": {
                            "type": "string",
                            "description": "Filter organizations founded on or before this date (ISO format)",
                        },
                        "stage": {
                            "type": "string",
                            "description": "Exact match for stage",
                        },
                        "general_funding_stage": {
                            "type": "string",
                            "description": "Exact match for general funding stage",
                        },
                        "org_status": {
                            "type": "string",
                            "description": "Exact match for organization status",
                        },
                        "org_type": {
                            "type": "string",
                            "description": "Exact match for organization type",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10 if not specified)",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Number of results to skip for pagination",
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

        # Build user prompt with company name context if available
        company_context = ""
        if company_name:
            company_context = f"\n\nNote: A company name '{company_name}' was identified in the query. Use the 'name' or 'name_ilike' parameter to search for this company."

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}{company_context}

Analyze this query and extract the relevant search parameters for finding organizations. Only include parameters that are clearly mentioned or implied in the query.

Call the get_organizations function with the extracted parameters."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[get_organizations_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,  # Lower temperature for more consistent parameter extraction
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_organizations"},
                },
            )

            # Check if we got a function call result
            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "get_organizations":
                    params = result["arguments"]
                    # Filter out None values and set default limit
                    filtered_params = {k: v for k, v in params.items() if v is not None}

                    # If company name was identified but not in params, add it
                    if company_name and "name" not in filtered_params and "name_ilike" not in filtered_params:
                        filtered_params["name_ilike"] = company_name

                    if "limit" not in filtered_params:
                        filtered_params["limit"] = 10
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
        organizations: list[Dict[str, Any]],
        strategy: Dict[str, Any],
    ) -> AgentInsight:
        """Generate domain insight from organization data using LLM.

        Args:
            context: The agent context.
            organizations: List of organization records from the database.
            strategy: The search strategy used.

        Returns:
            AgentInsight object with domain interpretation and reasoning.
        """
        # Define function schema matching AgentInsight structure
        generate_insight_tool = {
            "type": "function",
            "function": {
                "name": "generate_insight",
                "description": "Generate domain insight from organization data",
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
        base_prompt = """You are an organization analysis expert. Analyze the organization data and generate insights that directly answer the user's query.

Your task:
- Summarize organization characteristics
- Identify patterns (location, stage, funding)
- Highlight notable organizations
- Compare organizations if multiple results
- State uncertainty where data is missing
- Directly answer the user's query in the summary

If no organizations are found, explain why (e.g., search criteria too narrow, no data available for the specified parameters)."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(
                add_temporal_context=False, add_markdown_instructions=False
            ),
        )

        # Build user prompt with data
        user_prompt_content = f"""User Query: {context.query}

Organization Data (JSON):
{json.dumps(organizations, indent=2, default=str)}

Search Strategy: {json.dumps(strategy, indent=2, default=str)}

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
                if organizations:
                    return AgentInsight(
                        summary=f"Found {len(organizations)} organization(s) but failed to generate insight.",
                        confidence=0.0,
                    )
                else:
                    return AgentInsight(
                        summary=f"I couldn't find any organizations matching your criteria. You asked about: {context.query}. Try adjusting your search parameters or broadening your criteria.",
                        confidence=0.0,
                    )
        except Exception as e:
            logger.warning(
                f"Failed to generate insight from LLM: {e}", exc_info=True
            )
            # Fallback insight
            if organizations:
                return AgentInsight(
                    summary=f"Found {len(organizations)} organization(s) but encountered an error generating insights.",
                    confidence=0.0,
                )
            else:
                return AgentInsight(
                    summary=f"I couldn't find any organizations matching your criteria. You asked about: {context.query}. Try adjusting your search parameters or broadening your criteria.",
                    confidence=0.0,
                )

