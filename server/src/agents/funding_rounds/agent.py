"""Funding rounds agent for handling funding round-related queries.

This agent is specialized in:
- Finding funding rounds for companies
- Getting funding round details (amounts, investors, stages, dates)
- Analyzing funding trends
- Searching funding rounds by various criteria (company names, dates, amounts, investors, stages, etc.)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from src.contracts.agent_io import AgentOutput, create_agent_output
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentInsight, AgentResponse, ResponseStatus
from src.prompts.prompt_manager import PromptOptions, get_prompt_manager
from src.tools.find_investor_portfolio import find_investor_portfolio
from src.tools.generate_llm_function_response import generate_llm_function_response
from src.tools.get_funding_rounds import get_funding_rounds
from src.tools.get_organizations import get_organizations
from src.tools.semantic_search_organizations import semantic_search_organizations

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

            # Prefer pre-extracted intent if available
            extracted = context.get_metadata("extracted_entities", {})
            query_intent = extracted.get("query_intent")

            if query_intent == "find_investor_portfolio":
                return await self._handle_investor_portfolio_query(context)

            # Fallback to LLM classification if intent not provided
            query_type = query_intent or await self._classify_query_type(context)

            if query_type == "investor_portfolio":
                return await self._handle_investor_portfolio_query(context)

            # Step 2: Identify and resolve company names to UUIDs
            resolved_uuids = await self._resolve_company_names(context)

            # Step 3: Extract other search parameters from the query
            search_params = await self._extract_search_parameters(context, resolved_uuids)

            # Step 4: Call get_funding_rounds tool with extracted parameters
            # Always include organizations for better insights
            search_params["include_organizations"] = True

            # Fan-out when multiple companies and no single org_uuid filter
            funding_rounds: list[Dict[str, Any]] = []
            tool_calls: list[Dict[str, Any]] = []
            funding_rounds_output = None

            if (
                resolved_uuids.get("company_pool")
                and len(resolved_uuids["company_pool"]) > 1
                and not search_params.get("org_uuid")
            ):
                funding_rounds, fanout_calls = await self._fanout_funding_rounds(
                    company_pool=resolved_uuids["company_pool"],
                    base_params=search_params,
                )
                tool_calls.extend(fanout_calls)
                # If all fan-out calls failed, return early with error
                if not funding_rounds and all(
                    call.get("result", {}).get("success") is False for call in fanout_calls
                ):
                    return create_agent_output(
                        content="",
                        agent_name=self.name,
                        agent_category=self.category,
                        status=ResponseStatus.ERROR,
                        error="Failed to retrieve funding rounds for all companies in fan-out.",
                    )
            else:
                funding_rounds_output = await get_funding_rounds(**search_params)

                # Check if tool execution was successful
                if not funding_rounds_output.success:
                    error_msg = funding_rounds_output.error or "Failed to retrieve funding rounds"
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

            # Step 5: Format results into natural language response
            response_content = await self._format_response(context, funding_rounds, search_params)

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

            # If we did single-call path, add that call metadata
            if (
                not any(call.get("name") == "get_funding_rounds" for call in tool_calls)
                and funding_rounds_output
            ):
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
        # Use pre-extracted companies if available
        extracted = context.get_metadata("extracted_entities", {})
        companies_data = extracted.get("companies", {})

        # Handle both dict and list formats (defensive coding)
        # Sometimes LLM returns companies as a list instead of dict when validation fails
        if isinstance(companies_data, list):
            # If it's a list, treat it as a list of company names
            logger.warning(
                f"companies_data is a list instead of dict, converting to dict format. "
                f"List: {companies_data}"
            )
            company_names = companies_data if companies_data else []
            company_uuids: List[str] = []
            companies_data = {
                "names": company_names,
                "uuids": company_uuids,
                "context": "",
            }
        elif isinstance(companies_data, dict):
            # Normal case: dict with names, uuids, context
            company_names = companies_data.get("names", [])
            company_uuids = companies_data.get("uuids", [])
        else:
            # Fallback: empty dict structure
            logger.warning(
                f"companies_data is unexpected type: {type(companies_data)}, using empty structure"
            )
            company_names = []
            company_uuids = []
            companies_data = {"names": [], "uuids": [], "context": ""}

        company_pool: list[str] = []

        if company_names or company_uuids:
            logger.info("Using pre-extracted companies for funding rounds agent")
            org_uuid = None

            # Check context to determine if this is a comparison query
            company_context = companies_data.get("context", "").lower()
            is_comparison = any(
                keyword in company_context
                for keyword in [
                    "compar",
                    "compare",
                    "both",
                    "multiple",
                    "versus",
                    "vs",
                    "and",
                ]
            )
            is_directed = any(
                keyword in company_context
                for keyword in ["specific", "single", "one", "the company"]
            )

            # For comparison queries or when multiple companies without explicit single-company context,
            # don't assign org_uuid - use company_pool for fan-out
            should_assign_single = is_directed and not is_comparison

            if is_comparison:
                logger.info(
                    f"Comparison query detected (context: '{company_context}'), "
                    f"using company_pool for fan-out instead of single org_uuid"
                )
            elif should_assign_single:
                logger.info(
                    f"Single company query detected (context: '{company_context}'), "
                    f"assigning org_uuid"
                )

            if company_uuids:
                # Add all UUIDs to company_pool
                for uuid in company_uuids:
                    if uuid:
                        company_pool.append(str(uuid))

                # Only assign single org_uuid if context indicates single company query
                if should_assign_single and len(company_uuids) >= 1:
                    org_uuid = str(company_uuids[0])
            elif company_names:
                # Resolve all company names and add to pool
                for name in company_names:
                    if name:
                        try:
                            orgs_output = await get_organizations(name_ilike=name, limit=3)
                            if orgs_output.success and orgs_output.result:
                                resolved_uuid = str(orgs_output.result[0].get("org_uuid"))
                                if resolved_uuid not in company_pool:
                                    company_pool.append(resolved_uuid)

                                # Only assign to org_uuid if single company query and first name
                                if should_assign_single and not org_uuid:
                                    org_uuid = resolved_uuid
                        except Exception as e:
                            logger.warning(f"Failed to resolve company '{name}': {e}")

            return {
                "org_uuid": org_uuid,
                "company_pool": company_pool,
                "semantic_search_calls": [],
                "get_organizations_calls": [],
            }

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
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Identify any company names mentioned in this query."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        resolved_uuids: Dict[str, Any] = {
            "org_uuid": None,
            "company_pool": [],
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
                                    resolved_uuid_str = str(org_uuid) if org_uuid else None
                                    resolved_uuids["org_uuid"] = resolved_uuid_str
                                    if (
                                        resolved_uuid_str
                                        and resolved_uuid_str not in resolved_uuids["company_pool"]
                                    ):
                                        resolved_uuids["company_pool"].append(resolved_uuid_str)
                                    resolved_uuids["get_organizations_calls"].append(
                                        {
                                            "parameters": {
                                                "name_ilike": company_name,
                                                "limit": 3,
                                            },
                                            "result": {
                                                "num_results": len(orgs),
                                                "matched_uuid": resolved_uuids["org_uuid"],
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
                            logger.warning(f"Failed to search sector '{sector_name}': {e}")

        except Exception as e:
            logger.warning(f"Failed to identify/resolve company names: {e}", exc_info=True)
            # Continue without resolved UUIDs - agent can still search by other criteria

        return resolved_uuids

    async def _fanout_funding_rounds(
        self,
        company_pool: list[str],
        base_params: Dict[str, Any],
        max_concurrency: int = 3,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """Execute get_funding_rounds per company concurrently with bounded fan-out.

        Returns a tuple of (flattened_funding_rounds, tool_calls_metadata).
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        funding_rounds: list[Dict[str, Any]] = []
        tool_calls: list[Dict[str, Any]] = []

        async def _run(company_uuid: str) -> None:
            params = {k: v for k, v in base_params.items() if k != "org_uuid"}
            params["org_uuid"] = company_uuid
            params["include_organizations"] = True

            async with semaphore:
                result = await get_funding_rounds(**params)

            tool_calls.append(
                {
                    "name": "get_funding_rounds",
                    "parameters": params,
                    "result": {
                        "num_results": (len(result.result or []) if result.success else 0),
                        "sample_ids": (
                            [fr.get("funding_round_uuid") for fr in (result.result or [])[:3]]
                            if result.success and result.result
                            else []
                        ),
                        "execution_time_ms": result.execution_time_ms,
                        "success": result.success,
                        "error": result.error,
                    },
                    "source_company_uuid": company_uuid,
                }
            )

            if result.success and result.result:
                funding_rounds.extend(result.result)

        # Deduplicate company UUIDs to avoid double calls
        seen = set()
        unique_pool = []
        for uuid in company_pool:
            if uuid and uuid not in seen:
                seen.add(uuid)
                unique_pool.append(uuid)

        await asyncio.gather(*[_run(uuid) for uuid in unique_pool])

        return funding_rounds, tool_calls

    async def _classify_query_type(self, context: AgentContext) -> str:
        """Use LLM to classify the query type.

        Determines whether the query is asking for:
        - investor_portfolio: Finding companies an investor has funded (portfolio query)
        - funding_rounds: Finding funding rounds for companies or other funding round queries

        Args:
            context: The agent context containing the user query.

        Returns:
            "investor_portfolio" if the query is about finding an investor's portfolio,
            "funding_rounds" otherwise.
        """
        classify_query_tool = {
            "type": "function",
            "function": {
                "name": "classify_funding_query",
                "description": "Classify a funding-related query to determine the appropriate tool to use.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["investor_portfolio", "funding_rounds"],
                            "description": "Type of query: 'investor_portfolio' if asking about companies an investor has funded (e.g., 'What companies has Sequoia invested in?', 'Show me Y Combinator's portfolio'), 'funding_rounds' for queries about funding rounds for companies, funding amounts, stages, dates, etc. (e.g., 'What funding rounds did Google raise?', 'Show me Series A rounds in 2023').",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why this query type was chosen.",
                        },
                    },
                    "required": ["query_type", "reasoning"],
                },
            },
        }

        system_prompt = """You are a query classifier for a funding rounds analysis system. Your job is to determine whether a query is asking for:
1. An investor's portfolio (companies they've invested in) - use "investor_portfolio"
2. Funding rounds information (rounds for companies, amounts, stages, dates) - use "funding_rounds"

Key distinctions:
- "What companies has [investor] invested in?" → investor_portfolio
- "[Investor] portfolio" → investor_portfolio
- "Companies [investor] has funded" → investor_portfolio
- "What funding rounds did [company] raise?" → funding_rounds
- "Show me Series A rounds" → funding_rounds
- "Funding rounds in 2023" → funding_rounds
- "How much did [company] raise?" → funding_rounds

When in doubt, prefer "funding_rounds" as it's the more general case."""

        user_prompt = f"""Classify this query: {context.query}"""

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[classify_query_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.1,  # Low temperature for consistent classification
                tool_choice={
                    "type": "function",
                    "function": {"name": "classify_funding_query"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "classify_funding_query":
                    query_type = result["arguments"].get("query_type", "funding_rounds")
                    reasoning = result["arguments"].get("reasoning", "")
                    logger.debug(f"Query classified as '{query_type}': {reasoning}")
                    return query_type

            # Default to funding_rounds if classification fails
            logger.warning("Query classification failed, defaulting to funding_rounds")
            return "funding_rounds"

        except Exception as e:
            logger.warning(
                f"Query classification error: {e}, defaulting to funding_rounds",
                exc_info=True,
            )
            return "funding_rounds"

    async def _fanout_investor_portfolios(
        self,
        investor_names: list[str],
        base_params: Dict[str, Any],
        max_concurrency: int = 3,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """Execute find_investor_portfolio per investor concurrently with bounded fan-out.

        Returns a tuple of (portfolio_results, tool_calls_metadata).
        Each portfolio_result is a dict with 'investor_name' and 'portfolio_data'.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        portfolio_results: list[Dict[str, Any]] = []
        tool_calls: list[Dict[str, Any]] = []

        async def _run(investor_name: str) -> None:
            params = {
                "investor_name": investor_name,
                "time_period_start": base_params.get("time_period_start"),
                "time_period_end": base_params.get("time_period_end"),
                "include_lead_only": base_params.get("include_lead_only", False),
            }

            async with semaphore:
                result = await find_investor_portfolio(**params)

            tool_calls.append(
                {
                    "name": "find_investor_portfolio",
                    "parameters": params,
                    "result": {
                        "num_companies": (
                            result.result.get("summary", {}).get("total_companies", 0)
                            if result.success and result.result
                            else 0
                        ),
                        "num_investments": (
                            result.result.get("summary", {}).get("total_investments", 0)
                            if result.success and result.result
                            else 0
                        ),
                        "execution_time_ms": result.execution_time_ms,
                        "success": result.success,
                        "error": result.error,
                    },
                    "source_investor_name": investor_name,
                }
            )

            if result.success and result.result:
                portfolio_results.append(
                    {
                        "investor_name": investor_name,
                        "portfolio_data": result.result,
                    }
                )

        # Deduplicate investor names to avoid double calls
        seen = set()
        unique_names = []
        for name in investor_names:
            if name and name.lower() not in seen:
                seen.add(name.lower())
                unique_names.append(name)

        await asyncio.gather(*[_run(name) for name in unique_names])

        return portfolio_results, tool_calls

    async def _handle_multi_investor_portfolio_query(
        self,
        context: AgentContext,
        investor_names: list[str],
        time_period_start: Optional[str],
        time_period_end: Optional[str],
    ) -> AgentOutput:
        """Handle queries about multiple investor portfolios with fan-out.

        Args:
            context: The agent context containing the user query.
            investor_names: List of investor/company names to query.
            time_period_start: Optional start date for filtering.
            time_period_end: Optional end date for filtering.

        Returns:
            AgentOutput with aggregated investor portfolio information.
        """
        base_params = {
            "time_period_start": time_period_start,
            "time_period_end": time_period_end,
            "include_lead_only": False,
        }

        portfolio_results, tool_calls = await self._fanout_investor_portfolios(
            investor_names=investor_names,
            base_params=base_params,
        )

        # If all fan-out calls failed, return early with error
        if not portfolio_results and all(
            call.get("result", {}).get("success") is False for call in tool_calls
        ):
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error="Failed to retrieve investor portfolios for all investors in fan-out.",
            )

        # Format the aggregated portfolio response
        response_content = await self._format_multi_portfolio_response(context, portfolio_results)

        return create_agent_output(
            content=response_content,
            agent_name=self.name,
            agent_category=self.category,
            status=ResponseStatus.SUCCESS,
            tool_calls=tool_calls,
            metadata={
                "query": context.query,
                "investor_names": investor_names,
                "num_investors": len(portfolio_results),
                "portfolio_summaries": [
                    {
                        "investor_name": pr["investor_name"],
                        "summary": pr["portfolio_data"].get("summary", {}),
                    }
                    for pr in portfolio_results
                ],
            },
        )

    async def _handle_investor_portfolio_query(self, context: AgentContext) -> AgentOutput:
        """Handle queries about investor portfolios.

        Args:
            context: The agent context containing the user query.

        Returns:
            AgentOutput with investor portfolio information.
        """
        # Check for pre-extracted companies first
        extracted = context.get_metadata("extracted_entities", {})
        companies_data = extracted.get("companies", {})

        # Handle both dict and list formats (defensive coding)
        if isinstance(companies_data, list):
            company_names = companies_data if companies_data else []
            companies_data = {"names": company_names, "uuids": [], "context": ""}
        elif isinstance(companies_data, dict):
            company_names = companies_data.get("names", [])
        else:
            company_names = []
            companies_data = {"names": [], "uuids": [], "context": ""}

        # Time period from extracted entities
        time_period = extracted.get("time_period", {})
        time_period_start = time_period.get("start")
        time_period_end = time_period.get("end")

        # If multiple companies identified, use fan-out
        if company_names and len(company_names) > 1:
            logger.info(
                f"Multiple investors detected ({len(company_names)}), using fan-out for investor portfolios"
            )
            return await self._handle_multi_investor_portfolio_query(
                context, company_names, time_period_start, time_period_end
            )

        # If single company identified, use it directly
        investor_name = None
        if company_names and len(company_names) == 1:
            investor_name = company_names[0]
            logger.info(f"Using pre-extracted investor name: {investor_name}")

        # Use LLM to extract investor name and parameters if not pre-extracted
        extract_investor_tool = {
            "type": "function",
            "function": {
                "name": "extract_investor_query_params",
                "description": "Extract investor name and parameters from a query about investor portfolios.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "investor_name": {
                            "type": "string",
                            "description": "Name of the investor mentioned in the query (e.g., 'Sequoia Capital', 'Andreessen Horowitz')",
                        },
                        "time_period_start": {
                            "type": "string",
                            "description": "Start date for filtering investments (ISO format, e.g., '2018-01-01T00:00:00'). Leave null if not mentioned.",
                        },
                        "time_period_end": {
                            "type": "string",
                            "description": "End date for filtering investments (ISO format, e.g., '2024-12-31T23:59:59'). Leave null if not mentioned.",
                        },
                        "include_lead_only": {
                            "type": "boolean",
                            "description": "If true, only include rounds where investor was lead. Leave null if not mentioned.",
                        },
                    },
                    "required": ["investor_name"],
                },
            },
        }

        system_prompt = """You are analyzing a query about an investor's portfolio. Extract the investor name and any time period or filtering criteria mentioned in the query.
        
Examples:
- "What companies has Sequoia Capital invested in?" → investor_name: "Sequoia Capital"
- "Show me Andreessen Horowitz's portfolio from 2020 to 2024" → investor_name: "Andreessen Horowitz", time_period_start: "2020-01-01T00:00:00", time_period_end: "2024-12-31T23:59:59"
- "What companies did Y Combinator lead invest in?" → investor_name: "Y Combinator", include_lead_only: true"""

        # If we already have investor_name from pre-extracted entities, skip LLM extraction
        if not investor_name:
            user_prompt = f"""User Query: {context.query}

Extract the investor name and any parameters from this query."""

            try:
                result = await generate_llm_function_response(
                    prompt=user_prompt,
                    tools=[extract_investor_tool],
                    system_prompt=system_prompt,
                    model="gpt-4.1-mini",
                    temperature=0.3,
                    tool_choice={
                        "type": "function",
                        "function": {"name": "extract_investor_query_params"},
                    },
                )

                if isinstance(result, dict) and "function_name" in result:
                    if result["function_name"] == "extract_investor_query_params":
                        args = result["arguments"]
                        investor_name = args.get("investor_name")
                        # Use LLM-extracted time period if not already set from pre-extracted
                        if not time_period_start:
                            time_period_start = args.get("time_period_start")
                        if not time_period_end:
                            time_period_end = args.get("time_period_end")
                else:
                    # LLM extraction failed, try fallback
                    investor_name = None
            except Exception as e:
                logger.warning(
                    f"LLM investor extraction failed: {e}, using pre-extracted if available"
                )

        # Now proceed with single investor query
        if not investor_name:
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error="Could not identify investor name in the query.",
            )

        try:
            # Call find_investor_portfolio tool
            portfolio_output = await find_investor_portfolio(
                investor_name=investor_name,
                time_period_start=time_period_start,
                time_period_end=time_period_end,
                include_lead_only=False,  # Default to False unless specified
            )

            if not portfolio_output.success:
                error_msg = portfolio_output.error or "Failed to retrieve investor portfolio"
                logger.error(f"find_investor_portfolio tool failed: {error_msg}")
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"Failed to retrieve investor portfolio: {error_msg}",
                )

            # Format the portfolio response
            portfolio_data = portfolio_output.result or {}
            response_content = await self._format_portfolio_response(context, portfolio_data)

            return create_agent_output(
                content=response_content,
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.SUCCESS,
                tool_calls=[
                    {
                        "name": "find_investor_portfolio",
                        "parameters": {
                            "investor_name": investor_name,
                            "time_period_start": time_period_start,
                            "time_period_end": time_period_end,
                            "include_lead_only": False,
                        },
                        "result": {
                            "num_companies": portfolio_data.get("summary", {}).get(
                                "total_companies", 0
                            ),
                            "num_investments": portfolio_data.get("summary", {}).get(
                                "total_investments", 0
                            ),
                            "execution_time_ms": portfolio_output.execution_time_ms,
                            "success": portfolio_output.success,
                        },
                    }
                ],
                metadata={
                    "query": context.query,
                    "investor_name": investor_name,
                    "portfolio_summary": portfolio_data.get("summary", {}),
                },
            )

        except Exception as e:
            error_msg = f"Failed to handle investor portfolio query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
            )

        except Exception as e:
            error_msg = f"Failed to handle investor portfolio query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
            )

    async def _format_portfolio_response(
        self, context: AgentContext, portfolio_data: Dict[str, Any]
    ) -> AgentInsight:
        """Generate domain insight from investor portfolio data using LLM.

        Args:
            context: The agent context.
            portfolio_data: Portfolio data from find_investor_portfolio tool.

        Returns:
            AgentInsight object with domain interpretation and reasoning.
        """
        generate_insight_tool = {
            "type": "function",
            "function": {
                "name": "generate_insight",
                "description": "Generate domain insight from investor portfolio data",
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

        prompt_manager = get_prompt_manager()
        base_prompt = """You are an investor portfolio analysis expert. Analyze the investor portfolio data and generate insights that directly answer the user's query.

Your task:
- Summarize the investor's portfolio (number of companies, investments, capital deployed)
- Highlight notable portfolio companies (by funding, stage, or other metrics)
- Identify investment patterns (stages, sectors, time periods)
- Compare portfolio size and activity if time periods are specified
- State uncertainty where data is missing
- Directly answer the user's query in the summary

If no portfolio companies are found, explain why (e.g., investor name not found, no investments in specified time period)."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        user_prompt_content = f"""User Query: {context.query}

Investor Portfolio Data (JSON):
{json.dumps(portfolio_data, indent=2, default=str)}

Analyze this data and generate insights that directly answer the user's query. Call the generate_insight function with your analysis."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        try:
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

            if isinstance(result, dict) and "function_name" in result:
                args = result["arguments"]
                return AgentInsight(
                    summary=args["summary"],
                    key_findings=args.get("key_findings"),
                    evidence=args.get("evidence"),
                    confidence=args.get("confidence"),
                )
            else:
                # Fallback
                summary = portfolio_data.get("summary", {})
                num_companies = summary.get("total_companies", 0)
                if num_companies > 0:
                    return AgentInsight(
                        summary=f"Found {num_companies} portfolio company/companies for {portfolio_data.get('investor_name', 'the investor')}.",
                        confidence=0.7,
                    )
                else:
                    return AgentInsight(
                        summary=f"Could not find any portfolio companies for {portfolio_data.get('investor_name', 'the specified investor')}.",
                        confidence=0.0,
                    )
        except Exception as e:
            logger.warning(f"Failed to generate portfolio insight from LLM: {e}", exc_info=True)
            summary = portfolio_data.get("summary", {})
            num_companies = summary.get("total_companies", 0)
            if num_companies > 0:
                return AgentInsight(
                    summary=f"Found {num_companies} portfolio company/companies for {portfolio_data.get('investor_name', 'the investor')}.",
                    confidence=0.7,
                )
            else:
                return AgentInsight(
                    summary=f"Could not find any portfolio companies for {portfolio_data.get('investor_name', 'the specified investor')}.",
                    confidence=0.0,
                )

    async def _format_multi_portfolio_response(
        self,
        context: AgentContext,
        portfolio_results: list[Dict[str, Any]],
    ) -> AgentInsight:
        """Generate domain insight from multiple investor portfolio data using LLM.

        Args:
            context: The agent context.
            portfolio_results: List of dicts with 'investor_name' and 'portfolio_data'.

        Returns:
            AgentInsight object with domain interpretation and reasoning.
        """
        generate_insight_tool = {
            "type": "function",
            "function": {
                "name": "generate_insight",
                "description": "Generate domain insight from multiple investor portfolio data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Human-readable insight summary answering the user's query, comparing portfolios across investors",
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Bullet points of key findings comparing the portfolios",
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

        prompt_manager = get_prompt_manager()
        base_prompt = """You are an investor portfolio analysis expert. Analyze multiple investor portfolio data and generate insights that directly answer the user's query.

Your task:
- Compare portfolios across all investors (number of companies, investments, capital deployed)
- Highlight notable portfolio companies for each investor (by funding, stage, or other metrics)
- Identify investment patterns and differences between investors (stages, sectors, time periods, investment sizes)
- Compare portfolio sizes, activity levels, and strategic focus
- State uncertainty where data is missing
- Directly answer the user's query in the summary, providing a comprehensive comparison

If no portfolio companies are found for any investor, explain why (e.g., investor name not found, no investments in specified time period)."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        user_prompt_content = f"""User Query: {context.query}

Investor Portfolio Data (JSON):
{json.dumps(portfolio_results, indent=2, default=str)}

Analyze this data and generate insights that directly answer the user's query. Compare the portfolios across all investors. Call the generate_insight function with your analysis."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        try:
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

            if isinstance(result, dict) and "function_name" in result:
                args = result["arguments"]
                return AgentInsight(
                    summary=args["summary"],
                    key_findings=args.get("key_findings"),
                    evidence=args.get("evidence"),
                    confidence=args.get("confidence"),
                )
            else:
                # Fallback
                total_companies = sum(
                    pr["portfolio_data"].get("summary", {}).get("total_companies", 0)
                    for pr in portfolio_results
                )
                investor_names = [pr["investor_name"] for pr in portfolio_results]
                if total_companies > 0:
                    return AgentInsight(
                        summary=f"Found portfolios for {len(portfolio_results)} investor(s): {', '.join(investor_names)}. Total portfolio companies: {total_companies}.",
                        confidence=0.7,
                    )
                else:
                    return AgentInsight(
                        summary=f"Could not find any portfolio companies for the specified investors: {', '.join(investor_names)}.",
                        confidence=0.0,
                    )
        except Exception as e:
            logger.warning(
                f"Failed to generate multi-portfolio insight from LLM: {e}",
                exc_info=True,
            )
            total_companies = sum(
                pr["portfolio_data"].get("summary", {}).get("total_companies", 0)
                for pr in portfolio_results
            )
            investor_names = [pr["investor_name"] for pr in portfolio_results]
            if total_companies > 0:
                return AgentInsight(
                    summary=f"Found portfolios for {len(portfolio_results)} investor(s): {', '.join(investor_names)}. Total portfolio companies: {total_companies}.",
                    confidence=0.7,
                )
            else:
                return AgentInsight(
                    summary=f"Could not find any portfolio companies for the specified investors: {', '.join(investor_names)}.",
                    confidence=0.0,
                )

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
        # First, use pre-extracted metadata when available to avoid redundant LLM calls
        extracted = context.get_metadata("extracted_entities", {})
        search_params: Dict[str, Any] = {}

        # Use resolved org uuid if present
        if resolved_uuids.get("org_uuid"):
            search_params["org_uuid"] = resolved_uuids["org_uuid"]

        # Time period
        time_period = extracted.get("time_period", {})
        if time_period.get("start"):
            search_params["investment_date_from"] = time_period["start"]
        if time_period.get("end"):
            search_params["investment_date_to"] = time_period["end"]

        # Amounts
        amounts = extracted.get("amounts", {})
        if amounts.get("fundraise_min") is not None:
            search_params["fundraise_amount_usd_min"] = amounts["fundraise_min"]
        if amounts.get("fundraise_max") is not None:
            search_params["fundraise_amount_usd_max"] = amounts["fundraise_max"]
        if amounts.get("valuation_min") is not None:
            search_params["valuation_usd_min"] = amounts["valuation_min"]
        if amounts.get("valuation_max") is not None:
            search_params["valuation_usd_max"] = amounts["valuation_max"]

        # Funding stages
        funding_stages = extracted.get("funding_stages", [])
        if funding_stages:
            search_params["general_funding_stage"] = funding_stages[0]

        # Investors
        investors = extracted.get("investors", {})
        investor_names = investors.get("names", [])
        if investor_names:
            search_params["investors_contains"] = investor_names[0]

        # Apply defaults if we gathered any params from metadata
        if search_params:
            if "limit" not in search_params:
                search_params["limit"] = 10
            return search_params

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
                            "enum": [
                                "investment_date",
                                "fundraise_amount_usd",
                                "valuation_usd",
                            ],
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
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Build prompt with resolved UUIDs if available
        uuid_info = []
        if resolved_uuids.get("org_uuid"):
            uuid_info.append(f"Organization UUID (already resolved): {resolved_uuids['org_uuid']}")

        uuid_context = (
            "\n".join(uuid_info) if uuid_info else "No company names were identified in the query."
        )

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Resolved Organization UUIDs:
{uuid_context}

Analyze this query and extract the relevant search parameters for finding funding rounds. Use the resolved UUIDs above if they are available. Only include parameters that are clearly mentioned or implied in the query.

Call the get_funding_rounds function with the extracted parameters."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
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
                    # Only use resolved UUID if it's a valid UUID string
                    resolved_uuid = resolved_uuids.get("org_uuid")
                    if resolved_uuid:
                        try:
                            # Validate that it's a valid UUID string
                            UUID(resolved_uuid)
                            filtered_params["org_uuid"] = resolved_uuid
                        except (ValueError, TypeError):
                            # Invalid UUID, log warning and skip it
                            logger.warning(
                                f"Invalid UUID from resolved_uuids: {resolved_uuid}, skipping"
                            )

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
            logger.warning(f"LLM parameter extraction failed: {e}, using defaults", exc_info=True)
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
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Build user prompt with data
        user_prompt_content = f"""User Query: {context.query}

Funding Rounds Data (JSON):
{json.dumps(funding_rounds, indent=2, default=str)}

Search Parameters: {json.dumps(search_params, indent=2, default=str)}

Analyze this data and generate insights that directly answer the user's query. Call the generate_insight function with your analysis."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
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
            logger.warning(f"Failed to generate insight from LLM: {e}", exc_info=True)
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
