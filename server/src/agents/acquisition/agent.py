"""Acquisition agent for handling acquisition-related queries.

This agent is specialized in:
- Finding companies that were acquired
- Getting acquisition details and terms
- Analyzing acquisition trends
- Searching acquisitions by various criteria (company names, dates, prices, etc.)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.contracts.agent_io import AgentOutput, create_agent_output
from src.core.agent_response import AgentInsight
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus
from src.prompts.prompt_manager import PromptOptions, get_prompt_manager
from src.tools.generate_llm_function_response import \
    generate_llm_function_response
from src.tools.get_acquisitions import get_acquisitions
from src.tools.get_organizations import get_organizations
from src.tools.semantic_search_organizations import \
    semantic_search_organizations

logger = logging.getLogger(__name__)


class AcquisitionAgent(AgentBase):
    """Agent specialized in handling acquisition-related queries and data retrieval.

    This agent uses LLM reasoning to understand user queries about acquisitions
    and calls the get_acquisitions tool to retrieve relevant data from the database.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the acquisition agent.

        Args:
            config_path: Path to the agent's YAML config file.
                If None, will attempt to find it automatically.
        """
        if config_path is None:
            # Default to acquisition_agent.yaml in configs/agents
            src_base = Path(__file__).parent.parent.parent.parent
            config_path = src_base / "configs" / "agents" / "acquisition_agent.yaml"

        super().__init__(config_path=config_path)

        # Store base prompts as instance variables for reuse
        # Base prompt for company name and sector identification
        self._identify_companies_prompt = """You are analyzing a query about company acquisitions. Your job is to identify:
1. Specific company names mentioned and determine whether they are:
   - The acquirer (the company doing the buying)
   - The acquiree (the company being bought)
2. Sector/industry names mentioned in the query (for semantic search)

IMPORTANT: The semantic_search_organizations tool is designed to query by SECTOR/INDUSTRY names, not specific company names. For specific company names, use get_organizations with name/name_ilike parameter.

Examples:
- "What did Google acquire?" → acquirer_name: "Google", acquiree_name: None, sector_name: None
- "Show me companies acquired by Microsoft" → acquirer_name: "Microsoft", acquiree_name: None, sector_name: None
- "Tell me about the acquisition of GitHub" → acquirer_name: None, acquiree_name: "GitHub", sector_name: None
- "Microsoft's acquisition of GitHub" → acquirer_name: "Microsoft", acquiree_name: "GitHub", sector_name: None
- "Acquisitions in the AI sector" → acquirer_name: None, acquiree_name: None, sector_name: "AI sector"

Only identify names that are clearly company names or sector/industry names. Leave fields as null if not mentioned."""

        # Base prompt for parameter extraction
        self._extract_params_prompt = """You are an acquisition search assistant. Your job is to analyze user queries about company acquisitions and extract relevant search parameters.

Rules:
- Only extract parameters that are explicitly mentioned or clearly implied in the query
- For date ranges, convert relative terms (e.g., "last year", "2023") to ISO format dates
- For prices, convert mentions like "over $1M" to acquisition_price_usd_min
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

    async def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the acquisition agent to handle acquisition-related queries.

        This method:
        1. Analyzes the user query to identify company names mentioned
        2. Resolves company names to UUIDs using semantic_search_organizations
        3. Extracts other search parameters (dates, prices, etc.)
        4. Calls the get_acquisitions tool with resolved UUIDs and parameters
        5. Formats the results into a natural language response
        6. Returns a structured response with tool calls tracked

        Args:
            context: The agent context containing the user query and metadata.

        Returns:
            AgentOutput containing:
            - content: Natural language response with acquisition information
            - status: SUCCESS if query processed successfully
            - metadata: Information about the search and results
            - tool_calls: List of tool calls made during execution
        """
        try:
            logger.info(f"Acquisition agent processing query: {context.query[:100]}")

            # Step 1: Identify and resolve company names to UUIDs
            resolved_uuids = await self._resolve_company_names(context)

            # Step 2: Extract other search parameters from the query
            search_params = await self._extract_search_parameters(
                context, resolved_uuids
            )

            # Step 3: Call get_acquisitions tool with extracted parameters
            # Always include organizations for better insights
            search_params["include_organizations"] = True
            acquisitions_output = await get_acquisitions(**search_params)

            # Check if tool execution was successful
            if not acquisitions_output.success:
                error_msg = (
                    acquisitions_output.error or "Failed to retrieve acquisitions"
                )
                logger.error(f"get_acquisitions tool failed: {error_msg}")
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"Failed to retrieve acquisitions: {error_msg}",
                )

            # Extract result from ToolOutput
            acquisitions = acquisitions_output.result or []

            # Step 4: Format results into natural language response
            response_content = await self._format_response(
                context, acquisitions, search_params
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
                    "name": "get_acquisitions",
                    "parameters": search_params,
                    "result": {
                        "num_results": len(acquisitions),
                        "sample_ids": (
                            [a.get("acquisition_uuid") for a in acquisitions[:3]]
                            if acquisitions
                            else []
                        ),
                        "execution_time_ms": acquisitions_output.execution_time_ms,
                        "success": acquisitions_output.success,
                    },
                }
            )

            logger.info(
                f"Acquisition agent completed: found {len(acquisitions)} acquisition(s)"
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
                    "num_results": len(acquisitions),
                    "search_parameters": search_params,
                    "resolved_companies": resolved_uuids,
                },
            )

        except Exception as e:
            error_msg = f"Acquisition agent failed to process query: {str(e)}"
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
            - acquirer_uuid: UUID of the acquiring company (if found)
            - acquiree_uuid: UUID of the acquired company (if found)
            - semantic_search_calls: List of semantic search tool calls made
        """
        # Use LLM to identify company names and sectors in the query
        identify_companies_tool = {
            "type": "function",
            "function": {
                "name": "identify_company_names",
                "description": "Identify company names mentioned in the query and determine their role (acquirer or acquiree), and identify sector/industry names.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "acquirer_name": {
                            "type": "string",
                            "description": "Name of the company that made/will make the acquisition (the buyer)",
                        },
                        "acquiree_name": {
                            "type": "string",
                            "description": "Name of the company that was/will be acquired (the target)",
                        },
                        "sector_name": {
                            "type": "string",
                            "description": "Sector or industry name mentioned in the query (e.g., 'AI companies', 'healthcare startups', 'fintech')",
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

Identify any company names mentioned in this query and determine if they are the acquirer or acquiree."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        resolved_uuids: Dict[str, Any] = {
            "acquirer_uuid": None,
            "acquiree_uuid": None,
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
                    acquirer_name = args.get("acquirer_name")
                    acquiree_name = args.get("acquiree_name")
                    sector_name = args.get("sector_name")

                    # Search for acquirer if name provided (use get_organizations for specific company names)
                    if acquirer_name:
                        try:
                            orgs_output = await get_organizations(
                                name_ilike=acquirer_name,
                                limit=3,  # Get top 3 matches
                            )
                            if orgs_output.success and orgs_output.result:
                                orgs = orgs_output.result
                                if orgs:
                                    # Use the top match - convert UUID to string
                                    org_uuid = orgs[0].get("org_uuid")
                                    resolved_uuids["acquirer_uuid"] = (
                                        str(org_uuid) if org_uuid else None
                                    )
                                    resolved_uuids["get_organizations_calls"].append(
                                        {
                                            "parameters": {
                                                "name_ilike": acquirer_name,
                                                "limit": 3,
                                            },
                                            "result": {
                                                "num_results": len(orgs),
                                                "matched_uuid": resolved_uuids[
                                                    "acquirer_uuid"
                                                ],
                                                "matched_name": orgs[0].get("name"),
                                                "execution_time_ms": orgs_output.execution_time_ms,
                                            },
                                        }
                                    )
                                    logger.info(
                                        f"Resolved acquirer '{acquirer_name}' to UUID: {resolved_uuids['acquirer_uuid']}"
                                    )
                            else:
                                logger.warning(
                                    f"get_organizations failed for acquirer '{acquirer_name}': {orgs_output.error}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to resolve acquirer name '{acquirer_name}': {e}"
                            )

                    # Search for acquiree if name provided (use get_organizations for specific company names)
                    if acquiree_name:
                        try:
                            orgs_output = await get_organizations(
                                name_ilike=acquiree_name,
                                limit=3,  # Get top 3 matches
                            )
                            if orgs_output.success and orgs_output.result:
                                orgs = orgs_output.result
                                if orgs:
                                    # Use the top match - convert UUID to string
                                    org_uuid = orgs[0].get("org_uuid")
                                    resolved_uuids["acquiree_uuid"] = (
                                        str(org_uuid) if org_uuid else None
                                    )
                                    resolved_uuids["get_organizations_calls"].append(
                                        {
                                            "parameters": {
                                                "name_ilike": acquiree_name,
                                                "limit": 3,
                                            },
                                            "result": {
                                                "num_results": len(orgs),
                                                "matched_uuid": resolved_uuids[
                                                    "acquiree_uuid"
                                                ],
                                                "matched_name": orgs[0].get("name"),
                                                "execution_time_ms": orgs_output.execution_time_ms,
                                            },
                                        }
                                    )
                                    logger.info(
                                        f"Resolved acquiree '{acquiree_name}' to UUID: {resolved_uuids['acquiree_uuid']}"
                                    )
                            else:
                                logger.warning(
                                    f"get_organizations failed for acquiree '{acquiree_name}': {orgs_output.error}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to resolve acquiree name '{acquiree_name}': {e}"
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

        Returns:
            Dictionary of parameters to pass to get_acquisitions tool.
        """
        # Define the function/tool schema for get_acquisitions
        get_acquisitions_tool = {
            "type": "function",
            "function": {
                "name": "get_acquisitions",
                "description": "Search for company acquisitions by various criteria. Extract parameters from the user query to search the acquisitions database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "acquisition_uuid": {
                            "type": "string",
                            "description": "Exact UUID of a specific acquisition (if mentioned)",
                        },
                        "acquiree_uuid": {
                            "type": "string",
                            "description": "UUID of the company that was acquired",
                        },
                        "acquirer_uuid": {
                            "type": "string",
                            "description": "UUID of the company that made the acquisition",
                        },
                        "acquisition_type": {
                            "type": "string",
                            "description": "Type of acquisition (e.g., 'acquisition', 'merger')",
                        },
                        "acquisition_announce_date": {
                            "type": "string",
                            "description": "Exact announce date in ISO format (YYYY-MM-DDTHH:MM:SS)",
                        },
                        "acquisition_announce_date_from": {
                            "type": "string",
                            "description": "Filter acquisitions announced on or after this date (ISO format)",
                        },
                        "acquisition_announce_date_to": {
                            "type": "string",
                            "description": "Filter acquisitions announced on or before this date (ISO format)",
                        },
                        "acquisition_price_usd": {
                            "type": "integer",
                            "description": "Exact acquisition price in USD",
                        },
                        "acquisition_price_usd_min": {
                            "type": "integer",
                            "description": "Minimum acquisition price in USD",
                        },
                        "acquisition_price_usd_max": {
                            "type": "integer",
                            "description": "Maximum acquisition price in USD",
                        },
                        "terms": {
                            "type": "string",
                            "description": "Exact match for acquisition terms",
                        },
                        "terms_ilike": {
                            "type": "string",
                            "description": "Case-insensitive partial match for terms (use for searching)",
                        },
                        "acquirer_type": {
                            "type": "string",
                            "description": "Type of acquirer (e.g., 'company', 'private_equity')",
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
                            "description": "Include nested organization details for acquiree and acquirer. Should be set to true to get full organization information for generating insights.",
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
        if resolved_uuids.get("acquirer_uuid"):
            uuid_info.append(
                f"Acquirer UUID (already resolved): {resolved_uuids['acquirer_uuid']}"
            )
        if resolved_uuids.get("acquiree_uuid"):
            uuid_info.append(
                f"Acquiree UUID (already resolved): {resolved_uuids['acquiree_uuid']}"
            )

        uuid_context = (
            "\n".join(uuid_info)
            if uuid_info
            else "No company names were identified in the query."
        )

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Resolved Company UUIDs:
{uuid_context}

Analyze this query and extract the relevant search parameters for finding acquisitions. Use the resolved UUIDs above if they are available. Only include parameters that are clearly mentioned or implied in the query.

Call the get_acquisitions function with the extracted parameters."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[get_acquisitions_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,  # Lower temperature for more consistent parameter extraction
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_acquisitions"},
                },
            )

            # Check if we got a function call result
            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "get_acquisitions":
                    params = result["arguments"]
                    # Filter out None values and set default limit
                    filtered_params = {k: v for k, v in params.items() if v is not None}

                    # Override with resolved UUIDs if available (they take precedence)
                    if resolved_uuids.get("acquirer_uuid"):
                        filtered_params["acquirer_uuid"] = resolved_uuids[
                            "acquirer_uuid"
                        ]
                    if resolved_uuids.get("acquiree_uuid"):
                        filtered_params["acquiree_uuid"] = resolved_uuids[
                            "acquiree_uuid"
                        ]

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
        acquisitions: list[Dict[str, Any]],
        search_params: Dict[str, Any],
    ) -> AgentInsight:
        """Generate domain insight from acquisition data using LLM.

        Args:
            context: The agent context.
            acquisitions: List of acquisition records from the database.
            search_params: Parameters used for the search.

        Returns:
            AgentInsight object with domain interpretation and reasoning.
        """
        # Define function schema matching AgentInsight structure
        generate_insight_tool = {
            "type": "function",
            "function": {
                "name": "generate_insight",
                "description": "Generate domain insight from acquisition data",
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
        base_prompt = """You are an acquisition analysis expert. Analyze the acquisition data and generate insights that directly answer the user's query.

Your task:
- Identify acquired companies (if data available)
- Group by era/time periods
- Highlight notable deals (price, strategic importance)
- Identify patterns (frequency, deal sizes, types)
- State uncertainty where data is missing
- Directly answer the user's query in the summary

If no acquisitions are found, explain why (e.g., search criteria too narrow, no data available for the specified parameters)."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(
                add_temporal_context=False, add_markdown_instructions=False
            ),
        )

        # Build user prompt with data
        user_prompt_content = f"""User Query: {context.query}

Acquisition Data (JSON):
{json.dumps(acquisitions, indent=2, default=str)}

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
                if acquisitions:
                    return AgentInsight(
                        summary=f"Found {len(acquisitions)} acquisition(s) but failed to generate insight.",
                        confidence=0.0,
                    )
                else:
                    return AgentInsight(
                        summary=f"I couldn't find any acquisitions matching your criteria. You asked about: {context.query}. Try adjusting your search parameters or broadening your criteria.",
                        confidence=0.0,
                    )
        except Exception as e:
            logger.warning(
                f"Failed to generate insight from LLM: {e}", exc_info=True
            )
            # Fallback insight
            if acquisitions:
                return AgentInsight(
                    summary=f"Found {len(acquisitions)} acquisition(s) but encountered an error generating insights.",
                    confidence=0.0,
                )
            else:
                return AgentInsight(
                    summary=f"I couldn't find any acquisitions matching your criteria. You asked about: {context.query}. Try adjusting your search parameters or broadening your criteria.",
                    confidence=0.0,
                )
