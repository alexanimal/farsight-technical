"""Query enrichment service for improving queries and extracting metadata.

This service uses LLM function calling to:
1. Improve/enrich user queries (especially for multi-turn conversations)
2. Extract structured metadata (companies, sectors, investors, time periods, etc.)
3. Identify query intent

The service is designed to be called at the workflow level before agents execute,
allowing agents to skip redundant LLM calls for metadata extraction.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.prompts.prompt_manager import PromptOptions, get_prompt_manager
from src.tools.get_organizations import get_organizations
from src.tools.generate_llm_function_response import generate_llm_function_response

logger = logging.getLogger(__name__)


class ExtractedQueryMetadata(BaseModel):
    """Structured metadata extracted from user query."""

    improved_query: str = Field(
        default="",
        description="Improved/enriched version of the query that resolves pronouns and adds context from conversation history",
    )
    original_query: str = Field(
        default="", description="Original user query as provided"
    )
    companies: Dict[str, Any] = Field(
        default_factory=dict,
        description="Companies mentioned in the query. Structure: {names: List[str], uuids: List[str], context: str}",
    )
    sectors: List[str] = Field(
        default_factory=list,
        description="Sector/industry names mentioned in the query (e.g., 'AI companies', 'fintech', 'healthcare startups')",
    )
    investors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Investors mentioned in the query. Structure: {names: List[str]}. Used for investor portfolio queries.",
    )
    time_period: Dict[str, Any] = Field(
        default_factory=dict,
        description="Time period mentioned in the query. Structure: {start: str (ISO format), end: str (ISO format), granularity: str (monthly/quarterly/yearly)}",
    )
    amounts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Amounts mentioned in the query. Structure: {fundraise_min: int, fundraise_max: int, valuation_min: int, valuation_max: int, acquisition_price_min: int, acquisition_price_max: int}",
    )
    funding_stages: List[str] = Field(
        default_factory=list,
        description="Funding stages mentioned in the query (e.g., 'seed', 'series_a', 'series_b', 'late_stage_venture', 'ipo')",
    )
    query_intent: Optional[str] = Field(
        default=None,
        description="Query intent classification. One of: 'find_funding_rounds', 'find_investor_portfolio', 'find_acquisitions', 'find_organizations', 'analyze_sector_trends'",
    )
    reasoning: str = Field(
        default="",
        description="Reasoning for query improvement and metadata extraction decisions",
    )


class QueryEnrichmentService:
    """Service for enriching user queries and extracting structured metadata."""

    def __init__(self):
        """Initialize the query enrichment service."""
        self._base_prompt = """You are a query enrichment and metadata extraction assistant. Your job is to:

1. **Improve the query**: 
   - Resolve pronouns (e.g., "it", "they", "that company") using conversation history
   - Add missing context from previous messages
   - Make the query more specific and clear
   - Preserve the original intent and meaning

2. **Extract structured metadata**:
   - Identify company names mentioned (extract as list)
   - Identify sector/industry names mentioned
   - Identify investor names mentioned (for investor portfolio queries)
   - Extract time periods (convert relative terms like "last year" to ISO dates)
   - Extract funding amounts (convert mentions like "over $1M" to numeric values)
   - Identify funding stages mentioned
   - Classify query intent

Rules:
- Only extract information that is explicitly mentioned or clearly implied
- For time periods, convert relative terms to ISO format dates (YYYY-MM-DDTHH:MM:SS)
- For amounts, convert to integers (USD)
- Be conservative - only extract what you're confident about
- Leave fields as empty/null if not mentioned
- Query intent should be one of: find_funding_rounds, find_investor_portfolio, find_acquisitions, find_organizations, analyze_sector_trends"""

    async def enrich_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Enrich query and extract metadata.

        This method uses a single LLM call to both improve the query text
        and extract structured metadata. If query improvement succeeds but
        metadata extraction fails, the improved query is still returned with
        empty metadata.

        Args:
            query: The user's query to enrich.
            conversation_history: Optional list of previous messages in the conversation.
                Each message should be a dict with 'role' and 'content' keys.

        Returns:
            Dictionary containing:
            - improved_query: str - The improved/enriched query
            - original_query: str - The original query
            - metadata: Dict - ExtractedQueryMetadata as a dictionary

        Raises:
            Exception: If enrichment completely fails, returns original query with empty metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided, returning as-is")
            return {
                "improved_query": query,
                "original_query": query,
                "metadata": ExtractedQueryMetadata(
                    improved_query=query, original_query=query
                ).model_dump(),
            }

        try:
            # Build function schema that matches ExtractedQueryMetadata
            extract_metadata_tool = {
                "type": "function",
                "function": {
                    "name": "extract_query_metadata",
                    "description": "Extract structured metadata from user query and improve the query text using conversation history.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "improved_query": {
                                "type": "string",
                                "description": "Improved/enriched version of the query that resolves pronouns and adds context from conversation history",
                            },
                            "original_query": {
                                "type": "string",
                                "description": "Original user query as provided",
                            },
                            "companies": {
                                "type": "object",
                                "description": "Companies mentioned in the query",
                                "properties": {
                                    "names": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of company names mentioned",
                                    },
                                    "uuids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of company UUIDs if already known (optional, can be empty)",
                                    },
                                    "context": {
                                        "type": "string",
                                        "description": "Context about how companies are mentioned (e.g., 'acquirer', 'acquiree', 'target company')",
                                    },
                                },
                            },
                            "sectors": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Sector/industry names mentioned (e.g., 'AI companies', 'fintech', 'healthcare startups')",
                            },
                            "investors": {
                                "type": "object",
                                "description": "Investors mentioned in the query (for investor portfolio queries)",
                                "properties": {
                                    "names": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of investor names mentioned",
                                    },
                                },
                            },
                            "time_period": {
                                "type": "object",
                                "description": "Time period mentioned in the query",
                                "properties": {
                                    "start": {
                                        "type": "string",
                                        "description": "Start date in ISO format (YYYY-MM-DDTHH:MM:SS) or null if not mentioned",
                                    },
                                    "end": {
                                        "type": "string",
                                        "description": "End date in ISO format (YYYY-MM-DDTHH:MM:SS) or null if not mentioned",
                                    },
                                    "granularity": {
                                        "type": "string",
                                        "enum": ["monthly", "quarterly", "yearly"],
                                        "description": "Time granularity if mentioned, or null",
                                    },
                                },
                            },
                            "amounts": {
                                "type": "object",
                                "description": "Amounts mentioned in the query",
                                "properties": {
                                    "fundraise_min": {
                                        "type": "integer",
                                        "description": "Minimum fundraise amount in USD or null",
                                    },
                                    "fundraise_max": {
                                        "type": "integer",
                                        "description": "Maximum fundraise amount in USD or null",
                                    },
                                    "valuation_min": {
                                        "type": "integer",
                                        "description": "Minimum valuation in USD or null",
                                    },
                                    "valuation_max": {
                                        "type": "integer",
                                        "description": "Maximum valuation in USD or null",
                                    },
                                    "acquisition_price_min": {
                                        "type": "integer",
                                        "description": "Minimum acquisition price in USD or null",
                                    },
                                    "acquisition_price_max": {
                                        "type": "integer",
                                        "description": "Maximum acquisition price in USD or null",
                                    },
                                },
                            },
                            "funding_stages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Funding stages mentioned (e.g., 'seed', 'series_a', 'series_b', 'late_stage_venture', 'ipo')",
                            },
                            "query_intent": {
                                "type": "string",
                                "enum": [
                                    "find_funding_rounds",
                                    "find_investor_portfolio",
                                    "find_acquisitions",
                                    "find_organizations",
                                    "analyze_sector_trends",
                                ],
                                "description": "Query intent classification",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Reasoning for query improvement and metadata extraction decisions",
                            },
                        },
                        "required": ["improved_query", "original_query", "reasoning"],
                    },
                },
            }

            # Build system prompt using prompt manager
            prompt_manager = get_prompt_manager()
            system_prompt = prompt_manager.build_system_prompt(
                base_prompt=self._base_prompt,
                options=PromptOptions(
                    add_temporal_context=False, add_markdown_instructions=False
                ),
            )

            # Build user prompt with conversation history if available
            user_prompt_parts = [f"User Query: {query}"]

            if conversation_history:
                history_lines = []
                for msg in conversation_history[-10:]:  # Last 10 messages for context
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    history_lines.append(f"{role.capitalize()}: {content}")

                if history_lines:
                    user_prompt_parts.append(
                        "\n\nPrevious conversation:\n" + "\n".join(history_lines)
                    )

            user_prompt_parts.append(
                "\n\nImprove this query and extract all relevant metadata. "
                "Resolve any pronouns using the conversation history. "
                "Call the extract_query_metadata function with your analysis."
            )

            user_prompt_content = "\n".join(user_prompt_parts)

            user_prompt = prompt_manager.build_user_prompt(
                user_query=user_prompt_content,
                options=PromptOptions(add_temporal_context=True),
            )

            # Call LLM with function calling
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[extract_metadata_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_query_metadata"},
                },
            )

            # Parse result
            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "extract_query_metadata":
                    args = result["arguments"]

                    # Ensure original_query is set
                    if "original_query" not in args or not args["original_query"]:
                        args["original_query"] = query

                    # Ensure improved_query is set (fallback to original if not provided)
                    if "improved_query" not in args or not args["improved_query"]:
                        args["improved_query"] = args.get("original_query", query)

                    # Create metadata object and validate
                    try:
                        metadata = ExtractedQueryMetadata(**args)
                        metadata_dict = metadata.model_dump()

                        logger.info(
                            f"Query enriched: '{query[:50]}...' -> "
                            f"'{metadata_dict['improved_query'][:50]}...'"
                        )

                        # Optionally resolve company names to UUIDs (non-blocking best-effort)
                        metadata_dict = await self._maybe_resolve_company_uuids(metadata_dict)

                        return {
                            "improved_query": metadata_dict["improved_query"],
                            "original_query": metadata_dict["original_query"],
                            "metadata": metadata_dict,
                        }
                    except Exception as e:
                        logger.warning(
                            f"Failed to validate extracted metadata: {e}, using raw args",
                            exc_info=True,
                        )
                        # Use raw args if validation fails
                        return {
                            "improved_query": args.get("improved_query", query),
                            "original_query": args.get("original_query", query),
                            "metadata": args,
                        }
                else:
                    logger.warning(
                        f"Unexpected function call: {result.get('function_name')}, "
                        f"using original query"
                    )
            else:
                logger.warning(
                    f"LLM did not make expected function call. Got: {type(result)}, "
                    f"using original query"
                )

        except Exception as e:
            logger.error(
                f"Query enrichment failed: {e}, using original query", exc_info=True
            )

        # Fallback: return original query with empty metadata
        empty_metadata = ExtractedQueryMetadata(
            improved_query=query, original_query=query
        )
        return {
            "improved_query": query,
            "original_query": query,
            "metadata": empty_metadata.model_dump(),
        }

    async def _maybe_resolve_company_uuids(
        self, metadata_dict: Dict[str, Any], max_names: int = 3
    ) -> Dict[str, Any]:
        """
        Best-effort optional resolution of company names to UUIDs.

        - Non-blocking: failures are logged and ignored.
        - Limited to a few names to avoid latency.
        """
        companies = metadata_dict.get("companies") or {}
        names = companies.get("names") or []

        # If UUIDs already present or no names, nothing to do
        existing_uuids = companies.get("uuids") or []
        if existing_uuids or not names:
            return metadata_dict

        resolved: List[str] = []
        for name in names[:max_names]:
            try:
                orgs_output = await get_organizations(name_ilike=name, limit=1)
                if orgs_output.success and orgs_output.result:
                    uuid = orgs_output.result[0].get("org_uuid")
                    if uuid:
                        resolved.append(str(uuid))
            except Exception as e:
                logger.warning(f"UUID resolution failed for '{name}': {e}")
                continue

        # Attach resolved UUIDs if any found; keep names regardless
        if resolved:
            metadata_dict["companies"] = {
                **companies,
                "uuids": resolved,
            }

        return metadata_dict

