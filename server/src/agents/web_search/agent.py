"""Web search agent for handling queries that require additional context from web search.

This agent is specialized in:
- Performing web searches to get additional context about queries
- Handling one or multiple search queries with parallel fan-out
- Enriching responses with web search information
- Providing up-to-date information that may not be in the database
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.core.agent_response import AgentInsight, ResponseStatus
from src.prompts.prompt_manager import PromptOptions, get_prompt_manager
from src.tools.generate_llm_function_response import generate_llm_function_response
from src.tools.web_search import web_search

logger = logging.getLogger(__name__)


class WebSearchAgent(AgentBase):
    """Agent for performing web searches to enrich query responses with additional context."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the web search agent.

        Args:
            config_path: Path to the agent's YAML config file.
                If None, will attempt to find it automatically.
        """
        if config_path is None:
            # Default to web_search_agent.yaml in configs/agents
            src_base = Path(__file__).parent.parent.parent.parent
            config_path = src_base / "configs" / "agents" / "web_search_agent.yaml"

        super().__init__(config_path=config_path)

        # Register base prompt with prompt manager
        prompt_manager = get_prompt_manager()
        base_prompt = """You are a web search agent that uses web search to provide additional context and information about user queries.

Your task is to:
1. Identify what information needs to be searched on the web
2. Extract one or more search queries from the user's question
3. Perform web searches to gather relevant, up-to-date information
4. Synthesize the search results into a coherent response that addresses the user's query

Rules:
- Extract clear, focused search queries from the user's question
- If the query contains multiple topics or questions, extract multiple search queries
- Use web search to find current, factual information
- Synthesize information from multiple sources when available
- Cite sources when possible
- Focus on providing accurate, relevant information that directly addresses the query"""

        prompt_manager.register_agent_prompt(
            agent_name=self.name, system_prompt=base_prompt, overwrite=True
        )

    @observe(as_type="agent")
    async def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the web search agent to handle queries requiring web search.

        This method:
        1. Extracts search queries from the user's query (one or many)
        2. Performs web searches in parallel if multiple queries are identified
        3. Synthesizes the results into a coherent response
        4. Returns a structured response with tool calls tracked

        Args:
            context: The agent context containing the user query and metadata.

        Returns:
            AgentOutput containing:
            - content: AgentInsight with web search results and synthesized response
            - status: SUCCESS if query processed successfully
            - metadata: Information about the searches performed
            - tool_calls: List of tool calls made during execution
        """
        try:
            logger.info(f"Web search agent processing query: {context.query[:100]}")

            # Step 1: Extract search queries from the user's query
            search_queries = await self._extract_search_queries(context)

            if not search_queries:
                logger.warning("No search queries extracted from user query")
                return create_agent_output(
                    content=AgentInsight(
                        summary="Unable to extract search queries from your question. Please rephrase your query to be more specific.",
                        confidence=0.0,
                    ),
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.SUCCESS,
                    metadata={"query": context.query, "num_searches": 0},
                )

            # Step 2: Perform web searches (fan-out if multiple queries)
            if len(search_queries) > 1:
                # Fan-out: perform searches in parallel
                search_results, tool_calls = await self._fanout_web_searches(
                    search_queries=search_queries,
                )
            else:
                # Single search
                search_result = await web_search(query=search_queries[0])
                tool_calls = [
                    {
                        "name": "web_search",
                        "parameters": {"query": search_queries[0]},
                        "result": {
                            "success": search_result.success,
                            "response_length": (
                                len(search_result.result.get("response", ""))
                                if search_result.result
                                else 0
                            ),
                            "execution_time_ms": search_result.execution_time_ms,
                            "error": search_result.error,
                        },
                    }
                ]
                search_results = (
                    [search_result.result] if search_result.success and search_result.result else []
                )

            # Step 3: Synthesize results into a coherent response
            response_content = await self._synthesize_response(
                context, search_queries, search_results
            )

            logger.info(f"Web search agent completed: performed {len(search_queries)} search(es)")

            # Return AgentOutput using contract helper
            return create_agent_output(
                content=response_content,
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.SUCCESS,
                tool_calls=tool_calls,
                metadata={
                    "query": context.query,
                    "num_searches": len(search_queries),
                    "search_queries": search_queries,
                },
            )

        except Exception as e:
            error_msg = f"Web search agent failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
            )

    async def _extract_search_queries(self, context: AgentContext) -> List[str]:
        """Extract search queries from the user's query.

        Uses LLM to identify what needs to be searched on the web and extracts
        one or more focused search queries.

        Args:
            context: The agent context containing the user query.

        Returns:
            List of search query strings. Empty list if extraction fails.
        """
        extract_queries_tool = {
            "type": "function",
            "function": {
                "name": "extract_search_queries",
                "description": "Extract one or more focused search queries from the user's question that should be searched on the web.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search queries to perform. Each query should be focused and specific. If the user's question contains multiple topics or questions, extract multiple queries. If it's a single focused question, extract one query.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why these queries were extracted and how they relate to the user's question.",
                        },
                    },
                    "required": ["queries", "reasoning"],
                },
            },
        }

        # Build system prompt using prompt manager
        prompt_manager = get_prompt_manager()
        system_prompt = prompt_manager.build_system_prompt(
            base_prompt="""You are a query extraction assistant. Your job is to extract focused search queries from user questions that need web search.

Rules:
- Extract clear, specific search queries that can be used to find information on the web
- If the user's question contains multiple topics or sub-questions, extract multiple queries
- Each query should be self-contained and focused on a single topic
- Make queries specific enough to yield relevant results
- If the question is a single focused question, extract one query
- Queries should be in natural language, suitable for web search""",
            options=PromptOptions(add_temporal_context=False, add_markdown_instructions=False),
        )

        # Build user prompt
        user_prompt_content = f"""User Query: {context.query}

Extract search queries that should be performed to answer this question. If the question contains multiple topics or sub-questions, extract multiple queries. If it's a single focused question, extract one query.

Call the extract_search_queries function with your analysis."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[extract_queries_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_search_queries"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "extract_search_queries":
                    queries = result["arguments"].get("queries", [])
                    if queries and isinstance(queries, list):
                        # Filter out empty queries
                        valid_queries = [q.strip() for q in queries if q and q.strip()]
                        logger.info(
                            f"Extracted {len(valid_queries)} search query(ies): {valid_queries}"
                        )
                        return valid_queries
                    else:
                        logger.warning("LLM returned empty or invalid queries list")
                        return []
                else:
                    logger.warning(f"Unexpected function call: {result.get('function_name')}")
                    return []
            else:
                logger.warning(f"LLM did not make expected function call. Got: {type(result)}")
                # Fallback: use the original query as a single search query
                return [context.query] if context.query else []

        except Exception as e:
            logger.warning(
                f"Failed to extract search queries: {e}, using original query as fallback",
                exc_info=True,
            )
            # Fallback: use the original query as a single search query
            return [context.query] if context.query else []

    async def _fanout_web_searches(
        self,
        search_queries: List[str],
        max_concurrency: int = 3,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute web_search per query concurrently with bounded fan-out.

        Args:
            search_queries: List of search query strings to execute.
            max_concurrency: Maximum number of concurrent searches (default: 3).

        Returns:
            Tuple of (search_results, tool_calls_metadata).
            - search_results: List of result dictionaries from successful searches
            - tool_calls: List of tool call metadata dictionaries
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        search_results: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []

        async def _run(query: str) -> None:
            async with semaphore:
                result = await web_search(query=query)

            tool_calls.append(
                {
                    "name": "web_search",
                    "parameters": {"query": query},
                    "result": {
                        "success": result.success,
                        "response_length": (
                            len(result.result.get("response", "")) if result.result else 0
                        ),
                        "execution_time_ms": result.execution_time_ms,
                        "error": result.error,
                    },
                    "source_query": query,
                }
            )

            if result.success and result.result:
                search_results.append(result.result)

        # Deduplicate queries to avoid double calls
        seen = set()
        unique_queries = []
        for query in search_queries:
            query_lower = query.lower().strip()
            if query_lower and query_lower not in seen:
                seen.add(query_lower)
                unique_queries.append(query)

        await asyncio.gather(*[_run(query) for query in unique_queries])

        return search_results, tool_calls

    async def _synthesize_response(
        self,
        context: AgentContext,
        search_queries: List[str],
        search_results: List[Dict[str, Any]],
    ) -> AgentInsight:
        """Synthesize web search results into a coherent response.

        Args:
            context: The agent context containing the original user query.
            search_queries: List of search queries that were performed.
            search_results: List of search result dictionaries from web_search tool.

        Returns:
            AgentInsight object with synthesized response.
        """
        if not search_results:
            return AgentInsight(
                summary="I performed web searches but was unable to retrieve results. Please try rephrasing your query or try again later.",
                confidence=0.0,
            )

        # If we have multiple search results, synthesize them
        if len(search_results) > 1:
            # Combine all responses
            all_responses = [
                result.get("response", "") for result in search_results if result.get("response")
            ]
            combined_text = "\n\n---\n\n".join(all_responses)

            # Use LLM to synthesize multiple results
            synthesize_tool = {
                "type": "function",
                "function": {
                    "name": "synthesize_search_results",
                    "description": "Synthesize multiple web search results into a coherent final answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Synthesized summary that directly answers the user's query using information from all search results",
                            },
                            "key_findings": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Key findings from the search results",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence in the synthesized answer",
                            },
                        },
                        "required": ["summary"],
                    },
                },
            }

            prompt_manager = get_prompt_manager()
            system_prompt = prompt_manager.build_system_prompt(
                base_prompt="""You are a synthesis assistant. Your job is to combine multiple web search results into a coherent, comprehensive answer that directly addresses the user's query.

Rules:
- Synthesize information from all search results
- Create a coherent narrative that answers the user's question
- Identify key findings from the search results
- If there are conflicting information, acknowledge it
- Focus on providing accurate, relevant information""",
                options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
            )

            user_prompt_content = f"""Original User Query: {context.query}

Search Queries Performed:
{chr(10).join(f'- {q}' for q in search_queries)}

Web Search Results:
{combined_text}

Synthesize these search results into a coherent answer that directly addresses the user's query. Call the synthesize_search_results function with your synthesis."""

            user_prompt = prompt_manager.build_user_prompt(
                user_query=user_prompt_content,
                options=PromptOptions(add_temporal_context=True),
            )

            try:
                result = await generate_llm_function_response(
                    prompt=user_prompt,
                    tools=[synthesize_tool],
                    system_prompt=system_prompt,
                    model="gpt-4.1-mini",
                    temperature=0.3,
                    tool_choice={
                        "type": "function",
                        "function": {"name": "synthesize_search_results"},
                    },
                )

                if isinstance(result, dict) and "function_name" in result:
                    args = result["arguments"]
                    return AgentInsight(
                        summary=args.get("summary", combined_text[:500]),
                        key_findings=args.get("key_findings"),
                        confidence=args.get("confidence", 0.7),
                        evidence={
                            "num_searches": len(search_queries),
                            "search_queries": search_queries,
                        },
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to synthesize multiple results: {e}, using first result",
                    exc_info=True,
                )

        # Fallback: use the first result or combine all results
        if search_results:
            first_result = search_results[0]
            response_text = first_result.get("response", "")
            if len(search_results) > 1:
                # Combine all responses as fallback
                all_responses = [r.get("response", "") for r in search_results if r.get("response")]
                response_text = "\n\n---\n\n".join(all_responses)

            return AgentInsight(
                summary=response_text,
                key_findings=[f"Found information from {len(search_results)} search result(s)"],
                confidence=0.7,
                evidence={
                    "num_searches": len(search_queries),
                    "search_queries": search_queries,
                },
            )

        return AgentInsight(
            summary="Unable to synthesize search results.",
            confidence=0.0,
        )
