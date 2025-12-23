"""Orchestration agent for planning and routing queries to specialized agents.

This agent is responsible for:
- Analyzing user queries to determine intent
- Selecting appropriate specialized agents to handle the query
- Creating execution plans for the orchestrator workflow
- Determining execution strategy (sequential vs parallel)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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
from src.tools.generate_llm_function_response import generate_llm_function_response

logger = logging.getLogger(__name__)

# Import agent registry (no Temporal dependency)
try:
    from src.agents.registry import get_agent, get_agent_metadata, list_available_agents

    _AGENT_REGISTRY_AVAILABLE = True
except ImportError:
    _AGENT_REGISTRY_AVAILABLE = False
    logger.warning("Agent registry not available. Using fallback AVAILABLE_AGENTS dictionary.")

# Fallback available agents dictionary (used if registry is unavailable)
# This should match the agents registered in the system
_FALLBACK_AVAILABLE_AGENTS = {
    "acquisition": {
        "name": "acquisition",
        "description": "Handles queries about company acquisitions, mergers, and M&A activity. Handles M&A (mergers and acquisitions), company exits, buyouts, and acquisition details including prices, terms, dates, and participating companies.",
        "keywords": [
            "acquisition",
            "acquired",
            "merger",
            "m&a",
            "m and a",
            "takeover",
            "buyout",
            "exit",
            "exits",
            "company exit",
            "acquisition price",
            "acquisition terms",
            "acquirer",
            "acquiree",
            "target company",
            "buying company",
            "company bought",
            "company sold",
            "acquisition date",
            "acquisition announcement",
        ],
    },
    "organizations": {
        "name": "organizations",
        "description": "Handles queries about companies, organizations, and their details. NOTE: NOT for investor-related queries - use funding_rounds agent for questions about investors, investor portfolios, or companies that investors have funded.",
        "keywords": [
            "company",
            "organization",
            "startup",
            "firm",
            "business",
            "companies",
            "organization details",
            "company information",
            "find company",
            "company search",
        ],
    },
    "funding_rounds": {
        "name": "funding_rounds",
        "description": "Handles queries about funding rounds, investments, and fundraising. PRIMARY agent for investor-related queries including finding companies investors have funded, investor portfolios, and funding rounds by specific investors.",
        "keywords": [
            "funding",
            "investment",
            "round",
            "raise",
            "investor",
            "investors",
            "portfolio",
            "fundraise",
            "fundraising",
            "venture capital",
            "vc",
            "angel investor",
            "lead investor",
            "companies investors funded",
            "investor portfolio",
        ],
    },
    "sector_trends": {
        "name": "sector_trends",
        "description": "Analyzes funding trends within sectors/industries over time, identifying growth patterns, funding velocity changes, and market momentum indicators. Analyzes AGGREGATED funding data across multiple companies within a sector. NOTE: This agent is for SECTOR/INDUSTRY-level analysis, NOT for individual company funding trends - use the funding_rounds agent for queries about specific companies.",
        "keywords": [
            "sector",
            "industry",
            "trend",
            "trends",
            "funding velocity",
            "growth",
            "momentum",
            "market analysis",
            "sector analysis",
            "industry analysis",
            "funding trends",
            "market trends",
            "sector funding",
            "industry funding",
            "market momentum",
            "sector momentum",
            "funding patterns",
            "sector patterns",
            "aggregated funding",
            "sector comparison",
            "industry comparison",
        ],
    },
    "web_search": {
        "name": "web_search",
        "description": "Performs web searches to get additional context and up-to-date information about user queries. Uses LLM with web search capabilities to enrich responses with information that may not be available in the database. Can handle one or multiple search queries with parallel fan-out for efficiency.",
        "keywords": [
            "web search",
            "search",
            "current events",
            "latest news",
            "recent information",
            "up-to-date",
            "current information",
            "web information",
            "online search",
            "internet search",
            "additional context",
            "external information",
            "real-time information",
        ],
    },
}


def _get_available_agents() -> Dict[str, Dict[str, Any]]:
    """Get available agents dynamically from the agent registry.

    This function:
    1. Attempts to use the agent registry to discover all registered agents
    2. Loads each agent's config file to get description and metadata
    3. Extracts keywords from metadata or generates them from the agent name/description
    4. Falls back to hardcoded dictionary if registry is unavailable

    Returns:
        Dictionary mapping agent names to their metadata (name, description, keywords).
    """
    if not _AGENT_REGISTRY_AVAILABLE:
        logger.debug("Using fallback AVAILABLE_AGENTS dictionary")
        return _FALLBACK_AVAILABLE_AGENTS.copy()

    try:
        available_agents = {}
        agent_names = list_available_agents()

        # Filter out the orchestration agent itself (it shouldn't orchestrate itself)
        agent_names = [name for name in agent_names if name != "orchestration"]

        for agent_name in agent_names:
            # Try to get metadata directly (more efficient)
            agent_metadata = get_agent_metadata(agent_name)
            if agent_metadata is None:
                logger.warning(f"Agent '{agent_name}' listed but not found in registry")
                continue

            # Use metadata if available
            try:
                description = agent_metadata.get(
                    "description", f"Agent for handling {agent_name} queries"
                )
                metadata = agent_metadata.get("metadata", {})

                # If metadata not fully loaded, try to load config file directly
                if not description or description == f"Agent for handling {agent_name} queries":
                    agent_info = get_agent(agent_name)
                    if agent_info:
                        agent_class, config_path = agent_info
                        with open(config_path, "r", encoding="utf-8") as f:
                            config_data = yaml.safe_load(f)
                            description = config_data.get("description", description)
                            metadata = config_data.get("metadata", metadata)

                # Extract keywords from metadata or generate from description
                keywords = []
                if "keywords" in metadata:
                    keywords = metadata["keywords"]
                else:
                    # Generate keywords from agent name, description, and use cases
                    # Start with agent name and its variations
                    keywords = [agent_name]

                    # Add words from description (filter common words)
                    common_words = {
                        "the",
                        "and",
                        "or",
                        "for",
                        "with",
                        "about",
                        "handling",
                        "queries",
                        "agent",
                        "specialized",
                        "related",
                        "data",
                        "retrieval",
                    }
                    desc_words = description.lower().replace(",", " ").replace(".", " ").split()
                    keywords.extend(
                        [
                            w.strip()
                            for w in desc_words
                            if w.strip() not in common_words and len(w.strip()) > 3
                        ]
                    )

                    # Extract from use cases if available
                    if "use_cases" in metadata:
                        use_cases = metadata["use_cases"]
                        for use_case in use_cases:
                            # Extract key terms from use case descriptions
                            words = use_case.lower().replace(",", " ").replace(".", " ").split()
                            keywords.extend(
                                [
                                    w.strip()
                                    for w in words
                                    if w.strip() not in common_words and len(w.strip()) > 3
                                ]
                            )

                # Deduplicate and limit keywords
                keywords = list(dict.fromkeys(keywords))[:10]  # Keep first 10 unique keywords

                available_agents[agent_name] = {
                    "name": agent_name,
                    "description": description,
                    "keywords": keywords,
                }

            except Exception as e:
                logger.warning(
                    f"Failed to load config for agent '{agent_name}': {e}. "
                    f"Using fallback description."
                )
                # Fallback: use agent name and basic description
                available_agents[agent_name] = {
                    "name": agent_name,
                    "description": f"Agent for handling {agent_name} queries",
                    "keywords": [agent_name],
                }

        if not available_agents:
            logger.warning("No agents found in registry, using fallback")
            return _FALLBACK_AVAILABLE_AGENTS.copy()

        logger.info(
            f"Discovered {len(available_agents)} agents from registry: {list(available_agents.keys())}"
        )
        return available_agents

    except Exception as e:
        logger.warning(f"Failed to discover agents from registry: {e}. Using fallback.")
        return _FALLBACK_AVAILABLE_AGENTS.copy()


class OrchestrationAgent(AgentBase):
    """Orchestration agent that plans and routes queries to specialized agents.

    This agent analyzes user queries and determines which specialized agents
    should be executed to handle the query. It returns a structured plan that
    the orchestrator workflow can use to coordinate agent execution.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the orchestration agent.

        Args:
            config_path: Path to the agent's YAML config file.
                If None, will attempt to find it automatically.
        """
        if config_path is None:
            # Default to orchestration_agent.yaml in configs/agents
            src_base = Path(__file__).parent.parent.parent.parent
            config_path = src_base / "configs" / "agents" / "orchestration_agent.yaml"

        super().__init__(config_path=config_path)

        # Register base prompt with prompt manager
        prompt_manager = get_prompt_manager()
        base_prompt = """You are an orchestration agent that analyzes user queries and determines which specialized agents should handle them.

Your task is to:
1. Analyze the user query to understand the intent
2. Select the appropriate agent(s) from the available agents list
3. Determine if agents should run sequentially (one after another) or in parallel
4. Provide reasoning for your decisions

Rules:
- IMPORTANT: If you mention an agent in your reasoning as being relevant to the query, you MUST include that agent's name in the agents array
- Select ALL agents that are relevant to the query, even if only one agent matches
- Use "sequential" if agents need to build on each other's results
- Use "parallel" if agents can work independently
- Only return an empty agents list if NO agents are relevant to the query
- Be specific and accurate in your reasoning
- Provide a confidence score based on how clear the query intent is
- The agents array should contain the exact agent name(s) from the available agents list"""

        prompt_manager.register_agent_prompt(
            agent_name=self.name, system_prompt=base_prompt, overwrite=True
        )

    @observe(as_type="agent")
    async def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the orchestration agent in planning or consolidation mode.

        This method supports three modes:
        1. **Planning mode** (default): Creates an execution plan
        2. **Consolidation mode**: Consolidates agent insights into final answer
        3. **Evaluation mode**: Evaluates response quality and creates re-plan if needed

        Args:
            context: The agent context containing the user query and metadata.
                - For planning: metadata may contain "available_agents"
                - For consolidation: metadata must contain "agent_responses" (List[AgentOutput])

        Returns:
            AgentOutput containing:
            - Planning mode: Dict with "agents" (list of agent names) and "execution_mode"
            - Consolidation mode: AgentInsight with consolidated final answer
            - status: SUCCESS if operation completed successfully
            - metadata: Additional information about the process
        """
        try:
            # Check mode from metadata
            mode = context.get_metadata("mode")

            if mode == "consolidation":
                # Consolidation mode: combine agent insights
                logger.info("Orchestration agent in consolidation mode")
                agent_responses = context.get_metadata("agent_responses", [])

                if not agent_responses:
                    logger.warning("No agent responses provided for consolidation")
                    return create_agent_output(
                        content="",
                        agent_name=self.name,
                        agent_category=self.category,
                        status=ResponseStatus.ERROR,
                        error="No agent responses provided for consolidation",
                    )

                # Consolidate insights
                consolidated_insight = await self._consolidate_responses(context, agent_responses)

                logger.info("Orchestration agent completed consolidation")

                return create_agent_output(
                    content=consolidated_insight,
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.SUCCESS,
                    metadata={
                        "query": context.query,
                        "mode": "consolidation",
                        "num_agents_consolidated": len(agent_responses),
                    },
                )
            elif mode == "evaluation_and_replanning":
                # Evaluation mode: evaluate response quality and create re-plan
                logger.info("Orchestration agent in evaluation and replanning mode")
                agent_responses = context.get_metadata("agent_responses", [])
                execution_history = context.get_metadata("execution_history", [])

                if not agent_responses:
                    logger.warning("No agent responses provided for evaluation")
                    return create_agent_output(
                        content="",
                        agent_name=self.name,
                        agent_category=self.category,
                        status=ResponseStatus.ERROR,
                        error="No agent responses provided for evaluation",
                    )

                # Evaluate response quality
                evaluation = await self._evaluate_response_quality(
                    context, agent_responses, execution_history
                )

                # If not satisfactory, create re-plan
                if not evaluation.get("satisfactory", False):
                    logger.info("Response not satisfactory, creating re-plan")
                    replan = await self._create_replan(context, evaluation, agent_responses)

                    return create_agent_output(
                        content="",
                        agent_name=self.name,
                        agent_category=self.category,
                        status=ResponseStatus.SUCCESS,
                        metadata={
                            "query": context.query,
                            "mode": "evaluation_and_replanning",
                            "evaluation": evaluation,
                            "replan": replan,
                        },
                    )
                else:
                    # Response is satisfactory
                    logger.info("Response is satisfactory, no re-planning needed")
                    return create_agent_output(
                        content="",
                        agent_name=self.name,
                        agent_category=self.category,
                        status=ResponseStatus.SUCCESS,
                        metadata={
                            "query": context.query,
                            "mode": "evaluation_and_replanning",
                            "evaluation": evaluation,
                            "replan": None,
                        },
                    )
            else:
                # Planning mode (default): create execution plan
                logger.info(f"Orchestration agent analyzing query: {context.query[:100]}")

                # Get available agents from metadata, or discover dynamically, or use fallback
                available_agents = (
                    context.get_metadata("available_agents") or _get_available_agents()
                )

                # Analyze query and create plan using LLM
                plan = await self._create_execution_plan(context, available_agents)

                # Build plan summary for AgentInsight
                agents_list = plan["agents"]
                execution_mode = plan.get("execution_mode", "sequential")
                reasoning = plan.get("reasoning", "")
                confidence = plan.get("confidence", 0.0)

                # Create summary describing the plan
                if agents_list:
                    agents_str = ", ".join(agents_list)
                    summary = f"Created execution plan with {len(agents_list)} agent(s): {agents_str}. Execution mode: {execution_mode}."
                else:
                    summary = "Created execution plan with no agents selected."

                # Create AgentInsight for the plan
                from src.core.agent_response import AgentInsight

                plan_insight = AgentInsight(
                    summary=summary,
                    key_findings=(
                        [
                            (
                                f"Selected {len(agents_list)} agent(s): {agents_str}"
                                if agents_list
                                else "No agents selected"
                            ),
                            f"Execution mode: {execution_mode}",
                        ]
                        if agents_list
                        else ["No agents selected for execution"]
                    ),
                    evidence={
                        "agents": agents_list,
                        "execution_mode": execution_mode,
                        "reasoning": reasoning,
                        "confidence": confidence,
                    },
                    confidence=confidence,
                )

                logger.info(
                    f"Orchestration agent created plan: {len(plan['agents'])} agents, "
                    f"mode: {plan.get('execution_mode', 'sequential')}"
                )

                # Return AgentOutput using contract helper
                # Store the plan dict in metadata for workflow to extract
                return create_agent_output(
                    content=plan_insight,
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.SUCCESS,
                    metadata={
                        "query": context.query,
                        "plan_created": True,
                        "num_agents": len(plan["agents"]),
                        "execution_mode": plan.get("execution_mode", "sequential"),
                        # Store plan dict in metadata for workflow extraction
                        "execution_plan": {
                            "agents": plan["agents"],
                            "execution_mode": plan.get("execution_mode", "sequential"),
                            "reasoning": plan.get("reasoning", ""),
                            "confidence": plan.get("confidence", 0.0),
                        },
                    },
                )

        except Exception as e:
            error_msg = f"Orchestration agent failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
            )

    async def _create_execution_plan(
        self, context: AgentContext, available_agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create an execution plan using LLM function calling for structured output.

        Args:
            context: The agent context.
            available_agents: Dictionary of available agents and their descriptions.

        Returns:
            Dictionary containing:
            - agents: List of agent names to execute
            - execution_mode: "sequential" or "parallel"
            - reasoning: Explanation of the plan
            - confidence: Confidence score (0.0-1.0)
        """
        # If query intent was pre-extracted, use it to shortcut planning
        extracted = context.get_metadata("extracted_entities", {}) or {}
        query_intent = extracted.get("query_intent")

        intent_to_agents: Dict[str, List[str]] = {}

        if query_intent in intent_to_agents:
            agents = [a for a in intent_to_agents[query_intent] if a in available_agents]
            if agents:
                reasoning = (
                    f"Pre-extracted intent '{query_intent}' maps to agents: {', '.join(agents)}"
                )
                logger.info(reasoning)
                return {
                    "agents": agents,
                    "execution_mode": "sequential",
                    "reasoning": reasoning,
                    "confidence": 0.75,
                }

        # Build agents description for prompt
        agents_description = "\n".join(
            [
                f"- {name}: {info['description']} (keywords: {', '.join(info.get('keywords', []))})"
                for name, info in available_agents.items()
            ]
        )

        # Build list of available agent names for the schema
        agent_names = list(available_agents.keys())

        # Define the function/tool schema for structured output
        create_plan_tool = {
            "type": "function",
            "function": {
                "name": "create_execution_plan",
                "description": "Create an execution plan for handling a user query by selecting appropriate agents and determining execution strategy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of agent names to execute. Only include agents that are relevant to the query.",
                            "enum": agent_names,  # Constrain to valid agent names
                        },
                        "execution_mode": {
                            "type": "string",
                            "enum": ["sequential", "parallel"],
                            "description": "Execution strategy: 'sequential' if agents need to build on each other's results, 'parallel' if they can work independently.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why these agents were selected and why this execution mode was chosen.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence score between 0.0 and 1.0 indicating how confident the plan is.",
                        },
                    },
                    "required": ["agents", "execution_mode", "reasoning", "confidence"],
                },
            },
        }

        # Build system prompt using prompt manager with dynamic available agents section
        prompt_manager = get_prompt_manager()
        base_prompt = f"""You are an orchestration agent that analyzes user queries and determines which specialized agents should handle them.

Your task is to:
1. Analyze the user query to understand the intent
2. Select the appropriate agent(s) from the available agents list
3. Determine if agents should run sequentially (one after another) or in parallel
4. Provide reasoning for your decisions

Available agents:
{agents_description}

Rules:
- IMPORTANT: If you mention an agent in your reasoning as being relevant to the query, you MUST include that agent's name in the agents array
- Select ALL agents that are relevant to the query, even if only one agent matches
- Use "sequential" if agents need to build on each other's results
- Use "parallel" if agents can work independently
- Only return an empty agents list if NO agents are relevant to the query
- Be specific and accurate in your reasoning
- Provide a confidence score based on how clear the query intent is
- The agents array should contain the exact agent name(s) from the available agents list above"""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Include conversation history if available
        history_text = ""
        if context.conversation_history:
            history_for_llm = context.get_conversation_history_for_llm(max_messages=50)
            if history_for_llm:
                history_lines = []
                for msg in history_for_llm:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    history_lines.append(f"{role.capitalize()}: {content}")
                history_text = "\n\nPrevious conversation:\n" + "\n".join(history_lines) + "\n"

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}
{history_text}
Analyze this query and create an execution plan. Consider:
- What information is the user seeking?
- Which agents have the capabilities to provide this information?
- Should agents run sequentially or in parallel?
- If there's conversation history, how does this query relate to previous messages?

IMPORTANT: If you determine that an agent is relevant to answering this query, you MUST include that agent's name in the agents array. Do not leave the agents array empty if you identify relevant agents in your reasoning.

Call the create_execution_plan function with your analysis."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        # Call LLM with function calling for structured output
        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[create_plan_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,  # Lower temperature for more consistent planning
                tool_choice={
                    "type": "function",
                    "function": {"name": "create_execution_plan"},
                },  # Force the function call
            )

            # Check if we got a function call result
            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "create_execution_plan":
                    plan_data = result["arguments"]
                    logger.debug(f"Raw LLM plan data: {plan_data}")
                    # Validate and normalize plan
                    plan = self._validate_and_normalize_plan(plan_data, available_agents)
                    if len(plan_data.get("agents", [])) == 0 and len(plan["agents"]) > 0:
                        logger.info(
                            f"Fixed empty agents list: LLM returned empty but validation added {plan['agents']}"
                        )
                    return plan
                else:
                    logger.warning(
                        f"Unexpected function call: {result.get('function_name')}, using fallback"
                    )
                    return self._create_fallback_plan(context, available_agents)
            else:
                logger.warning(
                    f"LLM did not make expected function call. Got: {type(result)}, using fallback"
                )
                return self._create_fallback_plan(context, available_agents)

        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using fallback", exc_info=True)
            return self._create_fallback_plan(context, available_agents)

    def _validate_and_normalize_plan(
        self, plan_data: Dict[str, Any], available_agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate and normalize the execution plan from LLM.

        Args:
            plan_data: Raw plan data from LLM.
            available_agents: Dictionary of available agents.

        Returns:
            Validated and normalized plan dictionary.
        """
        # Extract agents list
        agents = plan_data.get("agents", [])
        if not isinstance(agents, list):
            agents = []

        # Filter to only include valid agent names
        valid_agents = [agent for agent in agents if agent in available_agents]

        # Get reasoning
        reasoning = plan_data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = ""

        # Safety check: If reasoning mentions an agent but it's not in the agents list, add it
        # This handles cases where the LLM recognizes relevance but doesn't include the agent
        if reasoning and not valid_agents:
            reasoning_lower = reasoning.lower()
            for agent_name in available_agents.keys():
                # Check if agent name or its description keywords are mentioned in reasoning
                agent_info = available_agents[agent_name]
                agent_desc_lower = agent_info.get("description", "").lower()

                # Check if agent name appears in reasoning
                if agent_name.lower() in reasoning_lower:
                    if agent_name not in valid_agents:
                        logger.info(
                            f"Found agent '{agent_name}' mentioned in reasoning but not in agents list. Adding it."
                        )
                        valid_agents.append(agent_name)
                # Also check if key terms from the agent's description appear
                elif any(
                    keyword.lower() in reasoning_lower for keyword in agent_info.get("keywords", [])
                ):
                    if agent_name not in valid_agents:
                        logger.info(
                            f"Found agent '{agent_name}' keywords in reasoning but not in agents list. Adding it."
                        )
                        valid_agents.append(agent_name)

        # Validate execution mode
        execution_mode = plan_data.get("execution_mode", "sequential")
        if execution_mode not in ["sequential", "parallel"]:
            execution_mode = "sequential"

        # Validate confidence
        confidence = plan_data.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            confidence = 0.5

        return {
            "agents": valid_agents,
            "execution_mode": execution_mode,
            "reasoning": reasoning,
            "confidence": float(confidence),
        }

    def _create_fallback_plan(
        self, context: AgentContext, available_agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a fallback plan using keyword matching.

        This is used when LLM planning fails.

        Args:
            context: The agent context.
            available_agents: Dictionary of available agents.

        Returns:
            Fallback plan dictionary.
        """
        query_lower = context.query.lower()
        selected_agents = []

        # Simple keyword matching
        for agent_name, agent_info in available_agents.items():
            keywords = agent_info.get("keywords", [])
            if any(keyword.lower() in query_lower for keyword in keywords):
                selected_agents.append(agent_name)

        # Default to sequential execution
        execution_mode = "sequential"

        # If multiple agents selected, consider parallel if they're independent
        if len(selected_agents) > 1:
            # For now, default to sequential for safety
            # This could be enhanced with more sophisticated logic
            execution_mode = "sequential"

        return {
            "agents": selected_agents,
            "execution_mode": execution_mode,
            "reasoning": f"Fallback plan based on keyword matching in query",
            "confidence": 0.6,
        }

    async def _consolidate_responses(
        self,
        context: AgentContext,
        agent_responses: List[AgentOutput],
    ) -> AgentInsight:
        """Consolidate multiple agent insights into a single final answer.

        Args:
            context: The agent context containing the original user query.
            agent_responses: List of AgentOutput objects (or dicts) from specialized agents.

        Returns:
            AgentInsight object with consolidated final answer.
        """
        # Extract insights from agent responses
        # Handle both AgentOutput objects and serialized dicts
        agent_insights = []
        for resp in agent_responses:
            # Convert dict to AgentOutput if needed
            if isinstance(resp, dict):
                try:
                    resp = AgentOutput(**resp)
                except Exception as e:
                    logger.warning(f"Failed to parse agent response dict: {e}")
                    continue

            # Extract AgentInsight from content
            content = resp.content
            if isinstance(content, AgentInsight):
                agent_insights.append(
                    {
                        "agent_name": resp.agent_name,
                        "agent_category": resp.agent_category,
                        "insight": content.model_dump(),
                    }
                )
            elif isinstance(content, dict):
                # Try to parse as AgentInsight if it's a dict with the right structure
                try:
                    insight = AgentInsight(**content)
                    agent_insights.append(
                        {
                            "agent_name": resp.agent_name,
                            "agent_category": resp.agent_category,
                            "insight": insight.model_dump(),
                        }
                    )
                except Exception:
                    # Not an AgentInsight dict - might be planning mode output, skip it
                    logger.debug(f"Skipping non-insight content from {resp.agent_name}")
                    continue
            elif isinstance(content, str):
                # Skip string content (errors or old format)
                logger.debug(f"Skipping string content from {resp.agent_name}")
                continue

        if not agent_insights:
            logger.warning("No agent insights found to consolidate")
            return AgentInsight(
                summary=f"No insights were available to consolidate from {len(agent_responses)} agent response(s).",
                confidence=0.0,
            )

        # Define function schema matching AgentInsight structure
        consolidate_insight_tool = {
            "type": "function",
            "function": {
                "name": "consolidate_insights",
                "description": "Consolidate multiple agent insights into a single final answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Consolidated human-readable summary answering the user's query",
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key findings from all agents",
                        },
                        "evidence": {
                            "type": "object",
                            "description": "Consolidated evidence from all agents",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Overall confidence in the consolidated insight",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }

        # Build system prompt using prompt_manager
        prompt_manager = get_prompt_manager()
        base_prompt = """You are a consolidation agent. Your task is to combine insights from multiple specialized agents into a single, coherent final answer.

Your task:
- Compose a coherent final answer that directly addresses the user's query
- Synthesize information from all agent insights
- Handle conflicting information by acknowledging discrepancies
- Acknowledge limitations and uncertainty where appropriate
- Create a well-structured summary that flows naturally"""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Include conversation history if available
        history_text = ""
        if context.conversation_history:
            history_for_llm = context.get_conversation_history_for_llm(max_messages=50)
            if history_for_llm:
                history_lines = []
                for msg in history_for_llm:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    history_lines.append(f"{role.capitalize()}: {content}")
                history_text = "\n\nPrevious conversation:\n" + "\n".join(history_lines) + "\n"

        # Build user prompt with all agent insights
        user_prompt_content = f"""User Query: {context.query}
{history_text}
Agent Insights:
{json.dumps(agent_insights, indent=2, default=str)}

Consolidate these insights into a single final answer that directly addresses the user's query. If there's conversation history, ensure your answer is coherent with previous messages. Call the consolidate_insights function with your consolidated analysis."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        try:
            # Call LLM with function calling
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[consolidate_insight_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "consolidate_insights"},
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
                # Fallback
                return AgentInsight(
                    summary=f"Consolidated insights from {len(agent_responses)} agents but failed to generate final summary.",
                    confidence=0.0,
                )
        except Exception as e:
            logger.warning(f"Failed to consolidate insights from LLM: {e}", exc_info=True)
            # Fallback
            return AgentInsight(
                summary=f"Consolidated insights from {len(agent_responses)} agents but encountered an error during consolidation.",
                confidence=0.0,
            )

    async def _evaluate_response_quality(
        self,
        context: AgentContext,
        agent_responses: List[Any],
        execution_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate the quality and completeness of agent responses.

        This method uses an LLM to evaluate if the current answer is satisfactory
        for the user's query, identifying gaps and missing information.

        Args:
            context: The agent context containing the original user query.
            agent_responses: List of AgentOutput objects (or dicts) from agents.
            execution_history: List of execution events from the workflow.

        Returns:
            Dictionary containing:
            - satisfactory: bool indicating if answer is satisfactory
            - reasoning: str explaining the evaluation
            - missing_information: List[str] of missing information gaps
            - confidence: float (0.0-1.0) confidence in the evaluation
        """
        # Extract insights from agent responses
        agent_insights = []
        for resp in agent_responses:
            if isinstance(resp, dict):
                content = resp.get("content", {})
                if isinstance(content, dict) and "summary" in content:
                    agent_insights.append(
                        {
                            "agent_name": resp.get("agent_name", "unknown"),
                            "summary": content.get("summary", ""),
                            "key_findings": content.get("key_findings", []),
                        }
                    )
                elif isinstance(content, str):
                    agent_insights.append(
                        {
                            "agent_name": resp.get("agent_name", "unknown"),
                            "summary": content,
                        }
                    )

        if not agent_insights:
            logger.warning("No agent insights found for evaluation")
            return {
                "satisfactory": False,
                "reasoning": "No agent insights were available to evaluate",
                "missing_information": ["No agent responses available"],
                "confidence": 0.0,
            }

        # Define function schema for evaluation
        evaluate_quality_tool = {
            "type": "function",
            "function": {
                "name": "evaluate_response_quality",
                "description": "Evaluate if the current answer is satisfactory for the user's query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "satisfactory": {
                            "type": "boolean",
                            "description": "Whether the answer is satisfactory and complete",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why the answer is or isn't satisfactory",
                        },
                        "missing_information": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of missing information or gaps in the answer",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in the evaluation (0.0-1.0)",
                        },
                    },
                    "required": [
                        "satisfactory",
                        "reasoning",
                        "missing_information",
                        "confidence",
                    ],
                },
            },
        }

        # Build system prompt
        prompt_manager = get_prompt_manager()
        base_prompt = """You are an evaluation agent. Your task is to evaluate if the current answer is satisfactory for the user's query.

Consider:
- Does the answer directly address the query?
- Is the information complete and comprehensive?
- Are there obvious gaps or missing information?
- Would additional research improve the answer?
- Is the answer accurate and well-supported?

Be critical but fair. An answer is satisfactory if it adequately addresses the query, even if it could be improved."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Build user prompt
        user_prompt_content = f"""User Query: {context.query}

Agent Responses:
{json.dumps(agent_insights, indent=2, default=str)}

Execution History:
{json.dumps(execution_history, indent=2, default=str)}

Evaluate if the current answer is satisfactory. Consider:
- Does it fully answer the user's query?
- Are there missing pieces of information?
- Would additional research help?

Call the evaluate_response_quality function with your evaluation."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[evaluate_quality_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "evaluate_response_quality"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "evaluate_response_quality":
                    evaluation = result["arguments"]
                    logger.info(
                        f"Evaluation result: satisfactory={evaluation.get('satisfactory')}, "
                        f"confidence={evaluation.get('confidence')}"
                    )
                    return evaluation
                else:
                    logger.warning(f"Unexpected function call: {result.get('function_name')}")
            else:
                logger.warning(f"LLM did not make expected function call. Got: {type(result)}")

            # Fallback: assume not satisfactory if evaluation fails
            return {
                "satisfactory": False,
                "reasoning": "Evaluation failed, assuming answer needs improvement",
                "missing_information": ["Unable to evaluate response quality"],
                "confidence": 0.5,
            }

        except Exception as e:
            logger.warning(
                f"Failed to evaluate response quality: {e}, using fallback",
                exc_info=True,
            )
            return {
                "satisfactory": False,
                "reasoning": f"Evaluation error: {str(e)}",
                "missing_information": ["Evaluation failed"],
                "confidence": 0.5,
            }

    async def _create_replan(
        self,
        context: AgentContext,
        evaluation: Dict[str, Any],
        agent_responses: List[Any],
    ) -> Dict[str, Any]:
        """Create a new execution plan based on evaluation results.

        This method creates a plan to address missing information identified
        in the evaluation. It may include new agents or refine queries for existing agents.

        Args:
            context: The agent context.
            evaluation: Evaluation result from _evaluate_response_quality().
            agent_responses: List of previous agent responses.

        Returns:
            Dictionary containing:
            - agents: List of agent names to execute
            - execution_mode: "sequential" or "parallel"
            - reasoning: Explanation of the re-plan
            - confidence: Confidence score (0.0-1.0)
        """
        missing_info = evaluation.get("missing_information", [])
        evaluation_reasoning = evaluation.get("reasoning", "")

        # Get available agents
        available_agents = context.get_metadata("available_agents") or _get_available_agents()

        # Build agents description for prompt
        agents_description = "\n".join(
            [
                f"- {name}: {info['description']} (keywords: {', '.join(info.get('keywords', []))})"
                for name, info in available_agents.items()
            ]
        )

        agent_names = list(available_agents.keys())

        # Define function schema for re-planning
        create_replan_tool = {
            "type": "function",
            "function": {
                "name": "create_replan",
                "description": "Create a new execution plan to address missing information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of agent names to execute. Focus on agents that can address the missing information.",
                            "enum": agent_names,
                        },
                        "execution_mode": {
                            "type": "string",
                            "enum": ["sequential", "parallel"],
                            "description": "Execution strategy for the re-plan",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why these agents were selected and how they address the missing information",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence that this plan will address the missing information",
                        },
                    },
                    "required": ["agents", "execution_mode", "reasoning", "confidence"],
                },
            },
        }

        # Build system prompt
        prompt_manager = get_prompt_manager()
        base_prompt = f"""You are a re-planning agent. Your task is to create a new execution plan to address missing information identified in an evaluation.

Available agents:
{agents_description}

Your task:
- Analyze the missing information from the evaluation
- Select agents that can best address these gaps
- Create a focused plan to fill the information gaps
- Consider which agents were already used (to avoid redundancy unless needed)"""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Build user prompt
        user_prompt_content = f"""Original Query: {context.query}

Evaluation Results:
- Satisfactory: {evaluation.get('satisfactory', False)}
- Reasoning: {evaluation_reasoning}
- Missing Information: {json.dumps(missing_info, indent=2)}

Previous Agent Responses: {len(agent_responses)} agent(s) have already executed.

Create a new execution plan to address the missing information. Focus on agents that can fill the identified gaps. Call the create_replan function with your plan."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[create_replan_tool],
                system_prompt=system_prompt,
                model="gpt-4o-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "create_replan"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "create_replan":
                    replan_data = result["arguments"]
                    # Validate and normalize plan
                    replan = self._validate_and_normalize_plan(replan_data, available_agents)
                    logger.info(
                        f"Created re-plan: {len(replan['agents'])} agents, "
                        f"mode: {replan.get('execution_mode', 'sequential')}"
                    )
                    return replan
                else:
                    logger.warning(f"Unexpected function call: {result.get('function_name')}")
            else:
                logger.warning(f"LLM did not make expected function call. Got: {type(result)}")

            # Fallback: return empty plan
            return {
                "agents": [],
                "execution_mode": "sequential",
                "reasoning": "Re-planning failed, using fallback",
                "confidence": 0.3,
            }

        except Exception as e:
            logger.warning(f"Failed to create re-plan: {e}, using fallback", exc_info=True)
            return {
                "agents": [],
                "execution_mode": "sequential",
                "reasoning": f"Re-planning error: {str(e)}",
                "confidence": 0.3,
            }
