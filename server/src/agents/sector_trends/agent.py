"""Sector trends agent for analyzing funding trends within sectors.

This agent is specialized in:
- Analyzing funding trends within specific sectors over time
- Identifying growth patterns, funding velocity changes, and market momentum
- Detecting funding patterns (peaks, troughs, seasonality, cycles)
- Providing insights on sector funding dynamics
"""

import json
import logging
from datetime import datetime, timedelta
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
from src.core.agent_response import AgentInsight
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentResponse, ResponseStatus
from src.prompts.prompt_manager import PromptOptions, get_prompt_manager
from src.tools.generate_llm_function_response import (
    generate_llm_function_response
)
from src.tools.semantic_search_organizations import (
    semantic_search_organizations
)
from src.tools.aggregate_funding_trends import aggregate_funding_trends
from src.tools.calculate_funding_velocity import calculate_funding_velocity
from src.tools.identify_funding_patterns import identify_funding_patterns

logger = logging.getLogger(__name__)


class SectorTrendsAgent(AgentBase):
    """Agent specialized in analyzing sector funding trends and patterns.

    This agent uses LLM reasoning to understand user queries about sector funding
    trends and calls specialized tools to analyze funding data over time.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the sector trends agent.

        Args:
            config_path: Path to the agent's YAML config file.
                If None, will attempt to find it automatically.
        """
        if config_path is None:
            # Default to sector_trends_agent.yaml in configs/agents
            src_base = Path(__file__).parent.parent.parent.parent
            config_path = src_base / "configs" / "agents" / "sector_trends_agent.yaml"

        super().__init__(config_path=config_path)

        # Store base prompts as instance variables for reuse
        # Base prompt for extracting sector and time parameters
        self._extract_params_prompt = """You are analyzing a query about sector funding trends. Your job is to extract:
1. Sector/industry name (e.g., "AI companies", "fintech", "healthcare startups")
2. Time period (start date, end date) - convert relative terms like "last 2 years" to ISO dates
3. Granularity (monthly, quarterly, yearly) - default to quarterly if not specified
4. Minimum funding amount filter (if mentioned)

Rules:
- Only extract parameters that are explicitly mentioned or clearly implied
- For date ranges, convert relative terms (e.g., "last 2 years", "2023") to ISO format dates
- Default time period: last 2 years if not specified
- Default granularity: quarterly if not specified
- Leave parameters as None/null if not mentioned
- Be conservative - only extract what you're confident about"""

        # Register prompts with prompt manager
        prompt_manager = get_prompt_manager()
        prompt_manager.register_agent_prompt(
            agent_name=f"{self.name}_extract_params",
            system_prompt=self._extract_params_prompt,
            overwrite=True,
        )

    @observe(as_type="agent")
    async def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the sector trends agent to analyze funding trends.

        This method:
        1. Extracts sector name and time parameters from the query
        2. Uses semantic_search_organizations to find companies in the sector
        3. Aggregates funding trends by time period
        4. Calculates funding velocity metrics
        5. Identifies funding patterns
        6. Formats the results into a natural language response

        Args:
            context: The agent context containing the user query and metadata.

        Returns:
            AgentOutput containing:
            - content: Natural language response with trend analysis
            - status: SUCCESS if query processed successfully
            - metadata: Information about the analysis and results
            - tool_calls: List of tool calls made during execution
        """
        try:
            logger.info(f"Sector trends agent processing query: {context.query[:100]}")

            tool_calls = []

            # Step 1: Extract parameters from query
            params = await self._extract_parameters(context)
            sector_name = params.get("sector_name")
            time_period_start = params.get("time_period_start")
            time_period_end = params.get("time_period_end")
            granularity = params.get("granularity", "quarterly")
            min_funding_amount = params.get("min_funding_amount")

            if not sector_name:
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error="Could not identify a sector/industry name from the query. Please specify a sector (e.g., 'AI companies', 'fintech', 'healthcare startups').",
                )

            # Step 2: Find companies in the sector using semantic search
            logger.info(f"Searching for companies in sector: {sector_name}")
            semantic_output = await semantic_search_organizations(
                text=sector_name,
                top_k=50,  # Get more companies for trend analysis
            )

            if not semantic_output.success:
                error_msg = semantic_output.error or "Failed to search for companies"
                logger.error(f"semantic_search_organizations failed: {error_msg}")
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"Failed to find companies in sector '{sector_name}': {error_msg}",
                )

            organizations = semantic_output.result or []
            if not organizations:
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"No companies found in sector '{sector_name}'. Try a different sector name.",
                )

            # Extract organization UUIDs
            org_uuids = [
                str(org.get("org_uuid")) for org in organizations
                if org.get("org_uuid")
            ]

            if not org_uuids:
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error="Found companies but could not extract organization UUIDs.",
                )

            logger.info(f"Found {len(org_uuids)} companies in sector '{sector_name}'")

            tool_calls.append({
                "name": "semantic_search_organizations",
                "parameters": {"text": sector_name, "top_k": 50},
                "result": {
                    "num_results": len(organizations),
                    "execution_time_ms": semantic_output.execution_time_ms,
                    "success": semantic_output.success,
                },
            })

            # Step 3: Aggregate funding trends
            logger.info(f"Aggregating funding trends for {len(org_uuids)} companies")
            aggregate_params = {
                "org_uuids": org_uuids,
                "time_period_start": time_period_start,
                "time_period_end": time_period_end,
                "granularity": granularity,
            }
            if min_funding_amount:
                aggregate_params["min_funding_amount"] = min_funding_amount

            trends_output = await aggregate_funding_trends(**aggregate_params)

            if not trends_output.success:
                error_msg = trends_output.error or "Failed to aggregate funding trends"
                logger.error(f"aggregate_funding_trends failed: {error_msg}")
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"Failed to aggregate funding trends: {error_msg}",
                )

            trends_data = trends_output.result or {}
            trend_data_list = trends_data.get("trend_data", [])

            if not trend_data_list:
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"No funding data found for companies in sector '{sector_name}' for the specified time period.",
                )

            tool_calls.append({
                "name": "aggregate_funding_trends",
                "parameters": aggregate_params,
                "result": {
                    "num_periods": len(trend_data_list),
                    "execution_time_ms": trends_output.execution_time_ms,
                    "success": trends_output.success,
                },
            })

            # Step 4: Calculate funding velocity
            logger.info("Calculating funding velocity metrics")
            velocity_output = await calculate_funding_velocity(
                trend_data=trend_data_list,
                moving_average_periods=3,
                calculate_cagr=True,
            )

            velocity_data = {}
            if velocity_output.success:
                velocity_data = velocity_output.result or {}
                tool_calls.append({
                    "name": "calculate_funding_velocity",
                    "parameters": {
                        "trend_data": f"[{len(trend_data_list)} periods]",
                        "moving_average_periods": 3,
                        "calculate_cagr": True,
                    },
                    "result": {
                        "execution_time_ms": velocity_output.execution_time_ms,
                        "success": velocity_output.success,
                    },
                })
            else:
                logger.warning(f"calculate_funding_velocity failed: {velocity_output.error}")

            # Step 5: Identify funding patterns
            logger.info("Identifying funding patterns")
            patterns_output = await identify_funding_patterns(
                trend_data=trend_data_list,
                granularity=granularity,
                anomaly_threshold=2.0,
                detect_seasonality=True,
                min_periods_for_cycles=8,
            )

            patterns_data = {}
            if patterns_output.success:
                patterns_data = patterns_output.result or {}
                tool_calls.append({
                    "name": "identify_funding_patterns",
                    "parameters": {
                        "trend_data": f"[{len(trend_data_list)} periods]",
                        "granularity": granularity,
                        "anomaly_threshold": 2.0,
                        "detect_seasonality": True,
                    },
                    "result": {
                        "execution_time_ms": patterns_output.execution_time_ms,
                        "success": patterns_output.success,
                    },
                })
            else:
                logger.warning(f"identify_funding_patterns failed: {patterns_output.error}")

            # Step 6: Format response
            response_content = await self._format_response(
                context,
                sector_name,
                trends_data,
                velocity_data,
                patterns_data,
                granularity,
            )

            logger.info("Sector trends agent completed successfully")

            # Return AgentOutput
            return create_agent_output(
                content=response_content,
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.SUCCESS,
                tool_calls=tool_calls,
                metadata={
                    "query": context.query,
                    "sector_name": sector_name,
                    "num_companies": len(org_uuids),
                    "num_periods": len(trend_data_list),
                    "time_period": {
                        "start": time_period_start,
                        "end": time_period_end,
                    },
                    "granularity": granularity,
                },
            )

        except Exception as e:
            error_msg = f"Sector trends agent failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_agent_output(
                content="",
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.ERROR,
                error=error_msg,
            )

    async def _extract_parameters(self, context: AgentContext) -> Dict[str, Any]:
        """Extract parameters from user query using LLM function calling.

        Args:
            context: The agent context containing the user query.

        Returns:
            Dictionary of extracted parameters.
        """
        # Use pre-extracted metadata when available
        extracted = context.get_metadata("extracted_entities", {})
        params: Dict[str, Any] = {}

        sectors = extracted.get("sectors", [])
        if sectors:
            params["sector_name"] = sectors[0]

        time_period = extracted.get("time_period", {})
        if time_period.get("start"):
            params["time_period_start"] = time_period["start"]
        if time_period.get("end"):
            params["time_period_end"] = time_period["end"]
        if time_period.get("granularity"):
            params["granularity"] = time_period["granularity"]

        amounts = extracted.get("amounts", {})
        if amounts.get("fundraise_min") is not None:
            params["min_funding_amount"] = amounts["fundraise_min"]

        if params:
            if "granularity" not in params:
                params["granularity"] = "quarterly"
            return params

        extract_params_tool = {
            "type": "function",
            "function": {
                "name": "extract_sector_trend_params",
                "description": "Extract sector name, time period, granularity, and other parameters from a query about sector funding trends.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector_name": {
                            "type": "string",
                            "description": "Sector or industry name (e.g., 'AI companies', 'fintech', 'healthcare startups'). Required.",
                        },
                        "time_period_start": {
                            "type": "string",
                            "description": "Start date for trend analysis in ISO format (e.g., '2022-01-01T00:00:00'). Defaults to 2 years ago if not specified.",
                        },
                        "time_period_end": {
                            "type": "string",
                            "description": "End date for trend analysis in ISO format (e.g., '2024-12-31T23:59:59'). Defaults to today if not specified.",
                        },
                        "granularity": {
                            "type": "string",
                            "description": "Time granularity: 'monthly', 'quarterly', or 'yearly'. Default: 'quarterly'",
                            "enum": ["monthly", "quarterly", "yearly"],
                        },
                        "min_funding_amount": {
                            "type": "integer",
                            "description": "Minimum funding amount in USD to include (filters out small rounds). Optional.",
                        },
                    },
                    "required": ["sector_name"],
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

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Extract the sector name, time period, granularity, and other parameters from this query about sector funding trends."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=False),
        )

        try:
            result = await generate_llm_function_response(
                prompt=user_prompt,
                tools=[extract_params_tool],
                system_prompt=system_prompt,
                model="gpt-4.1-mini",
                temperature=0.3,
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_sector_trend_params"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "extract_sector_trend_params":
                    params = result["arguments"]
                    # Set defaults for time period if not provided
                    if not params.get("time_period_start"):
                        # Default to 2 years ago
                        two_years_ago = datetime.now() - timedelta(days=730)
                        params["time_period_start"] = two_years_ago.strftime("%Y-%m-%dT00:00:00")
                    if not params.get("time_period_end"):
                        # Default to today
                        params["time_period_end"] = datetime.now().strftime("%Y-%m-%dT23:59:59")
                    if not params.get("granularity"):
                        params["granularity"] = "quarterly"
                    # Filter out None values
                    return {k: v for k, v in params.items() if v is not None}
                else:
                    logger.warning(
                        f"Unexpected function call: {result.get('function_name')}, using defaults"
                    )
            else:
                logger.warning(
                    f"LLM did not make expected function call. Got: {type(result)}"
                )
        except Exception as e:
            logger.warning(
                f"LLM parameter extraction failed: {e}, using defaults", exc_info=True
            )

        # Return defaults if extraction failed
        two_years_ago = datetime.now() - timedelta(days=730)
        return {
            "sector_name": None,  # Will cause error if not found
            "time_period_start": two_years_ago.strftime("%Y-%m-%dT00:00:00"),
            "time_period_end": datetime.now().strftime("%Y-%m-%dT23:59:59"),
            "granularity": "quarterly",
        }

    async def _format_response(
        self,
        context: AgentContext,
        sector_name: str,
        trends_data: Dict[str, Any],
        velocity_data: Dict[str, Any],
        patterns_data: Dict[str, Any],
        granularity: str,
    ) -> AgentInsight:
        """Generate domain insight from sector trend data using LLM.

        Args:
            context: The agent context.
            sector_name: Name of the sector analyzed.
            trends_data: Aggregated funding trends data.
            velocity_data: Funding velocity metrics.
            patterns_data: Identified funding patterns.
            granularity: Time granularity used.

        Returns:
            AgentInsight object with domain interpretation and reasoning.
        """
        generate_insight_tool = {
            "type": "function",
            "function": {
                "name": "generate_insight",
                "description": "Generate domain insight from sector funding trend data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Human-readable insight summary answering the user's query about sector funding trends",
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Bullet points of key findings about funding trends, velocity, and patterns",
                        },
                        "evidence": {
                            "type": "object",
                            "description": "Supporting data from tool calls (trends, velocity, patterns)",
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
        base_prompt = """You are a sector funding trends analysis expert. Analyze the funding trend data and generate insights that directly answer the user's query.

Your task:
- Summarize overall funding trends in the sector
- Identify growth patterns and momentum (accelerating, decelerating, stable)
- Highlight peak periods and troughs
- Identify seasonal patterns or cycles if present
- Compare funding velocity across time periods
- State uncertainty where data is limited
- Directly answer the user's query in the summary

Focus on actionable insights for finance professionals: investment timing, market sentiment, competitive intelligence, risk assessment."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(
                add_temporal_context=False, add_markdown_instructions=False
            ),
        )

        # Build user prompt with data
        user_prompt_content = f"""User Query: {context.query}

Sector: {sector_name}
Time Granularity: {granularity}

Funding Trends Data (JSON):
{json.dumps(trends_data, indent=2, default=str)}

Funding Velocity Metrics (JSON):
{json.dumps(velocity_data, indent=2, default=str)}

Funding Patterns (JSON):
{json.dumps(patterns_data, indent=2, default=str)}

Analyze this data and generate insights that directly answer the user's query about sector funding trends. Call the generate_insight function with your analysis."""

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
                return AgentInsight(
                    summary=f"Analyzed funding trends for {sector_name} but failed to generate detailed insight.",
                    confidence=0.0,
                )
        except Exception as e:
            logger.warning(
                f"Failed to generate insight from LLM: {e}", exc_info=True
            )
            # Fallback insight
            return AgentInsight(
                summary=f"Analyzed funding trends for {sector_name} but encountered an error generating insights.",
                confidence=0.0,
            )

