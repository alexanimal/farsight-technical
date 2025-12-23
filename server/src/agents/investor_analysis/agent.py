"""Investor analysis agent for analyzing investor portfolio performance.

This agent is specialized in:
- Analyzing investor portfolio performance and metrics
- Calculating exit rates, ROI, and time to exit
- Analyzing sector concentration in portfolios
- Identifying investment patterns (stages, velocity, timing)
- Comparing portfolio performance to market benchmarks
- Providing insights on investor track records
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
from src.core.agent_base import AgentBase
from src.core.agent_context import AgentContext
from src.core.agent_response import AgentInsight, AgentResponse, ResponseStatus
from src.prompts.prompt_manager import PromptOptions, get_prompt_manager
from src.tools.analyze_sector_concentration import analyze_sector_concentration
from src.tools.calculate_portfolio_metrics import calculate_portfolio_metrics
from src.tools.compare_to_market_benchmarks import compare_to_market_benchmarks
from src.tools.find_investor_portfolio import find_investor_portfolio
from src.tools.generate_llm_function_response import generate_llm_function_response
from src.tools.identify_investment_patterns import identify_investment_patterns

logger = logging.getLogger(__name__)


class InvestorAnalysisAgent(AgentBase):
    """Agent specialized in analyzing investor portfolio performance and patterns.

    This agent uses LLM reasoning to understand user queries about investor portfolios
    and calls specialized tools to analyze portfolio performance, sector concentration,
    investment patterns, and benchmark comparisons.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the investor analysis agent.

        Args:
            config_path: Path to the agent's YAML config file.
                If None, will attempt to find it automatically.
        """
        if config_path is None:
            # Default to investor_analysis_agent.yaml in configs/agents
            src_base = Path(__file__).parent.parent.parent.parent
            config_path = src_base / "configs" / "agents" / "investor_analysis_agent.yaml"

        super().__init__(config_path=config_path)

        # Store base prompts as instance variables for reuse
        # Base prompt for extracting investor name and time parameters
        self._extract_params_prompt = """You are analyzing a query about investor portfolio performance. Your job is to extract:
1. Investor name (e.g., "Sequoia Capital", "Andreessen Horowitz", "Accel Partners")
2. Time period (start date, end date) - convert relative terms like "last 5 years" to ISO dates
3. Whether to include only exited investments (if mentioned)

Rules:
- Only extract parameters that are explicitly mentioned or clearly implied
- For date ranges, convert relative terms (e.g., "last 5 years", "2020-2024") to ISO format dates
- Default time period: last 5 years if not specified
- Leave parameters as None/null if not mentioned
- Be conservative - only extract what you're confident about
- Investor name is required"""

        # Register prompts with prompt manager
        prompt_manager = get_prompt_manager()
        prompt_manager.register_agent_prompt(
            agent_name=f"{self.name}_extract_params",
            system_prompt=self._extract_params_prompt,
            overwrite=True,
        )

    @observe(as_type="agent")
    async def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the investor analysis agent to analyze portfolio performance.

        This method:
        1. Extracts investor name and time parameters from the query
        2. Uses find_investor_portfolio to discover portfolio companies
        3. Calculates portfolio performance metrics
        4. Analyzes sector concentration
        5. Identifies investment patterns
        6. Compares to market benchmarks
        7. Formats the results into a natural language response

        Args:
            context: The agent context containing the user query and metadata.

        Returns:
            AgentOutput containing:
            - content: Natural language response with portfolio analysis
            - status: SUCCESS if query processed successfully
            - metadata: Information about the analysis and results
            - tool_calls: List of tool calls made during execution
        """
        try:
            logger.info(f"Investor analysis agent processing query: {context.query[:100]}")

            tool_calls = []

            # Step 1: Extract parameters from query
            params = await self._extract_parameters(context)
            investor_name = params.get("investor_name")
            time_period_start = params.get("time_period_start")
            time_period_end = params.get("time_period_end")
            include_exits_only = params.get("include_exits_only", False)

            if not investor_name:
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error="Could not identify an investor name from the query. Please specify an investor (e.g., 'Sequoia Capital', 'Andreessen Horowitz').",
                )

            # Step 2: Find investor portfolio
            logger.info(f"Finding portfolio for investor: {investor_name}")
            portfolio_params = {
                "investor_name": investor_name,
            }
            if time_period_start:
                portfolio_params["time_period_start"] = time_period_start
            if time_period_end:
                portfolio_params["time_period_end"] = time_period_end

            portfolio_output = await find_investor_portfolio(**portfolio_params)

            if not portfolio_output.success:
                error_msg = portfolio_output.error or "Failed to find investor portfolio"
                logger.error(f"find_investor_portfolio failed: {error_msg}")
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"Failed to find portfolio for investor '{investor_name}': {error_msg}",
                )

            portfolio_data = portfolio_output.result or {}
            portfolio_companies = portfolio_data.get("portfolio_companies", [])

            if not portfolio_companies:
                return create_agent_output(
                    content="",
                    agent_name=self.name,
                    agent_category=self.category,
                    status=ResponseStatus.ERROR,
                    error=f"No portfolio companies found for investor '{investor_name}'. The investor may not exist in the database or may not have any investments in the specified time period.",
                )

            logger.info(f"Found {len(portfolio_companies)} portfolio companies for {investor_name}")

            tool_calls.append(
                {
                    "name": "find_investor_portfolio",
                    "parameters": portfolio_params,
                    "result": {
                        "num_companies": len(portfolio_companies),
                        "execution_time_ms": portfolio_output.execution_time_ms,
                        "success": portfolio_output.success,
                    },
                }
            )

            # Step 3: Calculate portfolio metrics
            logger.info("Calculating portfolio performance metrics")
            metrics_params = {
                "portfolio_companies": portfolio_companies,
            }
            if time_period_start:
                metrics_params["time_period_start"] = time_period_start
            if time_period_end:
                metrics_params["time_period_end"] = time_period_end
            if include_exits_only:
                metrics_params["include_exits_only"] = include_exits_only

            metrics_output = await calculate_portfolio_metrics(**metrics_params)

            portfolio_metrics: Dict[str, Any] = {}
            if metrics_output.success:
                portfolio_metrics = metrics_output.result or {}
                tool_calls.append(
                    {
                        "name": "calculate_portfolio_metrics",
                        "parameters": {
                            "portfolio_companies": f"[{len(portfolio_companies)} companies]",
                            "time_period_start": time_period_start,
                            "time_period_end": time_period_end,
                            "include_exits_only": include_exits_only,
                        },
                        "result": {
                            "execution_time_ms": metrics_output.execution_time_ms,
                            "success": metrics_output.success,
                        },
                    }
                )
            else:
                logger.warning(f"calculate_portfolio_metrics failed: {metrics_output.error}")

            # Step 4: Analyze sector concentration
            logger.info("Analyzing sector concentration")
            sector_output = await analyze_sector_concentration(
                portfolio_companies=portfolio_companies,
            )

            sector_concentration: Dict[str, Any] = {}
            if sector_output.success:
                sector_concentration = sector_output.result or {}
                tool_calls.append(
                    {
                        "name": "analyze_sector_concentration",
                        "parameters": {
                            "portfolio_companies": f"[{len(portfolio_companies)} companies]",
                        },
                        "result": {
                            "execution_time_ms": sector_output.execution_time_ms,
                            "success": sector_output.success,
                        },
                    }
                )
            else:
                logger.warning(f"analyze_sector_concentration failed: {sector_output.error}")

            # Step 5: Identify investment patterns
            logger.info("Identifying investment patterns")
            patterns_params = {
                "portfolio_companies": portfolio_companies,
            }
            if time_period_start:
                patterns_params["time_period_start"] = time_period_start
            if time_period_end:
                patterns_params["time_period_end"] = time_period_end

            patterns_output = await identify_investment_patterns(**patterns_params)

            investment_patterns: Dict[str, Any] = {}
            if patterns_output.success:
                investment_patterns = patterns_output.result or {}
                tool_calls.append(
                    {
                        "name": "identify_investment_patterns",
                        "parameters": {
                            "portfolio_companies": f"[{len(portfolio_companies)} companies]",
                            "time_period_start": time_period_start,
                            "time_period_end": time_period_end,
                        },
                        "result": {
                            "execution_time_ms": patterns_output.execution_time_ms,
                            "success": patterns_output.success,
                        },
                    }
                )
            else:
                logger.warning(f"identify_investment_patterns failed: {patterns_output.error}")

            # Step 6: Compare to market benchmarks
            logger.info("Comparing to market benchmarks")
            benchmark_output = await compare_to_market_benchmarks(
                portfolio_metrics=portfolio_metrics,
                time_period_start=time_period_start,
                time_period_end=time_period_end,
            )

            benchmark_comparison: Dict[str, Any] = {}
            if benchmark_output.success:
                benchmark_comparison = benchmark_output.result or {}
                tool_calls.append(
                    {
                        "name": "compare_to_market_benchmarks",
                        "parameters": {
                            "portfolio_metrics": "portfolio_metrics object",
                            "time_period_start": time_period_start,
                            "time_period_end": time_period_end,
                        },
                        "result": {
                            "execution_time_ms": benchmark_output.execution_time_ms,
                            "success": benchmark_output.success,
                        },
                    }
                )
            else:
                logger.warning(f"compare_to_market_benchmarks failed: {benchmark_output.error}")

            # Step 7: Format response
            response_content = await self._format_response(
                context,
                investor_name,
                portfolio_data,
                portfolio_metrics,
                sector_concentration,
                investment_patterns,
                benchmark_comparison,
            )

            logger.info("Investor analysis agent completed successfully")

            # Return AgentOutput
            return create_agent_output(
                content=response_content,
                agent_name=self.name,
                agent_category=self.category,
                status=ResponseStatus.SUCCESS,
                tool_calls=tool_calls,
                metadata={
                    "query": context.query,
                    "investor_name": investor_name,
                    "num_companies": len(portfolio_companies),
                    "time_period": {
                        "start": time_period_start,
                        "end": time_period_end,
                    },
                },
            )

        except Exception as e:
            error_msg = f"Investor analysis agent failed to process query: {str(e)}"
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

        # Handle investors structure: can be dict with "names" key or list (defensive coding)
        investors_data = extracted.get("investors", {})
        if isinstance(investors_data, dict):
            investor_names = investors_data.get("names", [])
        elif isinstance(investors_data, list):
            investor_names = investors_data
        else:
            investor_names = []

        if investor_names:
            params["investor_name"] = investor_names[0]

        time_period = extracted.get("time_period", {})
        if isinstance(time_period, dict):
            if time_period.get("start"):
                params["time_period_start"] = time_period["start"]
            if time_period.get("end"):
                params["time_period_end"] = time_period["end"]

        if params and params.get("investor_name"):
            # Set defaults for time period if not provided
            if not params.get("time_period_start"):
                five_years_ago = datetime.now() - timedelta(days=1825)
                params["time_period_start"] = five_years_ago.strftime("%Y-%m-%dT00:00:00")
            if not params.get("time_period_end"):
                params["time_period_end"] = datetime.now().strftime("%Y-%m-%dT23:59:59")
            return params

        extract_params_tool = {
            "type": "function",
            "function": {
                "name": "extract_investor_analysis_params",
                "description": "Extract investor name, time period, and other parameters from a query about investor portfolio performance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "investor_name": {
                            "type": "string",
                            "description": "Name of the investor to analyze (e.g., 'Sequoia Capital', 'Andreessen Horowitz', 'Accel Partners'). Required.",
                        },
                        "time_period_start": {
                            "type": "string",
                            "description": "Start date for analysis period in ISO format (e.g., '2018-01-01T00:00:00'). Defaults to 5 years ago if not specified.",
                        },
                        "time_period_end": {
                            "type": "string",
                            "description": "End date for analysis period in ISO format (e.g., '2024-12-31T23:59:59'). Defaults to today if not specified.",
                        },
                        "include_exits_only": {
                            "type": "boolean",
                            "description": "Whether to only analyze exited investments. Default: false.",
                        },
                    },
                    "required": ["investor_name"],
                },
            },
        }

        # Build system prompt using prompt manager
        prompt_manager = get_prompt_manager()
        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=self._extract_params_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Build user prompt using prompt manager
        user_prompt_content = f"""User Query: {context.query}

Extract the investor name, time period, and other parameters from this query about investor portfolio performance."""

        user_prompt = prompt_manager.build_user_prompt(
            user_query=user_prompt_content,
            options=PromptOptions(add_temporal_context=True),
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
                    "function": {"name": "extract_investor_analysis_params"},
                },
            )

            if isinstance(result, dict) and "function_name" in result:
                if result["function_name"] == "extract_investor_analysis_params":
                    params = result["arguments"]
                    # Set defaults for time period if not provided
                    if not params.get("time_period_start"):
                        five_years_ago = datetime.now() - timedelta(days=1825)
                        params["time_period_start"] = five_years_ago.strftime("%Y-%m-%dT00:00:00")
                    if not params.get("time_period_end"):
                        params["time_period_end"] = datetime.now().strftime("%Y-%m-%dT23:59:59")
                    if "include_exits_only" not in params:
                        params["include_exits_only"] = False
                    # Filter out None values
                    return {k: v for k, v in params.items() if v is not None}
                else:
                    logger.warning(
                        f"Unexpected function call: {result.get('function_name')}, using defaults"
                    )
            else:
                logger.warning(f"LLM did not make expected function call. Got: {type(result)}")
        except Exception as e:
            logger.warning(f"LLM parameter extraction failed: {e}, using defaults", exc_info=True)

        # Return defaults if extraction failed
        five_years_ago = datetime.now() - timedelta(days=1825)
        return {
            "investor_name": None,  # Will cause error if not found
            "time_period_start": five_years_ago.strftime("%Y-%m-%dT00:00:00"),
            "time_period_end": datetime.now().strftime("%Y-%m-%dT23:59:59"),
            "include_exits_only": False,
        }

    async def _format_response(
        self,
        context: AgentContext,
        investor_name: str,
        portfolio_data: Dict[str, Any],
        portfolio_metrics: Dict[str, Any],
        sector_concentration: Dict[str, Any],
        investment_patterns: Dict[str, Any],
        benchmark_comparison: Dict[str, Any],
    ) -> AgentInsight:
        """Generate domain insight from investor portfolio data using LLM.

        Args:
            context: The agent context.
            investor_name: Name of the investor analyzed.
            portfolio_data: Portfolio data from find_investor_portfolio.
            portfolio_metrics: Portfolio metrics from calculate_portfolio_metrics.
            sector_concentration: Sector concentration analysis.
            investment_patterns: Investment pattern insights.
            benchmark_comparison: Market benchmark comparison.

        Returns:
            AgentInsight object with domain interpretation and reasoning.
        """
        generate_insight_tool = {
            "type": "function",
            "function": {
                "name": "generate_insight",
                "description": "Generate domain insight from investor portfolio analysis data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Human-readable insight summary answering the user's query about investor portfolio performance",
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Bullet points of key findings about portfolio performance, sector concentration, investment patterns, and benchmark comparisons",
                        },
                        "evidence": {
                            "type": "object",
                            "description": "Supporting data from tool calls (metrics, sectors, patterns, benchmarks)",
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
        base_prompt = """You are an investor portfolio analysis expert. Analyze the portfolio data and generate insights that directly answer the user's query.

Your task:
- Summarize overall portfolio performance (exit rate, ROI, time to exit)
- Analyze sector concentration and diversification
- Identify investment patterns (preferred stages, velocity, timing)
- Compare performance to market benchmarks
- Highlight strengths and weaknesses
- State uncertainty where data is limited
- Directly answer the user's query in the summary

Focus on actionable insights for finance professionals: due diligence, benchmarking, strategy insights, relationship building."""

        system_prompt = prompt_manager.build_system_prompt(
            base_prompt=base_prompt,
            options=PromptOptions(add_temporal_context=True, add_markdown_instructions=True),
        )

        # Build user prompt with data
        user_prompt_content = f"""User Query: {context.query}

Investor: {investor_name}

Portfolio Data (JSON):
{json.dumps(portfolio_data, indent=2, default=str)}

Portfolio Metrics (JSON):
{json.dumps(portfolio_metrics, indent=2, default=str)}

Sector Concentration (JSON):
{json.dumps(sector_concentration, indent=2, default=str)}

Investment Patterns (JSON):
{json.dumps(investment_patterns, indent=2, default=str)}

Benchmark Comparison (JSON):
{json.dumps(benchmark_comparison, indent=2, default=str)}

Analyze this data and generate insights that directly answer the user's query about investor portfolio performance. Call the generate_insight function with your analysis."""

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
                return AgentInsight(
                    summary=f"Analyzed portfolio for {investor_name} but failed to generate detailed insight.",
                    confidence=0.0,
                )
        except Exception as e:
            logger.warning(f"Failed to generate insight from LLM: {e}", exc_info=True)
            # Fallback insight
            return AgentInsight(
                summary=f"Analyzed portfolio for {investor_name} but encountered an error generating insights.",
                confidence=0.0,
            )
